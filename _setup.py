#!/usr/bin/env python3
"""
_setup.py  -  find your GPU, pick a PyTorch build, install what is missing.

Shared by every launcher: `imgdedup.sh setup`, `Check-Image-Tools.bat`, and
the per-stage .bat files when a probe fails. One implementation, because
this project's worst bug class was three launchers drifting apart - and code
that CHANGES the user's environment is the last place to repeat that.

It never installs anything silently. Every install prints the exact command
first and waits for a yes.

Only the Embed stage cares about the GPU. Collect is disk-bound and Analyze
is numpy, so "CPU" there is correct, not a compromise.

    python _setup.py              interactive: report, then offer to fix
    python _setup.py --check      report only, exit 1 if something is missing
    python _setup.py --yes        assume yes (for scripted use)
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

IS_WIN = os.name == 'nt'
IS_MAC = sys.platform == 'darwin'

# One PCI vendor table serves both platforms: Windows PNPDeviceID carries the
# same IDs Linux exposes in sysfs.
PCI_VENDORS = {0x1002: 'AMD', 0x10DE: 'NVIDIA', 0x8086: 'Intel'}

# Fallbacks only. The real versions are discovered from the PyTorch index at
# run time (see newest_index) because they move: the stable ROCm index went
# 6.4 -> 7.0 -> 7.1 -> 7.2 within a few releases.
FALLBACK_CUDA = 'cu128'
FALLBACK_ROCM = 'rocm7.2'

# AMD's ROCm-on-Windows wheels. Pinned URLs, not an index, and cp312 ONLY -
# a full-ABI wheel that no other Python can load (see the caveat below).
ROCM_WIN_REL = '7.2.1'
ROCM_WIN_DRIVER = '26.2.2'
ROCM_WIN_BASE = 'https://repo.radeon.com/rocm/windows/rocm-rel-' + ROCM_WIN_REL
ROCM_WIN_WHEELS = [
    ROCM_WIN_BASE + '/rocm_sdk_core-%s-py3-none-win_amd64.whl' % ROCM_WIN_REL,
    ROCM_WIN_BASE + '/rocm_sdk_devel-%s-py3-none-win_amd64.whl' % ROCM_WIN_REL,
    ROCM_WIN_BASE + '/rocm_sdk_libraries_custom-%s-py3-none-win_amd64.whl' % ROCM_WIN_REL,
    ROCM_WIN_BASE + '/rocm-%s.tar.gz' % ROCM_WIN_REL,
]
ROCM_WIN_TORCH = [
    ROCM_WIN_BASE + '/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl',
]


# ----------------------------------------------------------- GPU detection --
def detect_gpus():
    """[(vendor, name)] for every graphics adapter, whether or not a compute
    driver is installed. Deliberately does NOT use rocm-smi / nvidia-smi to
    find cards: those only exist once the toolkit is already installed, which
    is precisely the case we are trying to fix. They are used separately, to
    tell 'card present' from 'driver usable'."""
    out = []
    if IS_WIN:
        try:
            p = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 'Get-CimInstance Win32_VideoController | '
                 'Select-Object Name,PNPDeviceID | ConvertTo-Json -Compress'],
                capture_output=True, text=True, timeout=40)
            data = json.loads(p.stdout or 'null')
            if isinstance(data, dict):
                data = [data]
            for d in (data or []):
                pnp = (d.get('PNPDeviceID') or '').upper()
                m = re.search(r'VEN_([0-9A-F]{4})', pnp)
                vid = int(m.group(1), 16) if m else 0
                out.append((PCI_VENDORS.get(vid, 'other'),
                            d.get('Name') or 'unknown'))
        except Exception:
            pass
    elif not IS_MAC:
        # sysfs PCI vendor ids - no lspci, no root, no vendor tooling.
        # card*-* entries are connectors, not devices.
        for card in sorted(glob.glob('/sys/class/drm/card*')):
            if re.search(r'card\d+-', os.path.basename(card)):
                continue
            try:
                with open(os.path.join(card, 'device', 'vendor')) as f:
                    vid = int(f.read().strip(), 16)
            except Exception:
                continue
            name = PCI_VENDORS.get(vid, 'other')
            model = ''
            for attr in ('product_name', 'device'):
                try:
                    with open(os.path.join(card, 'device', attr)) as f:
                        model = f.read().strip()
                        break
                except Exception:
                    pass
            out.append((name, model or os.path.basename(card)))
    return out


def driver_ready(vendor):
    """Is a usable compute driver present for this vendor (not just a card)?"""
    probes = {'NVIDIA': ['nvidia-smi'], 'AMD': ['rocm-smi', 'rocminfo']}
    for exe in probes.get(vendor, []):
        try:
            if subprocess.run([exe], capture_output=True,
                              timeout=20).returncode == 0:
                return True
        except Exception:
            continue
    if vendor == 'NVIDIA' and not IS_WIN:
        return os.path.isdir('/proc/driver/nvidia/gpus')
    return False


# -------------------------------------------------------- index discovery --
def _ver_key(s):
    """Sort ROCm/CUDA suffixes numerically. AMD's minor version passed 9, so
    'rocm7.14' is NEWER than 'rocm7.2' - float() and string compare both get
    this backwards, which would silently install an older stack."""
    return tuple(int(x) for x in re.findall(r'\d+', s))


def newest_index(prefix, timeout=15):
    """Newest .../whl/<prefix>N index actually published, or None offline.
    The listing is a plain PEP-503 anchor page."""
    try:
        from urllib.request import urlopen
        with urlopen('https://download.pytorch.org/whl/', timeout=timeout) as r:
            html = r.read().decode('utf-8', 'replace')
    except Exception:
        return None
    found = set(re.findall(r'>\s*(%s[\d.]+)\s*/?\s*<' % prefix, html))
    found |= set(re.findall(r'href="[^"]*?(%s[\d.]+)/' % prefix, html))
    return max(found, key=_ver_key) if found else None


# --------------------------------------------------------------- backends --
class Backend(object):
    def __init__(self, key, label, args, note='', blocked=''):
        self.key = key
        self.label = label
        self.args = args          # pip args for torch (transformers added later)
        self.note = note
        self.blocked = blocked    # non-empty = cannot be used here, and why

    @property
    def usable(self):
        return not self.blocked


def backends_for(vendors, online=True):
    """Every torch build worth offering on this machine, best first."""
    pv = sys.version_info[:2]
    out = []

    if 'NVIDIA' in vendors:
        idx = (newest_index('cu') if online else None) or FALLBACK_CUDA
        out.append(Backend(
            'cuda', 'NVIDIA GPU (CUDA)',
            ['torch', '--index-url', 'https://download.pytorch.org/whl/' + idx],
            note='needs a current NVIDIA driver'))

    if 'AMD' in vendors:
        if IS_WIN:
            # cp312-cp312 is a FULL-ABI wheel; no other Python can load it,
            # and --ignore-requires-python only skips the check, not the ABI.
            if pv == (3, 12):
                out.append(Backend(
                    'rocm-win', 'AMD GPU (ROCm %s for Windows)' % ROCM_WIN_REL,
                    ROCM_WIN_WHEELS + ROCM_WIN_TORCH,
                    note='needs the %s graphics driver' % ROCM_WIN_DRIVER))
            else:
                out.append(Backend(
                    'rocm-win', 'AMD GPU (ROCm for Windows)', [],
                    blocked=('AMD ships Windows wheels for Python 3.12 only '
                             '(you are on %d.%d). Install Python 3.12 '
                             'alongside, run this setup with it, then point '
                             'the Embed stage at it with IMGDEDUP_PYTHON - '
                             'the other stages can stay on %d.%d.'
                             % (pv[0], pv[1], pv[0], pv[1]))))
        elif IS_MAC:
            out.append(Backend('rocm', 'AMD GPU (ROCm)', [],
                               blocked='ROCm is Linux-only.'))
        else:
            idx = (newest_index('rocm') if online else None) or FALLBACK_ROCM
            out.append(Backend(
                'rocm', 'AMD GPU (ROCm, %s)' % idx,
                ['torch', '--index-url',
                 'https://download.pytorch.org/whl/' + idx],
                note='needs the amdgpu/ROCm kernel driver'))

    if 'Intel' in vendors and not IS_MAC:
        out.append(Backend(
            'xpu', 'Intel GPU (XPU)',
            ['torch', '--index-url', 'https://download.pytorch.org/whl/xpu'],
            note='Arc / recent iGPUs; the Embed stage uses it via torch.xpu'))

    if IS_MAC:
        out.append(Backend('mps', 'Apple GPU (Metal, built into stock torch)',
                           ['torch']))

    out.append(Backend(
        'cpu', 'CPU only (always works, slower)',
        ['torch', '--index-url', 'https://download.pytorch.org/whl/cpu']))
    return out


# ------------------------------------------------------- installed state --
PROBE = r'''
import json
o = {}
try:
    import PIL; o['pillow'] = getattr(PIL, '__version__', '?') if PIL.__file__ else 'EMPTY'
except Exception as e: o['pillow'] = 'ERR'
try:
    import numpy; o['numpy'] = numpy.__version__ if numpy.__file__ else 'EMPTY'
except Exception: o['numpy'] = 'ERR'
try:
    import cv2; o['opencv'] = getattr(cv2, '__version__', '?')
except Exception: o['opencv'] = 'ERR'
try:
    import transformers; o['transformers'] = transformers.__version__ if transformers.__file__ else 'EMPTY'
except Exception: o['transformers'] = 'ERR'
try:
    import torch
    o['torch'] = torch.__version__
    o['hip'] = getattr(torch.version, 'hip', None)
    o['cuda_build'] = getattr(torch.version, 'cuda', None)
    o['cuda_ok'] = bool(torch.cuda.is_available())
    try: o['xpu_ok'] = bool(torch.xpu.is_available())
    except Exception: o['xpu_ok'] = False
    try: o['mps_ok'] = bool(torch.backends.mps.is_available())
    except Exception: o['mps_ok'] = False
    o['gpu_name'] = torch.cuda.get_device_name(0) if o['cuda_ok'] else ''
except Exception:
    o['torch'] = 'ERR'
print('@@' + json.dumps(o))
'''


def installed_state(exe=None):
    exe = exe or sys.executable
    try:
        p = subprocess.run([exe, '-c', PROBE], capture_output=True,
                           text=True, timeout=180)
        for line in (p.stdout or '').splitlines():
            if line.startswith('@@'):
                return json.loads(line[2:])
    except Exception:
        pass
    return {}


def torch_flavour(st):
    """What KIND of torch is installed. HIP is checked FIRST: a ROCm build
    reports torch.cuda.is_available() == True (HIP reuses the cuda namespace)
    and torch.version.cuda is not a reliable discriminator - PyTorch's own
    collect_env.py overrides it inside the HIP branch."""
    if st.get('torch', 'ERR') == 'ERR':
        return 'none', 'not installed'
    if st.get('hip'):
        return 'rocm', 'ROCm/HIP %s' % st['hip']
    if st.get('cuda_build'):
        return 'cuda', 'CUDA %s' % st['cuda_build']
    if st.get('xpu_ok'):
        return 'xpu', 'Intel XPU'
    if st.get('mps_ok'):
        return 'mps', 'Apple Metal'
    return 'cpu', 'CPU-only build'


def missing_packages(st, want_embed=True):
    need = []
    for key, pkg in (('pillow', 'pillow'), ('numpy', 'numpy'),
                     ('opencv', 'opencv-python-headless')):
        if st.get(key, 'ERR') in ('ERR', 'EMPTY'):
            need.append(pkg)
    if want_embed:
        if st.get('transformers', 'ERR') in ('ERR', 'EMPTY'):
            need.append('transformers')
        if st.get('torch', 'ERR') == 'ERR':
            need.append('torch')
    return need


# ------------------------------------------------------------ installing --
def pip_base(exe=None):
    cmd = [exe or sys.executable, '-m', 'pip', 'install']
    # --user is invalid inside a venv, and pip errors out rather than
    # ignoring it. sys.prefix != sys.base_prefix is the venv signal.
    if sys.prefix == sys.base_prefix and not os.environ.get('VIRTUAL_ENV'):
        cmd.append('--user')
    return cmd


def show_and_run(cmd, assume_yes, what):
    print('')
    print('  About to install %s with:' % what)
    print('    ' + ' '.join(('"%s"' % c if ' ' in c else c) for c in cmd))
    if not assume_yes:
        try:
            ans = input('  Proceed? [y/N]: ').strip().lower()
        except EOFError:
            ans = ''
        if ans not in ('y', 'yes'):
            print('  Skipped.')
            return False
    print('')
    rc = subprocess.call(cmd)
    print('')
    if rc == 0:
        print('  OK.')
        return True
    print('  pip exited %d - nothing else was attempted.' % rc)
    return False


def main():
    ap = argparse.ArgumentParser(description='Set up the image toolkit.')
    ap.add_argument('--check', action='store_true', help='report only')
    ap.add_argument('--yes', action='store_true', help='assume yes')
    ap.add_argument('--offline', action='store_true',
                    help='skip index discovery, use pinned versions')
    args = ap.parse_args()

    print('')
    print('  Image toolkit setup')
    print('  ' + '-' * 62)
    print('  Python %d.%d.%d  (%s)'
          % (sys.version_info[:3] + (sys.executable,)))

    gpus = detect_gpus()
    vendors = set(v for v, _ in gpus)
    if gpus:
        for v, name in gpus:
            ready = driver_ready(v)
            tag = '' if v == 'other' else ('  [compute driver: %s]'
                                           % ('yes' if ready else 'not found'))
            print('  GPU: %-7s %s%s' % (v, name[:44], tag))
    else:
        print('  GPU: none detected')

    st = installed_state()
    flav, desc = torch_flavour(st)
    print('')
    print('  Installed now:')
    for key, label in (('pillow', 'pillow'), ('numpy', 'numpy'),
                       ('opencv', 'opencv'), ('transformers', 'transformers')):
        val = st.get(key, 'ERR')
        print('    %-14s %s' % (label, 'missing' if val in ('ERR', 'EMPTY')
                                else val))
    print('    %-14s %s' % ('torch', desc if flav != 'none' else 'missing'))
    if flav in ('cuda', 'rocm', 'xpu', 'mps'):
        live = st.get('cuda_ok') or st.get('xpu_ok') or st.get('mps_ok')
        print('    %-14s %s' % ('  -> GPU',
                                (st.get('gpu_name') or 'available') if live
                                else 'NOT usable (driver missing or too old)'))

    need = missing_packages(st)
    if not need:
        print('')
        print('  Everything the toolkit needs is present. Nothing to do.')
        return 0
    print('')
    print('  Missing: ' + ', '.join(need))
    if args.check:
        return 1

    plain = [p for p in need if p != 'torch']
    if plain:
        show_and_run(pip_base() + plain, args.yes, ', '.join(plain))

    if 'torch' not in need:
        return 0

    opts = backends_for(vendors, online=not args.offline)
    print('')
    print('  PyTorch build to install (only the Embed stage uses the GPU):')
    usable = [b for b in opts if b.usable]
    for i, b in enumerate(usable, 1):
        print('    %d) %s%s' % (i, b.label, '  - ' + b.note if b.note else ''))
    for b in opts:
        if not b.usable:
            print('    -) %s: %s' % (b.label, b.blocked))
    if args.yes:
        pick = usable[0]
    else:
        try:
            raw = input('  Choose [1-%d, default 1]: ' % len(usable)).strip()
        except EOFError:
            raw = ''
        try:
            pick = usable[int(raw) - 1] if raw else usable[0]
        except (ValueError, IndexError):
            print('  Not a listed choice - nothing installed.')
            return 1
    print('  -> %s' % pick.label)
    if not show_and_run(pip_base() + pick.args, args.yes, 'torch (%s)' % pick.key):
        return 1
    if st.get('transformers', 'ERR') in ('ERR', 'EMPTY') \
            and 'transformers' not in plain:
        show_and_run(pip_base() + ['transformers'], args.yes, 'transformers')

    after = installed_state()
    flav2, desc2 = torch_flavour(after)
    print('')
    print('  Now installed: torch %s' % desc2)
    if flav2 in ('cuda', 'rocm') and not after.get('cuda_ok'):
        print('  NOTE: the build is right but no GPU is visible yet - that is')
        print('        a driver matter, not a package one.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('')
        print('  Cancelled. Nothing was installed.')
        sys.exit(130)
