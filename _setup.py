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
    python _setup.py --offline    skip index discovery, use the pinned suffixes
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
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
FALLBACK_CUDA = 'cu132'

# PyTorch's CUDA 12.8 and newer builds carry kernels for Turing (compute
# 7.5) and later only, and CUDA 13 cannot compile for anything older. On a
# GTX 9xx or 10xx, a Titan X/Xp/V or a Quadro M/P, torch.cuda.is_available()
# still says True and the first kernel then fails with "no kernel image is
# available". Such a card gets the newest index at or below CUDA 12.6.
LEGACY_CUDA_MAX = (126,)
FALLBACK_CUDA_LEGACY = 'cu126'
_LEGACY_NVIDIA = re.compile(
    r'GTX\s*(9\d\d|10\d\d)|\bGT\s*10\d\d|TITAN\s*(X|XP|V)\b|'
    r'QUADRO\s*[KMP]\d|TESLA\s*[KMPV]\d|\bMX\s*[1-3]\d\d', re.I)
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
                capture_output=True, text=True, errors='replace', timeout=40)
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


def nvidia_compute_caps():
    """Compute capabilities nvidia-smi reports, e.g. [(6, 1)]; [] when it
    is not installed or too old to know the field."""
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap',
                            '--format=csv,noheader'], capture_output=True,
                           text=True, errors='replace', timeout=20)
    except Exception:
        return []
    caps = []
    if p.returncode == 0:
        for line in (p.stdout or '').splitlines():
            m = re.match(r'\s*(\d+)\.(\d+)\s*$', line)
            if m:
                caps.append((int(m.group(1)), int(m.group(2))))
    return caps


def nvidia_is_legacy(names):
    """True when an NVIDIA card here predates Turing (compute 7.5).
    nvidia-smi's answer wins; the model name decides before a driver is
    installed, which is exactly when setup runs."""
    caps = nvidia_compute_caps()
    if caps:
        return min(caps) < (7, 5)
    return any(_LEGACY_NVIDIA.search(n or '') for n in names)


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
    # rocm-smi and rocminfo are not part of AMD's Windows driver, and Intel
    # had no probe at all, so both always read "compute driver: not found" -
    # a working Arc included. Each vendor's compute runtime is looked for
    # where its driver puts it instead.
    sysdir = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32')
    if vendor == 'AMD':
        if IS_WIN:
            return bool(glob.glob(os.path.join(sysdir, 'amdhip64*.dll')))
        return os.path.exists('/dev/kfd')          # the ROCm kernel driver
    if vendor == 'Intel':
        if IS_WIN:
            return os.path.exists(os.path.join(sysdir, 'ze_loader.dll'))
        try:
            import ctypes.util
            return bool(ctypes.util.find_library('ze_loader'))   # Level Zero
        except Exception:
            return False
    return False


# -------------------------------------------------------- index discovery --
def _ver_key(s):
    """Sort ROCm/CUDA suffixes numerically. AMD's minor version passed 9, so
    'rocm7.14' is NEWER than 'rocm7.2' - float() and string compare both get
    this backwards, which would silently install an older stack."""
    return tuple(int(x) for x in re.findall(r'\d+', s))


def _wheel_tags():
    """(python tag, platform substring) a torch wheel needs to install here."""
    py = 'cp%d%d' % sys.version_info[:2]
    if IS_WIN:
        plat = 'win_amd64'
    elif IS_MAC:
        plat = 'macosx'
    else:
        plat = 'aarch64' if 'aarch64' in platform.machine() else 'x86_64'
    return py, plat


# transformers 5 switches torch off below this ("PyTorch >= 2.4 is
# required"), so an older wheel installs cleanly and then does nothing.
MIN_TORCH = (2, 4)


def _index_best_torch(idx, timeout=15):
    """The newest torch version .../whl/<idx>/torch/ offers for this Python
    and platform, as a tuple; None when it offers none or cannot be read."""
    py, plat = _wheel_tags()
    try:
        from urllib.request import urlopen
        url = 'https://download.pytorch.org/whl/%s/torch/' % idx
        with urlopen(url, timeout=timeout) as r:
            html = r.read().decode('utf-8', 'replace')
    except Exception:
        return None
    pat = r'torch-(\d+(?:\.\d+)*)[^"<]*?-%s-%s-[^"<]*%s[^"<]*\.whl' % (py, py, plat)
    vers = [tuple(int(x) for x in m.group(1).split('.'))
            for m in re.finditer(pat, html)]
    return max(vers) if vers else None


def newest_index(prefix, timeout=15, max_key=None):
    """The .../whl/<prefix>N index to install torch from: of the newest few,
    the one whose best wheel for this Python and platform is the newest
    torch, and at least MIN_TORCH. None when the listing cannot be reached
    (offline); '' when it can but no index has a usable wheel here.

    A directory existing proves nothing - cu134 was published holding only
    torch 2.0 aarch64 wheels - and "has a wheel" was not enough either:
    that 2.0.1 wheel still won on Linux aarch64 with Python 3.10 and 3.11,
    and transformers then refused it. The listing is a plain PEP-503 anchor
    page; only a few candidates are checked, because each check downloads
    that index's wheel list."""
    try:
        from urllib.request import urlopen
        with urlopen('https://download.pytorch.org/whl/', timeout=timeout) as r:
            html = r.read().decode('utf-8', 'replace')
    except Exception:
        return None
    found = set(re.findall(r'>\s*(%s[\d.]+)\s*/?\s*<' % prefix, html))
    found |= set(re.findall(r'href="[^"]*?(%s[\d.]+)/' % prefix, html))
    if max_key is not None:
        found = set(f for f in found if _ver_key(f) <= max_key)
    best = None
    for idx in sorted(found, key=_ver_key, reverse=True)[:4]:
        v = _index_best_torch(idx, timeout)
        if v is None or v < MIN_TORCH:
            continue
        if best is None or v > best[0]:       # ties keep the newer index
            best = (v, idx)
    return best[1] if best else ''


def no_wheel_note():
    pv = sys.version_info[:2]
    return ('the newest indexes on download.pytorch.org have no PyTorch %d.%d+ '
            'build for Python %d.%d on this platform%s'
            % (MIN_TORCH[0], MIN_TORCH[1], pv[0], pv[1],
                                    ' (current PyTorch releases need Python 3.10'
                                    ' or newer)'
                                    if pv < (3, 10) else ''))


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


def backends_for(vendors, online=True, names=()):
    """Every torch build worth offering on this machine, best first."""
    pv = sys.version_info[:2]
    out = []

    if 'NVIDIA' in vendors:
        legacy = nvidia_is_legacy(names)
        if legacy:
            idx = newest_index('cu', max_key=LEGACY_CUDA_MAX) if online else None
            fallback = FALLBACK_CUDA_LEGACY
        else:
            idx = newest_index('cu') if online else None
            fallback = FALLBACK_CUDA
        if idx == '':
            # Reached, and nothing fits: the fallback would fail the same way
            # in pip, as a bare "No matching distribution".
            out.append(Backend('cuda', 'NVIDIA GPU (CUDA)', [],
                               blocked=no_wheel_note()))
        else:
            idx = idx or fallback
            label = ('NVIDIA GPU (CUDA %s build - this card predates Turing)' % (
                '.'.join((idx[2:-1], idx[-1])) if idx[2:].isdigit() else idx)
                if legacy else 'NVIDIA GPU (CUDA)')
            out.append(Backend(
                'cuda', label,
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
            idx = newest_index('rocm') if online else None
            if idx == '':
                out.append(Backend('rocm', 'AMD GPU (ROCm)', [],
                                   blocked=no_wheel_note()))
            else:
                idx = idx or FALLBACK_ROCM
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
    # the BUILD, not the device: an XPU wheel without its driver is still an
    # XPU wheel, and was reported as "CPU-only" and sent round a reinstall
    o['xpu_build'] = bool(getattr(torch.version, 'xpu', None))
    try: o['mps_built'] = bool(torch.backends.mps.is_built())
    except Exception: o['mps_built'] = False
    o['gpu_name'] = torch.cuda.get_device_name(0) if o['cuda_ok'] else ''
    if o['cuda_ok'] and not o['hip']:
        try:
            cap = tuple(torch.cuda.get_device_capability(0))
            o['cuda_cc'] = '%d.%d' % cap
            fits = False
            for a in torch.cuda.get_arch_list():
                kind, _, num = a.partition('_')
                num = ''.join(ch for ch in num if ch.isdigit())
                if len(num) < 2:
                    continue
                mj, mn = int(num[:-1]), int(num[-1])
                if (kind == 'sm' and mj == cap[0] and mn <= cap[1]) or \
                        (kind == 'compute' and (mj, mn) <= cap):
                    fits = True
            o['cuda_fits'] = fits
        except Exception:
            pass
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
    if st.get('xpu_build') or st.get('xpu_ok'):
        return 'xpu', 'Intel XPU'
    if st.get('mps_built') or st.get('mps_ok'):
        return 'mps', 'Apple Metal'
    return 'cpu', 'CPU-only build'


def wrong_build(flav, st, vendors):
    """Why the installed torch can never use this machine's GPU, or ''.

    A CUDA build on an AMD-only machine read as a driver problem, and setup
    then said "Nothing to do" because every package was present; the doctor
    sent a CPU-only build round "uninstall, then setup" with no way back in.
    What is wrong is the build, so setup offers to replace it."""
    gpu = set(vendors) & {'NVIDIA', 'AMD', 'Intel'}
    if not gpu:
        return ''
    made_for = {'cuda': 'NVIDIA', 'rocm': 'AMD', 'xpu': 'Intel'}.get(flav)
    if made_for and made_for not in gpu:
        return ('made for %s GPUs, and the GPU here is %s.'
                % (made_for, ', '.join(sorted(gpu))))
    if flav == 'cuda' and st.get('cuda_fits') is False:
        return ('with no kernels for this card (compute %s); PyTorch\'s CUDA '
                '12.8 and newer builds start at 7.5.' % st.get('cuda_cc', '?'))
    if flav == 'cpu' and any(b.usable and b.key != 'cpu'
                             for b in backends_for(gpu, online=False)):
        return 'which can never use the %s GPU here.' % ', '.join(sorted(gpu))
    return ''


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


# -------------------------------------------------- externally managed --
# PEP 668. A distro-managed Python carries a marker file, and pip REFUSES to
# install into it:
#     error: externally-managed-environment
# The trap is that --user does NOT exempt you - pip rejects that too, which
# is exactly how this script used to dead-end on Arch: it appended --user
# because it saw no venv, pip refused, and setup printed "pip exited 1"
# with no way forward. Arch, Debian 12+, Ubuntu 23.04+, Fedora 38+ and
# Homebrew all ship the marker. A venv is exempt by design, so that is the
# route offered first.

def in_venv():
    # Deliberately pip's own test (running_under_virtualenv), because pip is
    # the thing that will accept or refuse. $VIRTUAL_ENV is NOT part of it
    # and must not be added: an activated venv exports that variable, but the
    # interpreter actually running can still be the system one, and then pip
    # refuses while we would have believed ourselves exempt.
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix)


def em_marker():
    """Path to the PEP 668 marker, or None if pip may install here."""
    if in_venv():
        return None
    try:
        import sysconfig
        p = os.path.join(sysconfig.get_path('stdlib'), 'EXTERNALLY-MANAGED')
    except Exception:
        return None
    return p if os.path.isfile(p) else None


def distro_advice():
    """The distro's OWN words, read from the marker. PEP 668 defines it as an
    INI file with an [externally-managed] section and an Error key, and that
    is the text pip prints. Quoting it beats guessing: it is written by
    whoever marked this interpreter, so it names the right package manager
    even on a distro this script has never heard of."""
    p = em_marker()
    if not p:
        return ''
    try:
        import configparser
        cp = configparser.ConfigParser(interpolation=None)
        cp.read(p, encoding='utf-8')
        return cp.get('externally-managed', 'Error', fallback='').strip()
    except Exception:
        return ''


def _distro_id():
    """ID plus ID_LIKE from /etc/os-release, lowercased. Empty off Linux."""
    try:
        with open('/etc/os-release', encoding='utf-8', errors='replace') as f:
            osr = dict(ln.rstrip('\n').split('=', 1) for ln in f if '=' in ln)
    except OSError:
        return ''
    return (osr.get('ID', '') + ' ' + osr.get('ID_LIKE', '')).lower().replace('"', '')


def module_ok(mod, exe=None):
    """Can EXE import MOD? Asked by running it rather than by inspecting
    paths - the same functional-probe habit as the rest of the toolkit,
    for the same reason: a gutted package still imports as an empty
    namespace package."""
    exe = exe or sys.executable
    try:
        return subprocess.call(
            [exe, '-c', 'import ' + mod],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except Exception:
        return False


def pip_version(exe=None):
    """pip's version, or None when this interpreter has none.

    pip is NOT part of Python on most Linux distros. Arch splits it into
    python-pip, Debian into python3-pip, and a venv built --without-pip has
    none at all. Unchecked, the first install dies with "No module named
    pip", which reads like a broken toolkit rather than a missing system
    package - so ask first and name the package."""
    exe = exe or sys.executable
    try:
        out = subprocess.check_output([exe, '-m', 'pip', '--version'],
                                      stderr=subprocess.STDOUT, timeout=60)
        txt = out.decode('utf-8', 'replace').strip()
        bits = txt.split()
        return bits[1] if len(bits) > 1 and bits[0] == 'pip' else (txt[:40] or '?')
    except Exception:
        return None


def venv_usable(exe=None):
    """Can this interpreter actually build a working venv? `venv` alone is
    not enough - `ensurepip` is what puts pip INSIDE the new environment,
    and Debian famously splits both out into python3-venv. When it is
    present, a venv is a way to get pip on a machine that has none."""
    return module_ok('venv', exe) and module_ok('ensurepip', exe)


# Packages providing pip and the venv machinery. None means "already inside
# the base python package here". Checked against each distro's own package
# database rather than recalled, because three of the four are unobvious:
#   Arch      venv AND ensurepip (with its bundled wheel) are in `python`,
#             so `pacman -S python` alone can build a venv WITH pip in it.
#   Debian    the classic trap, and narrower than usually told: the venv
#             module itself is in the base python3; python3-venv adds only
#             ensurepip plus the wheels. So `python3 -m venv` imports fine
#             and then dies with "ensurepip is not available".
#   Fedora    venv and ensurepip are in python3-libs, which hard-Requires
#             the pip wheel package - so venv works standalone there too.
#   openSUSE  has NO plain python3-pip. The packages are version-flavoured
#             (python314-pip), hence the %(v)s below.
BOOTSTRAP = (
    (('arch', 'manjaro', 'endeavouros'), 'sudo pacman -S',
     {'pip': 'python-pip', 'venv': None}),
    (('debian', 'ubuntu'), 'sudo apt install',
     {'pip': 'python3-pip', 'venv': 'python3-venv'}),
    (('fedora', 'rhel', 'centos'), 'sudo dnf install',
     {'pip': 'python3-pip', 'venv': None}),
    (('opensuse', 'suse'), 'sudo zypper install',
     {'pip': 'python%(v)s-pip', 'venv': None}),
)


def is_distro_python3():
    """True when THIS interpreter is the distribution's own python3 - the one
    its python3-* packages are built for. imgdedup.sh runs setup with the
    newest python3.X it finds, which on Ubuntu 24.04 with a deadsnakes 3.13
    is not python3 (3.12): "apt install python3-venv" then installed the
    3.12 module, and re-running setup showed the same message again."""
    try:
        other = shutil.which('python3')
        if not other:
            return False
        if os.path.realpath(other) == os.path.realpath(sys.executable):
            return True
        out = subprocess.check_output(
            [other, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'],
            stderr=subprocess.DEVNULL, timeout=30).decode('ascii', 'replace')
        return out.strip() == '%d.%d' % sys.version_info[:2]
    except Exception:
        return False


def bootstrap_hint(what):
    """The command that installs pip ('pip') or the venv machinery ('venv')
    on this distro, or None when we cannot say."""
    ident = _distro_id()
    for keys, pm, names in BOOTSTRAP:
        if any(k in ident for k in keys):
            pkg = names.get(what)
            if 'debian' in keys and not is_distro_python3():
                # versioned for a second Python (deadsnakes and the like);
                # its pip comes with the venv, python3-pip is the other one's
                pkg = 'python%d.%d-venv' % sys.version_info[:2]
            if not pkg:
                return None
            if '%(v)s' in pkg:
                pkg = pkg % {'v': '%d%d' % sys.version_info[:2]}
            return '%s %s' % (pm, pkg)
    return None


def report_no_pip(can_venv):
    """Explain a missing pip in terms of the thing the user has to install,
    and say whether the venv route can still rescue the run."""
    print('')
    print('  pip is not available to this Python.')
    print('    %s -m pip  ->  No module named pip' % sys.executable)
    print('')
    # Same symptom, different causes - say the one that actually applies
    # rather than explaining Linux packaging to someone on Windows.
    if in_venv():
        print('  This is a virtual environment built without pip')
        print('  (python -m venv --without-pip), so it never had one.')
    elif _distro_id():
        print('  That is normal on Linux: pip is packaged separately from')
        print('  Python itself, so a base install genuinely has none.')
    else:
        print('  pip is normally bundled with Python here, so it was either')
        print('  deselected during installation or removed afterwards.')
        print('  Re-running the Python installer and ticking pip fixes it.')
    cmd = bootstrap_hint('pip')
    if cmd:
        print('')
        print('  On this distro:')
        print('    %s' % cmd)
    if can_venv:
        print('')
        print('  A virtual environment brings its own pip, so setup can')
        print('  still continue without touching anything system-wide.')
    else:
        vcmd = bootstrap_hint('venv')
        print('')
        print('  The venv module is missing too, so there is no way around')
        print('  it from here - install the package above first.')
        if vcmd:
            print('  This distro splits that out as well:')
            print('    %s' % vcmd)


def distro_packages():
    """(install command, {our name: distro name}) for the running distro, or
    None if unrecognised. Only the three CPU packages are mapped: torch and
    transformers are either absent from the official repos or far enough
    behind PyPI that pointing someone at them would be unkind."""
    ident = _distro_id()
    if not ident:
        return None
    table = (
        (('arch', 'manjaro', 'endeavouros'), 'sudo pacman -S',
         {'pillow': 'python-pillow', 'numpy': 'python-numpy',
          'opencv': 'python-opencv'}),
        (('debian', 'ubuntu'), 'sudo apt install',
         {'pillow': 'python3-pil', 'numpy': 'python3-numpy',
          'opencv': 'python3-opencv'}),
        (('fedora', 'rhel', 'centos'), 'sudo dnf install',
         {'pillow': 'python3-pillow', 'numpy': 'python3-numpy',
          'opencv': 'python3-opencv'}),
        (('opensuse', 'suse'), 'sudo zypper install',
         {'pillow': 'python3-Pillow', 'numpy': 'python3-numpy',
          'opencv': 'python3-opencv'}),
    )
    for keys, cmd, names in table:
        if any(k in ident for k in keys):
            return cmd, names
    return None


def dropped_path(arg, here):
    """The path a folder dropped on one of the .bat launchers really had.

    Explorer quotes a dropped path only when it contains a space, so cmd
    splits D:\\Photos&Videos at the "&" - and D:\\Paris,2019 at the comma,
    and drops a "^" - before the .bat starts: the argument that arrives is
    D:\\Photos, a DIFFERENT folder that may well exist, and that is the one
    that got scanned. The launchers keep cmd's untouched command line in
    IMGDEDUP_RAWCMD; the dropped path is the first argument after the
    launcher there. It is used only when the launcher is one of this
    toolkit's, and the path extends the argument that arrived and exists.
    Anything else returns ARG unchanged."""
    raw = os.environ.get('IMGDEDUP_RAWCMD')
    if not raw or not IS_WIN:
        return arg
    low, here_n = raw.lower(), os.path.normcase(os.path.abspath(here))
    i = low.find('.bat"')
    while i != -1:
        bat = raw[raw.rfind('"', 0, i) + 1:i + 4]
        if os.path.normcase(os.path.dirname(os.path.abspath(bat))) == here_n:
            break
        i = low.find('.bat"', i + 1)
    if i == -1:
        return arg
    rest = raw[i + 5:].strip()
    if rest.endswith('"'):
        rest = rest[:-1].rstrip()         # the quote closing cmd /c "..."
    if not rest:
        return arg
    if rest.startswith('"'):
        end = rest.find('"', 1)
        cand = rest[1:end] if end != -1 else rest[1:]
    else:
        cand = rest.split(' ')[0]         # unquoted means it had no space
    if not cand or not os.path.exists(cand):
        return arg
    got = os.path.normcase(os.path.abspath(arg)) if arg else ''
    want = os.path.normcase(os.path.abspath(cand))
    if got == want:
        return arg
    if got and not want.replace('^', '').startswith(got.rstrip('\\.')):
        return arg
    return cand


def pip_hint(pkg, exe=None, is_venv=None, is_managed=None):
    """The install line to PRINT for EXE. Kept here so the four stage scripts
    do not each hard-code advice that is wrong on Arch.

    A caller advising about a DIFFERENT interpreter than the one running must
    pass that interpreter's own answers - the doctor reads them off its
    per-interpreter probe. Omit both and this introspects the running process
    instead, which is right for the stage scripts (they only ever advise about
    themselves) and wrong for the doctor, whose launcher hands it a .venv while
    the advice is about a system Python. That mismatch printed a bare
    `pip install` on Arch, which pip refuses: externally-managed-environment.

    Named is_venv/is_managed rather than in_venv/managed so the parameters do
    not shadow the module functions of those names, which the fallback calls."""
    exe = exe or sys.executable
    if is_venv is None and is_managed is None:
        is_venv, is_managed = in_venv(), bool(em_marker())
    if is_venv:
        return '"%s" -m pip install %s' % (exe, pkg)      # a venv owns itself
    if is_managed:
        # marked by a distribution - or by uv or Homebrew, which do it on
        # Windows and macOS too, where "./imgdedup.sh" cannot run
        return ('this Python is marked as managed by another tool (a Linux '
                'distribution, uv, Homebrew), so pip will refuse.\n       '
                'Run the setup helper instead:  %s'
                % ('"%s" "%s"' % (exe, os.path.abspath(__file__)) if IS_WIN
                   else './imgdedup.sh setup'))
    # No --user: pip picks a user install by itself when site-packages is
    # not writable (a Python in Program Files), and forcing it elsewhere
    # sent a conda or pyenv Python's packages into ~/.local - where the
    # system Python of the same version reads them too.
    return '"%s" -m pip install %s' % (exe, pkg)


# ------------------------------------------------------------ installing --
def pip_base(exe=None, break_system=False):
    cmd = [exe or sys.executable, '-m', 'pip', 'install']
    if in_venv():
        return cmd                 # a venv owns itself; no flag wanted
    if em_marker():
        # Reaching here means the venv and distro routes were shown and the
        # user chose this one deliberately - never add the flag silently.
        if break_system:
            cmd.append('--break-system-packages')
        return cmd
    # No --user (see pip_hint): pip falls back to it by itself when it must.
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


def venv_python(path):
    """The interpreter inside a venv, on either platform's layout."""
    for rel in (os.path.join('bin', 'python'),
                os.path.join('Scripts', 'python.exe')):
        p = os.path.join(path, rel)
        if os.path.isfile(p):
            return p
    return None


def offer_managed_routes(need, args):
    """Explain PEP 668 and let the user pick a way forward.

    Returns (proceed, rc). proceed is True when the caller should go on to
    install with pip into THIS interpreter (the user knowingly chose
    --break-system-packages); otherwise everything that is going to happen
    already has, and rc is the exit code: 0 only when something was
    installed. Every route used to end in exit 0 - a failed venv, a route
    the user still had to run by hand - and imgdedup.sh then said "Setup
    finished"."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(here, '.venv')
    dp = distro_packages()

    print('')
    print('  This Python is managed by your distribution (PEP 668):')
    print('    ' + (em_marker() or ''))
    print('  pip will refuse to install into it. Note that --user does NOT')
    print('  exempt you - pip rejects that too, which catches most people out.')
    print('  (pip blocks uninstall the same way, so removing a package that')
    print('  came from your package manager also has to go through it.)')
    advice = distro_advice()
    if advice:
        print('')
        print('  Your distribution\'s own words:')
        for ln in advice.splitlines():
            if ln.strip():
                print('    | ' + ln.rstrip())
    print('')
    # Do not recommend a route this machine cannot take. Debian 12 with
    # python3-pip installed but python3-venv missing is an entirely ordinary
    # state, and it is exactly the machine PEP 668 forces down this path.
    can_venv = venv_usable()
    routes = []
    if can_venv:
        routes.append('1')
        print('  1) Make a virtual environment for this toolkit  [recommended]')
        print('       "%s" -m venv "%s"' % (sys.executable, venv_dir))
        print('     Self-contained, needs no root, touches nothing your')
        print('     package manager owns, and imgdedup.sh prefers it')
        print('     automatically.')
    else:
        print('  1) Make a virtual environment           [NOT POSSIBLE YET]')
        print('     This Python cannot build one: the venv/ensurepip modules')
        print('     are missing, which is how Debian and Ubuntu package it.')
        vcmd = bootstrap_hint('venv')
        if vcmd:
            print('     Install that first, then re-run setup:')
            print('       %s' % vcmd)
    # route 2's packages are built for the distribution's python3 only, so
    # a second Python cannot load them; and the pip name 'opencv-python-
    # headless' has to be looked up as 'opencv', or OpenCV was never mapped
    # and the route vanished whenever only it and the embed packages were due
    if dp and is_distro_python3():
        pm, names = dp
        key = {'opencv-python-headless': 'opencv'}
        mapped = [names[key.get(p, p)] for p in need if key.get(p, p) in names]
        if mapped:
            routes.append('2')
            print('')
            print('  2) Install from your package manager instead')
            print('       %s %s' % (pm, ' '.join(mapped)))
            rest = [p for p in need if key.get(p, p) not in names]
            if rest:
                print('     Does not cover %s - that still wants a venv.'
                      % ', '.join(rest))
    routes.append('3')
    print('')
    print('  3) Install into the system Python anyway')
    print('       pip install --break-system-packages ...')
    print('     The exact thing your distro is trying to prevent: pip and')
    print('     your package manager can then disagree about the same files.')
    print('')

    default = '1' if can_venv else ('2' if '2' in routes else '3')
    if args.yes:
        # never pick 3 unattended - it can break a package-managed system
        if not can_venv and '2' not in routes:
            print('  --yes: no route can be taken unattended here - nothing '
                  'was installed.')
            return False, 1
        choice = '1' if can_venv else '2'
        print('  --yes: taking route %s.' % choice)
    else:
        try:
            choice = input('  Choose [%s, default %s]: '
                           % ('/'.join(routes), default)).strip() or default
        except EOFError:
            choice = default
    if choice == '1' and not can_venv:
        print('')
        print('  That route is not available until the venv module is')
        print('  installed - nothing was done.')
        return False, 1

    if choice == '2' and '2' in routes:
        print('')
        print('  Nothing installed. Run the command above, then re-run setup.')
        return False, 1
    if choice == '3':
        print('')
        print('  Proceeding with --break-system-packages, as chosen.')
        return True, 0
    if choice != '1':
        print('')
        print('  Not a listed choice - nothing installed.')
        return False, 1

    return False, create_venv_and_rerun(args)


def create_venv_and_rerun(args):
    """Build .venv beside the toolkit and re-run setup inside it. Returns
    the exit code to pass on - the child's, or 1 when the venv could not be
    made. Whatever was going to be installed has been, by the child, and
    the caller must not carry on installing here."""
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '.venv')
    existed = os.path.isdir(venv_dir)

    def give_up(msg):
        """Abandon a half-built venv, and take it with us.

        A stub venv is worse than none at all: `python -m venv` writes the
        interpreter and pyvenv.cfg BEFORE provisioning pip, so an aborted
        run leaves something that looks like a working environment and has
        no way to install anything. The launchers prefer a .venv beside
        them over any Python on PATH, so leaving it would let one failed
        setup capture every later run - including the setup meant to fix
        it. Only ever remove what this call created."""
        print('')
        print('  ' + msg)
        vcmd = bootstrap_hint('venv')
        if vcmd:
            print('  This distro packages that separately:')
            print('    %s' % vcmd)
        else:
            print('  On Debian/Ubuntu this usually means python3-venv is')
            print('  missing; add it and try again.')
        if not existed and os.path.isdir(venv_dir):
            try:
                shutil.rmtree(venv_dir)
                print('')
                print('  Removed the half-built %s, so the launchers will'
                      % venv_dir)
                print('  not prefer it over a working Python.')
            except OSError as exc:
                print('')
                print('  Could NOT remove %s (%s).' % (venv_dir, exc))
                print('  Delete it by hand: while it exists the launchers')
                print('  will keep choosing it.')
        return 1

    print('')
    print('  Creating %s' % venv_dir)
    if subprocess.call([sys.executable, '-m', 'venv', venv_dir]) != 0:
        return give_up('venv creation failed.')
    vpy = venv_python(venv_dir)
    if not vpy:
        return give_up('The venv was created but holds no interpreter.')
    if pip_version(vpy) is None:
        return give_up('The venv was created WITHOUT pip, so nothing can be '
                       'installed into it (ensurepip was unavailable).')
    print('  OK. Re-running setup inside it:')
    print('    "%s" "%s"' % (vpy, os.path.abspath(__file__)))
    rc = subprocess.call([vpy, os.path.abspath(__file__)]
                         + (['--yes'] if args.yes else [])
                         + (['--offline'] if args.offline else []))
    print('')
    if rc != 0:
        print('  Setup inside the venv did not finish (exit %d).' % rc)
        return rc
    print('  From now on the toolkit uses that venv - imgdedup.sh looks for')
    print('  .venv beside itself before falling back to the system Python.')
    return 0


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
    # pip ships separately from Python on most distros, so establish that it
    # exists before anything tries to run it.
    pipv, can_venv = pip_version(), venv_usable()
    print('  pip %-12s venv %s' % (pipv or 'MISSING',
                                   'ok' if can_venv else 'MISSING'))

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
    why = wrong_build(flav, st, vendors) if 'torch' not in need else ''
    if why:
        print('')
        print('  torch here is a %s build, %s' % (desc, why))
        need.append('torch')
    if not need:
        print('')
        print('  Everything the toolkit needs is present. Nothing to do.')
        return 0
    print('')
    print('  Missing: ' + ', '.join(need))
    if args.check:
        return 1

    # No pip at all is a different failure from pip refusing: name the
    # missing package rather than letting "No module named pip" surface.
    if pipv is None:
        report_no_pip(can_venv)
        if not can_venv:
            return 1
        if args.yes:
            ans = 'y'
        else:
            print('')
            try:
                ans = input('  Create the virtual environment now? [Y/n]: '
                            ).strip().lower() or 'y'
            except EOFError:
                ans = 'y'
        if ans not in ('y', 'yes'):
            print('  Nothing installed.')
            return 1
        return create_venv_and_rerun(args)

    # PEP 668 distros refuse pip outright; ask before doing anything.
    breaksys = False
    if em_marker():
        proceed, rc = offer_managed_routes(need, args)
        if not proceed:
            return rc
        breaksys = True

    plain = [p for p in need if p != 'torch']
    # The return value used to be discarded, so a refused install still fell
    # through to "return 0" below and the launcher reported success having
    # installed nothing - which is how Arch failed silently.
    if plain and not show_and_run(pip_base(break_system=breaksys) + plain,
                                  args.yes, ', '.join(plain)):
        return 1

    if 'torch' not in need:
        return 0

    opts = backends_for(vendors, online=not args.offline,
                        names=[n for v, n in gpus if v == 'NVIDIA'])
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
        # Range-checked rather than indexed: "0" is int 0, and usable[-1]
        # is the LAST entry, so typing zero installed a build nobody asked
        # for - on a mixed-vendor machine, quite possibly the wrong one.
        # "-1" reached the one before it. Both looked like the menu had
        # been obeyed.
        try:
            n = int(raw) if raw else 1
        except ValueError:
            n = 0
        if not 1 <= n <= len(usable):
            print('  Not a listed choice - nothing installed.')
            return 1
        pick = usable[n - 1]
    print('  -> %s' % pick.label)
    if flav != 'none':
        # pip takes "torch" as already satisfied by the build that is there
        # and installs nothing from the new index, so that one goes first
        un = [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch']
        if breaksys:
            un.append('--break-system-packages')
        if not show_and_run(un, args.yes, 'the %s torch build' % desc):
            return 1
    if not show_and_run(pip_base(break_system=breaksys) + pick.args, args.yes,
                        'torch (%s)' % pick.key):
        return 1
    if st.get('transformers', 'ERR') in ('ERR', 'EMPTY') \
            and 'transformers' not in plain:
        show_and_run(pip_base(break_system=breaksys) + ['transformers'],
                     args.yes, 'transformers')

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
