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


def bootstrap_hint(what):
    """The command that installs pip ('pip') or the venv machinery ('venv')
    on this distro, or None when we cannot say."""
    ident = _distro_id()
    for keys, pm, names in BOOTSTRAP:
        if any(k in ident for k in keys):
            pkg = names.get(what)
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


def pip_hint(pkg, exe=None, in_venv=None, managed=None):
    """The install line to PRINT for EXE. Kept here so the four stage scripts
    do not each hard-code advice that is wrong on Arch.

    A caller that is advising about a DIFFERENT interpreter than the one
    running must pass that interpreter's own `in_venv` / `managed` answers -
    the doctor gets them from its per-interpreter probe. Without them this
    falls back to introspecting the running process, which is right for the
    stage scripts (they only ever advise about themselves) and wrong for the
    doctor, whose launcher hands it a .venv while the advice is about a
    system Python. That mismatch printed a bare `pip install` on Arch, which
    pip refuses with error: externally-managed-environment."""
    exe = exe or sys.executable
    if in_venv is None and managed is None:
        in_venv, managed = globals()['in_venv'](), bool(em_marker())
    if in_venv:
        return '"%s" -m pip install %s' % (exe, pkg)      # a venv owns itself
    if managed:
        return ('this Python is managed by your distribution, so pip will '
                'refuse.\n       Run the setup helper instead:  '
                './imgdedup.sh setup')
    return '"%s" -m pip install --user %s' % (exe, pkg)


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

    Returns True if the caller should go on to install with pip into THIS
    interpreter (i.e. the user knowingly chose --break-system-packages), and
    False when everything that is going to happen already has."""
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
    if dp:
        pm, names = dp
        mapped = [names[p] for p in need if p in names]
        if mapped:
            routes.append('2')
            print('')
            print('  2) Install from your package manager instead')
            print('       %s %s' % (pm, ' '.join(mapped)))
            rest = [p for p in need if p not in names]
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
        return False

    if choice == '2' and '2' in routes:
        print('')
        print('  Nothing installed. Run the command above, then re-run setup.')
        return False
    if choice == '3':
        print('')
        print('  Proceeding with --break-system-packages, as chosen.')
        return True
    if choice != '1':
        print('')
        print('  Not a listed choice - nothing installed.')
        return False

    return create_venv_and_rerun(args)


def create_venv_and_rerun(args):
    """Build .venv beside the toolkit and re-run setup inside it. Always
    returns False: whatever was going to be installed has been, by the
    child process, and the caller must not carry on installing here."""
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
        return False

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
    subprocess.call([vpy, os.path.abspath(__file__)]
                    + (['--yes'] if args.yes else [])
                    + (['--offline'] if args.offline else []))
    print('')
    print('  From now on the toolkit uses that venv - imgdedup.sh looks for')
    print('  .venv beside itself before falling back to the system Python.')
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
        create_venv_and_rerun(args)
        return 0

    # PEP 668 distros refuse pip outright; ask before doing anything.
    breaksys = False
    if em_marker():
        if not offer_managed_routes(need, args):
            return 0
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
