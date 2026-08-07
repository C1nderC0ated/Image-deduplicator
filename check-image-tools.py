#!/usr/bin/env python3
"""
check-image-tools.py  -  read-only environment doctor.

Finds every Python interpreter it can on this machine, asks each one what it
can ACTUALLY do (not merely what imports), and prints the exact command to
fix whatever is missing. Touches nothing.

Note on "EMPTY": an installed package whose files have been deleted leaves
its directory behind, and Python happily imports that directory as an empty
"namespace package". A plain `import torch` therefore succeeds against a
gutted install. This script checks module.__file__ and runs a real
functional test, so hollow installs are reported as broken rather than fine.

Usage:  python check-image-tools.py
"""
import json
import os
import re
import subprocess
import sys

# Paths with exotic characters must not crash a redirected console.
# stdout only: stderr already defaults to backslashreplace, which cannot
# raise, and 'replace' would only destroy detail in tracebacks.
for _s in (sys.stdout,):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

IS_WIN = os.name == 'nt'
SHOW = ['PIL', 'numpy', 'torch', 'torchvision', 'transformers', 'pillow_heif']


def _hint(pkg, exe=None, info=None):
    """How to install PKG into EXE, phrased for that interpreter.

    Routed through _setup so the advice lives in one place. Hard-coding
    '--user' is wrong on a distro-managed Python (Arch, Debian 12+, Fedora
    38+): pip refuses it, and the answer there is a virtual environment.

    `info` is the probe result for `exe`, and it carries that interpreter's
    own answers to "am I in a venv" and "am I externally managed". Passing
    them is what keeps the advice honest: this used to be inferred from the
    interpreter *running the doctor*, which since the launchers began
    preferring a .venv is almost never the one being advised about. On Arch
    that printed a plain `pip install` for the system Python - a command it
    refuses outright with error: externally-managed-environment."""
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        if d not in sys.path:
            sys.path.insert(0, d)
        from _setup import pip_hint
        info = info or {}
        return pip_hint(pkg, exe, is_venv=info.get('in_venv'),
                        is_managed=info.get('managed'))
    except Exception:
        return '"%s" -m pip install --user %s' % (exe or sys.executable, pkg)


def _installable(info):
    """Can pip install into the interpreter this probe result describes?
    A venv always owns itself; otherwise a PEP 668 marker means no."""
    return bool(info.get('in_venv')) or not info.get('managed')


PROBE = r'''
import json, sys
o = {'v': '%d.%d.%d' % sys.version_info[:3], 'exe': sys.executable,
     'bits': 64 if sys.maxsize > 2**32 else 32,
     'ft': not getattr(sys, '_is_gil_enabled', lambda: True)()}
m = {}
for name in ['PIL', 'numpy', 'torch', 'torchvision', 'transformers', 'pillow_heif']:
    try:
        mod = __import__(name)
        if getattr(mod, '__file__', None) is None:
            m[name] = 'EMPTY'
        else:
            m[name] = str(getattr(mod, '__version__', '?'))
    except Exception as e:
        m[name] = 'ERR:' + type(e).__name__ + ': ' + str(e)[:80]
o['mods'] = m
try:
    from PIL import Image, ImageOps
    Image.new('RGB', (4, 4)).convert('L')
    o['collect_ok'] = True
except Exception as e:
    o['collect_ok'] = False
    o['collect_err'] = type(e).__name__ + ': ' + str(e)[:110]
try:
    import torch
    from transformers import CLIPModel, CLIPImageProcessor
    torch.zeros(1) + 1
    o['embed_ok'] = True
    o['torchver'] = str(torch.__version__)
    o['cuda'] = bool(torch.cuda.is_available())
    o['hip'] = getattr(torch.version, 'hip', None)
    o['cuda_build'] = getattr(torch.version, 'cuda', None)
except Exception as e:
    o['embed_ok'] = False
    o['embed_err'] = type(e).__name__ + ': ' + str(e)[:110]
# pip is packaged separately from Python on most Linux distros, so an
# interpreter can be perfectly good and still have no way to install
# anything. venv matters for the same reason: where pip is missing but
# ensurepip is present, a virtual environment is the way to get one.
try:
    import pip
    o['pip'] = str(getattr(pip, '__version__', '?'))
except Exception:
    o['pip'] = None
try:
    import venv, ensurepip           # noqa: F401
    o['venv'] = True
except Exception:
    o['venv'] = False
# The two facts that decide whether pip may install here, asked of THIS
# interpreter rather than of whichever one is running the doctor. The two
# are routinely different - the launcher prefers a .venv while the advice
# is about a system Python - and reading them off the wrong side is how a
# command pip refuses gets printed as the fix.
# Read them together, never 'managed' alone: inside a venv sysconfig still
# points at the base stdlib, so a venv on Arch reports managed=True while
# pip installs into it happily. in_venv wins; see _installable in this file.
o['in_venv'] = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
try:
    import os as _os, sysconfig as _sc
    o['managed'] = _os.path.isfile(
        _os.path.join(_sc.get_path('stdlib'), 'EXTERNALLY-MANAGED'))
except Exception:
    o['managed'] = False
print('@@' + json.dumps(o))
'''


def run(cmd, timeout=180):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or '') + (p.stderr or '')
    except Exception as e:
        return -1, str(e)


def probe(cmd):
    rc, out = run(list(cmd) + ['-c', PROBE])
    for line in out.splitlines():
        if line.startswith('@@'):
            try:
                return json.loads(line[2:])
            except Exception:
                pass
    return {'error': (out.strip().splitlines() or ['no output'])[-1][:160]}


def gpu_inventory():
    """[(vendor, name, driver_ready)] for every graphics adapter, via the
    shared setup helper: PCI vendor IDs, so an AMD or Intel card is seen
    even with no vendor toolchain installed. nvidia-smi/rocm-smi are used
    only to tell 'card present' from 'compute driver usable'."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _setup
        return [(v, n, _setup.driver_ready(v)) for v, n in _setup.detect_gpus()]
    except Exception:
        return []
    finally:
        if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
            sys.path.pop(0)


def candidates():
    found, seen = [], set()

    def add(cmd, label):
        key = ' '.join(cmd)
        if key not in seen:
            seen.add(key)
            found.append((cmd, label))

    # In launcher order, because the verdict is first-match-wins and its whole
    # job is to predict what the launchers will actually pick:
    #   IMGDEDUP_PYTHON  ->  .venv beside the toolkit  ->  PATH
    # The .venv was previously invisible here. It is not on PATH and the
    # launchers exec it directly rather than activating it, so $VIRTUAL_ENV -
    # the only venv this ever looked at - is unset. On any PEP 668 distro that
    # is exactly where setup puts every package, so the doctor would report
    # "nothing can load torch + transformers" about a machine whose very next
    # `./imgdedup.sh embed` runs fine.
    override = os.environ.get('IMGDEDUP_PYTHON')
    if override:
        add([override], 'IMGDEDUP_PYTHON')
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (os.path.join('.venv', 'bin', 'python'),
                os.path.join('.venv', 'Scripts', 'python.exe')):
        p = os.path.join(here, rel)
        if os.path.exists(p):
            add([p], 'toolkit .venv (what the launchers use)')
    if not IS_WIN:
        # No `py` launcher off Windows. Interpreters live on PATH under
        # versioned names, and pyenv / deadsnakes / Homebrew installs are
        # only reachable that way, so ask for each one by name.
        for minor in range(14, 8, -1):
            exe = 'python3.%d' % minor
            rc, out = run([exe, '-c', 'import sys;print(sys.executable)'], timeout=30)
            if rc == 0 and out.strip():
                p = out.strip().splitlines()[-1]
                if os.path.exists(p):
                    add([p], 'on PATH as ' + exe)
        for extra in (os.path.join(os.path.expanduser('~'), '.local', 'bin', 'python3'),
                      '/usr/local/bin/python3', '/opt/homebrew/bin/python3'):
            if os.path.exists(extra):
                add([extra], extra)
        venv = os.environ.get('VIRTUAL_ENV')
        if venv:
            p = os.path.join(venv, 'bin', 'python')
            if os.path.exists(p):
                add([p], 'active virtualenv ($VIRTUAL_ENV)')
    rc, out = run(['py', '-0p'], timeout=30) if IS_WIN else (1, '')
    if rc == 0:
        # Paths may contain spaces, so never split the whole line on whitespace.
        # Take the tag up to the first run of blanks, then the rest is the path.
        for line in out.splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            # The DEFAULT interpreter is flagged like " -V:3.14 *   C:\..."
            # - the marker must neither hide the line nor glue "* " onto
            # the path (it used to do one or the other, so the one
            # interpreter you actually use was missing from this report).
            m = re.match(r'\s*(\S+)(\s+\*)?\s{2,}(.+?)\s*$', line)
            if not m:
                m = re.match(r'\s*(\S+)(\s+\*)?\s+([A-Za-z]:\\.+?)\s*$', line)
            if not m:
                continue
            tag, star, path = m.group(1), m.group(2), m.group(3).strip('"')
            if path.lower().endswith(('python.exe', 'pythonw.exe')) and os.path.exists(path):
                add([path], 'py launcher: ' + tag + (' (default)' if star else ''))
    for exe in ('python', 'python3'):
        rc, out = run([exe, '-c', 'import sys;print(sys.executable)'], timeout=30)
        if rc == 0 and out.strip():
            p = out.strip().splitlines()[-1]
            if os.path.exists(p):
                add([p], 'on PATH as ' + exe)
    return found


def main():
    print('')
    print('  Image-tools environment check')
    print('  ' + '-' * 64)
    print('  (read-only - this changes nothing)')
    print('')
    inv = gpu_inventory()
    gpus = [n for v, n, _ in inv if v in ('NVIDIA', 'AMD', 'Intel')]
    vendors = set(v for v, _, _ in inv)
    if gpus:
        for v, name, ready in inv:
            if v == 'other':
                continue
            print('  GPU: %-7s %s   [compute driver: %s]'
                  % (v, name[:44], 'yes' if ready else 'not found'))
    else:
        print('  GPU: none detected (only the Embed stage would use one)')
    print('')

    cands = candidates()
    if not cands:
        print('  No Python interpreter could be found at all.')
        print('  Install Python from https://www.python.org/downloads/ and retry.')
        return 1

    results = [(cmd, label, probe(cmd)) for cmd, label in cands]
    hollow_seen = False
    best_collect = best_embed = None

    for cmd, label, info in results:
        print('  ' + label)
        print('    ' + (info.get('exe') or cmd[0]))
        if 'error' in info:
            print('    [DEAD] interpreter did not run: ' + info['error'])
            print('')
            continue
        print('    Python %s  (%d-bit)%s'
              % (info['v'], info.get('bits', 0),
                 '  free-threaded' if info.get('ft') else ''))
        # pip is packaged separately from Python on most Linux distros, so a
        # perfectly healthy interpreter can still have no way to install
        # anything - worth saying plainly rather than letting it surface
        # later as "No module named pip".
        if info.get('pip'):
            print('      [ok   ] %-13s %s' % ('pip', info['pip']))
        elif info.get('venv'):
            print('      [MISS ] %-13s not installed - but venv is, so a '
                  'virtual environment can supply one' % 'pip')
        else:
            print('      [MISS ] %-13s not installed, and venv is missing too'
                  % 'pip')
        mods = info.get('mods', {})
        for name in SHOW:
            val = mods.get(name, 'ERR:not probed')
            if val == 'EMPTY':
                hollow_seen = True
                print('      [EMPTY] %-13s directory exists but has no files' % name)
            elif val.startswith('ERR:'):
                if name == 'torchvision':
                    # torchvision is probed because a PRESENT-but-broken one
                    # takes transformers (and the embedder) down with it.
                    # NOTHING in this toolkit needs it, so absent is the
                    # healthy state, not a gap to fill.
                    if 'ModuleNotFound' in val:
                        print('      [ok   ] %-13s not installed - fine, '
                              'nothing here needs it' % name)
                    else:
                        print('      [BROKEN] %-12s %s' % (name, val[4:][:66]))
                else:
                    print('      [MISS  ] %-13s %s' % (name, val[4:][:66]))
            else:
                note = '  (not needed by this toolkit)' if name == 'torchvision' else ''
                print('      [ok   ] %-13s %s%s' % (name, val[:70], note))

        if info.get('collect_ok'):
            print('      -> collector test:  PASS')
            if best_collect is None:
                best_collect = (cmd, label, info)
        else:
            print('      -> collector test:  FAIL  ' + info.get('collect_err', '')[:60])
        if info.get('embed_ok'):
            print('      -> embedder test:   PASS   torch %s, CUDA %s'
                  % (info.get('torchver', '?'), 'yes' if info.get('cuda') else 'no'))
            if best_embed is None:
                best_embed = (cmd, label, info)
        else:
            err = info.get('embed_err', '')
            print('      -> embedder test:   FAIL  ' + err[:60])
            tv = mods.get('torchvision', '')
            if tv.startswith('ERR:') and 'ModuleNotFound' not in tv:
                exe = info.get('exe') or cmd[0]
                print('         Cause: torchvision is installed but will not load -')
                print('         its compiled _C matches a torch build that is no longer')
                print('         here (the usual aftermath of a torch reinstall). Fix:')
                print('           "%s" -m pip uninstall torchvision' % exe)
                print('         (nothing in this toolkit needs it), or reinstall the pair:')
                print('           ' + _hint(
                    '--force-reinstall torch torchvision --index-url '
                    'https://download.pytorch.org/whl/cu132', exe, info))
        print('')

    print('  ' + '=' * 64)
    print('  VERDICT')
    print('  ' + '=' * 64)
    if best_collect:
        print('  collect-image-inventory.py  -> USE: %s' % best_collect[1])
        print('       %s' % (best_collect[2].get('exe') or best_collect[0][0]))
    else:
        print('  collect-image-inventory.py  -> no interpreter has a working Pillow.')
        print('       Fix:  ' + _hint('pillow'))
    print('')
    if best_embed:
        print('  embed-images.py             -> USE: %s' % best_embed[1])
        print('       %s' % (best_embed[2].get('exe') or best_embed[0][0]))
        tv = best_embed[2].get('torchver', '')
        hip = best_embed[2].get('hip')
        # HIP first: a ROCm build reports cuda.is_available() == True and
        # torch.version.cuda is not a reliable discriminator.
        kind = ('ROCm/HIP %s' % hip if hip else
                'CUDA %s' % best_embed[2].get('cuda_build')
                if best_embed[2].get('cuda_build') else 'CPU-only')
        if best_embed[2].get('cuda'):
            print('       GPU acceleration available  (%s build).' % kind)
        else:
            exe = best_embed[2].get('exe') or best_embed[0][0]
            if gpus and kind == 'CPU-only':
                print('       CPU only - BUT a GPU is present (%s).' % gpus[0])
                print('       A CPU-only wheel (%s) can never use it; that is' % tv)
                print('       decided when the wheel is installed. To switch:')
                print('         "%s" -m pip uninstall torch' % exe)
                print('         "%s" %s_setup.py    (picks the right build)'
                      % (exe, '' if IS_WIN else ''))
            elif gpus:
                print('       CPU only - a %s build is installed but no device is'
                      % kind)
                print('       visible; check the driver for %s.'
                      % (', '.join(sorted(vendors - {'other'})) or 'your GPU'))
            else:
                print('       CPU only - slower, but fine (no GPU detected).')
        print('')
        exe_pin = best_embed[2].get('exe') or best_embed[0][0]
        if IS_WIN:
            print('  If Embed-Images.bat picks the wrong one, force it:')
            print('       set IMGDEDUP_PYTHON=%s' % exe_pin)
        else:
            print('  If ./imgdedup.sh picks the wrong one, force it:')
            print('       export IMGDEDUP_PYTHON=%s' % exe_pin)
    else:
        print('  embed-images.py             -> nothing can load torch + transformers.')
        pick = None
        for cmd, label, info in results:
            if 'error' in info or info.get('ft'):
                continue
            try:
                mm = tuple(int(x) for x in info.get('v', '0.0').split('.')[:2])
            except ValueError:
                continue
            if mm >= (3, 9) and info.get('collect_ok'):
                pick = (cmd, label, info)
                break
        if pick:
            exe = pick[2].get('exe') or pick[0][0]
            print('')
            print('     Install into: %s (Python %s)' % (pick[1], pick[2]['v']))
            if not _installable(pick[2]):
                # No pip command can succeed against this interpreter, so the
                # per-GPU menu below would be three identical copies of the
                # same refusal under headings that promise commands.
                print('       ' + _hint('torch', exe, pick[2]))
            else:
                print('       CPU only:')
                print('         ' + _hint('torch --index-url '
                                          'https://download.pytorch.org/whl/cpu',
                                          exe, pick[2]))
                print('       NVIDIA GPU (CUDA):')
                print('         ' + _hint('torch --index-url '
                                          'https://download.pytorch.org/whl/cu132',
                                          exe, pick[2]))
                print('       Then, either way:')
                print('         ' + _hint('transformers', exe, pick[2]))
            if _installable(pick[2]) and pick[2]['v'].startswith('3.14'):
                print('')
                print('     NOTE: on Python 3.14 the cu121 index has NO wheels.')
                print('           Use cu132 above, not cu121.')

    if hollow_seen:
        print('')
        print('  ' + '=' * 64)
        print('  WARNING: at least one package showed as [EMPTY].')
        print('  Its folder is still there but every file inside is gone, so')
        print('  Python imports it as an empty namespace package and a plain')
        print('  "import torch" would wrongly appear to succeed. That install')
        print('  is broken, not present. Reinstall the package, or point the')
        print('  tools at a different interpreter (see VERDICT above).')
        if IS_WIN:
            print('  If this is under your Downloads folder, check whether Windows')
            print('  Storage Sense is set to auto-delete files there.')
    print('')

    # Offer to fix what the report just found. The doctor itself stays
    # read-only: it never installs, it only hands off to _setup.py, which
    # shows every command and asks before running it.
    if not (best_collect and best_embed):
        offer_setup(best_collect or best_embed)
    return 0


def offer_setup(best):
    setup = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_setup.py')
    if not os.path.exists(setup):
        return
    exe = (best[2].get('exe') or best[0][0]) if best else sys.executable
    print('  ' + '=' * 64)
    print('  Something above is missing. Setup can install it: it detects')
    print('  your GPU, asks which PyTorch build you want, and shows every')
    print('  command before running it.')
    if not sys.stdin.isatty():
        print('')
        print('  Run:  "%s" "%s"' % (exe, setup))
        print('')
        return
    try:
        ans = input('  Run setup now? [y/N]: ').strip().lower()
    except EOFError:
        ans = ''
    if ans in ('y', 'yes'):
        print('')
        subprocess.call([exe, setup])
    else:
        print('')
        print('  Skipped. Run it any time:  "%s" "%s"' % (exe, setup))
        print('')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
