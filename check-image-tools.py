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
        return '"%s" -m pip install %s' % (exe or sys.executable, pkg)


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
    from transformers import CLIPModel
    # Ask for the Pillow backend by name, exactly as the embedder does.
    # Plain CLIPImageProcessor prints a torchvision warning and then returns
    # this same class anyway, and that line lands in the middle of the
    # doctor's own output.
    try:
        from transformers import CLIPImageProcessorPil as CLIPImageProcessor
    except ImportError:
        from transformers import CLIPImageProcessor
    torch.zeros(1) + 1
    o['embed_ok'] = True
    o['torchver'] = str(torch.__version__)
    o['cuda'] = bool(torch.cuda.is_available())
    o['hip'] = getattr(torch.version, 'hip', None)
    o['cuda_build'] = getattr(torch.version, 'cuda', None)
    # A CUDA 12.8+ build has no kernels for a card older than Turing (7.5),
    # yet reports it available; the first kernel then fails. CUDA's rule: a
    # binary for the same major and a lower-or-equal minor runs, and PTX
    # ('compute_') runs on anything newer.
    if o['cuda'] and not o['hip']:
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
                if kind == 'sm' and mj == cap[0] and mn <= cap[1]:
                    fits = True
                if kind == 'compute' and (mj, mn) <= cap:
                    fits = True
            o['cuda_fits'] = fits
        except Exception:
            pass
    # The embedder runs on Intel Arc and on Apple Metal too. Reporting only
    # CUDA told an Arc owner "CPU only - BUT a GPU is present" and handed
    # them a reinstall command, about a setup that was already using the
    # GPU. Collected here so the report can say which one.
    try:
        o['xpu'] = bool(torch.xpu.is_available())
    except Exception:
        o['xpu'] = False
    try:
        o['mps'] = bool(torch.backends.mps.is_available())
    except Exception:
        o['mps'] = False
    # the BUILD as well as the device: an XPU wheel whose driver is missing
    # is still an XPU wheel, and was reported as CPU-only with a reinstall
    # command for the very wheel already installed
    o['xpu_build'] = bool(getattr(torch.version, 'xpu', None))
    try:
        o['mps_built'] = bool(torch.backends.mps.is_built())
    except Exception:
        o['mps_built'] = False
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


def _decode(b):
    """Child output as text. The py launcher writes UTF-8 to a pipe, and
    the Python probes are told to (PYTHONIOENCODING in run); anything else
    is read in the ANSI code page. Decoding everything in the ANSI code
    page garbled a path with 'u-umlaut' so it was dropped, and one byte
    undefined in cp1252 - from 'L-stroke', 'A-acute' or most Cyrillic -
    failed the whole decode: every py-listed interpreter vanished and the
    doctor said no Python existed at all."""
    if not b:
        return ''
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        import locale
        return b.decode(locale.getpreferredencoding(False) or 'utf-8',
                        'replace')


def _setup_line(exe):
    """The command that runs setup under EXE, with an absolute path: the
    old relative "_setup.py" failed from any other directory."""
    return '"%s" "%s"' % (exe, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '_setup.py'))


def run(cmd, timeout=180):
    try:
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        return p.returncode, _decode(p.stdout) + _decode(p.stderr)
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
    #   Linux/macOS   IMGDEDUP_PYTHON -> .venv -> PATH        (imgdedup.sh)
    #   Windows       IMGDEDUP_PYTHON -> py    -> .venv -> PATH
    #                                                     (_pick-python.bat)
    # The two differ on purpose - a PEP 668 distro forces every package into
    # the venv so it has to win there, while on Windows `py` is the idiomatic
    # entry point and a stray .venv should not hijack it - so this cannot be
    # one order. Hard-coding the Linux one made the label "what the launchers
    # use" false on Windows, which is worse than not reporting it at all.
    #
    # The .venv was invisible here until recently. It is not on PATH and the
    # launchers exec it directly rather than activating it, so $VIRTUAL_ENV -
    # the only venv this ever looked at - is unset. On any PEP 668 distro that
    # is exactly where setup puts every package, so the doctor would report
    # "nothing can load torch + transformers" about a machine whose very next
    # `./imgdedup.sh embed` runs fine.
    override = os.environ.get('IMGDEDUP_PYTHON')
    if override:
        add([override], 'IMGDEDUP_PYTHON')
    here = os.path.dirname(os.path.abspath(__file__))

    def add_toolkit_venv():
        label = ('toolkit .venv (below the py launcher here)' if IS_WIN
                 else 'toolkit .venv (what the launchers use)')
        for rel in (os.path.join('.venv', 'bin', 'python'),
                    os.path.join('.venv', 'Scripts', 'python.exe')):
            p = os.path.join(here, rel)
            if os.path.exists(p):
                add([p], label)

    if not IS_WIN:
        add_toolkit_venv()
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
    if IS_WIN:
        # After the py launcher, before bare PATH - mirroring
        # _pick-python.bat exactly.
        add_toolkit_venv()
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
                    print('      [MISS ] %-13s %s' % (name, val[4:][:66]))
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
                print('         Nothing in this toolkit needs it. (The old second option,')
                print('         reinstalling the pair from the CUDA index, swapped a ROCm or')
                print('         Intel torch for a CUDA one.)')
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
        # torch.version.cuda is not a reliable discriminator. XPU and Metal
        # come after both, because neither sets torch.version.cuda and a
        # build that does set it is the one being described.
        info = best_embed[2]
        xpu_b = info.get('xpu') or info.get('xpu_build')
        mps_b = info.get('mps') or info.get('mps_built')
        kind = ('ROCm/HIP %s' % hip if hip else
                'CUDA %s' % info.get('cuda_build')
                if info.get('cuda_build') else
                'Intel XPU' if xpu_b else
                'Apple Metal' if mps_b else 'CPU-only')
        made_for = ('AMD' if hip else 'NVIDIA' if info.get('cuda_build')
                    else 'Intel' if xpu_b else None)
        gpu_v = vendors & {'NVIDIA', 'AMD', 'Intel'}
        setup_line = _setup_line(info.get('exe') or best_embed[0][0])
        if best_embed[2].get('cuda') and best_embed[2].get('cuda_fits') is False:
            print('       CPU only - this %s build has no kernels for the GPU'
                  % kind)
            print('       (compute %s; PyTorch\'s CUDA 12.8 and newer builds start'
                  % best_embed[2].get('cuda_cc'))
            print('       at 7.5). The CUDA 12.6 build still supports this card.')
            # setup, not a bare pip line: pip refuses a distro-managed Python,
            # and setup already knows this card and the index to use
            print('       Setup replaces the build with that one, and asks first:')
            print('         ' + setup_line)
        elif (best_embed[2].get('cuda') or best_embed[2].get('xpu')
                or best_embed[2].get('mps')):
            print('       GPU acceleration available  (%s build).' % kind)
        elif made_for and gpu_v and made_for not in gpu_v:
            # a CUDA build on an AMD-only machine is not a driver problem
            print('       CPU only - this %s build is made for %s GPUs, and the'
                  % (kind, made_for))
            print('       GPU here is %s. Setup replaces it with the right build:'
                  % ', '.join(sorted(gpu_v)))
            print('         ' + setup_line)
        else:
            if gpus and kind == 'CPU-only':
                v = info.get('v', '')
                print('       CPU only - BUT a GPU is present (%s).' % gpus[0])
                if IS_WIN and gpu_v == {'AMD'} and not v.startswith('3.12'):
                    # nothing to switch TO: removing torch to reinstall the
                    # same CPU build was all the old advice could achieve
                    print('       AMD ships its Windows GPU builds for Python 3.12 only;')
                    print('       see "AMD GPUs" in the README.')
                else:
                    print('       A CPU-only wheel (%s) can never use it. Setup replaces'
                          % tv)
                    print('       it with the build for this GPU, and asks first:')
                    print('         ' + setup_line)
            elif gpus:
                print('       CPU only - a %s build is installed but no device is'
                      % kind)
                print('       visible; check the driver for %s.'
                      % (', '.join(sorted(vendors - {'other'})) or 'your GPU'))
            else:
                print('       CPU only - slower, but fine (no GPU detected).')
        print('')
        exe_pin = best_embed[2].get('exe') or best_embed[0][0]
        # quoted, so a path with spaces (or "&") can be pasted as it stands:
        # unquoted, export stopped at the first space and set at the "&"
        if IS_WIN:
            print('  If Embed-Images.bat picks the wrong one, force it:')
            print('       set "IMGDEDUP_PYTHON=%s"' % exe_pin)
        else:
            import shlex
            print('  If ./imgdedup.sh picks the wrong one, force it:')
            print('       export IMGDEDUP_PYTHON=%s' % shlex.quote(exe_pin))
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
            if not pick[2].get('pip'):
                # pip commands for an interpreter with no pip all fail with
                # "No module named pip", right under its own [MISS] pip line
                print('       It has no pip, so nothing installs into it as it is.')
                print('       Setup makes a virtual environment, which brings its own:')
                print('         ' + _setup_line(exe))
            elif not _installable(pick[2]):
                # No pip command can succeed against this interpreter, so the
                # per-GPU menu would be identical copies of the same refusal.
                print('       ' + _hint('torch', exe, pick[2]))
            else:
                # setup reads the current index list and the GPU; a fixed
                # CUDA index here went stale (and was wrong for older cards)
                print('       Setup picks the PyTorch build for your GPU, and asks first:')
                print('         ' + _setup_line(exe))
                print('       Or by hand, CPU only:')
                print('         ' + _hint('torch --index-url '
                                          'https://download.pytorch.org/whl/cpu',
                                          exe, pick[2]))
                print('         ' + _hint('transformers', exe, pick[2]))

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
