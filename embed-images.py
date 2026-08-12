#!/usr/bin/env python3
"""
embed-images.py  -  optional semantic tier for the image inventory.

Reads an image-inventory*.jsonl produced by collect-image-inventory.py,
opens each ORIGINAL image, and computes a CLIP embedding. Output is
image-embeddings.jsonl next to the inventory: one line per image,
keyed by the file's SHA-256 so renames/moves don't matter.

The analyzer uses these to find semantic matches that pixel comparison
misses: aggressive crops, filtered/recolored edits, mirrored saves,
re-captioned memes.

READ-ONLY with respect to your images. Resumable: re-running skips
every sha already present in the output file. Everything stays on this
machine unless you pass --share.

Usage:
    python embed-images.py <inventory.jsonl | folder containing one>
Options:
    --model NAME    HF model id (default: openai/clip-vit-base-patch32;
                    first run downloads ~600 MB into the HF cache)
    --device D      auto (default) / cuda / cpu. auto prefers the GPU and
                    explains exactly why when it cannot use one
    --root DIR      override the image root recorded in the inventory
    --batch N       batch size (default: 64 on GPU, 8 on CPU)
    --workers N     image decode/preprocess threads (default: auto).
                    Decoding runs ahead of the model on a thread pool, so
                    the GPU never waits for the disk.
    --gpu-preprocess
                    resize opaque 2x-or-larger downscales and normalize on
                    the GPU with antialiased bicubic interpolation. Smaller,
                    transparent, and animated images stay on the exact Pillow
                    path. Faster on a CPU-bound run, but opt-in:
                    the downscale kernel is not pixel-identical to Pillow and
                    produces a separate, resume-incompatible vector set.
    --no-draft      decode JPEGs at full resolution instead of letting
                    libjpeg downscale to ~4x the model input during decode
                    (draft is the same trick the collector uses for
                    thumbnails; disable only if you want bit-identical
                    decoding at any speed)
    --share         also copy the output into the shared _inventory folder,
                    where an AI assistant (e.g. Claude) can pick it up.
                    Off by default - everything stays local.
    --mirror-dir D  copy the output to a custom folder instead ('' = off)

EXIF orientation is applied before embedding - the same correction the
collector applies to thumbnails - and the output header records it as
"pre": see PRE_TAG below. Version history lives in CHANGES.md.
"""
import argparse
import base64
import collections
import glob
import json
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import TimeoutError as FuturesTimeout

# Exotic characters in filenames must not crash a redirected console.
# stdout only: stderr already defaults to backslashreplace, which cannot
# raise, and 'replace' would only destroy detail in tracebacks.
for _s in (sys.stdout,):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

def default_mirror_dir():
    """Where --share puts a copy; must match the collector's choice.
    IMGDEDUP_SHARE_DIR overrides. (The old hardcoded Windows path would
    silently create a directory named 'C:\\Users\\...' on Linux, where a
    backslash is an ordinary filename character.)"""
    env = os.environ.get('IMGDEDUP_SHARE_DIR')
    if env:
        return env
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        base = os.path.join(os.path.expanduser('~'), 'Library',
                            'Application Support')
    else:
        base = os.environ.get('XDG_DATA_HOME') or os.path.join(
            os.path.expanduser('~'), '.local', 'share')
    return os.path.join(base, 'image-inventory', '_inventory')


MIRROR_DIR = default_mirror_dir()

def _hint(pkg, exe=None):
    """How to install PKG here. Routed through _setup (stdlib-only) because
    '--user' is refused outright on a distro-managed Python - Arch, Debian
    12+, Fedora 38+ - where a virtual environment is the way through."""
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        if d not in sys.path:
            sys.path.insert(0, d)
        from _setup import pip_hint
        return pip_hint(pkg, exe)
    except Exception:
        return '"%s" -m pip install --user %s' % (exe or sys.executable, pkg)


try:
    from PIL import Image, ImageOps
except ImportError:
    print('Pillow missing:  ' + _hint('pillow'))
    sys.exit(2)

HEIF_OK = False
try:
    import pillow_heif                      # optional; the collector registers
    pillow_heif.register_heif_opener()      # this too - without it every HEIC
    HEIF_OK = True                          # it inventoried fails to embed
    try:
        pillow_heif.register_avif_opener()
    except Exception:
        pass
except Exception:
    pass
try:
    import torch
except ImportError:
    print('PyTorch is not available in this Python interpreter.')
    print('Run Check-Image-Tools.bat - it lists every Python on this machine')
    print('and prints the exact pip command for the one you want to use.')
    sys.exit(2)
import numpy as np           # torch depends on it, so it is here if torch is


def _defuse_stale_torchvision():
    """torchvision ships a compiled _C.pyd linked against one exact torch
    build. Reinstalling torch (e.g. CPU wheel -> CUDA wheel) without also
    reinstalling torchvision leaves a torchvision that still *imports* as far
    as Python's presence check is concerned, but detonates on load with
    'entry point not found' / 'DLL load failed while importing _C'.

    transformers only checks that torchvision is PRESENT, then uses it - so
    one stale package takes CLIP down with it. Absent torchvision is fine
    (transformers falls back to its Pillow image path), so if it is present
    and broken we hide it, which is strictly better than crashing.

    Returns the exception when it was broken, else None."""
    import importlib.util
    if importlib.util.find_spec('torchvision') is None:
        return None                                  # absent: Pillow path, fine
    try:
        import torchvision  # noqa: F401
        return None                                  # present and healthy
    except Exception as exc:
        sys.modules['torchvision'] = None            # make it look absent
        for name in [k for k in list(sys.modules) if k.startswith('torchvision.')]:
            del sys.modules[name]
        return exc


_TV_BROKEN = _defuse_stale_torchvision()

try:
    from transformers import CLIPModel
    try:
        # The Pillow-backed processor, ASKED FOR BY NAME.
        #
        # Plain CLIPImageProcessor prints, once per run:
        #   `CLIPImageProcessor` requires torchvision (not installed);
        #   falling back to `CLIPImageProcessorPil` ...
        # and then hands back a CLIPImageProcessorPil regardless - verified,
        # the two construct the identical class. So naming it directly costs
        # nothing and removes a warning about a package this toolkit
        # deliberately does not need.
        #
        # Deliberately unconditional, rather than preferring the torchvision
        # backend where it happens to exist: the two backends resample
        # differently, and vectors that depend on which optional package a
        # machine has installed are not comparable between machines. Stable
        # embeddings are worth more here than a marginally faster resize.
        from transformers import CLIPImageProcessorPil as CLIPImageProcessor
    except ImportError:
        # older transformers: no Pil-suffixed name, and no warning either
        from transformers import CLIPImageProcessor
except ImportError as exc:
    import importlib.util
    if importlib.util.find_spec('transformers') is None:
        print('transformers is not installed for this Python:')
        print('  ' + _hint('transformers'))
    else:
        print('transformers is installed but could not be loaded:')
        print('  ' + type(exc).__name__ + ': ' + str(exc)[:200])
        print('')
        print('If that mentions a DLL or an entry point, a compiled package no')
        print('longer matches the installed torch. Reinstall the pair together:')
        print('  ' + _hint('--force-reinstall torch torchvision --index-url '
                           'https://download.pytorch.org/whl/cu132'))
    sys.exit(2)

Image.MAX_IMAGE_PIXELS = 300_000_000

# See collect-image-inventory.py: Pillow's raw stderr warning reads as a
# crash and as an accusation about the user's own file. Suppressed at
# module level rather than per image, because catch_warnings() is
# process-global and not thread-safe. The >2x DecompressionBombError is a
# different class and still raises.
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

# Stamped into the output header as "pre", and checked on resume. Bump this
# whenever preprocessing changes what the model actually sees, so a resumed
# file cannot silently mix two populations of vectors.
#   exif      EXIF orientation applied before embedding
#   +pil      the Pillow image processor asked for BY NAME. Previously
#             CLIPImageProcessor resolved to the torchvision backend wherever
#             torchvision happened to be installed, and the two resample
#             differently - so the same images could embed differently on two
#             machines. Also covers the high-bit-depth rescale below.
#   +flat     transparent pixels composited onto white instead of having
#             their alpha silently dropped.
PRE_TAG = 'exif+pil+flat'
GPU_PRE_TAG = PRE_TAG + '+torch-aa-down2x-opaque+jpeg-draft'
GPU_FULL_JPEG_PRE_TAG = PRE_TAG + '+torch-aa-down2x-opaque+jpeg-full'
FULL_JPEG_PRE_TAG = PRE_TAG + '+jpeg-full'

# Deferred GPU resize keeps source pixels alive until a model batch flushes.
# Bound one image, the accumulated model batch, and the decode run-ahead so a
# folder of large lossless files cannot turn a speed option into unbounded
# host memory use. Files above the per-image ceiling simply take the normal
# Pillow path. The future window is derived from the worst-case RGB tensor,
# making its count cap a byte cap even when every decode finishes early.
GPU_RAW_IMAGE_PIXELS = 4_000_000        # 12 MB as RGB uint8
GPU_RAW_BATCH_BYTES = 256 * 1024 * 1024
GPU_RAW_QUEUE_BYTES = 768 * 1024 * 1024
GPU_RAW_QUEUE_RESULTS = max(1, GPU_RAW_QUEUE_BYTES //
                            (GPU_RAW_IMAGE_PIXELS * 3))


def _strict_pre_tag(tag):
    """Preprocessing populations which must never be mixed on resume."""
    return bool(tag and ('+torch-aa-down2x-opaque' in tag or
                         '+jpeg-full' in tag))


def default_workers():
    return max(2, min(8, os.cpu_count() or 4))


def _gpu_hint():
    """(vendors_present, names) without needing any vendor toolchain - the
    same PCI-vendor detection the setup helper uses. Imported softly so the
    embedder still runs if _setup.py is not beside it."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _setup
        gpus = _setup.detect_gpus()
        return set(v for v, _ in gpus), [n for _, n in gpus]
    except Exception:
        return set(), []
    finally:
        if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
            sys.path.pop(0)


def torch_build():
    """Which torch build this is. HIP is checked FIRST: a ROCm build reports
    torch.cuda.is_available() == True (HIP deliberately reuses the torch.cuda
    namespace, and 'rocm' is not a device string), and torch.version.cuda is
    NOT a reliable discriminator - PyTorch's own collect_env.py overrides it
    inside the HIP branch. Getting this backwards told AMD users with a
    working GPU that their setup could never use one."""
    if getattr(torch.version, 'hip', None):
        return 'rocm', 'ROCm/HIP ' + str(torch.version.hip)
    if getattr(torch.version, 'cuda', None):
        return 'cuda', 'CUDA ' + str(torch.version.cuda)
    try:
        if torch.xpu.is_available():
            return 'xpu', 'Intel XPU'
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return 'mps', 'Apple Metal'
    except Exception:
        pass
    return 'cpu', 'CPU-only build (' + torch.__version__ + ')'


def _install_hint(vendors):
    """The right reinstall line for the hardware actually present."""
    if 'NVIDIA' in vendors:
        return '    ' + _hint('torch --index-url '
                              'https://download.pytorch.org/whl/cu132')
    if 'AMD' in vendors:
        if os.name == 'nt':
            return ('    AMD ships ROCm wheels for Windows on Python 3.12 only;'
                    ' run:  python _setup.py')
        return '    ' + _hint('torch --index-url '
                              'https://download.pytorch.org/whl/rocm7.2')
    if 'Intel' in vendors:
        return '    ' + _hint('torch --index-url '
                              'https://download.pytorch.org/whl/xpu')
    return '    python _setup.py   (picks the right build for this machine)'


def gpu_device():
    """The torch device string for whatever accelerator this build can use.
    ROCm uses 'cuda' - that is not a bug, HIP reuses the namespace."""
    build, _ = torch_build()
    if build in ('cuda', 'rocm'):
        return 'cuda' if torch.cuda.is_available() else ''
    if build == 'xpu':
        return 'xpu'
    if build == 'mps':
        return 'mps'
    return ''


def resolve_device(requested):
    """Pick an accelerator and, on a CPU fallback, say exactly WHY. The usual
    cause is a build that cannot use the hardware present - which the wheel
    decides at install time, not the code at run time."""
    build, desc = torch_build()
    dev = gpu_device()
    vendors, names = _gpu_hint()

    if requested == 'cpu':
        print('Device: cpu (forced by --device cpu)')
        return 'cpu'
    if requested != 'auto':
        if requested == dev or (requested == 'cuda' and dev == 'cuda'):
            print('Device: %s (%s, %s)' % (dev, _dev_name(dev), desc))
            return dev
        print('--device %s was requested, but this torch cannot use it here:'
              % requested)
        print('  installed build: %s' % desc)
        if names:
            print('  hardware present: %s' % ', '.join(names[:2]))
        print(_install_hint(vendors))
        sys.exit(2)

    if dev:
        print('Device: %s (%s, %s)' % (dev, _dev_name(dev), desc))
        return dev

    print('Device: cpu')
    print('  Why not the GPU: torch here is a %s.' % desc)
    if names:
        print('  Hardware present: %s' % ', '.join(names[:2]))
        if build == 'cpu':
            print('  A CPU-only wheel can never use it - that is fixed by')
            print('  installing a different build, not by any setting:')
            print(_install_hint(vendors))
        else:
            print('  The build is right but no device is visible - check the')
            print('  driver is installed and current.')
    return 'cpu'


def _dev_name(dev):
    try:
        if dev == 'cuda':
            return torch.cuda.get_device_name(0)
        if dev == 'xpu':
            return torch.xpu.get_device_name(0)
    except Exception:
        pass
    return dev


def find_inventory(arg):
    if os.path.isfile(arg):
        return arg
    if os.path.isdir(arg):
        cands = [p for p in glob.glob(os.path.join(glob.escape(arg),
                                                   'image-inventory*.jsonl'))]
        if cands:
            return max(cands, key=os.path.getmtime)
    return None


def load_inventory(path):
    """Returns (root, records). Follows .partN siblings of the given file."""
    d, b = os.path.split(path)
    b = b[:-len('.jsonl')]
    if '.part' in b:                     # basename only: a folder named
        b = b.split('.part')[0]          # 'archive.part' must not truncate
    stem = os.path.join(glob.escape(d), glob.escape(b)) if d else glob.escape(b)
    paths = sorted(set(glob.glob(stem + '.jsonl') + glob.glob(stem + '.part*.jsonl')))
    root = None
    recs = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if isinstance(r, dict):
                    if 'root' in r and root is None:
                        root = r['root']
                    if r.get('kind') is None and 'p' in r and 'sha' in r:
                        recs.append(r)
    first = {}
    uniq = []
    for r in recs:
        f = first.get(r['sha'])
        if f is None:
            first[r['sha']] = r
            uniq.append(r)
        elif r['p'] != f['p']:
            # byte-identical twin under another path: remember it, so a
            # missing or unreadable first path can fall back to a healthy
            # copy instead of erroring while a perfect source sits on disk
            f.setdefault('_alts', []).append(r['p'])
    return root, uniq, paths


def model_input_edge(proc):
    """Shortest-edge target of the image processor, for draft decoding."""
    sz = getattr(proc, 'size', None)
    if isinstance(sz, dict):
        for k in ('shortest_edge', 'height', 'width'):
            if isinstance(sz.get(k), int):
                return sz[k]
    if isinstance(sz, int):
        return sz
    return 224


def _cfg(obj, *keys):
    """First present key of a transformers SizeDict / plain dict."""
    for k in keys:
        v = getattr(obj, k, None) if not isinstance(obj, dict) else obj.get(k)
        if isinstance(v, int):
            return v
    return None


def build_fast_preprocess(proc):
    """A bit-identical stand-in for `proc(images=im)`, or None.

    The processor spends most of its time not resizing. Measured on 1,500
    images from a real library: 22.96 ms inside `proc()` against 14.35 ms
    for the same four operations called directly, so roughly 8.6 ms per
    image is framework overhead - validation, list wrapping, dtype probing,
    channel-format negotiation - on a stage that is CPU-bound, not GPU-bound
    (36 ms of preprocessing per image against 10.5 ms of wall time means the
    card is waiting on threads).

    Matching it EXACTLY means copying two details that look like noise:

      * rescale upcasts to float64 before multiplying and only then casts
        down to float32. Multiplying in float32 directly is off by one ULP.
      * normalize then runs in float32, mean and std cast to the image
        dtype - not the other way around.

    Get either wrong and the vectors shift by ~5e-07, which is small enough
    to look like nothing and still change which pairs the analyzer nominates.
    So this refuses to engage unless every setting matches what it
    replicates, and the caller proves it byte-for-byte before use."""
    size, crop = getattr(proc, 'size', None), getattr(proc, 'crop_size', None)
    edge = _cfg(size, 'shortest_edge')
    ch = _cfg(crop, 'height')
    cw = _cfg(crop, 'width')
    if edge is None or ch is None or cw != ch:
        return None
    if _cfg(size, 'height') or _cfg(size, 'width') or _cfg(size, 'longest_edge'):
        return None                      # a fixed-size processor, not this one
    if not all(getattr(proc, f, False) for f in
               ('do_resize', 'do_center_crop', 'do_rescale', 'do_normalize')):
        return None
    try:
        # A processor may carry resample=None, and int(None) raises rather
        # than declining - which would take the whole stage down instead of
        # falling back, exactly inverting what this function is for.
        if int(getattr(proc, 'resample', -1)) != int(Image.Resampling.BICUBIC):
            return None
    except (TypeError, ValueError):
        return None
    scale = getattr(proc, 'rescale_factor', None)
    mean, std = getattr(proc, 'image_mean', None), getattr(proc, 'image_std', None)
    if not isinstance(scale, float) or not mean or not std:
        return None
    if len(mean) != 3 or len(std) != 3:
        return None
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    bicubic = Image.Resampling.BICUBIC

    # The rescale+normalize chain maps each uint8 value through the same
    # scalar arithmetic every time, so run that arithmetic 256 times here -
    # in the EXACT op order the processor uses (float64 upcast for the
    # multiply, float32 for normalize; see the docstring for why one ULP
    # matters) - and the per-pixel work becomes three table gathers written
    # straight into the (3, H, W) planes the model wants. That also deletes
    # the transpose+ascontiguousarray copy the old chain needed. The caller
    # still proves the whole path byte-for-byte against proc() before use,
    # so a numpy that disagrees falls back rather than shipping a shifted
    # vector.
    v64 = (np.arange(256, dtype=np.float64) * scale).astype(np.float32)
    luts = [(v64 - mean[c]) / std[c] for c in range(3)]

    def fast(im):
        w, h = im.size
        # transformers' get_resize_output_image_size with default_to_square
        # False: scale the short side to `edge`, truncate the long side.
        if w <= h:
            nw, nh = edge, int(edge * h / w)
        else:
            nh, nw = edge, int(edge * w / h)
        im = im.resize((nw, nh), bicubic)
        left, top = (nw - cw) // 2, (nh - ch) // 2
        a = np.asarray(im.crop((left, top, left + cw, top + ch)))
        out = np.empty((3, ch, cw), dtype=np.float32)
        for c in range(3):
            np.take(luts[c], a[:, :, c], out=out[c])
        return torch.from_numpy(out)

    return fast


def build_gpu_preprocess(proc, device, pixel_budget=32_000_000):
    """Build the opt-in torch resize/normalize path, or return None.

    Decode, EXIF handling, high-bit-depth scaling, and alpha compositing stay
    on the CPU. This function takes the resulting RGB uint8 CHW tensors and
    moves the expensive resize and normalization to the accelerator.

    `antialias=True` is load-bearing. At the library's measured median 5.7x
    downscale, torch's default fixed four-tap cubic aliases badly enough to
    make the same image fall below the Tier A cosine gate. Antialiased bicubic
    measured median 0.9997 / minimum 0.9953 against Pillow preprocessing, but
    is still deliberately opt-in because torch's cubic kernel (a=-0.75) is
    not Pillow's (a=-0.5).

    Images in one model batch need not share a source shape. Equal shapes are
    grouped so common sprite/photo dimensions resize together, and very large
    groups are sliced by `pixel_budget` to cap the temporary float32 buffer.
    The returned tensor is already on DEVICE in NCHW float32 form.
    """
    size, crop = getattr(proc, 'size', None), getattr(proc, 'crop_size', None)
    edge = _cfg(size, 'shortest_edge')
    ch = _cfg(crop, 'height')
    cw = _cfg(crop, 'width')
    if edge is None or ch is None or cw != ch:
        return None
    if _cfg(size, 'height') or _cfg(size, 'width') or _cfg(size, 'longest_edge'):
        return None
    if not all(getattr(proc, f, False) for f in
               ('do_resize', 'do_center_crop', 'do_rescale', 'do_normalize')):
        return None
    scale = getattr(proc, 'rescale_factor', None)
    mean = getattr(proc, 'image_mean', None)
    std = getattr(proc, 'image_std', None)
    if not isinstance(scale, float) or not mean or not std:
        return None
    if len(mean) != 3 or len(std) != 3:
        return None

    mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    budget = max(1, int(pixel_budget))

    def gpu(tensors):
        if not tensors:
            return torch.empty((0, 3, ch, cw), dtype=torch.float32,
                               device=device)
        groups = collections.defaultdict(list)
        for i, t in enumerate(tensors):
            if (not torch.is_tensor(t) or t.dtype != torch.uint8 or
                    t.ndim != 3 or t.shape[0] != 3):
                raise ValueError('GPU preprocessing needs RGB uint8 CHW tensors')
            groups[tuple(t.shape)].append((i, t))

        out = torch.empty((len(tensors), 3, ch, cw), dtype=torch.float32,
                          device=device)
        with torch.inference_mode():
            for (_, h, w), items in groups.items():
                # Same geometry as transformers' Pillow processor: put the
                # short edge at EDGE, truncate the long edge, center crop.
                if w <= h:
                    nw, nh = edge, int(edge * h / w)
                else:
                    nh, nw = edge, int(edge * w / h)
                # Bound both the source and resized float32 working sets.
                per = max(1, budget // max(h * w, nh * nw))
                for start in range(0, len(items), per):
                    chunk = items[start:start + per]
                    cpu = torch.stack([t for _, t in chunk])
                    if device == 'cuda':
                        cpu = cpu.pin_memory()
                        x = cpu.to(device, non_blocking=True)
                    else:
                        x = cpu.to(device)
                    x = x.to(dtype=torch.float32)
                    x = torch.nn.functional.interpolate(
                        x, size=(nh, nw), mode='bicubic', align_corners=False,
                        antialias=True)
                    # Bicubic overshoots at sharp edges. Pillow writes the
                    # resize back to uint8 before normalization, so clip and
                    # quantize here too. Omitting the rounding preserves
                    # fractional samples Pillow never feeds to CLIP and made
                    # sparse graphics diverge much more than photographs.
                    x.clamp_(0.0, 255.0).round_()
                    left, top = (nw - cw) // 2, (nh - ch) // 2
                    x = x[:, :, top:top + ch, left:left + cw]
                    x.mul_(scale).sub_(mean).div_(std)
                    indices = torch.tensor([i for i, _ in chunk],
                                           dtype=torch.long, device=device)
                    out.index_copy_(0, indices, x)
        return out

    return gpu


def _result(fut):
    """An untimed Future.result() cannot be interrupted by Ctrl+C on
    Windows; the short timed wait keeps KeyboardInterrupt deliverable.
    (The collector's drain uses the same trick, for the same reason.)"""
    while True:
        try:
            return fut.result(timeout=0.5)
        except FuturesTimeout:
            pass


def bounded_map(pool, fn, items, window):
    """pool.map that keeps at most `window` tasks in flight and yields
    results in order - so decoding stays ahead of the model without
    loading every image into memory at once."""
    futs = collections.deque()
    it = iter(items)
    for x in it:
        futs.append(pool.submit(fn, x))
        if len(futs) >= window:
            yield _result(futs.popleft())
    while futs:
        yield _result(futs.popleft())


def main():
    ap = argparse.ArgumentParser(description='Compute CLIP embeddings for an image inventory.')
    ap.add_argument('inventory', help='inventory .jsonl, or a folder containing one')
    ap.add_argument('--model', default='openai/clip-vit-base-patch32')
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    ap.add_argument('--root', help='override image root folder')
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--workers', type=int, default=0,
                    help='decode/preprocess threads (default: auto)')
    ap.add_argument('--fp16', action='store_true',
                    help='run the model in float16 on a GPU (~2.9x faster). '
                         'Vectors shift slightly: measured max pairwise-cosine '
                         'change 0.0006, which can move a pair sitting exactly '
                         'on the Tier A cosine floor into the review tier. Off '
                         'by default for that reason')
    ap.add_argument('--gpu-preprocess', action='store_true',
                    help='resize opaque 2x-or-larger downscales and normalize '
                         'on the GPU with antialiased bicubic interpolation '
                         '(smaller/transparent/animated images retain Pillow). '
                         'Faster but not pixel-identical, '
                         'so it creates a resume-incompatible vector set and '
                         'is off by default')
    ap.add_argument('--no-draft', action='store_true',
                    help='decode JPEGs at full resolution (slower, no draft)')
    ap.add_argument('--share', action='store_true',
                    help='copy the output into the shared _inventory folder; off by default')
    ap.add_argument('--mirror-dir', default='',
                    help="copy the output to this folder instead ('' = off)")
    args = ap.parse_args()

    inv = find_inventory(args.inventory)
    if not inv:
        print('No image-inventory*.jsonl found at: ' + args.inventory)
        sys.exit(2)
    root, recs, parts = load_inventory(inv)
    root = args.root or root
    print('Inventory: ' + inv + ('  (+%d parts)' % (len(parts) - 1) if len(parts) > 1 else ''))
    print('Image root: ' + str(root))
    print('Unique images: ' + str(len(recs)))
    if not root or not os.path.isdir(root):
        print('Image root not found - pass --root <folder with the images>.')
        sys.exit(2)

    # Name the output after the inventory, so several inventories can share a
    # folder (the _inventory mirror does exactly that) without overwriting.
    base = os.path.basename(inv)
    for suffix in ('.jsonl',):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    if '.part' in base:
        base = base.split('.part')[0]
    stem = base.replace('image-inventory', 'image-embeddings')
    if stem == base:
        stem = base + '-embeddings'
    out = os.path.join(os.path.dirname(os.path.abspath(inv)), stem + '.jsonl')

    # Resolved HERE, before the resume guards, because the precision that
    # actually gets written depends on it. --fp16 is honoured only on a GPU,
    # so a run that asks for it and lands on the CPU writes fp32 vectors -
    # and the guard used to compare against the FLAG, so it saw fp16 ==
    # fp16, waved the run through, and appended fp32 vectors to a file whose
    # header says fp16. That is precisely the silent mixing it exists to
    # stop, and it was invisible because the opposite direction (an fp16
    # file resumed without the flag) was caught correctly.
    device = resolve_device(args.device)
    use_half = args.fp16 and device in ('cuda', 'xpu')
    use_gpu_preprocess = args.gpu_preprocess and device != 'cpu'
    if use_gpu_preprocess:
        pre_tag = (GPU_FULL_JPEG_PRE_TAG if args.no_draft else GPU_PRE_TAG)
    else:
        pre_tag = FULL_JPEG_PRE_TAG if args.no_draft else PRE_TAG

    done = set()
    prev_err = {}                     # sha -> last recorded error message
    prev_model = prev_dim = prev_pre = prev_prec = None
    if os.path.exists(out):
        with open(out, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get('schema') == 'img-emb/1':
                    prev_model = r.get('model')
                    prev_dim = r.get('dim')
                    prev_pre = r.get('pre')
                    prev_prec = r.get('prec')
                elif 'sha' in r and 'v' in r:
                    done.add(r['sha'])
                    prev_err.pop(r['sha'], None)
                elif 'sha' in r and 'err' in r:
                    # failures are NOT done: like the collector, retry them
                    # every run, so installing pillow-heif / restoring moved
                    # files and re-running fills exactly the gaps.
                    done.discard(r['sha'])
                    prev_err[r['sha']] = r.get('err')
        print('Resuming: ' + str(len(done)) + ' already embedded in ' + out)
        if prev_model and prev_model != args.model:
            print('')
            print('  [STOP] That file was built with a different model:')
            print('           existing: ' + str(prev_model))
            print('           now:      ' + args.model)
            print('  Vectors from two models are not comparable, so mixing them')
            print('  would silently corrupt the results. Either re-run with')
            print('    --model ' + str(prev_model))
            print('  or move/delete the existing file to start fresh.')
            sys.exit(2)
        # fp16 and fp32 vectors are not interchangeable at the
        # precision the Tier A cosine floor works at, so appending
        # one kind to a file of the other would quietly skew every
        # later comparison - the same trap the model check exists for.
        now_prec = 'fp16' if use_half else 'fp32'
        if done and prev_prec and prev_prec != now_prec:
            print('')
            print('  [STOP] that file was built with %s vectors; this run '
                  'would append %s.' % (prev_prec, now_prec))
            if args.fp16 and not use_half:
                # The subtle case: they DID pass --fp16, so "re-run with
                # --fp16" would be maddening advice. The flag is being
                # ignored because this run is not on a GPU.
                print('  --fp16 was given but only applies on a GPU, and this')
                print('  run is on %s. Either run it where the GPU is visible'
                      % device)
                print('  (see the device note above), or move the file aside')
                print('  and re-embed from scratch on this machine.')
            else:
                print('  Re-run %s --fp16, or move the file aside to start '
                      'fresh.' % ('with' if prev_prec == 'fp16' else 'without'))
            sys.exit(2)
        if (done and prev_pre != pre_tag and
                (_strict_pre_tag(prev_pre) or _strict_pre_tag(pre_tag))):
            print('')
            print('  [STOP] that file was built with different preprocessing:')
            print('           existing: %s' % (prev_pre or 'unrecorded'))
            print('           now:      %s' % pre_tag)
            print('  --gpu-preprocess and --no-draft both change which pixels')
            print('  reach the model. Mixing preprocessing populations would')
            print('  make cosine comparisons inconsistent. Move the embeddings')
            print('  file aside and re-embed the whole inventory.')
            sys.exit(2)
        if done and prev_pre != pre_tag:
            # Cumulative: name every change the existing file predates, so a
            # very old file is told all of them and a nearly-current one is
            # told only what actually differs.
            print('')
            print('  [WARN] Those vectors were built with different preprocessing')
            print('         (%s, this run uses %s):'
                  % (prev_pre or 'an unrecorded version', pre_tag))
            print('')
            if not prev_pre:
                print('         - EXIF orientation is applied before embedding now,')
                print('           matching the thumbnails. Affects only images that')
                print('           carry an orientation tag.')
            if prev_pre in (None, '', 'exif'):
                print('         - The image processor is asked for by name')
                print('           (CLIPImageProcessorPil). Where torchvision was')
                print('           installed the old code quietly used ITS backend,')
                print('           and the two resample differently. Built on a')
                print('           machine without torchvision, nothing changed.')
                print('         - 16-bit and float images used to clip to a white')
                print('           square; they are rescaled properly now.')
            if prev_pre in (None, '', 'exif', 'exif+pil'):
                print('         - Transparent pixels are composited onto white')
                print('           rather than having their alpha dropped, so a')
                print('           cut-out now embeds as what you actually see.')
            if prev_pre == 'exif+pil+flat+big':
                # Shortlived tag: for one commit, images >= 80 MP were
                # shrunk during decode. That is gone - the memory saving now
                # comes from changes that alter no pixels at all - so those
                # vectors differ only for images that large.
                print('         - Images at or above 80 MP were shrunk during')
                print('           decode; they no longer are, so only images')
                print('           that big embed differently.')
            print('')
            print('         Every one of these is conditional on the images')
            print('         involved, which is why this warns instead of')
            print('         stopping - unlike the model and precision checks,')
            print('         which change every vector. For a fully consistent')
            print('         file, move or delete it and re-embed from scratch.')
            print('')
    todo = [r for r in recs if r['sha'] not in done]
    if not todo:
        print('Nothing new to embed. Done.')
        return

    if _TV_BROKEN is not None:
        print('')
        print('  [WARN] torchvision is installed but broken, so it was ignored:')
        print('           ' + type(_TV_BROKEN).__name__ + ': ' + str(_TV_BROKEN)[:150])
        print('         This is the classic aftermath of reinstalling torch without')
        print('         reinstalling torchvision - its compiled _C.pyd is linked')
        print('         against the torch build you replaced.')
        print('         Embedding continues on the Pillow image path (same results,')
        print('         marginally slower preprocessing). To clean it up, either:')
        print('           "' + sys.executable + '" -m pip uninstall torchvision')
        print('         (this tool never needs it), or reinstall the matched pair:')
        print('           ' + _hint('--force-reinstall torch torchvision '
                                    '--index-url '
                                    'https://download.pytorch.org/whl/cu132'))
        print('')

    batch = args.batch or (64 if device == 'cuda' else 8)
    workers = args.workers if args.workers > 0 else default_workers()
    print('Batch size: ' + str(batch) + '   decode threads: ' + str(workers))
    print('Loading model ' + args.model + ' (first run downloads it) ...')
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(args.model)
    dim = getattr(model.config, 'projection_dim', 0) or 0
    print('Model ready.' + (' Declared embedding dim: ' + str(dim) if dim else ''))

    input_edge = model_input_edge(proc)
    draft_px = 0 if args.no_draft else input_edge * 4

    # The exact Pillow path remains available even in GPU mode: upscaling is
    # cheap and is where the two cubic kernels differ most, so small images
    # stay byte-identical to the default rather than taking a quality risk for
    # no meaningful speed gain.
    fast_prep = build_fast_preprocess(proc)
    if fast_prep is not None:
        try:
            rng = np.random.default_rng(0)
            for w, h in ((640, 480), (480, 640), (224, 224),
                         (1000, 233), (37, 41)):
                probe = Image.fromarray(
                    rng.integers(0, 256, (h, w, 3), dtype=np.uint8), 'RGB')
                if not torch.equal(
                        proc(images=probe,
                             return_tensors='pt')['pixel_values'][0],
                        fast_prep(probe)):
                    fast_prep = None
                    break
        except Exception:
            fast_prep = None

    gpu_prep = None
    if use_gpu_preprocess:
        gpu_prep = build_gpu_preprocess(proc, device)
        if gpu_prep is None:
            print('')
            print('  [STOP] --gpu-preprocess does not support this model\'s image')
            print('         processor settings. No vectors were written.')
            sys.exit(2)
        try:
            rng = np.random.default_rng(0)
            probe_images = [Image.fromarray(rng.integers(
                0, 256, (h, w, 3), dtype=np.uint8), 'RGB')
                for w, h in ((640, 480), (480, 640), (233, 1000))]
            probes = [torch.from_numpy(np.array(im, copy=True)).permute(
                2, 0, 1).contiguous() for im in probe_images]
            got = gpu_prep(probes)
            crop_edge = _cfg(getattr(proc, 'crop_size', None), 'height')
            if (got.shape != (len(probes), 3, crop_edge, crop_edge) or
                    got.dtype != torch.float32 or not torch.isfinite(got).all()):
                raise ValueError('unexpected probe output')
            reference = proc(images=probe_images,
                             return_tensors='pt')['pixel_values']
            if float((got.cpu() - reference).abs().mean()) > 0.02:
                raise ValueError('GPU resize differs too far from Pillow')
        except Exception as exc:
            print('')
            print('  [STOP] --gpu-preprocess failed its startup probe:')
            print('         %s: %s' % (type(exc).__name__, str(exc)[:160]))
            print('         No vectors were written.')
            sys.exit(2)
        print('Preprocess: GPU antialiased bicubic for opaque 2x+ downscales')
        print('            (opt-in; Pillow retained for other images)')
    else:
        if args.gpu_preprocess:
            print('(--gpu-preprocess ignored: it requires a GPU)')
    if fast_prep is None:
        print('   note: preprocessing through the image processor '
              '(the direct path does not match this model exactly).')

    def prep_one(r):
        """Decode + preprocess one image off the main thread.
        Returns (rec, pixel_tensor, error_or_None, path_used). Tries every
        known byte-identical path for this sha before giving up.

        The path travels with the result because it is no longer necessarily
        r['p']: reporting the first path while describing the last path's
        failure named a file that was merely missing and hid the one that
        was actually broken."""
        last = 'no readable path'
        last_rel = r['p']
        for rel in [r['p']] + r.get('_alts', []):
            # inventories store '/' so they travel between OSes
            full = os.path.join(root, *rel.split('/'))
            try:
                with Image.open(full) as im:
                    # Inventory metadata can be absent or stale. Ask the
                    # opened file too, so a legacy animated GIF/WebP cannot
                    # enter the GPU resize population that the A/B safety
                    # gate intentionally excluded. Pillow calls multi-picture
                    # JPEGs animated; the collector intentionally treats MPO
                    # as a still JPEG, so preserve that exception here.
                    opened_animated = (im.format != 'MPO' and
                                       bool(getattr(im, 'is_animated', False)))
                    is_animation = bool(r.get('anim') or opened_animated)
                    # 'MPO' as well - see process_one in the collector: a
                    # phone's dual-camera .jpg is a JPEG that Pillow
                    # relabels, and gating on 'JPEG' alone silently gave up
                    # the fast path on it.
                    if draft_px and im.format in ('JPEG', 'MPO'):
                        # libjpeg can decode at 1/2, 1/4, 1/8 scale; asking
                        # for ~4x the model input keeps resampling quality
                        # intact while skipping most of the decode work.
                        im.draft(None, (draft_px, draft_px))
                    # See make_thumb: exif_transpose copies the whole image
                    # even with no orientation tag. in_place skips that and
                    # returns None, so the call is bare.
                    ImageOps.exif_transpose(im, in_place=True)
                    # See make_thumb in collect-image-inventory.py: Pillow
                    # CLIPS high-bit-depth data at 255 rather than rescaling
                    # it, turning every 16-bit image into a white square.
                    # The collector and the embedder must agree here, or the
                    # pixel score and the CLIP vector describe different
                    # pictures.
                    if im.mode in ('I;16', 'I;16L', 'I;16B', 'I;16N'):
                        # see make_thumb: point() rejects the byte-order
                        # variants, and I;16B is just a big-endian TIFF
                        if im.mode != 'I;16':
                            im = im.convert('I')
                        im = im.point(lambda v: v * (1 / 257)).convert('L')
                    elif im.mode in ('I', 'F'):
                        lo, hi = im.getextrema()
                        if not (abs(lo) < 3.0e38 and abs(hi) < 3.0e38):
                            # Inf/NaN collapses the scale to 0 and blacks
                            # out the image - see make_thumb. Refused for
                            # the same reason, and refused HERE too so the
                            # embedder and the collector agree about which
                            # files exist.
                            raise ValueError('image has no finite pixel '
                                             'range (contains Inf or NaN)')
                        span = (hi - lo) or 1
                        im = im.point(
                            lambda v: (v - lo) * (255.0 / span)).convert('L')
                    # See make_thumb in collect-image-inventory.py: alpha is
                    # composited onto white, not dropped. The two must agree
                    # exactly, or the pixel score and the CLIP vector are
                    # describing different pictures.
                    had_alpha = (im.mode in ('RGBA', 'LA', 'PA', 'La')
                                 or 'transparency' in im.info)
                    if had_alpha:
                        # One buffer instead of three - see make_thumb.
                        if im.mode == 'La':
                            # see make_thumb: La converts only to LA
                            im = im.convert('LA')
                        rgba = im if im.mode == 'RGBA' else im.convert('RGBA')
                        bg = Image.new('RGB', rgba.size, (255, 255, 255))
                        bg.paste(rgba, (0, 0), rgba)
                        im = bg
                    if im.mode != 'RGB':
                        # convert() on an already-RGB image is documented to
                        # return self.copy() - a full-frame allocate+memcpy
                        # of the DRAFTED image (a 4000x3000 JPEG drafts to
                        # 2000x1500: both edges must stay >= 896, so 1/2 is
                        # the deepest scale libjpeg may pick, ~9 MB copied
                        # for nothing). JPEG decodes straight to RGB, so the
                        # dominant case paid it on every image; the alpha
                        # branch above already ends in an RGB `bg`.
                        im = im.convert('RGB')
                    if (gpu_prep is not None and not is_animation
                            and not had_alpha
                            and min(im.size) >= input_edge * 2
                            and im.width * im.height <= GPU_RAW_IMAGE_PIXELS):
                        # Keep transfer bandwidth at one byte/channel. The
                        # accelerator performs float conversion, resize,
                        # rescale, and normalize together at batch flush.
                        # Animated files deliberately retain the default
                        # vector path: on a real corpus, a tiny embedding
                        # shift moved which of two background-dominated GIF
                        # pairs crossed the automatic-delete boundary. Their
                        # sampled-frame fingerprints were too coarse to veto
                        # either false match, so animation safety must not
                        # depend on a different resize kernel.
                        a = np.array(im, dtype=np.uint8, copy=True)
                        px = torch.from_numpy(a).permute(2, 0, 1).contiguous()
                    else:
                        px = (fast_prep(im) if fast_prep is not None else
                              proc(images=im,
                                   return_tensors='pt')['pixel_values'][0])
                return r, px, None, rel
            except Exception as e:
                last, last_rel = rel + ': ' + str(e)[:200], rel
        return r, None, last, last_rel

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    ok = err = 0
    flushed_batches = 0
    consec_fail = 0
    header_written = bool(done)
    pool = ThreadPoolExecutor(max_workers=workers)
    with open(out, 'a', encoding='utf-8') as f:

        # use_half was decided up with the device, before the resume guard
        if args.fp16 and not use_half:
            print('(--fp16 ignored: it only applies on a GPU)')
        elif use_half:
            print('Precision: float16 (--fp16). Vectors differ slightly from a '
                  'float32 run;')
            print('           do not mix them in one embeddings file.')

        def infer(tensors):
            if gpu_prep is not None:
                raw = [(i, t) for i, t in enumerate(tensors)
                       if t.dtype == torch.uint8]
                exact = [(i, t) for i, t in enumerate(tensors)
                         if t.dtype != torch.uint8]
                if raw and not exact:
                    inp = gpu_prep([t for _, t in raw])
                elif exact and not raw:
                    inp = torch.stack([t for _, t in exact])
                    if device == 'cuda':
                        inp = inp.pin_memory().to(device, non_blocking=True)
                    else:
                        inp = inp.to(device)
                else:
                    crop_edge = exact[0][1].shape[-1]
                    inp = torch.empty((len(tensors), 3, crop_edge, crop_edge),
                                      dtype=torch.float32, device=device)
                    moved = torch.stack([t for _, t in exact])
                    if device == 'cuda':
                        moved = moved.pin_memory().to(device, non_blocking=True)
                    else:
                        moved = moved.to(device)
                    resized = gpu_prep([t for _, t in raw])
                    exact_ix = torch.tensor([i for i, _ in exact],
                                            dtype=torch.long, device=device)
                    raw_ix = torch.tensor([i for i, _ in raw],
                                          dtype=torch.long, device=device)
                    inp.index_copy_(0, exact_ix, moved)
                    inp.index_copy_(0, raw_ix, resized)
            else:
                inp = torch.stack(tensors)
                if device == 'cuda':
                    inp = inp.pin_memory().to(device, non_blocking=True)
                else:
                    inp = inp.to(device)
            with torch.inference_mode():
                if use_half:
                    with torch.autocast(device, dtype=torch.float16):
                        v = model.get_image_features(pixel_values=inp)
                else:
                    v = model.get_image_features(pixel_values=inp)
                # transformers 4.x returns the projected tensor directly;
                # 5.x returns a model-output whose pooler_output holds
                # the (already projected) image features.
                if not torch.is_tensor(v):
                    po = getattr(v, 'pooler_output', None)
                    v = po if po is not None else v[0]
                v = torch.nn.functional.normalize(v.float(), dim=-1).cpu()
            return v.to(torch.float16).numpy()

        def flush_batch(metas, tensors, final):
            # metas holds (record, path_actually_read) pairs - the path can
            # differ from record['p'] when a byte-identical twin supplied
            # the pixels, and the file we name must be the file we read.
            nonlocal ok, err, header_written, flushed_batches, consec_fail
            if metas:
                try:
                    v16 = infer(tensors)
                    consec_fail = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # One pathological image must not kill (and, via resume,
                    # permanently wedge) the whole run: retry one by one so
                    # only the offender is recorded as an error.
                    if len(metas) == 1:
                        msg = str(e)[:200]
                        r0, used0 = metas[0]
                        if prev_err.get(r0['sha']) != msg:
                            f.write(json.dumps({'sha': r0['sha'],
                                                'p': used0, 'err': msg},
                                               ensure_ascii=False) + '\n')
                            prev_err[r0['sha']] = msg
                        err += 1
                        consec_fail += 1
                        if consec_fail >= 8:
                            # every single image failing INSIDE the model is
                            # not an image problem - it is an OOM / driver /
                            # context problem. Stop loudly; resume retries.
                            print('')
                            print('  [STOP] %d consecutive inference failures - this is '
                                  'systemic' % consec_fail)
                            print('         (GPU out of memory, driver, or torch state), '
                                  'not bad images.')
                            print('         Last error: ' + msg[:150])
                            print('         Fix the cause and re-run; failed files are '
                                  'retried automatically.')
                            f.flush()
                            sys.exit(1)
                        return
                    print('  (batch of %d failed: %s - retrying one by one)'
                          % (len(metas), str(e)[:80]))
                    for r1, t1 in zip(metas, tensors):
                        flush_batch([r1], [t1], False)
                    return
                real_dim = int(v16.shape[1])
                if not header_written:
                    f.write(json.dumps({'schema': 'img-emb/1', 'model': args.model,
                                        'dim': real_dim, 'root': root, 'pre': pre_tag,
                                        'prec': 'fp16' if use_half else 'fp32',
                                        'started': int(time.time() * 1000)}) + '\n')
                    header_written = True
                    if dim and dim != real_dim:
                        print('  (note: model config said dim %d, actual vectors are %d;'
                              ' recording the actual width)' % (dim, real_dim))
                elif prev_dim and prev_dim != real_dim:
                    print('')
                    print('  [STOP] existing file has dim %d but this model produces %d.'
                          % (prev_dim, real_dim))
                    sys.exit(2)
                for (r, used), vec in zip(metas, v16):
                    b64 = base64.b64encode(vec.tobytes()).decode('ascii')
                    f.write(json.dumps({'sha': r['sha'], 'p': used, 'v': b64},
                                       ensure_ascii=False) + '\n')
                    ok += 1
            flushed_batches += 1
            if flushed_batches % 10 == 0 or final:
                el = time.time() - t0
                rate = (ok + err) / el if el > 0 else 0
                left = len(todo) - ok - err
                rem = left / rate if rate > 0 else 0
                print('  %d/%d  (%.1f img/s, ~%dm %ds left)'
                      % (ok + err, len(todo), rate, rem // 60, rem % 60),
                      flush=True)
            f.flush()

        metas, tensors, raw_bytes = [], [], 0
        try:
            for r, px, e, used in bounded_map(
                    pool, prep_one, todo,
                    window=(GPU_RAW_QUEUE_RESULTS
                            if gpu_prep is not None
                            else max(batch * 2, 16))):
                if e is not None:
                    if prev_err.get(r['sha']) != e:
                        # only record a failure once per distinct message -
                        # a permanent failure must not grow the file each run
                        f.write(json.dumps({'sha': r['sha'], 'p': used,
                                            'err': e}, ensure_ascii=False) + '\n')
                        prev_err[r['sha']] = e
                    err += 1
                    continue
                px_bytes = px.numel() if px.dtype == torch.uint8 else 0
                if (metas and px_bytes
                        and raw_bytes + px_bytes > GPU_RAW_BATCH_BYTES):
                    flush_batch(metas, tensors, False)
                    metas, tensors, raw_bytes = [], [], 0
                metas.append((r, used))
                tensors.append(px)
                raw_bytes += px_bytes
                if len(metas) >= batch:
                    flush_batch(metas, tensors, False)
                    metas, tensors, raw_bytes = [], [], 0
            flush_batch(metas, tensors, True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    print('')
    print('Done: %d embedded, %d unreadable, %.1f min.' % (ok, err, (time.time() - t0) / 60))
    print('Output: ' + out)

    mdir = args.mirror_dir or (MIRROR_DIR if args.share else '')
    if mdir:
        try:
            os.makedirs(mdir, exist_ok=True)
            tag = os.path.basename(root.rstrip('\\/')) or 'root'
            # mirror under the real output name, so image-embeddings-2.jsonl
            # does not overwrite the mirror of image-embeddings.jsonl
            shutil.copy2(out, os.path.join(mdir, tag + ' - ' + os.path.basename(out)))
            print('Copied to the shared folder for your AI assistant: ' + mdir)
        except Exception as e:
            print('(shared copy skipped: ' + str(e)[:120] + ')')
    print('')
    print('Next: run Analyze-Inventory.bat - it picks these embeddings up')
    print('automatically. Local by default; --share is the opt-in handoff to')
    print('an AI assistant (e.g. Claude).')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('')
        print('Aborted. Re-running resumes where it stopped.')
        sys.exit(130)
