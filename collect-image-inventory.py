#!/usr/bin/env python3
"""
collect-image-inventory.py  -  stage 1: scan a folder into an inventory.

Walks a folder tree and, for every image it finds, writes one JSON line to
image-inventory.jsonl:  path, size, mtime, SHA-256, format, dimensions,
key EXIF fields, JPEG quality fingerprint, AI-generation text chunks (PNG),
and a small base64 thumbnail (EXIF-rotation corrected). Files are hashed,
decoded and thumbnailed on a thread pool, each read from disk exactly once;
records still stream out in scan order, so the output is deterministic.
Resumable: unchanged files (same size+mtime, same --thumb) are carried over
from previous inventories, newest inventory first. Paths are stored with
forward slashes, so an inventory travels between operating systems.

READ-ONLY: never modifies, moves, or deletes your images. Its own output
files are the only thing it writes. Everything stays on this machine unless
you pass --share (see below).

Usage:
    python collect-image-inventory.py               scan the script's own folder
    python collect-image-inventory.py D:\\Photos    scan a given folder

Options:
    --out FILE        output path (default: <scanned folder>\\image-inventory.jsonl)
    --thumb N         thumbnail max side, px            (default 128)
    --workers N       parallel worker threads           (default: auto,
                      min(CPU cores, 8); each file is read once and
                      hashed + decoded + thumbnailed off the main thread)
    --split-mb N      start a new .partN file when the current one
                      exceeds N megabytes               (default 200)
    --resume          reuse records from previous image-inventory*.jsonl
                      in the scanned folder for unchanged files
    --no-resume       never reuse, ignore previous inventories
                      (without either flag you are asked, when previous
                       inventories exist and the console is interactive)
    --share           also copy the output into the shared _inventory
                      folder, where an AI assistant (e.g. Claude) can pick
                      it up. Off by default - everything stays local.
    --mirror-dir DIR  copy the output to a custom folder instead ('' = off)

Version history lives in CHANGES.md.
"""
import argparse
import base64
import collections
import glob
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

# Exotic characters in filenames must not crash a redirected console
# (interactively Windows writes UTF-8; a piped/redirected stream may be
# cp1251 and would die on the first character it cannot encode).
# stdout only: stderr already defaults to backslashreplace, which cannot
# raise, and 'replace' would turn a traceback's recoverable \uXXXX escapes
# into unrecoverable '?'.
try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

if sys.version_info < (3, 9):
    print("This script needs Python 3.9 or newer.")
    sys.exit(2)

def _hint(pkg):
    """How to install PKG here. Routed through _setup (stdlib-only, so it
    imports before Pillow exists) because '--user' is refused outright on a
    distro-managed Python - Arch, Debian 12+, Fedora 38+ - where the answer
    is a virtual environment instead."""
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        if d not in sys.path:
            sys.path.insert(0, d)
        from _setup import pip_hint
        return pip_hint(pkg)
    except Exception:
        return '"%s" -m pip install --user %s' % (sys.executable, pkg)


try:
    from PIL import Image, ImageOps
except ImportError:
    print("Pillow is not installed for this Python interpreter.")
    print("Fix:  " + _hint('pillow'))
    sys.exit(2)

HEIF_OK = False
try:
    import pillow_heif                      # optional; enables .heic/.heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
    try:
        pillow_heif.register_avif_opener()  # separate call; .avif needs it on
    except Exception:                       # Pillow builds without native AVIF
        pass
except Exception:
    pass

Image.MAX_IMAGE_PIXELS = 300_000_000

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tif', '.tiff',
        '.heic', '.heif', '.avif'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules',
             '$recycle.bin', 'system volume information', '_inventory',
             # Linux/macOS equivalents. The trash ones matter: without them a
             # scan walks straight into ~/.local/share/Trash and inventories
             # files the user already deleted, which then come back as
             # "duplicates" of the originals they were deleted for.
             '.trash', 'lost+found', '.thumbnails', '.cache'}
SKIP_DIR_PREFIXES = ('.trash-',)        # freedesktop per-volume trash dirs
SCHEMA = 'img-inv/3'
def default_mirror_dir():
    """Where --share puts a copy. The old hardcoded Windows path was not
    merely wrong on Linux, it was silently destructive: backslash is a legal
    filename character there, so os.makedirs would happily create one
    directory literally named 'C:\\Users\\...' in the working directory.
    IMGDEDUP_SHARE_DIR overrides; otherwise a per-platform data location."""
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

# Files at or below this size are read into memory once and both hashed and
# decoded from that buffer (one disk read instead of two). Larger files fall
# back to streamed hashing plus a second read by the decoder.
ONE_READ_LIMIT = 32 * 1024 * 1024

EXIF_SUBIFD = 0x8769
TAG_DT_ORIG = 36867
TAG_DT = 306
TAG_MAKE = 271
TAG_MODEL = 272
TAG_SOFT = 305
TAG_ORIENT = 274

PNG_TEXT_KEYS = ('parameters', 'prompt', 'workflow', 'software',
                 'comment', 'description', 'source')


def default_workers():
    return max(2, min(8, os.cpu_count() or 4))


def strip_jsonl(path):
    """Path without a trailing .jsonl (case-insensitive); the extension that
    was actually there, or '.jsonl' as the default for odd --out names."""
    stem, ext = os.path.splitext(path)
    if ext.lower() == '.jsonl':
        return stem, ext
    return path, '.jsonl'


def exif_str(v):
    if v is None:
        return ''
    if isinstance(v, bytes):
        v = v.decode('ascii', 'replace')
    return str(v).replace('\x00', '').strip()[:120]


def make_thumb(im, thumb_px, fast=False):
    """Returns (fmt_flag, tw, th, b64). JPEG by default; lossless WebP when
    that is actually smaller (flat/UI content compresses better losslessly)."""
    im2 = ImageOps.exif_transpose(im)
    # High-bit-depth images must be RESCALED, not converted. Pillow's
    # I;16 -> L path CLIPS at 255, so every pixel above 1/257 of full scale
    # becomes pure white and a 16-bit photo thumbnails to a near-solid white
    # square. That is not merely ugly. Measured: two unrelated 16-bit images
    # both go 99.7% white, score MAD 0.71 against a Tier A gate of 4.0, and
    # are reported as duplicates of each other - a false DELETE
    # recommendation, which is the one outcome this tool must never produce.
    # The same pair as 8-bit scores 72.3 and is correctly rejected.
    #
    # 1/257, not 1/256: 257 is 65535/255, and it is what makes a 16-bit
    # image byte-identical to its own 8-bit export (verified, max diff 0),
    # so the two are correctly recognised as the same picture.
    if im2.mode in ('I;16', 'I;16L', 'I;16B', 'I;16N'):
        im2 = im2.point(lambda v: v * (1 / 257)).convert('L')
    elif im2.mode in ('I', 'F'):
        # 32-bit int and float carry no defined range, so normalise by what
        # is actually in the image rather than assuming one.
        lo, hi = im2.getextrema()
        span = (hi - lo) or 1
        im2 = im2.point(lambda v: (v - lo) * (255.0 / span)).convert('L')
    # Alpha must be COMPOSITED, not dropped. convert('RGB') discards the
    # alpha band and keeps whatever RGB happens to sit under transparent
    # pixels - colour no human has ever seen, because every viewer paints
    # those pixels as background. Measured: two cut-outs that look identical
    # (same black square, transparent background, junk RGB of (255,0,0,0)
    # vs (0,255,0,0) underneath) scored MAD 146 against each other and were
    # never reported as duplicates. The same artwork saved once transparent
    # and once flattened onto white missed each other the same way - a
    # headline use case for a deduplicator.
    #
    # White, because that is what viewers and file managers flatten onto,
    # and it must match embed-images.py exactly or the pixel score and the
    # CLIP vector end up describing different pictures.
    #
    # Fully-opaque RGBA is untouched: compositing an alpha=255 image over
    # anything returns that image, so ordinary screenshots and PNG exports
    # keep the pixels they already had. This also subsumes the old
    # palette-transparency hop - going via RGBA is exactly what Pillow's
    # "should be converted to RGBA images" warning asks for.
    if (im2.mode in ('RGBA', 'LA', 'PA', 'La')
            or 'transparency' in im2.info):
        rgba = im2.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        im2 = Image.alpha_composite(bg, rgba)
    if im2.mode != 'RGB':
        im2 = im2.convert('RGB')
    im2.thumbnail((thumb_px, thumb_px), Image.LANCZOS)
    bj = io.BytesIO()
    im2.save(bj, 'JPEG', quality=78)
    best, flag = bj.getvalue(), ''
    if fast:
        # The lossless-WebP attempt is 44x the cost of the JPEG encode and
        # takes a photo library's per-image work from 13 ms to 35 ms. It is
        # worth it by default - it genuinely wins on flat/UI content - but
        # no cheap test predicts the winner: measured over 180 images, JPEG
        # size overlaps in both directions and a smooth gradient has every
        # pixel unique yet still compresses 30x better losslessly. So this
        # is a choice, not a guess.
        return flag, im2.size[0], im2.size[1], base64.b64encode(best).decode('ascii')
    try:
        bw = io.BytesIO()
        # For lossless WebP, Pillow's `quality` is the compression-EFFORT
        # knob, not a fidelity knob - the pixels are identical at any value.
        # 100 selects the slowest encoder setting for a few percent of size;
        # 75 encodes several times faster and the JPEG-vs-WebP winner (a
        # 5-14x margin either way on real content) practically never flips.
        im2.save(bw, 'WEBP', lossless=True, quality=75)
        if len(bw.getvalue()) < len(best):
            best, flag = bw.getvalue(), 'w'
    except Exception:
        pass
    return flag, im2.size[0], im2.size[1], base64.b64encode(best).decode('ascii')


def process_one(full, rel, thumb_px, fast=False):
    rec = {'p': rel}
    st = os.stat(full)
    rec['b'] = st.st_size
    rec['mt'] = int(st.st_mtime * 1000)

    h = hashlib.sha256()
    data = None
    if st.st_size <= ONE_READ_LIMIT:
        with open(full, 'rb') as f:
            data = f.read()
        h.update(data)
    else:
        with open(full, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    rec['sha'] = h.hexdigest()

    src = io.BytesIO(data) if data is not None else full
    with Image.open(src) as im:
        rec['fmt'] = im.format or ''
        rec['w'], rec['h'] = im.size
        if getattr(im, 'is_animated', False):
            rec['anim'] = int(getattr(im, 'n_frames', 2))

        # JPEG quality fingerprint: sum of the luma quantization table.
        # Lower = finer quantization = higher quality / less recompressed.
        try:
            q = getattr(im, 'quantization', None)
            if q and 0 in q:
                rec['qsum'] = int(sum(q[0]))
        except Exception:
            pass

        # PNG text chunks (AI generation parameters etc.)
        try:
            txt = getattr(im, 'text', None)
            if txt:
                keep = {}
                for k, v in txt.items():
                    if k.lower() in PNG_TEXT_KEYS and isinstance(v, str) and v.strip():
                        keep[k[:24]] = v.strip()[:300]
                    if len(keep) >= 4:
                        break
                if keep:
                    rec['txt'] = keep
        except Exception:
            pass

        try:
            ex = im.getexif()
        except Exception:
            ex = None
        if ex:
            dt = ''
            try:
                dt = exif_str(ex.get_ifd(EXIF_SUBIFD).get(TAG_DT_ORIG))
            except Exception:
                pass
            if not dt:
                dt = exif_str(ex.get(TAG_DT))
            cam = (exif_str(ex.get(TAG_MAKE)) + ' ' + exif_str(ex.get(TAG_MODEL))).strip()
            sw = exif_str(ex.get(TAG_SOFT))
            ori = ex.get(TAG_ORIENT)
            if dt:
                rec['dt'] = dt
            if cam:
                rec['cam'] = cam
            if sw:
                rec['sw'] = sw
            if isinstance(ori, int) and ori != 1:
                rec['ori'] = ori

        if im.format == 'JPEG':
            im.draft(None, (thumb_px * 4, thumb_px * 4))
        flag, tw, th, b64 = make_thumb(im, thumb_px, fast)
        rec['tw'], rec['th'], rec['tb'] = tw, th, b64
        if flag:
            rec['tf'] = flag
    return rec


def work_one(full, rel, thumb_px, fast=False):
    """Thread-pool task: never raises (KeyboardInterrupt stays in the main
    thread). Returns (kind, record, ext_on_error)."""
    try:
        return 'ok', process_one(full, rel, thumb_px, fast), None
    except Exception as e:
        try:
            msg = type(e).__name__ + ': ' + str(e)
            # decoding runs from memory, so PIL names a BytesIO instead of the
            # file - put the filename back and drop the run-random 0x address.
            # The replacement MUST be a callable: as a string it would be
            # parsed as a regex template, and repr() of a name containing a
            # non-printable (U+00A0 from a browser paste, a soft hyphen, a
            # BOM) emits \xNN, which the template parser rejects - taking
            # down the whole scan from inside the handler meant to record
            # one unreadable file.
            msg = re.sub(r'<_io\.BytesIO object at 0x[0-9a-fA-F]+>',
                         lambda _m: repr(rel), msg)
        except Exception as inner:          # nothing here may raise
            msg = type(e).__name__ + ' (message unprintable: ' \
                + type(inner).__name__ + ')'
        rec = {'p': rel, 'err': msg[:300]}
        return 'err', rec, os.path.splitext(rel)[1].lower()


def walk_images(root, skip_names):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d.lower() not in SKIP_DIRS
                       and not d.lower().startswith(SKIP_DIR_PREFIXES)
                       and not d.startswith('.')]
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() not in EXTS:
                continue
            full = os.path.join(dirpath, name)
            if os.path.normcase(os.path.abspath(full)) in skip_names:
                continue
            yield full


def load_previous(root, thumb):
    """path -> record from any previous image-inventory*.jsonl in root.
    Read oldest-first so that when several inventories describe the same
    path, the NEWEST file's record wins. A file whose header records a
    different --thumb (or none at all, e.g. v1) is listed as superseded but
    its records are never reused - mixing thumbnail sizes would silently
    shift the analyzer's pixel scores."""
    prev = {}
    files = sorted(glob.glob(os.path.join(glob.escape(root), 'image-inventory*.jsonl')))
    for fp in sorted(files, key=lambda p: (os.path.getmtime(p), p)):
        try:
            with open(fp, encoding='utf-8') as f:
                file_thumb = None
                recs = {}
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(r, dict):
                        continue
                    if r.get('kind') == 'header' or 'schema' in r:
                        if isinstance(r.get('thumb'), int):
                            file_thumb = r['thumb']
                        continue
                    if r.get('kind') is None \
                            and 'p' in r and 'sha' in r and 'tb' in r:
                        recs[r['p']] = r
                if file_thumb == thumb:
                    prev.update(recs)
        except Exception:
            pass
    return files, prev


class PartWriter:
    # Binary writing with an explicit os.linesep keeps the bytes identical
    # to the old text-mode output while making tell() cheap: on a text
    # stream every tell() flushes the buffer, which turned each record into
    # its own OS write call.
    def __init__(self, base_out, split_mb, header):
        self.stem, self.ext = strip_jsonl(base_out)
        self.limit = max(1, int(split_mb)) * 1048576
        self.header = dict(header)
        self.part = 0
        self.paths = []
        self.f = None
        self.nl = os.linesep.encode('ascii')
        self._open_next()

    def _open_next(self):
        if self.f:
            self.f.close()
        self.part += 1
        path = (self.stem + self.ext) if self.part == 1 else \
            self.stem + '.part' + str(self.part) + self.ext
        self.paths.append(path)
        self.f = open(path, 'wb')
        hdr = dict(self.header)
        hdr['kind'] = 'header'
        hdr['part'] = self.part
        self.f.write(json.dumps(hdr).encode('utf-8') + self.nl)

    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False).encode('utf-8') + self.nl)
        if self.f.tell() > self.limit:
            self._open_next()

    def close(self, footer):
        self.f.write(json.dumps(footer).encode('utf-8') + self.nl)
        self.f.close()


def main():
    ap = argparse.ArgumentParser(description='Build a read-only image inventory (JSONL).')
    ap.add_argument('folder', nargs='?', help="folder to scan (default: this script's folder)")
    ap.add_argument('--out', help='output file path')
    ap.add_argument('--thumb', type=int, default=128, help='thumbnail max side, px')
    ap.add_argument('--fast-thumbs', action='store_true',
                    help='skip the lossless-WebP thumbnail attempt (~2.7x '
                         'faster scanning). Thumbnails are then always JPEG; '
                         'flat/UI content loses the smaller lossless copy, so '
                         'its stored pixels differ slightly from a default run '
                         '- do not mix the two in one resumed inventory')
    ap.add_argument('--workers', type=int, default=0,
                    help='parallel worker threads (default: auto)')
    ap.add_argument('--split-mb', type=float, default=200.0,
                    help='roll to a new .partN file past this size')
    ap.add_argument('--resume', action='store_true',
                    help='reuse records for unchanged files from previous inventories')
    ap.add_argument('--no-resume', action='store_true',
                    help='ignore previous inventories')
    ap.add_argument('--share', action='store_true',
                    help='also copy the output into the shared _inventory folder '
                         'for an AI assistant (e.g. Claude); off by default')
    ap.add_argument('--mirror-dir', default='',
                    help="copy the output to this folder instead ('' = off)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(args.folder) if args.folder else script_dir
    if not os.path.isdir(root):
        print('Not a folder: ' + root)
        sys.exit(2)

    prev_files, prev = ([], {})
    if not args.no_resume:
        prev_files, prev = load_previous(root, args.thumb)
    use_resume = False
    if prev:
        if args.resume:
            use_resume = True
        elif sys.stdin.isatty():
            print('Found a previous inventory here ('
                  + str(len(prev)) + ' records in ' + str(len(prev_files)) + ' file(s)).')
            ans = input('Reuse it for unchanged files? [Y/n]: ').strip().lower()
            use_resume = ans in ('', 'y', 'yes')
        else:
            print('Previous inventory found; pass --resume to reuse it. Rescanning fresh.')

    out = args.out or os.path.join(root, 'image-inventory.jsonl')
    # Canonicalize FIRST (an --out without .jsonl gets it appended), so the
    # path the collision guard checks is byte-identical to the path
    # PartWriter will actually open with 'wb'.
    base_stem, base_ext = strip_jsonl(out)
    out, ver = base_stem + base_ext, 2

    def _taken(path):
        # A name is taken if the file exists OR orphaned .partN siblings do -
        # rollover opens parts with 'w' and must never truncate an old run.
        stem, ext = strip_jsonl(path)
        return os.path.exists(path) or \
            glob.glob(glob.escape(stem) + '.part*' + ext)
    while _taken(out):
        out = base_stem + '-' + str(ver) + base_ext
        ver += 1

    skip_names = {os.path.normcase(os.path.abspath(p)) for p in prev_files}
    skip_names.add(os.path.normcase(os.path.abspath(out)))

    print('Scanning: ' + root)
    files = list(walk_images(root, skip_names))
    total = len(files)
    print('Found ' + str(total) + ' image files.')
    if total == 0:
        print('Nothing to do.')
        return
    if not HEIF_OK:
        n_heif = sum(1 for f in files
                     if os.path.splitext(f)[1].lower() in ('.heic', '.heif', '.avif'))
        if n_heif:
            print('NOTE: ' + str(n_heif) + ' HEIC/HEIF/AVIF files present but the codec')
            print('      is not installed; they will be listed as unreadable.')
            print('      Fix:  ' + _hint('pillow-heif'))
    workers = args.workers if args.workers > 0 else default_workers()
    print('Reading files (hash + thumbnail, %d worker%s). Progress below;'
          % (workers, '' if workers == 1 else 's'))
    print('Ctrl+C aborts safely - a partial inventory is still usable.')

    t0 = time.time()
    ok = err = reused = 0
    nbytes = 0
    written = 0
    unreadable_ext = {}
    unreadable_why = {}          # exception type -> [count, [(path, msg), ..]]
    header = {'schema': SCHEMA, 'root': root, 'files': total,
              'thumb': args.thumb, 'started': int(time.time() * 1000)}
    w = PartWriter(out, args.split_mb, header)
    pool = ThreadPoolExecutor(max_workers=workers)
    # In-flight window: results are written strictly in scan order, so the
    # output is byte-for-byte the order a serial run would produce.
    window = max(workers * 4, 8)
    pending = collections.deque()   # ('reused', rec) tuples or Futures

    def drain(block=False, keep=0):
        # Write ready results from the head of the queue, preserving order.
        # With block=True, additionally wait until at most `keep` items
        # remain in flight (so the pool stays busy while the writer catches
        # up, instead of stalling behind one slow file).
        nonlocal ok, err, reused, nbytes, written
        while pending:
            head = pending[0]
            ready = isinstance(head, tuple) or head.done()
            if not ready and (not block or len(pending) <= keep):
                break
            pending.popleft()
            if isinstance(head, tuple):
                kind, rec, ext = head
            else:
                while True:      # timed wait keeps Ctrl+C responsive on
                    try:         # Windows, where a bare result() blocks it
                        kind, rec, ext = head.result(timeout=0.5)
                        break
                    except FuturesTimeout:
                        continue
            if kind == 'ok':
                ok += 1
                nbytes += rec['b']
            elif kind == 'reused':
                ok += 1
                reused += 1
                nbytes += rec.get('b', 0)
            else:
                err += 1
                unreadable_ext[ext] = unreadable_ext.get(ext, 0) + 1
                # Keep the REASON, not just the count. It was already being
                # written to the JSONL and then never shown, so the only
                # thing on screen was "N unreadable" plus an extension
                # histogram - which is why "some images give an error" gets
                # reported as a guess. Grouped by exception type, because
                # MemoryError and "truncated" want completely different
                # answers from the user.
                why = str(rec.get('err', 'unknown'))
                kind_of = why.split(':', 1)[0].strip() or 'unknown'
                slot = unreadable_why.setdefault(kind_of, [0, []])
                slot[0] += 1
                if len(slot[1]) < 3:
                    slot[1].append((rec.get('p', '?'), why[:140]))
            w.write(rec)
            written += 1
            if written % 250 == 0 or written == total:
                print('  ' + str(written) + '/' + str(total)
                      + '   (' + str(reused) + ' reused, ' + str(err) + ' unreadable, '
                      + str(int(time.time() - t0)) + 's)', flush=True)

    completed = False
    try:
        for full in files:
            # Stored with forward slashes regardless of platform. A Windows
            # scan used to write "sub\file.jpg", which on Linux is not a
            # path at all but a single filename containing backslashes - so
            # an inventory could not be carried between machines, which is
            # exactly what --share exists to do. Every consumer normalises
            # back to the local separator when it opens the file.
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            rec = None
            if use_resume and rel in prev:
                old = prev[rel]
                try:
                    st = os.stat(full)
                    if st.st_size == old.get('b') and int(st.st_mtime * 1000) == old.get('mt'):
                        rec = old
                except OSError:
                    rec = None
            if rec is not None:
                pending.append(('reused', rec, None))
            else:
                pending.append(pool.submit(work_one, full, rel, args.thumb,
                                           args.fast_thumbs))
            if len(pending) > window:
                drain(block=True, keep=window)
            else:
                drain(block=False)
        drain(block=True, keep=0)
        completed = True
    finally:
        if not completed:
            try:                       # salvage whatever already finished,
                drain(block=False)     # so an abort loses nothing computed
            except Exception:
                pass
        w.close({'kind': 'footer', 'done': completed, 'ok': ok, 'errors': err,
                 'reused': reused, 'bytes': nbytes, 'parts': len(w.paths),
                 'elapsed_s': round(time.time() - t0, 1)})
        pool.shutdown(wait=False, cancel_futures=True)

    print('')
    print('Done: ' + str(ok) + ' images inventoried ('
          + str(reused) + ' reused), ' + str(err) + ' unreadable.')
    if unreadable_ext:
        print('Unreadable by extension: ' + json.dumps(unreadable_ext))
        if not HEIF_OK and any(e in unreadable_ext for e in ('.heic', '.heif', '.avif')):
            print('  -> install pillow-heif and re-run with --resume to fill these in.')
    if unreadable_why:
        print('')
        print('Why they failed:')
        for kind_of, (n, samples) in sorted(unreadable_why.items(),
                                            key=lambda kv: -kv[1][0]):
            print('  %s  (%d)' % (kind_of, n))
            for path, msg in samples:
                print('     %s' % path)
                print('       %s' % msg)
            if n > len(samples):
                print('     ... and %d more' % (n - len(samples)))
        if 'MemoryError' in unreadable_why:
            # Very large non-JPEG files are the usual cause: JPEG gets the
            # libjpeg draft fast path, PNG/TIFF/WebP/BMP are decoded whole,
            # once per worker thread.
            print('  -> ran out of memory decoding. Large PNG/TIFF/WebP are')
            print('     read at full size, once per worker; re-run with')
            print('     --workers 2 (or 1) to cut the peak.')
        print('  Every reason is also stored in the inventory, as "err".')
    tot_mb = sum(os.path.getsize(p) for p in w.paths) / 1048576.0
    for p in w.paths:
        print('Output: ' + p)
    print('Total inventory size: %.1f MB in %d file(s).' % (tot_mb, len(w.paths)))
    if use_resume and prev_files:
        old_mb = sum(os.path.getsize(p) for p in prev_files if os.path.exists(p)) / 1048576.0
        print('')
        print('Superseded by this run (%.1f MB, safe to delete):' % old_mb)
        for p in prev_files:
            print('   ' + p)

    def _under(child, parent):
        try:
            return os.path.commonpath([os.path.normcase(os.path.abspath(child)),
                                       os.path.normcase(os.path.abspath(parent))]) \
                   == os.path.normcase(os.path.abspath(parent))
        except ValueError:
            return False          # different drives

    mdir = args.mirror_dir or (MIRROR_DIR if args.share else '')
    # The "already reachable there" skip only makes sense for the built-in
    # shared folder (scanning inside the tree that HOLDS _inventory). An
    # explicit --mirror-dir is an instruction, not a hint: honour it even
    # when it sits beside the scanned folder - otherwise --mirror-dir D:\out
    # while scanning D:\Photos silently copies nothing, because every path
    # on the drive is "under" D:\.
    if mdir and not args.mirror_dir and _under(root, os.path.dirname(mdir)):
        print('(shared copy skipped: the scanned folder is inside the mirror')
        print(" folder's parent, so the output is already reachable there)")
    elif mdir:
        try:
            os.makedirs(mdir, exist_ok=True)
            tag = os.path.basename(root) or 'root'
            for p in w.paths:
                shutil.copy2(p, os.path.join(mdir, tag + ' - ' + os.path.basename(p)))
            print('Copied to the shared folder for your AI assistant: ' + mdir)
        except Exception as e:
            print('(shared copy skipped: ' + str(e)[:120] + ')')
    print('')
    print('Next: run Analyze-Inventory.bat on this folder for the duplicate report.')
    print('You might also ask your AI assistant (e.g. Claude) to generate the')
    print('report - see --share - but by default everything stays on this machine.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('')
        print('Aborted. The partial inventory file is still usable.')
        sys.exit(130)
