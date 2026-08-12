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
    --lossless-thumbs also try a lossless WebP thumbnail and keep it when
                      smaller. ~2.2x slower; changed no duplicate decision
                      across 36,410 images. Worth it for collections that
                      are mostly screenshots, UI captures or pixel art
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
import math
import os
import re
import shutil
import sys
import time
import warnings
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

# Pillow writes this straight to stderr, unbuffered, so it landed ABOVE our
# own banner:
#   PIL/Image.py:3578: DecompressionBombWarning: Image size (324000000
#   pixels) exceeds limit of 300000000 pixels, could be decompression bomb
#   DOS attack.
# To someone scanning their own holiday photos that reads as a crash, and
# as an accusation that their file is an attack. The guard is aimed at
# untrusted uploads; here the user owns every file being scanned.
#
# Suppressed at module level rather than with catch_warnings(), which is
# process-global state and NOT thread-safe - the collector runs eight
# workers, so trapping per image would let one thread's filter leak into
# another's. Instead the size is read from the lazy Image.open header (see
# process_one), and anything oversized is reported in our own voice at the
# end. DecompressionBombError, the >2x case, is a separate class and still
# raises and is still recorded as unreadable.
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

# Flagged as "very large" for the summary. Well below the bomb limit on
# purpose: this is the size at which decoding starts to dominate memory,
# not the size at which Pillow gets suspicious.
HUGE_PX = 80_000_000

# Formats that turn up in an actual image GALLERY. Every one is decoded by
# Pillow itself, so none can become "unreadable" noise just for being listed.
EXTS = {
    # photos and general images
    '.jpg', '.jpeg', '.jfif', '.jpe', '.png', '.webp', '.tif', '.tiff',
    '.bmp',
    # phone cameras (.hif is Canon's name for the same container).
    # These need pillow-heif; without it they are the one group that lands
    # as unreadable.
    '.heic', '.heif', '.hif', '.avif',
    # animated
    '.gif', '.apng',
    # Targa - Warcraft 3 and a good many older games screenshot to it, which
    # is why it is here and .dds is not: a screenshot belongs in a gallery,
    # a texture belongs to the game.
    '.tga',
    # QOI. Not a gallery format by origin - nothing consumer writes it, no
    # browser renders it, and its real users are game and embedded work -
    # but it is current rather than legacy, Pillow decodes it, and the
    # asymmetry decides it: listing it costs one entry, leaving it out means
    # a QOI file is skipped without a word.
    '.qoi',
}

# What is deliberately NOT scanned, and why - each of these was in the list
# at some point and taken back out:
#
#   .dds .icb .vda .vst      game textures and Targa's texture-side aliases.
#                            Asset folders reuse the same file on purpose,
#                            so every "duplicate" found there is intended.
#   .ico .cur .icns          icons and cursors. Multi-resolution containers
#                            that are UI furniture, not pictures.
#   .psd                     a working file, not a gallery image - and the
#                            one format here Pillow can read but not write,
#                            so it could never be tested end to end.
#   .pcx .sgi .dib           legacy encodings nothing writes into a photo
#   .ppm .pgm .pbm .pnm      library today. (.qoi sat here once and came
#                            back: it is niche, but it is not legacy, and
#                            skipping a file silently costs more than one
#                            entry in a set.)
#   .jp2 .j2k .jpf .jpx      JPEG 2000: archival and medical, effectively
#                            never a consumer gallery format.
#   .pdf .eps .fits .grib    Pillow opens these, but scanning a photo
#                            library for PostScript and weather data finds
#                            no duplicates and plenty of noise.
#
# Camera RAW (.cr2 .nef .arw .dng .orf .rw2) is absent for a different
# reason: Pillow cannot decode it at all - verified, none are registered. It
# would need rawpy/libraw, plus a real design decision, because a RAW
# carries an embedded JPEG preview and "is this a duplicate" has two
# different answers depending on which of the two you mean.
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


def thumb_size(w, h, box):
    """The size Image.thumbnail would pick, without calling it.

    Lifted from Pillow's own implementation (round_aspect and all) because
    make_thumb must RESIZE into a new image rather than thumbnail the
    source in place. Thumbnailing in place mutates the open ImageFile, and
    the later save() then calls _ensure_mutable() -> load(), at which point
    a lazy plugin re-decodes at full resolution into a buffer that is now
    128px - pillow_heif raises `ValueError: tile cannot extend outside
    image` and the file is dropped from the inventory as unreadable. That
    hit every multi-frame HEIC, i.e. iPhone bursts and Live Photos.

    Copying before thumbnailing also fixes it, but costs a full-resolution
    duplicate of every already-RGB image - which is exactly the allocation
    the in-place exif_transpose was added to avoid. Resizing costs nothing
    and is byte-identical; self_test pins that against Pillow directly, so
    if this ever drifts from the real implementation a test fails instead
    of every thumbnail quietly moving.

    Returns None when the image is already within the box, which is
    thumbnail()'s no-op case.
    """
    x, y = box
    if x >= w and y >= h:
        return None

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    aspect = w / h
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(x / aspect,
                         key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    return x, y


FRAME_SAMPLES = 25


def frame_signature(im):
    """A compact fingerprint of an animation BEYOND its first frame.

    The thumbnail is frame 0 only, which makes every animation that starts
    the same look identical. Measured: two entirely different GIFs sharing
    a first frame score MAD 0.0000 and were reported as automatic
    duplicates - and a still extracted from a GIF matched the GIF itself
    just as perfectly. This is the signal that tells them apart.

    Twenty-five samples, because GIF seeking is a REPLAY from frame 0, not
    random access: measured 2.6 ms to reach frame 1 and 128 ms to reach
    frame 119 of the same file. Once the walk to the last frame is paid
    for, extra samples along the way are FREE - K=5 and K=25 both measure
    ~15 ms on a 120-frame GIF - so sampling meanly buys nothing.

    It was five, and five was too few to see with. Five samples of a
    60-frame animation look at frames 0, 15, 30, 44 and 59 - 8% of the
    timeline - and a change anywhere else is invisible. Measured against
    the same animation with a short stretch replaced: six of eight cases
    scored 0.0000, meaning byte-identical fingerprints, so two animations
    that genuinely differ were declared the same. Sharing frame 0 they also
    share a thumbnail and a CLIP vector, so nothing else would have caught
    it either and one of them was a Tier A delete.

    At K=25 all eight are caught, the weakest at 59.0 against a cut of 48.

    Segment averaging - fold every frame into one of K buckets, so nothing
    is unsampled - was tried and rejected. It fixes coverage and breaks
    trim tolerance: the buckets cover different frames when the length
    changes, so a genuinely trimmed copy scored 13.9 while the weakest real
    difference scored 12.4. Point sampling picks frames by FRACTION of the
    timeline, which is what survives a trim, and that is worth keeping.

    RGB, not greyscale. Greyscale was the first attempt and it is
    colour-blind in exactly the way that matters here: two animations whose
    frames differ only in hue collapse onto the same luma. Measured, with
    an identical first frame so nothing else could separate them - pure red
    against pure green, both luma 76, scored 0.00 on a cut of 12.0, and
    crimson/blue and magenta/olive scored 8.5 and 4.0. All three would have
    been reported as automatic duplicates. In RGB the same pairs score 64,
    62 and 85.

    8x8x3 per frame: 192 bytes, so five frames add ~1.3 KB of base64
    against a 128px thumbnail's few KB. Stills return '' and pay nothing.

    The walk is strictly ASCENDING and then rewinds to 0. That is not
    incidental: Pillow 12.3 raises on a backwards seek to a middle frame of
    an APNG, so any implementation that visits frames out of order breaks
    on them.
    """
    n = int(getattr(im, 'n_frames', 1) or 1)
    if n < 2:
        return ''
    # ALWAYS emit FRAME_SAMPLES tiles, repeating an index when the animation
    # is shorter than that. A set comprehension here deduplicated the
    # indices, so a 2-frame GIF produced 384 bytes, 3-frame 576, 4-frame 768
    # and anything longer 960 - and two fingerprints of different lengths
    # are positionally incomparable, which made the comparison abstain and
    # wave the pair through on frame 0 alone. Fixed size means sample k is
    # the same fraction of the timeline on both sides, so a 3-frame and a
    # 6-frame animation are genuinely compared.
    want = [int(round(t * (n - 1) / (FRAME_SAMPLES - 1)))
            for t in range(FRAME_SAMPLES)]
    out = bytearray()
    try:
        tiles = {}
        for idx in sorted(set(want)):     # seek ascending - APNG demands it
            im.seek(idx)
            tiles[idx] = im.convert('RGB').resize(
                (8, 8), Image.BILINEAR).tobytes()
        for idx in want:
            out += tiles[idx]
    except Exception:
        return ''                 # unseekable: say nothing rather than guess
    finally:
        try:
            im.seek(0)            # every later step expects frame 0
        except Exception:
            pass
    return base64.b64encode(bytes(out)).decode('ascii')


def truncated(data):
    """True when the container itself says bytes are missing.

    Deliberately NOT done by asking Pillow, and deliberately not paired
    with ImageFile.LOAD_TRUNCATED_IMAGES. Measured, on this toolkit's own
    pipeline: with that flag on, Pillow fills the undecoded remainder with
    a CONSTANT - grey for JPEG, black for PNG - so two unrelated photos cut
    to 1200 bytes both become the same flat square and score MAD 0.0710
    against a Tier A gate of 4.0. Intact, that pair scores 97.21. It is the
    16-bit white-square bug again, and worse (0.07 vs 0.71).

    And it buys nothing: a truncated file does not even match its own
    intact original - self-MAD 8.4 / 37.8 / 67.3 with 90% / 50% / 10% of
    the bytes kept, every one of them outside the gate. So partial pixels
    are never compared. The file is reported as damaged instead, by name,
    which is the part that was actually missing.

    Reads only the bytes already in hand for the SHA-256, so it is free.
    Pure function of those bytes - no Pillow, no global state, nothing to
    race across the eight workers. Anything unrecognised returns False:
    this may only ever ACCUSE a file it is certain about.
    """
    n = len(data)
    if n < 12:
        # Too small to identify. A 0-byte or stub file is "unreadable",
        # which is what it will be reported as - calling it truncated would
        # be a guess, and this function may only accuse what it is sure of.
        return False
    if data[:3] == b'\xff\xd8\xff':       # JPEG
        i = 2
        while i + 4 <= n:
            if data[i] != 0xFF:
                return False              # lost marker alignment; do not guess
            m = data[i + 1]
            if m == 0xFF:
                i += 1
                continue
            if m == 0x01 or 0xD0 <= m <= 0xD8:
                i += 2
                continue
            if m == 0xD9:
                return False
            seg = int.from_bytes(data[i + 2:i + 4], 'big')
            if seg < 2:
                return False
            if m == 0xDA:
                # Entropy data cannot legally contain a bare FFD9 (an FF is
                # stuffed as FF00, or is a restart marker), so an FFD9 at or
                # past the scan is the true end. Searching backwards makes
                # the intact case O(1), and requiring it past the scan start
                # skips the EOI inside an EXIF thumbnail - which lives in an
                # APPn segment and otherwise reads as a complete file.
                return data.rfind(b'\xff\xd9') < i + 2 + seg
            i += 2 + seg
        return True
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        i = 8
        while i + 8 <= n:
            ln = int.from_bytes(data[i:i + 4], 'big')
            typ = data[i + 4:i + 8]
            end = i + 12 + ln             # length + type + payload + CRC
            if end > n:
                return True
            if typ == b'IEND':
                return False              # trailing junk is not our business
            i = end
        return True
    if data[:4] == b'RIFF':               # WebP and friends
        return n < 8 + int.from_bytes(data[4:8], 'little')
    return False


def make_thumb(im, thumb_px, fast=True):
    """Returns (fmt_flag, tw, th, b64). JPEG by default; lossless WebP when
    that is actually smaller (flat/UI content compresses better losslessly)."""
    # exif_transpose ends in `return image.copy()` even when the image has
    # NO orientation tag - a full-resolution duplicate of the whole thing,
    # for nothing. That copy is the single largest avoidable allocation in
    # this function. Skipping it: a 144 Mpx PNG peaks 1183 -> 607 MB, 36
    # Mpx 318 -> 174 MB, and the output is BYTE-IDENTICAL, so no stored
    # thumbnail moves and no threshold is needed.
    #
    # in_place returns None, hence the bare call. It mutates the caller's
    # image, which is safe here: make_thumb is the last thing process_one
    # does with it.
    ImageOps.exif_transpose(im, in_place=True)
    im2 = im
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
        # point() is implemented for I, I;16 and F only - the byte-order
        # variants raise "point operation not supported for this mode".
        # I;16B is not exotic: it is what a big-endian 16-bit TIFF opens
        # as, which is ordinary scanner and Adobe output, and the raise
        # took make_thumb down so the file was never inventoried at all -
        # no sha, no thumbnail, never compared against anything.
        if im2.mode != 'I;16':
            im2 = im2.convert('I')
        im2 = im2.point(lambda v: v * (1 / 257)).convert('L')
    elif im2.mode in ('I', 'F'):
        # 32-bit int and float carry no defined range, so normalise by what
        # is actually in the image rather than assuming one.
        lo, hi = im2.getextrema()
        if not (abs(lo) < 3.0e38 and abs(hi) < 3.0e38):
            # One Inf or NaN pixel makes the span infinite, the scale 0.0
            # and EVERY pixel black - and two unrelated images then score a
            # perfect match, so one is offered for deletion. That is the
            # exact failure this rescale exists to prevent. (NaN fails the
            # comparison too, which is what we want.)
            #
            # Refusing is the safe answer, not a cop-out: the file is
            # recorded as unreadable with this reason and never compared,
            # rather than compared wrongly. Clamping in place is not
            # available - Pillow's point() accepts only affine expressions
            # for I and F, and Collect is deliberately Pillow-only, with no
            # numpy to fall back on.
            raise ValueError(
                'image has no finite pixel range (contains Inf or NaN)')
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
        # Three full-resolution buffers used to live in these three lines:
        # convert('RGBA') copies even when the image is ALREADY RGBA,
        # Image.new allocates a second, and alpha_composite returns a
        # third. A 144 Mpx RGBA peaked at 2911 MB. Pasting the image
        # through its own alpha as a mask, straight onto an RGB
        # background, does the whole job in one - and is bit-exact, which
        # was checked across all 256 alphas x 256 colours rather than
        # sampled. Combined with the transpose fix: 2911 -> 1183 MB.
        if im2.mode == 'La':
            # Premultiplied greyscale+alpha converts to exactly one thing in
            # Pillow: LA. La->L, ->RGB, ->RGBA and ->P all raise "conversion
            # from La to L not supported", so going straight to RGBA below
            # would throw. RGBa, the colour equivalent, has no such problem -
            # this is a quirk of one mode, not a family. No image FILE decodes
            # to La (Pillow cannot write it as TIFF, PNG or WebP either), so
            # this is a latent trap rather than something a scan hits, but it
            # costs one line to close.
            im2 = im2.convert('LA')
        rgba = im2 if im2.mode == 'RGBA' else im2.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, (0, 0), rgba)
        im2 = bg
    if im2.mode != 'RGB':
        im2 = im2.convert('RGB')
    # resize, not thumbnail: see thumb_size. Same pixels, and it never
    # mutates the open file.
    _t = thumb_size(im2.width, im2.height, (thumb_px, thumb_px))
    im2 = im2.resize(_t, Image.LANCZOS, reducing_gap=2.0) if _t else im2.copy()
    bj = io.BytesIO()
    # 78, and 128 px above, set the noise floor of every pixel comparison
    # the analyzer makes. What reaches its 4.0 gate is the DIFFERENTIAL
    # between two similar images, not the error in one, and measured over
    # 290 real pairs that is:
    #
    #     q      median   p95    p99    bytes/img
    #     74      0.983  2.796  3.151     3334
    #     78      0.610  1.484  1.905     3588
    #     80      0.351  1.146  1.520     3732   <- here
    #     82      0.399  1.239  1.946     3919
    #     85      0.863  4.363  5.402     4250
    #     92      0.235  0.895  1.829     5565
    #
    # 80 rather than 78: 42% less median noise and 20% less at p99 for 4%
    # more storage. It beats q92 at p99 while costing a third as much extra.
    #
    # The curve is NOT monotonic - 85 is worse than 78 on every percentile,
    # and 88 and 92 both have fatter tails than 80 - so this is a measured
    # point, not "higher is better". Two samples would have suggested the
    # opposite; an earlier reading of only 78 and 92 concluded "quality buys
    # more than size", which the full curve does not support.
    #
    # None of this is fixing a wrong answer. A full run of a 36,410-image
    # library flipped no verdict at 78, so the gain is margin against the
    # 4.0 gate rather than corrections.
    im2.save(bj, 'JPEG', quality=80)
    best, flag = bj.getvalue(), ''
    if fast:
        # The default, since v4.3.3. The lossless-WebP attempt below costs
        # 135x the JPEG encode on a real library (21.97 ms against 0.16 ms)
        # and wins 2 times in 2,000, so it was most of the scan for a tenth
        # of a percent of the thumbnails.
        #
        # Dropping it is not free in principle - those thumbnails become
        # lossy - so it was settled by running the whole pipeline both ways
        # on 36,410 images: same 350 list lines, same marks, same 3 Tier A
        # clusters, same 95 Tier B clusters. Three extra candidate PAIRS
        # appeared, which is the noise showing up in the prefilter and
        # stopping short of any verdict.
        #
        # What made it safe is that JPEG is deterministic: two copies of one
        # picture thumbnail to identical bytes, so the error cancels instead
        # of accumulating. Measured on real pairs, |MAD(lossless) -
        # MAD(jpeg)| tops out at 1.66 against a 4.0 gate.
        #
        # --lossless-thumbs brings it back for libraries where a larger
        # share of thumbnails qualify: screenshots, UI captures, pixel art.
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


def process_one(full, rel, thumb_px, fast=True):
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
    # Free: these are the bytes just hashed. Only for files read in one go;
    # the larger streaming path is covered by the check in work_one's error
    # branch, which is where a partial file almost always ends up anyway.
    if data is not None and truncated(data):
        rec['trunc'] = 1

    src = io.BytesIO(data) if data is not None else full
    with Image.open(src) as im:
        rec['fmt'] = im.format or ''
        rec['w'], rec['h'] = im.size
        # Read from the header - Image.open is lazy, so this costs nothing
        # and is known BEFORE the decode that might run out of memory.
        if rec['w'] * rec['h'] >= HUGE_PX:
            rec['huge'] = round(rec['w'] * rec['h'] / 1e6, 1)
        # MPO is excluded deliberately. A .jpg from a dual-camera or
        # 3D-capable phone carries a multi-picture header, so Pillow returns
        # an MpoImageFile with is_animated=True and n_frames=2 - but those
        # frames are the second lens or a depth capture of the SAME instant,
        # not an animation. Recording anim=2 for an ordinary still photo
        # inverts what that field is for.
        if getattr(im, 'is_animated', False) and im.format != 'MPO':
            rec['anim'] = int(getattr(im, 'n_frames', 2))
            # Only animations pay for this, and only once. Must run before
            # make_thumb, which transposes in place.
            fs = frame_signature(im)
            if fs:
                rec['fsig'] = fs

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

        # MPO too: it IS a JPEG underneath, and Pillow only relabels it
        # because of the multi-picture header. Gating on 'JPEG' alone meant
        # every dual-camera phone photo - exactly the libraries this tool
        # is pointed at - decoded at full resolution. Measured on a 3000px
        # frame: 750x750 drafted vs 3000x3000 undrafted, 16x the pixels.
        if im.format in ('JPEG', 'MPO'):
            im.draft(None, (thumb_px * 4, thumb_px * 4))
        flag, tw, th, b64 = make_thumb(im, thumb_px, fast)
        rec['tw'], rec['th'], rec['tb'] = tw, th, b64
        if flag:
            rec['tf'] = flag
    return rec


def work_one(full, rel, thumb_px, fast=True):
    """Thread-pool task: never raises (KeyboardInterrupt stays in the main
    thread). Returns (kind, record, ext_on_error)."""
    try:
        return 'ok', process_one(full, rel, thumb_px, fast), None
    except Exception as e:
        try:
            msg = type(e).__name__ + ': ' + str(e)
            # Say WHICH kind of broken. A partial file - interrupted
            # download, bad copy, half-synced cloud folder - is something
            # the user can go and re-fetch, while a genuinely corrupt one
            # is not. Only on the error path, so a healthy scan never pays
            # for the extra read.
            try:
                with open(full, 'rb') as _f:
                    if truncated(_f.read()):
                        msg = 'TruncatedFile: incomplete download or copy ' \
                              '(' + msg + ')'
            except Exception:
                pass
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


def walk_images(root, skip_names, on_relink=None):
    """Every image under root, each PHYSICAL file once.

    os.walk(followlinks=False) does not stop a Windows junction. Junctions
    are reparse points, not symlinks: os.path.islink() returns False for
    one, DirEntry.is_symlink() returns False, and the guard never fires. So
    `mklink /J D:\\Photos\\backup D:\\Photos` - which needs no admin rights,
    and which relocated or mirrored folders create routinely - made the
    scan walk the tree twice and inventory every photo under two names.

    That is not a harmless duplicate entry. The two names are ONE file, so
    the analyzer sees a perfect SHA match, pre-marks one copy X, and the
    recycler deletes it - taking the "survivor" with it, because the
    survivor is the same bytes on disk. Measured: 12 photos, one junction,
    no real duplicates anywhere; after the recycler ran, both directories
    were empty. A self-referential junction was worse, multiplying an
    8-image folder into 512 entries until MAX_PATH stopped the walk.

    The fix is to remember the RESOLVED directory and refuse to descend
    into one twice. realpath collapses junctions, symlinks and mount
    points alike, so this covers POSIX symlinks and directory loops with
    the same line rather than special-casing Windows. It is per-directory,
    so the cost is a syscall per folder, not per file.

    A junction pointing somewhere genuinely outside the tree still gets
    scanned - it is new content, and skipping it would lose files. Only a
    second route to something already walked is dropped.
    """
    seen_dirs = set()

    def resolved(p):
        try:
            return os.path.normcase(os.path.realpath(p))
        except OSError:
            return os.path.normcase(os.path.abspath(p))

    seen_dirs.add(resolved(root))
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            if (d.lower() in SKIP_DIRS
                    or d.lower().startswith(SKIP_DIR_PREFIXES)
                    or d.startswith('.')):
                continue
            r = resolved(os.path.join(dirpath, d))
            if r in seen_dirs:
                # A second path to a directory already walked. Announced
                # rather than skipped quietly: from the outside this looks
                # like files going missing from the scan.
                if on_relink is not None:
                    on_relink(os.path.join(dirpath, d), r)
                continue
            seen_dirs.add(r)
            keep.append(d)
        dirnames[:] = keep
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
    ap.add_argument('--lossless-thumbs', action='store_true',
                    help='also try a lossless WebP thumbnail and keep it when '
                         'it is smaller. Costs ~2.2x the scan time; measured '
                         'on a 36,410-image library it changed no duplicate '
                         'decision at all. Worth it for a collection that is '
                         'mostly screenshots, UI captures or pixel art, where '
                         'a larger share of thumbnails qualify')
    ap.add_argument('--fast-thumbs', action='store_true',
                    help=argparse.SUPPRESS)      # now the default; kept so
    #                                              existing commands still run
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
        # A phone or camera on a cable is the likeliest way to land here, and
        # "Not a folder" is a useless thing to say about it. Those connect
        # over MTP, which is a shell namespace rather than a filesystem:
        # Explorer shows "This PC\Phone\Internal storage\DCIM" and there is
        # no drive letter behind it, so os.walk and open() cannot address it
        # and neither can cmd. Dragging one onto a .bat hands over a shell
        # item, not a path.
        if not os.path.exists(root):
            print('')
            print('If that came from a phone or camera on a USB cable: those')
            print('appear over MTP, which is not a drive - there is no path')
            print('behind what Explorer shows, so nothing here can read it.')
            print('Copy the folder to this PC first and scan the copy, or put')
            print('the card in a reader, which does get a real drive letter.')
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
    relinks = []
    files = list(walk_images(root, skip_names,
                             lambda p, r: relinks.append((p, r))))
    total = len(files)
    print('Found ' + str(total) + ' image files.')
    if relinks:
        # Said out loud. A junction that doubles the tree used to end with
        # the recycler deleting both copies of every file, and the only
        # visible sign was a suspiciously tidy pile of "exact duplicates".
        print('')
        print('%d folder(s) are a second route to something already scanned'
              % len(relinks))
        print('(a junction, symlink or mount point). Walked once, not twice -')
        print('otherwise one file appears under two names and looks like its')
        print('own duplicate:')
        for p, r in relinks[:6]:
            print('   %s' % p)
            print('     -> %s' % r)
        if len(relinks) > 6:
            print('   ... and %d more' % (len(relinks) - 6))
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
    huge_seen = []               # (path, megapixels) for very large images
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
            if kind in ('ok', 'reused') and rec.get('huge'):
                huge_seen.append((rec.get('p', '?'), rec['huge']))
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
                                           not args.lossless_thumbs))
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
    if huge_seen:
        # Said in our own voice, replacing PIL's stderr warning about a
        # "decompression bomb DOS attack" - which is about untrusted
        # uploads and reads as an accusation when it is your own photo.
        huge_seen.sort(key=lambda t: -t[1])
        print('')
        print('Very large images (%d, biggest %.1f MP):' % (len(huge_seen),
                                                            huge_seen[0][1]))
        for path, mp in huge_seen[:3]:
            print('   %.1f MP  %s' % (mp, path))
        if len(huge_seen) > 3:
            print('   ... and %d more' % (len(huge_seen) - 3))
        print('  These decode fine but are memory-hungry, and are read once')
        print('  per worker. If a scan runs out of memory, --workers 2 is')
        print('  the lever. Recorded in the inventory as "huge".')
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
