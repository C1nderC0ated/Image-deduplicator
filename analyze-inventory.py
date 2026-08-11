#!/usr/bin/env python3
"""
analyze-inventory.py  -  find duplicates in an image inventory.

Reads image-inventory*.jsonl (and image-embeddings*.jsonl if present),
groups images into duplicate clusters, and writes next to the inventory:

    <name>-report.html         every cluster as pictures, keeper marked
    <name>-list.txt            one line per candidate: X = delete, . = keep
    Recycle-Duplicates.py      the recycler - every safety rule lives here
      + .bat / .sh             thin launchers that only find a Python

READ-ONLY with respect to your images. This script never deletes anything;
the generated recycler does, and only after you edit the list and confirm.
Deletions go to the OS trash (Recycle Bin / freedesktop Trash / ~/.Trash),
never a permanent delete.

    python analyze-inventory.py <inventory.jsonl | folder>

Options:
    --tier-a-mad N       pixel-difference ceiling for Tier A   (default 4.0)
    --tier-a-cos N       CLIP floor for Tier A                 (default 0.99)
    --tier-b-mad N       pixel floor for the review tier       (default 4.0)
    --tier-b-cos N       CLIP floor for the review tier        (default 0.90)
    --sig-cut N          8x8 signature prefilter ceiling       (default 8.0,
                         auto-raised to 2x --tier-a-mad when that is higher;
                         an explicit value is always used verbatim)
    --clip-neighbors N   nearest neighbours each image contributes from the
                         embeddings (default 16); bounds the candidate set
                         on large libraries. A binding cap is reported.
    --no-orient          skip rotation/mirror matching (faster; rotated and
                         mirrored copies will be missed)
    --no-embeddings      ignore embeddings even if present
    --self-test          run the built-in invariant tests and exit

TIERS
  A  duplicate      - same picture, re-encoded or resized. Pre-set to X.
  B  crop / variant - structurally the same, genuinely different pixels
                      (crops, rotations, recolours, re-rolls). Pre-set to
                      "." always - nothing here is deleted unreviewed.

THE INVARIANTS (enforced in code, not by memory)
  1. every file appears at most once as a deletion candidate
  2. a file chosen as a KEEPER never appears as a candidate anywhere
  3. every candidate's keeper exists and is not itself a candidate
  4. no cluster is emitted with fewer than two members
A violation aborts the run rather than writing a list that could delete a
file something else depends on. Tier B is built from pairs, and pairs chain
(A crops to B, B crops to C), which is exactly how invariant 2 was broken
once before; that is why it is asserted here.
"""
import argparse
import base64
import collections
import glob
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# Exotic characters in filenames must not crash a redirected console.
# stdout only: stderr already defaults to backslashreplace, which cannot
# raise, and 'replace' would only destroy detail in tracebacks.
for _s in (sys.stdout,):
    try:
        _s.reconfigure(errors='replace')
    except Exception:
        pass

def _hint(pkg):
    """How to install PKG on THIS machine.

    Routed through _setup so there is one answer instead of five, and so it
    can be wrong in only one place. Hard-coding '--user' is actively wrong
    on a distro-managed Python (Arch, Debian 12+, Fedora 38+): pip refuses
    it outright, and the way forward there is a virtual environment.
    _setup is stdlib-only, so this works before numpy or Pillow exist."""
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        if d not in sys.path:
            sys.path.insert(0, d)
        from _setup import pip_hint
        return pip_hint(pkg)
    except Exception:
        return '"%s" -m pip install --user %s' % (sys.executable, pkg)


try:
    import numpy as np
except ImportError:
    print('numpy is required:  ' + _hint('numpy'))
    sys.exit(2)
try:
    from PIL import Image
except ImportError:
    print('Pillow is required:  ' + _hint('pillow'))
    sys.exit(2)

_HAVE_CV2 = False
# OpenCV is optional (it powers the crop tier), but when present it also
# gives an exact, SIMD-accelerated absolute difference and a much faster
# JPEG/WebP decoder. Both were verified equivalent to the Pillow/numpy
# versions before adoption - see CHANGES v4.2f. Imported once here so the
# hot loops do not pay an import lookup per pair.
try:
    import cv2 as _cv2
except Exception:
    _cv2 = None


def absdiff_mean(A, B):
    """mean(|A - B|) for two uint8 images of the same shape.

    cv2.norm(NORM_L1) sums |A - B| over every element in one pass, with no
    intermediate image at all: no uint8 difference buffer, no float32 pair.
    Dividing by A.size (which counts channels) gives the mean back exactly -
    the sum is over integers, so it is not merely close to absdiff().mean()
    but equal to it, measured 0.000e+00 apart over 400 trials including
    identical inputs. It is also ~13x faster, which matters because this is
    the innermost call of the whole analyzer.

    The numpy fallback casts to float32 and accumulates in float32, so it can
    differ from the cv2 path around the 7th decimal. That difference is in
    cv2's FAVOUR - an exact integer sum beats float32 accumulation - and it
    is ~1e-6 against thresholds of 4.0. Measured over thousands of pairs: no
    tier verdict changes either way."""
    if _cv2 is not None:
        return _cv2.norm(A, B, _cv2.NORM_L1) / A.size
    return float(np.abs(A.astype(np.float32) - B.astype(np.float32)).mean())


def imdecode_rgb(raw):
    """Decode stored thumbnail bytes to a uint8 RGB array. cv2 is ~2.6x
    faster than Pillow here, and was verified PIXEL-IDENTICAL across 240
    JPEG and lossless-WebP thumbnails (both orientations, flat/photo/noise
    content) - byte-for-byte equal, not merely close."""
    if _cv2 is not None:
        m = _cv2.imdecode(np.frombuffer(raw, np.uint8), _cv2.IMREAD_COLOR)
        if m is not None:
            return _cv2.cvtColor(m, _cv2.COLOR_BGR2RGB)
    return np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'), dtype=np.uint8)


def default_workers():
    return max(2, min(8, os.cpu_count() or 4))


_T0 = [None]


def phase(msg):
    """Print how long the previous stage took, then announce this one.
    On a big library the run is long enough that silence is indistinguishable
    from a hang, and it should be obvious which stage is the expensive one."""
    import time as _t
    now = _t.time()
    if _T0[0] is not None:
        el = now - _T0[0]
        if el >= 0.05:
            print('    (%.1fs)' % el)
    _T0[0] = now
    if msg:
        print(msg, flush=True)


# ----------------------------------------------------------------- loading --
def find_inventory(arg):
    if os.path.isfile(arg):
        return arg
    if os.path.isdir(arg):
        c = glob.glob(os.path.join(glob.escape(arg), '*image-inventory*.jsonl'))
        c = [p for p in c if '.part' not in os.path.basename(p)] or c
        if c:
            return max(c, key=os.path.getmtime)
    return None


def load_inventory(path):
    d, b = os.path.split(path)
    b = b[:-len('.jsonl')]
    if '.part' in b:                     # basename only: a folder named
        b = b.split('.part')[0]          # 'archive.part' must not truncate
    stem = os.path.join(glob.escape(d), glob.escape(b)) if d else glob.escape(b)
    paths = sorted(set(glob.glob(stem + '.jsonl') + glob.glob(stem + '.part*.jsonl')))
    root, recs, errs = None, [], []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not isinstance(r, dict):
                    continue
                if 'root' in r and root is None:
                    root = r['root']
                # control lines carry 'kind'; older files are detected by shape
                if r.get('kind') or r.get('schema') or r.get('done'):
                    continue
                if 'p' in r and 'sha' in r and 'tb' in r:
                    recs.append(r)
                elif 'p' in r and 'err' in r:
                    errs.append(r)
    return root, recs, errs, paths


def load_embeddings(inv_path, recs):
    d = os.path.dirname(os.path.abspath(inv_path))
    # Prefer the embeddings file NAMED after this inventory (that is how
    # embed-images.py names its output); only fall back to newest-by-mtime
    # when no name matches - several inventories can share one folder. The
    # explicit `want` glob also finds custom-named outputs like
    # myscan-embeddings.jsonl that the generic pattern cannot see.
    base = os.path.basename(inv_path)
    if base.endswith('.jsonl'):
        base = base[:-len('.jsonl')]
    if '.part' in base:
        base = base.split('.part')[0]
    want = base.replace('image-inventory', 'image-embeddings')
    if want == base:
        want = base + '-embeddings'
    cands = set(glob.glob(os.path.join(glob.escape(d), '*image-embeddings*.jsonl')))
    cands |= set(glob.glob(os.path.join(glob.escape(d), glob.escape(want) + '.jsonl')))
    if not cands:
        return None, None
    named = [p for p in cands if os.path.basename(p) == want + '.jsonl']
    path = named[0] if named else max(cands, key=os.path.getmtime)
    vec, model = {}, None
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('schema') == 'img-emb/1':
                model = r.get('model')
            elif 'sha' in r and 'v' in r:
                vec[r['sha']] = np.frombuffer(base64.b64decode(r['v']),
                                              dtype=np.float16).astype(np.float32)
    have = sum(1 for r in recs if r['sha'] in vec)
    print('Embeddings: %s  (%s, %d/%d images covered)'
          % (os.path.basename(path), model, have, len(recs)))
    if have < len(recs) * 0.5:
        print('  Less than half the images are covered - ignoring them.')
        return None, path
    dims = {v.shape[0] for v in vec.values()}
    if len(dims) > 1:
        print('  Mixed vector widths %s - refusing to use these.' % sorted(dims))
        return None, path
    return vec, path


# ------------------------------------------------------------- comparison --
def _memory_budget():
    """How much RAM the thumbnails may take. Measured from what is actually
    free rather than guessed, because the alternative to holding them is
    millions of JPEG decodes - roughly a hundred times slower - and a fixed
    cap gets that decision wrong on both small and large machines. Leaves
    the sweep and the embeddings plenty of headroom."""
    free = None
    try:                                    # Windows
        import ctypes
        class _MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            free = int(m.ullAvailPhys)
    except Exception:
        pass
    if free is None:
        try:                                # Linux / macOS
            free = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except Exception:
            free = 0
    if not free:
        return 1_200_000_000                # no idea: a safe default
    return max(512_000_000, int(free * 0.55))


class ThumbStore:
    """Thumbnails decoded on demand, with a bounded cache.

    Keeping every thumbnail as pixels costs 1.24 GB at 36k images and was a
    hard MemoryError not far beyond that - on a machine with plenty of RAM
    free, because the peak lands on top of the sweep's own arrays. It is
    also almost entirely wasted: the sweep runs on 8x8 signatures, and only
    CANDIDATE pairs ever need real pixels. Those arrive in sorted order, so
    a small LRU keeps the hit rate high while the resident set stays flat
    no matter how large the library gets.

    Indexing is unchanged (TH[i] -> uint8 HxWx3), so every scorer works as
    written.
    """

    def __init__(self, recs, cap=8192, budget=None, workers=4):
        self._recs = recs
        self._cap = max(64, cap)
        self._d = collections.OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        self.preloaded = False
        # Decoding on demand is the safe default, but it is also the slow
        # one: candidate pairs are visited in (i, j) order, so j sweeps the
        # whole library and misses the cache almost every time - millions of
        # JPEG decodes. When the whole set comfortably fits a memory budget,
        # decode it once up front instead. Falls back to lazy on
        # MemoryError, so a machine that cannot take it still finishes.
        if budget is None:
            budget = _memory_budget()
        need = sum(r.get('tw', 128) * r.get('th', 128) * 3 for r in recs)
        if need <= budget:
            try:
                got = [None] * len(recs)

                def one(i):
                    got[i] = self._decode(i)

                with ThreadPoolExecutor(workers) as pool:
                    collections.deque(pool.map(one, range(len(recs))), maxlen=0)
                self._all = got
                self.preloaded = True
            except MemoryError:
                self._all = None
        else:
            self._all = None

    def _decode(self, i):
        return imdecode_rgb(base64.b64decode(self._recs[i]['tb']))

    def __getitem__(self, i):
        if self._all is not None:
            return self._all[i]
        with self._lock:
            a = self._d.get(i)
            if a is not None:
                self._d.move_to_end(i)
                self.hits += 1
                return a
            self.misses += 1
        a = self._decode(i)                     # decode outside the lock
        with self._lock:
            self._d[i] = a
            self._d.move_to_end(i)
            while len(self._d) > self._cap:
                self._d.popitem(last=False)
        return a

    def __len__(self):
        return len(self._recs)


def decode_signatures(recs, workers):
    """One streaming pass for the 8x8 colour signatures. The full-size
    pixels are deliberately NOT kept - see ThumbStore."""
    n = len(recs)
    C = np.zeros((n, 192), dtype=np.float32)

    def one(i):
        a = imdecode_rgb(base64.b64decode(recs[i]['tb']))
        C[i] = np.asarray(Image.fromarray(a).resize((8, 8), Image.LANCZOS),
                          dtype=np.float32).ravel()

    with ThreadPoolExecutor(workers) as pool:
        collections.deque(pool.map(one, range(n)), maxlen=0)
    return C


_EXACT_CHUNK = 65536


def _mean_abs_diff(M, ia, ib):
    """mean(|M[ia] - M[ib]|) per row pair, without building four temporaries
    the size of the whole chunk. The obvious one-liner allocates the two
    gathers, the difference and the absolute value - ~800 MB at the old
    chunk size, which is enough to fail on a busy machine even with plenty
    of RAM free."""
    a = M[ia]
    b = M[ib]
    np.subtract(a, b, out=a)
    np.abs(a, out=a)
    return a.mean(1)


def sweep_candidates(C, cut):
    """All-pairs prefilter on the 8x8 signatures, exactly equivalent to
    brute-force  mean|a-b| <= cut  over every pair, but fast:

    Since every |d| <= 255,  mean|d| <= cut  implies
        sum(d^2) <= max|d| * sum|d| <= 255 * (192*cut).
    So a BLAS squared-distance pass with that bound cannot lose a single
    true candidate; survivors are then re-scored with the exact same
    float32 mean-absolute-difference the brute force uses. --self-test
    asserts set equality against brute force.
    """
    n, w = C.shape
    # float32, not float64: the gram terms reach 192*255^2 ~ 1.25e7, where
    # float32 resolves to ~1, and the bound we compare against is ~4e5 - so
    # rounding is four orders of magnitude below the decision. The bound is
    # widened anyway (below), and every survivor is re-checked exactly, so
    # the only possible effect is a few extra exact checks. Halves the memory
    # traffic and roughly doubles the matmul.
    # Sort by mean signature value first. Because
    #     mean|a-b| >= |mean(a) - mean(b)|
    # a pair whose signature means differ by more than `cut` can never
    # qualify - so once the rows are ordered by mean, each row only has to
    # be compared against the rows within `cut` of it. That turns the sweep
    # from every-pair into a band, and it is exact: nothing inside the band
    # is skipped and nothing outside it could have passed.
    means = C.mean(1)
    order = np.argsort(means, kind='stable')
    Cs = np.ascontiguousarray(C[order], dtype=np.float32)
    ms = means[order]
    sq = (Cs * Cs).sum(1)
    # widen by a hair so float32 rounding can never drop a true candidate
    lim = 255.0 * cut * w * 1.0001 + 1000.0
    cand = []
    block = max(64, min(2048, int(8_000_000 // max(1, n))))
    for a in range(0, n, block):
        b = min(n, a + block)
        # rows a..b only need columns up to the last row still within `cut`
        hi = int(np.searchsorted(ms, ms[b - 1] + cut, side='right'))
        if hi <= a + 1:
            continue
        D2 = sq[a:b, None] + sq[None, a:hi] - 2.0 * (Cs[a:b] @ Cs[a:hi].T)
        ii, jj = np.nonzero(D2 <= lim)
        gi = ii + a
        gj = jj + a
        keep = gi < gj
        if not keep.any():
            continue
        gis, jjs = gi[keep], gj[keep]
        for s in range(0, len(gis), _EXACT_CHUNK):
            gs, js = gis[s:s + _EXACT_CHUNK], jjs[s:s + _EXACT_CHUNK]
            d = _mean_abs_diff(Cs, gs, js)             # exact, same as brute
            sel = d <= cut
            oi = order[gs[sel]]
            oj = order[js[sel]]
            lo = np.minimum(oi, oj)
            hh = np.maximum(oi, oj)
            cand.extend(zip(lo.tolist(), hh.tolist()))
    return cand


def sweep_candidates_cross(C, Ck, cut, rows=None):
    """Same lossless bound as sweep_candidates, but between two different
    signature sets (image i as stored vs image j re-oriented). With `rows`,
    only those rows of C are swept against ALL of Ck - used when embeddings
    cover most of the library and only the uncovered images still need the
    pixel path. Returned pairs use GLOBAL indices either way. i == j is
    dropped - an image only matches its own rotation when the picture is
    symmetric, and that is not a duplicate."""
    n, w = C.shape
    ridx = np.arange(n) if rows is None else np.asarray(rows)
    A = np.ascontiguousarray(C[ridx], dtype=np.float32)
    B = np.ascontiguousarray(Ck, dtype=np.float32)
    sa, sb = (A * A).sum(1), (B * B).sum(1)
    lim = 255.0 * cut * w * 1.0001 + 1000.0
    out = []
    block = max(64, min(2048, int(8_000_000 // max(1, n))))
    for a in range(0, len(A), block):
        b = min(len(A), a + block)
        D2 = sa[a:b, None] + sb[None, :] - 2.0 * (A[a:b] @ B.T)
        ii, jj = np.nonzero(D2 <= lim)
        gi = ridx[ii + a]
        keep = gi != jj
        if not keep.any():
            continue
        lis, gis, jjs = (ii + a)[keep], gi[keep], jj[keep]
        for s in range(0, len(gis), _EXACT_CHUNK):
            ls, gs, js = (lis[s:s + _EXACT_CHUNK], gis[s:s + _EXACT_CHUNK],
                          jjs[s:s + _EXACT_CHUNK])
            t = A[ls]
            np.subtract(t, B[js], out=t)
            np.abs(t, out=t)
            d = t.mean(1)
            sel = d <= cut
            out.extend(zip(gs[sel].tolist(), js[sel].tolist()))
    return out


def oriented_signatures(C, k):
    """The 8x8 signature grid re-oriented by DIHEDRAL[k], as a pure
    permutation of the existing signatures - no thumbnail decodes, no
    resampling. Exact for square thumbnails; for non-square ones the
    anisotropic resize makes it an approximation, which is why callers
    sweep with the full (not tightened) cut and every nominated pair is
    still re-scored exactly downstream."""
    n = len(C)
    G = C.reshape(n, 8, 8, 3)
    _name, fn = DIHEDRAL[k]
    return np.ascontiguousarray(
        fn(G.transpose(1, 2, 0, 3)).transpose(2, 0, 1, 3)).reshape(n, 192)


def mad_pair(TH, i, j):
    # Shares absdiff_mean rather than repeating its body: this is the single
    # hottest call in the analyzer (376,660 pairs on a 36k library), and it
    # had been left on the slow float32 path when the fast one was added.
    A, B = TH[i], TH[j]
    if A.shape != B.shape:
        B = np.asarray(Image.fromarray(B).resize(
            (A.shape[1], A.shape[0]), Image.LANCZOS), dtype=np.uint8)
    return absdiff_mean(A, B)


def compute_mads(TH, cand, workers):
    """mad_pair for every candidate pair, chunked across threads."""
    out = [0.0] * len(cand)
    if not cand:
        return out
    step = max(64, (len(cand) + workers - 1) // workers)

    def run(k):
        for t in range(k, min(k + step, len(cand))):
            i, j = cand[t]
            out[t] = mad_pair(TH, i, j)

    with ThreadPoolExecutor(workers) as pool:
        collections.deque(pool.map(run, range(0, len(cand), step)), maxlen=0)
    return out


# The eight ways a picture can be saved without changing what it shows:
# four rotations, each optionally mirrored. Index 0 is the identity.
DIHEDRAL = (
    ('as-is', lambda a: a),
    ('rot90', lambda a: np.rot90(a, 1)),
    ('rot180', lambda a: np.rot90(a, 2)),
    ('rot270', lambda a: np.rot90(a, 3)),
    ('mirror', lambda a: a[:, ::-1]),
    ('mirror+rot90', lambda a: np.rot90(a[:, ::-1], 1)),
    ('mirror+rot180', lambda a: np.rot90(a[:, ::-1], 2)),
    ('mirror+rot270', lambda a: np.rot90(a[:, ::-1], 3)),
)


def oriented_mad(TH, i, j):
    """Smallest mad over the eight orientations of j. A rotated or mirrored
    save is the same picture, but every pixel lands somewhere else, so the
    plain comparison scores it like an unrelated image (~55 on a 0-255
    scale). Returns (best_mad, label)."""
    A = TH[i]
    best, how = 255.0, 'as-is'
    for name, fn in DIHEDRAL[1:]:
        B = fn(TH[j])                       # a view; no copy yet
        if B.shape != A.shape:
            # Only shape-PRESERVING orientations can match a genuine rotated
            # copy: the collector thumbnails the rotated image, so a rotated
            # save already has a transposed thumbnail and the orientation
            # that lines it up restores the original shape. Resizing a
            # shape-changing rotation compares a distorted picture - work
            # that cannot succeed. Skipping it is 4.4x faster and changed no
            # verdict across thousands of measured pairs.
            continue
        m = absdiff_mean(A, np.ascontiguousarray(B))
        if m < best:
            best, how = m, name
    return best, how


def compute_oriented_mads(TH, pairs, workers):
    if not pairs:
        return {}
    out = {}
    step = max(16, (len(pairs) + workers - 1) // workers)

    def run(k):
        for t in range(k, min(k + step, len(pairs))):
            i, j = pairs[t]
            out[(i, j)] = oriented_mad(TH, i, j)

    with ThreadPoolExecutor(workers) as pool:
        collections.deque(pool.map(run, range(0, len(pairs), step)), maxlen=0)
    return out


_LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luma(a):
    """Rec.601 luma of a uint8 RGB thumbnail, as float32."""
    return (a[:, :, 0] * 0.299 + a[:, :, 1] * 0.587 + a[:, :, 2] * 0.114)


def luma_mad_pair(TH, i, j):
    """mad ignoring colour: catches a grayscale or recoloured copy, which
    differs hugely in RGB (~30) but barely at all in brightness (~2)."""
    A, B = TH[i], TH[j]
    if A.shape != B.shape:
        B = np.asarray(Image.fromarray(B).resize(
            (A.shape[1], A.shape[0]), Image.LANCZOS), dtype=np.uint8)
    # one matmul instead of three multiplies and two adds per channel, and
    # one mean instead of two: (la - la.mean()) - (lb - lb.mean()) is
    # algebraically d - d.mean() for d = la - lb
    d = (A @ _LUMA_W) - (B @ _LUMA_W)
    # normalise out a flat brightness offset - a desaturated copy is often
    # also slightly lifted or darkened, and that is not a different picture
    return float(np.abs(d - d.mean()).mean())


def compute_luma_mads(TH, pairs, workers):
    if not pairs:
        return {}
    out = {}
    step = max(64, (len(pairs) + workers - 1) // workers)

    def run(k):
        for t in range(k, min(k + step, len(pairs))):
            i, j = pairs[t]
            out[(i, j)] = luma_mad_pair(TH, i, j)

    with ThreadPoolExecutor(workers) as pool:
        collections.deque(pool.map(run, range(0, len(pairs), step)), maxlen=0)
    return out


def trim_bars(a, tol=1.0, keep=0.25, minfrac=0.08):
    """Drop a uniform band from the edges of a thumbnail.

    Phone screenshots of vertical artwork carry big black bars top and
    bottom, and those bars are IDENTICAL between unrelated pictures. They
    are also most of the frame, so template matching correlates on them and
    two unrelated screenshots score like a crop of one another. Measured on
    real images dropped into a shared 1080x2388 frame: with 30% bars at
    each end, 2% of unrelated pairs clear the 0.90 gate, and with 40% bars,
    11% do - against 1% for the same images with no bars.

    The bars carry no information, so they are removed before matching.

    Everything here is about NOT firing on ordinary pictures. `tol` of 1.0
    is strict enough that a plain sky or a dark vignette fails it; a band
    under `minfrac` of the dimension is left alone, because nudging one
    side of a pair by a few pixels costs alignment for no gain; and `keep`
    stops a genuinely flat image being reduced to a sliver. On 70 real
    images it fires on 3, and real crops are unaffected: 90% crops stay at
    100% accepted, 70% at 99%.
    """
    g = a.astype(np.float32).mean(2)
    h, w = g.shape
    t, b, l, r = 0, h, 0, w
    while t < int(h * (1 - keep)) and g[t].std() < tol \
            and abs(g[t].mean() - g[0].mean()) < tol:
        t += 1
    while b > int(h * keep) + 1 and g[b - 1].std() < tol \
            and abs(g[b - 1].mean() - g[h - 1].mean()) < tol:
        b -= 1
    while l < int(w * (1 - keep)) and g[:, l].std() < tol \
            and abs(g[:, l].mean() - g[:, 0].mean()) < tol:
        l += 1
    while r > int(w * keep) + 1 and g[:, r - 1].std() < tol \
            and abs(g[:, r - 1].mean() - g[:, w - 1].mean()) < tol:
        r -= 1
    if t < h * minfrac:
        t = 0
    if h - b < h * minfrac:
        b = h
    if l < w * minfrac:
        l = 0
    if w - r < w * minfrac:
        r = w
    if b - t < 8 or r - l < 8 or (t == 0 and b == h and l == 0 and r == w):
        return a
    return np.ascontiguousarray(a[t:b, l:r])


def gray_small(TH, x):
    """Grayscale copy of thumb x, downscaled to max side 64 (as float32).

    Half the thumbnail's 128 px, which looks like a cheapness that ought to
    cost detection. Measured on 150 real images against their own crops, it
    does not - and it is not even neutral:

        crop keeps   cap 64   cap 96   cap 128
        95%             99%      92%       90%
        90%             97%      97%       92%
        80%             84%      95%       97%
        average         94%      90%       93%

    Zero unrelated pairs cleared the gate at any cap, and full resolution
    costs 10.84 ms a pair against 5.59. The spread between caps at one crop
    size is wider than the difference between them, so read this as "64
    costs nothing" rather than "64 wins" - the plausible reason being that
    a resized crop cannot align pixel-perfectly at 128 px and tolerates the
    mismatch at 64.

    Letterbox bars are trimmed first - see trim_bars. This affects only the
    crop tier; the pixel scores and the signature sweep see the untouched
    thumbnail."""
    g = np.asarray(Image.fromarray(trim_bars(TH[x])).convert('L'),
                   dtype=np.float32)
    s = 64.0 / max(g.shape)
    if s < 1.0:
        g = np.asarray(Image.fromarray(g.astype('uint8')).resize(
            (max(8, int(g.shape[1] * s)), max(8, int(g.shape[0] * s))),
            Image.LANCZOS), dtype=np.float32)
    return g


# Template scales tried when searching one thumbnail inside the other. The
# grid must be fine near 1.0: a gentle crop (95% of the frame) needs ~0.95,
# and with the old 0.9/1.0 spacing it scored 0.72 against a 0.92 gate - a
# trivially-detectable crop missed purely by quantisation.
NCC_SCALES = (0.35, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92,
              0.95, 0.97, 1.0)
# 0.88/0.92/0.97 added after measuring the grid rather than reasoning about
# it. The old spacing left a hole either side of 0.9, and acceptance was
# NOT monotonic in crop size - a 90% crop was accepted 79% of the time
# while an 80% crop managed 84%. Non-monotonic behaviour is what a gap in
# the grid looks like, not a threshold that is set wrong. Filling it took
# 90% crops from 79% to 97% at the old gate. Costs ~20 s on a 164 s run.


def compute_nccs(TH, pairs, workers):
    """ncc containment for the given pairs, in parallel, with the per-image
    grayscale and per-(image, scale) template work computed once instead of
    once per pair."""
    if not pairs:
        return {}
    try:
        import cv2
    except Exception:
        # Exception, not ImportError, to match the module-level guard: a cv2
        # that imports and THEN fails (numpy ABI mismatch, missing libGL)
        # raises something else entirely, and crashing here would be worse
        # than losing the crop tier.
        return {p: 0.0 for p in pairs}

    grays = {}
    idxs = sorted({x for p in pairs for x in p})
    tcache = {}

    def template(idx, s, small):
        t = tcache.get((idx, s))
        if t is None:
            th, tw = int(small.shape[0] * s), int(small.shape[1] * s)
            t = np.ascontiguousarray(np.asarray(
                Image.fromarray(small.astype('uint8')).resize(
                    (tw, th), Image.LANCZOS), dtype=np.float32))
            tcache[(idx, s)] = t
        return t

    def one(p):
        i, j = p
        a, b = grays[i], grays[j]
        # Always try BOTH directions. Thumbnail area cannot tell you which
        # image contains the other: thumbnails are capped on the long side,
        # so area measures squareness, not size. Crop a 16:9 photo to square
        # and the CROP gets the larger thumbnail (128x128 vs 128x72) - pick
        # the container by area and you search the original inside its own
        # crop, which can never match, because the crop lacks the margins.
        # Since the score is the max over directions, trying both can only
        # raise it; the bounds check below skips most of the wrong direction,
        # so the cost is well under 2x.
        dirs = ((a, b, j), (b, a, i))
        best = 0.0
        for big, small, sidx in dirs:
            for s in NCC_SCALES:
                th, tw = int(small.shape[0] * s), int(small.shape[1] * s)
                if th < 8 or tw < 8 or th > big.shape[0] or tw > big.shape[1]:
                    continue
                t = template(sidx, s, small)
                best = max(best, float(cv2.matchTemplate(
                    big, t, cv2.TM_CCOEFF_NORMED).max()))
        return p, best

    res = {}
    with ThreadPoolExecutor(workers) as pool:
        for x, g in zip(idxs, pool.map(lambda x: gray_small(TH, x), idxs)):
            grays[x] = g
        for p, v in pool.map(one, pairs):
            res[p] = v
    return res


class UF(object):
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        g = {}
        for x in list(self.p):
            g.setdefault(self.find(x), []).append(x)
        return [sorted(v) for v in g.values() if len(v) > 1]


def quality_key(recs, i):
    """Which copy of a duplicate group to KEEP: most pixels, then largest
    file, then finest JPEG quantisation (a smaller qsum is finer).

    This decides which file gets proposed for deletion, which makes it the
    most consequential rule here - and unlike the thresholds around it, it
    has never been validated against anything. It is a plausible heuristic,
    written down as one so the next person does not mistake it for a
    measured result.

    The blind spot it cannot see: area wins first, so a heavily-compressed
    UPSCALE beats a pristine original. A 4000x3000 upscale of a 2000x1500
    photo is kept and the original is the copy marked X. Nothing downstream
    catches that, because both files are genuinely the same picture and the
    tiers are only asked whether they match, not which is better.

    Testing it properly needs a judgement about real files - "did this pick
    the worse copy" is not answerable from the inventory alone - which is
    why it is still unmeasured rather than quietly assumed fine.
    """
    r = recs[i]
    return (r['w'] * r['h'], r['b'], -r.get('qsum', 10 ** 9))


def anim_compatible(ra, rb):
    """False when two records cannot be the same picture BECAUSE of motion.

    The collector thumbnails frame 0 only, so an animation is invisible to
    every pixel test here. Measured: a still extracted from a GIF and the
    GIF itself score MAD 0.0000, and two entirely different animations that
    happen to share a first frame also score 0.0000. Both were being
    reported as automatic duplicates, and deleting the animation to keep
    the still throws away every frame after the first.

    README has promised this guard since `anim` was introduced ("the frame
    count is recorded so a still never silently 'duplicates' an
    animation"). The field was recorded and never read, so the promise was
    not kept. It is now.

    Differing frame counts are treated as different animations. That can
    cost a real match - an optimiser which drops duplicate frames changes
    the count - but such a pair is not discarded, it falls through to the
    review tier. Missing a duplicate leaves both files on disk; a false one
    puts a file on a delete list, and only one of those is recoverable.
    """
    a, b = ra.get('anim'), rb.get('anim')
    if bool(a) != bool(b):
        return False                  # a still can never stand in for motion
    if not (a and b):
        return True                   # two stills: nothing here applies
    if ra.get('fsig') and rb.get('fsig'):
        # Sampled frames are the real signal, so let them decide even when
        # the frame counts differ. Frame count alone was the first rule and
        # a real corpus disproved it: of 2739 GIFs, four pixel-identical
        # pairs were rejected purely for differing counts, and the
        # fingerprint scored them 0.97, 2.63, 2.94 and 10.25 - three of
        # those sit inside the 0.00-1.93 range of the 45 pairs that were
        # allowed through. They were re-encodes with a few frames trimmed,
        # not different animations.
        return frames_agree(ra, rb)
    return a == b                     # no fingerprint: fall back to the count


# Set from a real corpus of 2739 GIFs, not guessed. On that library:
#   same animation, matching frame counts   0.00 - 1.93   (45 pairs)
#   same animation, a few frames trimmed    0.97 - 2.94   (3 pairs)
#   same base clip, one a longer loop       10.25         (1 pair)
#   genuinely different animations          62 - 85       (measured separately)
# 6.0 sits in the gap above every confirmed re-encode and below the loop
# variant, which is the right side to err on: a trimmed-or-extended clip
# goes to review rather than onto a delete list, while the enormous margin
# to 62 keeps unrelated animations out regardless.
FRAME_CUT = 6.0

# A second, separate question: does any ONE sampled frame disagree badly?
# The mean above answers "do these animations look alike overall", and it is
# blind to a short stretch being different, because 24 agreeing frames
# divide the one that does not. Measured on 39 real animations against
# themselves with a stretch replaced, the mean alone missed 14 - 36% - and
# every one of those shares frame 0, so its thumbnail and CLIP vector agree
# too and nothing else would have caught it.
#
# Set from the same real corpus, not guessed:
#   same clip, re-encoded (70 pairs)      worst frame up to 47.09
#   same clip, stretch replaced (39)      worst frame up to 127.13
#   genuinely different clips (494)       worst frame never below 24.89
# 60 clears every real re-encode by 13 points and still catches 38 of 39
# planted differences. It costs nothing: the 2 re-encodes it blocks already
# fail the mean test, whose own worst real value is 6.65.
#
# Both must pass. The mean catches wholesale differences, the worst frame
# catches local ones, and neither is asked to do the other's job.
FRAME_WORST_CUT = 60.0


def frames_agree(ra, rb):
    """Do two animations still match once you look past frame 0?

    Returns True when one side has no fingerprint at all - there is
    genuinely nothing to say, and silence must not destroy a real match.

    A SIZE MISMATCH is different, and getting that wrong shipped a false
    delete. Fingerprints of unequal length are positionally incomparable,
    and this used to abstain on them, which reads as consent: an all-black
    2-sample fingerprint and an all-white 5-sample one were declared
    compatible, so unrelated animations sharing a first frame were
    pre-marked for the Recycle Bin. Collector fingerprints are fixed-size
    now, but inventories written before that fix still hold short ones, so
    fall back to the frame count instead of consenting.
    """
    fa, fb = ra.get('fsig'), rb.get('fsig')
    if not fa or not fb:
        return True
    try:
        A = np.frombuffer(base64.b64decode(fa), dtype=np.uint8)
        B = np.frombuffer(base64.b64decode(fb), dtype=np.uint8)
    except Exception:
        return True
    if A.size == 0 or B.size == 0:
        return True
    if A.size != B.size:
        return ra.get('anim') == rb.get('anim')
    d = np.abs(A.astype(np.float32) - B.astype(np.float32))
    if float(d.mean()) > FRAME_CUT:
        return False
    # ...and no single sampled frame may disagree badly, which is the part
    # the mean cannot see. Only when the fingerprint is tile-shaped: an
    # inventory written before this existed still gets the mean test alone
    # rather than a reshape that would misread its bytes.
    if d.size % 192 == 0:
        return float(d.reshape(-1, 192).mean(1).max()) <= FRAME_WORST_CUT
    return True


def shown_dims(r):
    """(w, h) as the picture actually appears, not as the file stores it.

    The collector records the file's RAW dimensions but thumbnails the
    EXIF-ROTATED image, so a portrait phone photo stores 4032x3024 and
    displays as portrait. Printing the raw pair beside a rotated thumbnail
    is merely confusing; feeding it to the "identical dimensions" test is
    worse - an EXIF-rotated original and a physically-rotated,
    EXIF-stripped copy of the SAME photo were labelled MIXED RESOLUTIONS
    despite having identical pixels.

    Orientations 5-8 are the ones that transpose (6 and 8 are the quarter
    turns, 5 and 7 those plus a mirror). Fixed here rather than in the
    collector so existing inventories keep working: 'ori' has been recorded
    all along, it was simply never read. Note keeper_key above is unaffected
    either way, since it ranks on w*h, which does not care about rotation.
    """
    w, h = r['w'], r['h']
    return (h, w) if r.get('ori') in (5, 6, 7, 8) else (w, h)


# ------------------------------------------------------------- invariants --
class InvariantError(Exception):
    pass


def check_invariants(tier_a, tier_b):
    """tier_a / tier_b: list of (keeper, [drops], [members])."""
    seen = {}
    keepers = set()
    for tier, name in ((tier_a, 'A'), (tier_b, 'B')):
        for k, drops, members in tier:
            if len(members) < 2:
                raise InvariantError('tier %s: cluster with %d member(s)' % (name, len(members)))
            if k in drops:
                raise InvariantError('tier %s: keeper %r is also its own drop' % (name, k))
            keepers.add(k)
            for d in drops:
                if d in seen:
                    raise InvariantError(
                        'file %r listed as a candidate twice (tier %s and tier %s)'
                        % (d, seen[d], name))
                seen[d] = name
    clash = keepers & set(seen)
    if clash:
        raise InvariantError('%d file(s) are a KEEPER and a deletion candidate: %s'
                             % (len(clash), sorted(clash)[:5]))
    for tier, name in ((tier_a, 'A'), (tier_b, 'B')):
        for k, drops, members in tier:
            if k in seen:
                raise InvariantError('tier %s: keeper %r is a candidate elsewhere' % (name, k))
            if not drops:
                raise InvariantError('tier %s: cluster with no candidates' % name)
    return True


def build_tier_b(groups, recs, tier_a):
    """Turn raw crop/variant groups into (tier_b, info_b).

    tier_b entries suggest deletions; info_b entries are shown for review
    with nothing pre-marked. Members that Tier A is already deleting are
    taken out - but a group must never be dropped just because that empties
    it, or a file whose ONLY relation was to those doomed copies would
    appear in no cluster anywhere. Their Tier A keeper holds the same
    pixels, so it stands in for them and the relation outlives the cleanup
    it describes.
    """
    a_drops = set(i for _, d, _ in tier_a for i in d)
    a_keeps = set(k for k, _, _ in tier_a)
    keeper_of = {}                       # Tier A drop -> the copy that stays
    for k, drops, _m in tier_a:
        for d in drops:
            keeper_of[d] = k

    tier_b, info_b = [], []
    for members in groups:
        surv = [i for i in members if i not in a_drops]     # already leaving
        if len(surv) < 2:
            stand = list(surv)
            for i in members:
                if i in a_drops:
                    kk = keeper_of.get(i)
                    if kk is not None and kk not in stand:
                        stand.append(kk)
            if len(stand) > 1:
                info_b.append(sorted(stand))
            continue
        k = max(surv, key=lambda i: quality_key(recs, i))
        drops = [i for i in surv if i != k and i not in a_keeps and i not in a_drops]
        if drops:
            tier_b.append((k, drops, surv))
        else:
            # No suggested deletions here: every member apart from the
            # elected keeper is itself a Tier A keeper. The old rule
            # discarded the whole group - hiding the crop/variant relation
            # between kept files, and sometimes hiding a free file whose
            # only cluster this was. Keep it: shown, reviewable, nothing
            # pre-marked.
            info_b.append(surv)
    return tier_b, info_b


def build_emission_plan(tier_a, tier_b, recs, info_b=None):
    """Decide, for every cluster, which members get an EDITABLE line and which
    are only referenced.

    info_b: optional list of member-lists for crop/variant groups with no
    suggested deletions (each member is a Tier A keeper, plus possibly the
    group's own elected keeper as a free file). Members already editable
    elsewhere become references; a free member still gets its editable
    line here - so the group is shown and reviewable either way.

    A file can legitimately belong to a Tier A cluster (exact duplicates) and
    also to a Tier B cluster (it is the uncropped original of something).
    Giving it an editable line in both lets the two lines disagree - and the
    Recycle script keys marks by filename, so the later line silently wins.
    That is exactly how a cluster showing one "." and one "X" was reported as
    "every copy is marked X".

    So: first cluster to claim a file owns its editable line (Tier A first,
    because exact duplication is the stronger relation). Later clusters show
    it as a reference comment naming where it can be edited.

    Returns [(cl_id, tier_key, keeper, editable, refs)].
    """
    plan = []
    home = {}
    cl_id = 0
    for tier, key in ((tier_a, 'A'), (tier_b, 'B'),
                      ([(None, [], m) for m in (info_b or [])], 'B')):
        for k, drops, members in tier:
            editable = [i for i in members if i not in home]
            refs = [i for i in members if i in home]
            cl_id += 1
            if not editable:
                # every member already has its editable line elsewhere.
                # Keep the cluster as a reference-only entry so the
                # relationship is still SHOWN (report tiles + list
                # comments); it gets no manifest rows and nothing can be
                # deleted here. (Used to be dropped silently.)
                plan.append((cl_id, key, None, [], refs))
                continue
            keeper = k if k in editable else max(
                editable, key=lambda i: quality_key(recs, i))
            for i in editable:
                home[i] = cl_id
            plan.append((cl_id, key, keeper, editable, refs))
    return plan, home


def check_emission(plan, home):
    """One editable line per file, and every reference resolves elsewhere."""
    seen = {}
    for cl_id, key, keeper, editable, refs in plan:
        if not editable:
            # reference-only cluster: information display, nothing editable
            if keeper is not None:
                raise InvariantError('cluster %d: reference-only cluster has a keeper' % cl_id)
            if not refs:
                raise InvariantError('cluster %d is empty' % cl_id)
        elif keeper not in editable:
            raise InvariantError('cluster %d: keeper is not one of its editable members' % cl_id)
        for i in editable:
            if i in seen:
                raise InvariantError(
                    'file %r would get an editable line in cluster %d and cluster %d - '
                    'the two lines could disagree' % (i, seen[i], cl_id))
            seen[i] = cl_id
        for i in refs:
            if i not in home:
                raise InvariantError('cluster %d references a file with no home' % cl_id)
            if home[i] == cl_id:
                raise InvariantError('cluster %d lists a file as both editable and reference' % cl_id)
    return True


# ------------------------------------------------------------- self tests --
def _sweep_reference(C, cut):
    """The old brute-force sweep, kept as the oracle for --self-test."""
    n = C.shape[0]
    ref = set()
    step = max(1, min(64, 8_000_000 // max(1, n)))
    for a in range(0, n, step):
        d = np.abs(C[a:a + step, None, :] - C[None, :, :]).mean(2)
        ii, jj = np.where(d <= cut)
        for x, y in zip(ii, jj):
            gi = a + int(x)
            if gi < int(y):
                ref.add((gi, int(y)))
    return ref


def cv2_fallback_tests():
    """Every OpenCV-free path, exercised on a machine that HAS OpenCV.

    This suite exists because absdiff_mean once shipped with its fallback
    calling itself - infinite recursion on every machine without OpenCV -
    and nothing here noticed. The rest of the suite runs where cv2 IS
    installed, so the fallback branch was simply never entered. A test that
    cannot reach a supported configuration is not covering it.

    Two switches are needed, and neither alone is enough:
        _cv2 = None                module global; reaches absdiff_mean and
                                   imdecode_rgb (and everything built on it)
        sys.modules['cv2'] = None  makes a later `import cv2` raise
                                   ImportError; reaches compute_nccs, which
                                   imports cv2 locally and would otherwise
                                   go on scoring happily while you believed
                                   you had taken OpenCV away
    """
    global _cv2
    print('OpenCV-free fallbacks')
    if _cv2 is None:
        print('  [SKIP] OpenCV is absent here - nothing to compare against')
        return True

    ok = True

    def case(name, fn, extra=''):
        """fn is a thunk, not a value, so a check that RAISES is reported as
        a failure instead of killing the run. That matters here more than
        elsewhere: the regression this suite was written for (absdiff_mean
        recursing into itself) throws rather than returning a wrong number,
        and a suite that dies on the first one hides every later check."""
        nonlocal ok
        try:
            good, why = bool(fn()), extra
        except Exception as exc:
            good, why = False, '%s: %s' % (type(exc).__name__, exc)
        print('  [%s] %s%s' % ('PASS' if good else 'FAIL', name,
                               '' if good else '   (%s)' % why))
        ok = ok and good

    rng = np.random.default_rng(20260807)

    def enc(a):
        # PNG, not JPEG: lossless, so the two decoders must agree EXACTLY
        # and any difference is a real defect rather than codec noise.
        b = io.BytesIO()
        Image.fromarray(a).save(b, 'PNG')
        return b.getvalue()

    base = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    shifted = np.roll(base, 1, axis=0)
    recs = [{'tb': base64.b64encode(enc(rng.integers(
                 0, 256, (32, 32, 3), dtype=np.uint8))).decode(),
             'tw': 32, 'th': 32} for _ in range(6)]
    sq = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    TH_sq = [sq, np.ascontiguousarray(np.rot90(sq, 1))]

    # reference values, taken WITH OpenCV
    ref_ad = absdiff_mean(base, shifted)
    ref_img = imdecode_rgb(enc(base))
    ref_sig = decode_signatures(recs, 3)
    ref_or = oriented_mad(TH_sq, 0, 1)

    had = 'cv2' in sys.modules
    saved_mod, saved_cv2 = sys.modules.get('cv2'), _cv2
    try:
        _cv2 = None
        sys.modules['cv2'] = None

        case('absdiff_mean falls back instead of recursing',
             lambda: np.isfinite(absdiff_mean(base, shifted)))
        case('absdiff_mean fallback agrees with cv2',
             lambda: abs(absdiff_mean(base, shifted) - ref_ad) < 1e-3)
        case('absdiff_mean scores identical inputs exactly 0',
             lambda: absdiff_mean(base, base.copy()) == 0.0)
        view = base.transpose(1, 0, 2)          # deliberately non-contiguous
        case('absdiff_mean accepts a non-contiguous view',
             lambda: np.isfinite(absdiff_mean(view, view.copy())))

        case('imdecode_rgb fallback is byte-identical to cv2',
             lambda: np.array_equal(imdecode_rgb(enc(base)), ref_img))
        case('imdecode_rgb fallback keeps uint8 HxWx3',
             lambda: imdecode_rgb(enc(base)).dtype == np.uint8
             and imdecode_rgb(enc(base)).shape == base.shape)
        red = np.zeros((8, 8, 3), np.uint8)
        red[..., 0] = 255
        # a BGR slip would pass every shape check and corrupt every signature
        case('imdecode_rgb fallback returns RGB, not BGR',
             lambda: tuple(int(v) for v in imdecode_rgb(enc(red))[0, 0])
             == (255, 0, 0))

        case('decode_signatures matches the cv2 result exactly',
             lambda: np.array_equal(decode_signatures(recs, 3), ref_sig))

        pre = ThumbStore(recs, budget=10 ** 9, workers=2)
        lazy = ThumbStore(recs, cap=64, budget=1, workers=2)
        case('ThumbStore preloads when the budget allows',
             lambda: pre.preloaded)
        case('ThumbStore goes lazy on a tight budget',
             lambda: not lazy.preloaded)
        case('both ThumbStore branches decode identically',
             lambda: all(np.array_equal(pre[i], lazy[i])
                         for i in range(len(recs))))

        # a LABEL flip matters more than numeric drift: it renames the match
        case('oriented_mad picks the same orientation without cv2',
             lambda: oriented_mad(TH_sq, 0, 1)[1] == ref_or[1])
        case('oriented_mad score agrees with cv2',
             lambda: abs(oriented_mad(TH_sq, 0, 1)[0] - ref_or[0]) < 1e-3)

        # the one that _cv2 = None alone would NOT have covered
        pairs = [(0, 1), (0, 2)]
        case('compute_nccs degrades to zeros rather than raising',
             lambda: set(compute_nccs(pre, pairs, 2)) == set(pairs)
             and all(v == 0.0 for v in compute_nccs(pre, pairs, 2).values()))
        case('compute_nccs still short-circuits on an empty pair list',
             lambda: compute_nccs(pre, [], 2) == {})
    except Exception as exc:
        # anything unguarded above (a constructor, say) still lands as a
        # reported failure rather than a traceback that skips the restore
        case('fallback suite ran to completion', lambda: False,
             '%s: %s' % (type(exc).__name__, exc))
    finally:
        _cv2 = saved_cv2
        if had:
            sys.modules['cv2'] = saved_mod
        else:
            sys.modules.pop('cv2', None)

    # If the restore leaked, every test after this one runs degraded and
    # silently means something else.
    case('OpenCV is restored afterwards',
         lambda: _cv2 is not None and __import__('cv2') is not None)
    return ok


def self_test():
    ok = True

    def case(name, tier_a, tier_b, should_raise):
        nonlocal ok
        try:
            check_invariants(tier_a, tier_b)
            raised = None
        except InvariantError as e:
            raised = str(e)
        good = (raised is not None) == should_raise
        print('  [%s] %s%s' % ('PASS' if good else 'FAIL', name,
                               '' if good else '   (got: %s)' % raised))
        ok = ok and good

    print('Invariant self-test')
    case('clean A+B accepted', [(1, [2], [1, 2])], [(3, [4], [3, 4])], False)
    case('keeper also dropped elsewhere', [(1, [2], [1, 2])], [(3, [1], [3, 1])], True)
    case('same file listed twice', [(1, [2], [1, 2])], [(3, [2], [3, 2])], True)
    case('keeper inside its own drops', [(1, [1], [1, 2])], [], True)
    case('one-member cluster', [(1, [], [1])], [], True)
    case('cluster with no candidates', [(1, [], [1, 2])], [], True)
    # chained tier-B pairs: A>B, B>C  must not leave B as both keeper and drop
    uf = UF()
    uf.union(1, 2)
    uf.union(2, 3)
    grp = uf.groups()
    # emission plan: a file shared by a Tier A and a Tier B cluster must get
    # exactly ONE editable line (the real bug: it got two, which disagreed)
    recs_stub = [{'w': 10, 'h': 10, 'b': 10, 'qsum': 1} for _ in range(5)]
    ta = [(0, [1], [0, 1])]
    tb = [(0, [2], [0, 2])]          # file 0 is in both
    plan, hm = build_emission_plan(ta, tb, recs_stub)
    ed = [e for _, _, _, e, _ in plan]
    flat = [i for group in ed for i in group]
    shared_once = len(flat) == len(set(flat)) and 0 in flat
    print('  [%s] file shared by tier A and B gets exactly one editable line'
          % ('PASS' if shared_once else 'FAIL'))
    ok = ok and shared_once
    ref_ok = any(0 in r for _, _, _, _, r in plan)
    print('  [%s] and is shown as a reference in the other cluster'
          % ('PASS' if ref_ok else 'FAIL'))
    ok = ok and ref_ok
    try:
        check_emission(plan, hm); em_ok = True
    except InvariantError:
        em_ok = False
    print('  [%s] emission plan passes its own check' % ('PASS' if em_ok else 'FAIL'))
    ok = ok and em_ok
    try:
        check_emission([(1, 'A', 0, [0, 1], []), (2, 'B', 0, [0, 2], [])], {0: 1, 1: 1, 2: 2})
        dbl = False
    except InvariantError:
        dbl = True
    print('  [%s] two editable lines for one file are rejected' % ('PASS' if dbl else 'FAIL'))
    ok = ok and dbl
    # a tier-B cluster whose members ALL have their editable lines elsewhere
    # must still appear in the plan as a reference-only entry (v3.8: it used
    # to be dropped silently, hiding the crop relationship from the report)
    ta2 = [(0, [1], [0, 1]), (2, [3], [2, 3])]
    tb2 = [(0, [2], [0, 2])]                 # both 0 and 2 claimed by tier A
    plan2, hm2 = build_emission_plan(ta2, tb2, recs_stub)
    ronly = [e for e in plan2 if not e[3]]
    got = (len(ronly) == 1 and ronly[0][1] == 'B' and ronly[0][2] is None
           and sorted(ronly[0][4]) == [0, 2])
    print('  [%s] reference-only cluster is kept as information, not dropped'
          % ('PASS' if got else 'FAIL'))
    ok = ok and got
    try:
        check_emission(plan2, hm2)
        em2 = True
    except InvariantError:
        em2 = False
    print('  [%s] reference-only cluster passes the emission check'
          % ('PASS' if em2 else 'FAIL'))
    ok = ok and em2
    # informational groups (crop relation between two Tier A keepers) enter
    # the plan through the info_b parameter and come out reference-only too
    plan3, hm3 = build_emission_plan(ta2, [], recs_stub, info_b=[[0, 2]])
    ronly3 = [e for e in plan3 if not e[3]]
    got3 = (len(ronly3) == 1 and ronly3[0][2] is None
            and sorted(ronly3[0][4]) == [0, 2])
    try:
        check_emission(plan3, hm3)
    except InvariantError:
        got3 = False
    print('  [%s] informational keeper-to-keeper group emitted the same way'
          % ('PASS' if got3 else 'FAIL'))
    ok = ok and got3

    # A crop group whose every OTHER member is a Tier A drop must not vanish:
    # file 3 relates only to file 1, and file 1 is a drop of Tier A cluster
    # {0 keeps, 1 drops}. Removing the drop leaves one member, and the old
    # rule discarded the group - so file 3, which is in no Tier A cluster,
    # appeared in NO cluster at all.
    recs4 = [{'w': 40 - i, 'h': 40, 'b': 100 - i, 'qsum': 1} for i in range(4)]
    ta4 = [(0, [1], [0, 1])]                       # 0 keeps, 1 is deleted
    tb4, ib4 = build_tier_b([[1, 3]], recs4, ta4)  # crop relation 1 <-> 3
    rescued = (tb4 == [] and ib4 == [[0, 3]])
    print('  [%s] a crop group left with one member keeps it, standing the '
          'Tier A keeper in for the deleted copy' % ('PASS' if rescued else 'FAIL'))
    ok = ok and rescued
    plan4, hm4 = build_emission_plan(ta4, tb4, recs4, ib4)
    homes4 = {i for _, _, _, ed, _ in plan4 for i in ed}
    try:
        check_emission(plan4, hm4)
        check_invariants(ta4, tb4)
        shown4 = 3 in homes4                       # the free file is editable
    except InvariantError:
        shown4 = False
    print('  [%s] and the rescued file gets its own editable line'
          % ('PASS' if shown4 else 'FAIL'))
    ok = ok and shown4
    # a group of only doomed copies of ONE Tier A cluster still collapses -
    # its whole story is already told by that cluster
    _, ib5 = build_tier_b([[1, 2]], recs4, [(0, [1, 2], [0, 1, 2])])
    print('  [%s] a group of drops from one Tier A cluster stays collapsed'
          % ('PASS' if ib5 == [] else 'FAIL'))
    ok = ok and ib5 == []

    print('  [%s] chained pairs collapse into one cluster of %d'
          % ('PASS' if grp == [[1, 2, 3]] else 'FAIL', len(grp[0]) if grp else 0))
    ok = ok and grp == [[1, 2, 3]]

    # The BLAS sweep must return EXACTLY the brute-force candidate set -
    # including the sparse-outlier pair that defeats a naive RMS-only filter
    # (one huge channel difference, tiny mean) and a pair sitting exactly on
    # the threshold.
    rng = np.random.default_rng(20260806)
    C = rng.integers(0, 256, size=(72, 192)).astype(np.float32)
    C[1] = C[0]; C[1][0] = min(255.0, C[1][0] + 200.0)     # sparse outlier
    C[3] = C[2]                                            # identical pair
    C[5] = np.clip(C[4] + 8.0, 0, 255)                     # near threshold
    C[7] = np.minimum(C[6] + 8.0, 255.0)
    # The sweep bands rows by mean signature value, so the dangerous case is
    # a real pair sitting just INSIDE that band: a flat offset of 7.9 moves
    # the mean by 7.9 against a cut of 8. An off-by-one in the band edge
    # would drop it, and nothing else in this suite would notice.
    C[9] = np.clip(C[8] + 7.9, 0, 255)
    C[11] = np.clip(C[10] - 7.9, 0, 255)
    new = set(sweep_candidates(C, 8.0))
    ref = _sweep_reference(C, 8.0)
    sweep_ok = (new == ref and (0, 1) in new and (2, 3) in new
                and (8, 9) in new and (10, 11) in new)
    print('  [%s] BLAS sweep == brute force on %d pairs (%d candidates)'
          % ('PASS' if sweep_ok else 'FAIL', 72 * 71 // 2, len(ref)))
    ok = ok and sweep_ok

    # The collector resizes into a new image instead of thumbnailing the
    # open file in place, using its own copy of Pillow's size arithmetic.
    # That is what stops a lazy plugin re-decoding at full resolution and
    # dropping the file (every multi-frame HEIC hit this). Pin the copy
    # against the real thing, so a Pillow change surfaces here rather than
    # as every thumbnail quietly moving.
    print('')
    print('Collector thumbnail sizing')
    try:
        import importlib.util
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'collect-image-inventory.py')
        _s = importlib.util.spec_from_file_location('_col_sz', _p)
        _c = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_c)
        shapes = [(800, 600), (600, 800), (1000, 1000), (1920, 1080),
                  (333, 777), (2000, 137), (100, 90), (1, 500), (128, 128)]
        bad = []
        for w, h in shapes:
            for box in (128, 256):
                want = Image.new('RGB', (w, h))
                want.thumbnail((box, box), Image.LANCZOS)
                got = _c.thumb_size(w, h, (box, box)) or (w, h)
                if tuple(want.size) != tuple(got):
                    bad.append('%dx%d@%d: pillow=%s ours=%s'
                               % (w, h, box, want.size, got))
        print('  [%s] thumb_size matches Image.thumbnail on %d shapes%s'
              % ('PASS' if not bad else 'FAIL', len(shapes) * 2,
                 '' if not bad else '   ' + '; '.join(bad[:2])))
        ok = ok and not bad

        # Every animation must produce the SAME NUMBER of samples, however
        # few frames it has. Deduplicating the sample indices made short
        # animations emit shorter fingerprints, which are positionally
        # incomparable - and the comparison below used to read that as
        # consent, pre-marking unrelated animations for deletion.
        want = _c.FRAME_SAMPLES
        lens = set()
        for n in (2, 3, 4, 5, 6, 9, 40):
            idx = [int(round(t * (n - 1) / (want - 1))) for t in range(want)]
            lens.add(len(idx))
        fixed = lens == {want}
        print('  [%s] frame fingerprint is %d samples for every length'
              % ('PASS' if fixed else 'FAIL', want))
        ok = ok and fixed
    except Exception as exc:
        print('  [FAIL] could not check: %s: %s' % (type(exc).__name__, exc))
        ok = False

    # ... and a mismatched length must never be read as agreement.
    try:
        short = base64.b64encode(b'\x00' * 384).decode()
        longer = base64.b64encode(b'\xff' * 960).decode()
        strict = not frames_agree({'fsig': short, 'anim': 2},
                                  {'fsig': longer, 'anim': 9})
        print('  [%s] a size-mismatched fingerprint is not consent'
              % ('PASS' if strict else 'FAIL'))
        ok = ok and strict
    except Exception as exc:
        print('  [FAIL] could not check: %s: %s' % (type(exc).__name__, exc))
        ok = False

    # One frame that clearly disagrees must veto the match, even when every
    # other sampled frame is identical. That is the case the mean cannot
    # see - it divides the difference by however many frames agreed - and
    # it missed 36% of them on a real corpus before the worst-frame test
    # was added.
    try:
        K = 25
        base = np.zeros((K, 192), dtype=np.uint8)
        base[:] = 100
        other = base.copy()
        other[K // 2] = 240              # a single sampled frame goes wrong
        ra = {'fsig': base64.b64encode(base.tobytes()).decode(), 'anim': 60}
        rb = {'fsig': base64.b64encode(other.tobytes()).decode(), 'anim': 60}
        vetoed = not frames_agree(ra, rb)
        print('  [%s] one clearly different frame is not consent'
              % ('PASS' if vetoed else 'FAIL'))
        ok = ok and vetoed
        # ...while an even, mild difference everywhere still reads as the
        # same animation, which is what a re-encode looks like.
        mild = base.copy().astype(np.int16) + 3
        rc = {'fsig': base64.b64encode(mild.astype(np.uint8).tobytes()).decode(),
              'anim': 60}
        agrees = frames_agree(ra, rc)
        print('  [%s] a mild difference everywhere still matches'
              % ('PASS' if agrees else 'FAIL'))
        ok = ok and agrees
    except Exception as exc:
        print('  [FAIL] could not check: %s: %s' % (type(exc).__name__, exc))
        ok = False

    # A name the list refuses to make editable must not be reachable by the
    # recycler's bulk Tier B answer either. The list promises "always kept"
    # about these; that promise has to survive every route to deletion, not
    # just the one it is printed next to.
    try:
        import re as _re
        import shutil as _sh
        import tempfile
        recs_n = [{'p': 'keeper.png', 'b': 900, 'sha': 'a' * 64, 'w': 40, 'h': 40},
                  {'p': 'bad\nname.png', 'b': 100, 'sha': 'b' * 64, 'w': 20, 'h': 20}]
        td = tempfile.mkdtemp()
        _n, _L, _at, sug, _all = write_list_and_script(
            os.path.join(td, 'l.txt'), os.path.join(td, 'r.py'),
            os.path.join(td, 'r.bat'), os.path.join(td, 'r.sh'),
            recs_n, [], [(0, [1], [0, 1])], td)
        src = open(os.path.join(td, 'r.py'), encoding='utf-8').read()
        man = json.loads(_re.search(r'MANIFEST = (\[.*?\n\])\n', src, _re.S).group(1))
        flagged = [r for r in man if r.get('sd')]
        safe = not sug and not flagged
        print('  [%s] an uneditable name is never a bulk-delete suggestion'
              % ('PASS' if safe else 'FAIL'))
        ok = ok and safe
        _sh.rmtree(td, ignore_errors=True)
    except Exception as exc:
        print('  [FAIL] could not check: %s: %s' % (type(exc).__name__, exc))
        ok = False

    # A path may legally contain "</script>" on Linux - one directory called
    # `a<` and a file called `script>`. Generate a real report through the
    # real writer and check the browser would still see one whole script.
    try:
        import shutil as _sh2
        import tempfile
        buf = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, 'JPEG')
        tb = base64.b64encode(buf.getvalue()).decode('ascii')
        evil = 'a</script>x.png'
        recs_s = [{'p': 'keep.png', 'b': 900, 'sha': 'a' * 64, 'w': 40, 'h': 40, 'tb': tb},
                  {'p': evil, 'b': 100, 'sha': 'b' * 64, 'w': 20, 'h': 20, 'tb': tb}]
        ta_s = [(0, [1], [0, 1])]
        td = tempfile.mkdtemp()
        _n, LL, at_s, sg_s, all_s = write_list_and_script(
            os.path.join(td, 'l.txt'), os.path.join(td, 'r.py'),
            os.path.join(td, 'r.bat'), os.path.join(td, 'r.sh'),
            recs_s, ta_s, [], td)
        pl_s, hm_s = build_emission_plan(ta_s, [], recs_s)
        rp = os.path.join(td, 'r.html')
        write_report(rp, recs_s, ta_s, [], td,
                     {'n': 2, 'exact': 0, 'headline': 'x', 'method': 'y'},
                     pl_s, hm_s, list_lines=LL, editable_at=at_s,
                     suggested_b=sg_s, tier_b_all=all_s, list_name='l.txt')
        page = open(rp, encoding='utf-8').read()
        body = page[page.find('<script>'):]
        # exactly one closing tag, and it comes after the code that runs
        whole = (body.count('</script>') == 1
                 and body.find('paint();') < body.find('</script>'))
        print('  [%s] a path cannot close the report\'s script block'
              % ('PASS' if whole else 'FAIL'))
        ok = ok and whole
        _sh2.rmtree(td, ignore_errors=True)
    except Exception as exc:
        print('  [FAIL] could not check: %s: %s' % (type(exc).__name__, exc))
        ok = False

    print('')
    # called first, then ANDed - the reverse would short-circuit the whole
    # fallback suite away the moment anything above it failed
    ok = cv2_fallback_tests() and ok

    print('Self-test: ' + ('ALL PASS' if ok else 'FAILURES'))
    return 0 if ok else 1


# ----------------------------------------------------------------- output --
def list_safe(p):
    """A path as it may appear in the hand-edited selection list: control
    characters escaped so a filename can never span lines or forge a mark
    line. Paths that come back unchanged are safe to emit raw."""
    return ''.join(ch if (ch >= ' ' and ch != '\x7f') else repr(ch)[1:-1]
                   for ch in p)



def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def small_b64(rec, maxw=150):
    im = Image.open(io.BytesIO(base64.b64decode(rec['tb']))).convert('RGB')
    if im.width > maxw:
        im = im.resize((maxw, max(1, int(im.height * maxw / im.width))), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=80)
    return base64.b64encode(b.getvalue()).decode('ascii')


def write_report(path, recs, tier_a, tier_b, root, stats, plan=None, home=None,
                 list_lines=None, editable_at=None, suggested_b=None,
                 tier_b_all=None, list_name=None, b_edges=None):
    def mb(x):
        return '%.2f MB' % (x / 1048576.0)

    def meta(i):
        r = recs[i]
        s = '%dx%d &middot; %s' % (shown_dims(r) + (mb(r['b']),))
        if 'qsum' in r:
            s += ' &middot; q%d' % r['qsum']
        return s
    P = []
    A = P.append
    dn = sum(len(d) for _, d, _ in tier_a)
    db = sum(recs[i]['b'] for _, d, _ in tier_a for i in d)
    A('<!doctype html><html lang="en"><meta charset="utf-8"><title>Duplicate report</title>')
    A('<meta name="viewport" content="width=device-width,initial-scale=1"><style>'
      # Dark by design: this report is looked at next to the pictures it is
      # about, and a dark page keeps the thumbnails - not the paper around
      # them - the brightest thing on screen. color-scheme darkens the
      # scrollbars too, so the frame matches the page.
      ':root{color-scheme:dark;'
      '--bg:#0f1116;--panel:#171a21;--panel2:#1c202a;'
      '--ink:#e7e9ef;--dim:#99a1b3;--line:#2a2f3a;'
      '--keep:#3ddc97;--drop:#ff7a7a;--rev:#f0b64b;--ref:#6f7787}'
      '*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);'
      'font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;'
      '-webkit-font-smoothing:antialiased}'
      '.wrap{max-width:1180px;margin:0 auto;padding:40px 24px 90px}'
      'h1{font-size:27px;margin:0 0 6px;letter-spacing:-.02em}'
      '.sub{color:var(--dim);margin:0 0 24px;max-width:80ch}'
      '.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:22px 0}'
      '.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px}'
      '.stat b{display:block;font-size:22px}.stat span{color:var(--dim);font-size:12.5px}'
      'h2{font-size:20px;margin:44px 0 4px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}'
      '.pill{font:700 11px/1.9 ui-sans-serif;letter-spacing:.05em;padding:0 9px;border-radius:5px}'
      '.lead{color:var(--dim);margin:0 0 14px;max-width:80ch}'
      '.cl{background:var(--panel);border:1px solid var(--line);border-radius:11px;'
      'margin:14px 0;overflow:hidden}'
      '.ch{padding:9px 15px;background:var(--panel2);border-bottom:1px solid var(--line);'
      'display:flex;justify-content:space-between;gap:14px;font-size:12.5px;color:var(--dim);flex-wrap:wrap}'
      '.ch b{color:var(--ink)}.strip{display:flex;gap:16px;padding:16px;overflow-x:auto}'
      '.it{flex:0 0 auto;width:150px}'
      '.it img{width:150px;border-radius:6px;border:3px solid transparent;display:block;'
      'background:#20242d}'
      '.it.k img{border-color:var(--keep)}.it.d img{border-color:var(--drop)}'
      '.it.n img{border-color:var(--rev)}.it.r img{border-color:var(--ref);opacity:.55}'
      '.it.r .lb{color:var(--dim)}'
      '.it.x img{border-color:#7d6a9e;border-style:dashed}'
      '.it.x .lb{color:#b3a0d4}'
      '.clid{font:700 12px/1.7 ui-monospace,Menlo,Consolas,monospace;'
      'background:#1e2634;color:#a9bde8;padding:1px 8px;border-radius:5px;'
      'text-decoration:none;border:1px solid #2e3a4e}'
      '.clid:hover{background:#27324a}'
      '.cnt{color:var(--dim);font-weight:400}'
      '.cl:target{box-shadow:0 0 0 3px #5b7cfa55;border-color:#5b7cfa}'
      '.lb{font:700 9.5px/1.6 ui-sans-serif;letter-spacing:.05em;margin-top:5px}'
      '.it.k .lb{color:var(--keep)}.it.d .lb{color:var(--drop)}.it.n .lb{color:var(--rev)}'
      '.fn{font:10.5px/1.35 ui-monospace,Menlo,Consolas,monospace;color:var(--dim);'
      'word-break:break-all;margin-top:3px}'
      'footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);'
      'color:var(--dim);font-size:13.5px;max-width:84ch}'
      'code{background:#232833;color:#cdd3e0;padding:1px 5px;border-radius:4px;font-size:12.5px}'
      # Review controls. Hidden unless the page was given a list to edit,
      # so a report written without one looks exactly as it did before.
      '.bar{position:sticky;top:0;z-index:9;background:#131720ee;backdrop-filter:blur(8px);'
      'border-bottom:1px solid var(--line);margin:0 -24px 8px;padding:11px 24px;'
      'display:flex;gap:10px;align-items:center;flex-wrap:wrap}'
      '.bar .sp{flex:1}'
      'button{font:600 13px/1 inherit;color:var(--ink);background:var(--panel2);'
      'border:1px solid var(--line);border-radius:7px;padding:8px 13px;cursor:pointer}'
      'button:hover{background:#252b38;border-color:#3a4358}'
      'button.go{background:#1d4d38;border-color:#2c6b4e;color:#c8f5e0}'
      'button.go:hover{background:#236049}'
      '.tally{font:600 13px/1 inherit;color:var(--dim)}'
      '.tally b{color:var(--drop);font-size:15px}'
      '.it.sel{outline:2px solid #5b7cfa;outline-offset:3px;border-radius:8px}'
      '.it .tg{margin-top:5px;width:100%;padding:5px 0;font-size:11px}'
      '.it.on .tg{background:#4a1f1f;border-color:#7a3030;color:#ffd6d6}'
      '.it .tg.no{background:#4a3a1f;border-color:#7a6030;color:#ffe9c0}'
      '.hint{color:var(--dim);font-size:12.5px;margin:0 0 14px}'
      'kbd{background:#232833;border:1px solid var(--line);border-bottom-width:2px;'
      'border-radius:4px;padding:0 5px;font:11px ui-monospace,monospace;color:#cdd3e0}'
      '</style><div class="wrap">')
    live = bool(list_lines and editable_at)
    clusters = {}                 # cluster id -> the paths that count as its
    #                               surviving copies, refs included
    if live:
        A('<div class="bar">'
          '<button id="ball">Mark all Tier B suggestions</button>'
          '<button id="bnone">Clear Tier B marks</button>'
          '<button id="breset">Reset to scan defaults</button>'
          '<span class="sp"></span>'
          '<span class="tally"><b id="nx">0</b> marked X</span>'
          '<button class="go" id="bdl">Download %s</button></div>'
          % esc(list_name or 'duplicates-list.txt'))
    A('<h1>Duplicate report</h1>')
    if live:
        A('<p class="hint">Click a thumbnail to mark or unmark it. '
          '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> move, <kbd>X</kbd> toggles. '
          'Nothing is deleted here: download the list, save it beside the images, '
          'and run the recycler. Any copy can be marked, including the suggested '
          'keeper &mdash; the last remaining copy in a cluster cannot, so a group '
          'is never emptied.</p>')
    if not tier_a and not tier_b:
        A('<p class="sub" style="color:var(--keep);font-weight:600">'
          'No duplicates found &mdash; every image in this folder is distinct. '
          'No selection list or deletion script was generated.</p>')
    A('<p class="sub">%s images in <code>%s</code>. %s</p>'
      % (stats['n'], esc(root), esc(stats['headline'])))
    A('<p class="sub">Cluster numbers match the selection list exactly &mdash; '
      'search the .txt for <code>cluster 91</code> to find the same group. '
      'Click a number to link straight to it.</p>')
    plan_by_key = {'A': [], 'B': []}
    if plan:
        for cl_id, key, keeper, editable, refs in plan:
            plan_by_key[key].append((cl_id, keeper, editable, refs))

    def shown_drops(key, tier):
        """How many files this tier actually offers HERE.

        Not the same as the number of relations it found: a Tier B drop that
        is already editable in a Tier A cluster is shown as a reference and
        cannot be marked here, so counting relations promised more to review
        than the page contains. It also has to agree with the review bar,
        which can only touch what it renders."""
        if not plan:
            return sum(len(d) for _, d, _ in tier)
        return sum(len([m for m in ed if m != kp])
                   for _c, kp, ed, _r in plan_by_key[key])

    A('<div class="stats">')
    for b, s in (('%d' % stats['n'], 'images'),
                 ('%d' % stats['exact'], 'exact (SHA) duplicates'),
                 ('%d' % len(tier_a), 'duplicate clusters'),
                 ('%d' % dn, 'droppable files'),
                 (mb(db), 'reclaimable'),
                 ('%d' % shown_drops('B', tier_b), 'crop / variant to review')):
        A('<div class="stat"><b>%s</b><span>%s</span></div>' % (b, s))
    A('</div>')

    for tier, key, colour, bg, title, lead in (
            (tier_a, 'A', '#3ddc97', '#12281e', 'DUPLICATE',
             'Same picture, re-encoded or resized. Keeper is highest resolution, then '
             'largest file, then finest JPEG quantization. Pre-set to <code>X</code>.'),
            (tier_b, 'B', '#f0b64b', '#2a2113', 'CROP / VARIANT',
             'Structurally the same picture, genuinely different pixels. Nothing here is '
             'deleted &mdash; pre-set to <code>.</code>. '
             '<b style="color:var(--rev)">REVIEW</b> matched the keeper directly; '
             '<b style="color:#b3a0d4">LINKED</b> (dashed) matched another member '
             'instead, so it is in this group by a chain and may have little to do '
             'with the keeper.')):
        groups = plan_by_key[key] if plan else [
            (None, k, [k] + drops, []) for k, drops, members in tier]
        if not groups:
            continue
        A('<h2><span class="pill" style="color:%s;background:%s">TIER %s &middot; %s</span> '
          '<span style="color:var(--dim);font-weight:400;font-size:15px">%d clusters &middot; '
          '%d candidates</span></h2>' % (colour, bg, key, title, len(groups),
                                         shown_drops(key, tier)))
        A('<p class="lead">%s</p>' % lead)
        for cl_id, keeper, editable, refs in groups:
            members = editable + refs
            dims = set(shown_dims(recs[i]) for i in members)
            note = 'mixed resolutions' if len(dims) > 1 else 'identical dimensions'
            label = ('cluster %d' % cl_id) if cl_id else '%d files' % len(members)
            A('<div class="cl" id="cluster-%s"><div class="ch">'
              '<b><a class="clid" href="#cluster-%s">%s</a> '
              '<span class="cnt">&middot; %d files</span></b>'
              '<span>%s</span></div><div class="strip">'
              % (cl_id, cl_id, esc(label), len(members), note))
            # The suggested keeper is a suggestion, not a fixture. It gets a
            # toggle like everything else; what the page refuses is emptying
            # a cluster, which is the rule that actually matters and the one
            # the recycler enforces too. Pinning the keeper instead meant
            # that preferring the other copy could not be expressed here at
            # all, and the whole point of the list is that the choice is
            # yours.
            if keeper is not None:
                krel = recs[keeper]['p']
                ktog = ''
                if live and krel in (editable_at or {}):
                    ktog = ('<button class="tg" data-p="%s">keep</button>'
                            % esc(krel))
                if cl_id is not None:
                    clusters.setdefault(cl_id, []).append(krel)
                A('<div class="it k"%s%s><img src="data:image/jpeg;base64,%s" alt="">'
                  '<div class="lb">KEEP</div><div class="fn">%s<br>%s</div>%s</div>'
                  % ((' data-p="%s"' % esc(krel)) if ktog else '',
                     (' data-cl="%s"' % cl_id) if ktog else '',
                     small_b64(recs[keeper]), meta(keeper), esc(krel[:44]), ktog))
            for d in [m for m in editable if m != keeper]:
                cls = 'd' if key == 'A' else 'n'
                lab = 'DROP' if key == 'A' else 'REVIEW'
                # A review cluster is a connected component, so a member can
                # be here because it matches the keeper, or because it
                # matches something that matches the keeper. Those are very
                # different claims and the report used to make them look
                # identical - which is how an eighteen-file cluster of
                # unrelated screenshots reads as a bug rather than as a
                # chain. Say which is which.
                if (key == 'B' and b_edges is not None and keeper is not None
                        and (keeper, d) not in b_edges):
                    cls, lab = 'x', 'LINKED'
                rel = recs[d]['p']
                on = ' on' if key == 'A' else ''
                tog = ''
                if live and rel in (editable_at or {}):
                    tog = ('<button class="tg" data-p="%s">%s</button>'
                           % (esc(rel), 'bin' if key == 'A' else 'keep'))
                if cl_id is not None:
                    clusters.setdefault(cl_id, []).append(rel)
                A('<div class="it %s%s"%s%s><img src="data:image/jpeg;base64,%s" alt="">'
                  '<div class="lb">%s</div><div class="fn">%s<br>%s</div>%s</div>'
                  % (cls, on,
                     (' data-p="%s"' % esc(rel)) if tog else '',
                     (' data-cl="%s"' % cl_id) if tog else '',
                     small_b64(recs[d]), lab, meta(d), esc(rel[:44]), tog))
            for r in refs:
                # Counted as a member here even though it is edited
                # elsewhere: the recycler treats an unmarked reference as a
                # surviving copy, so the page must too, or it would refuse
                # an edit the recycler would have allowed.
                if cl_id is not None and recs[r]['p'] in (editable_at or {}):
                    clusters.setdefault(cl_id, []).append(recs[r]['p'])
                A('<div class="it r"><img src="data:image/jpeg;base64,%s" alt="">'
                  '<div class="lb">IN CLUSTER %s</div><div class="fn">%s<br>%s</div></div>'
                  % (small_b64(recs[r]), home.get(r, '?') if home else '?',
                     meta(r), esc(recs[r]['p'][:44])))
            A('</div></div>')
    A('<footer><b>Method.</b> ' + esc(stats['method']) +
      '<br><br><b>Guarantees.</b> Every file appears at most once as a deletion candidate, '
      'and no file marked KEEP is a candidate anywhere. Both are asserted in code before '
      'these files are written; a violation aborts the run.</footer></div>')
    if live:
        # The page holds the list verbatim and only ever rewrites the first
        # character of a line it was told is editable. It cannot invent a
        # line, reorder one, or touch a comment, so the file it hands back
        # is the one this run wrote with some marks flipped - nothing else.
        def js(o):
            """JSON safe to sit inside a <script> element.

            A path may legally contain "</script>" on Linux - one directory
            called `a<` and a file called `script>` is all it takes - and a
            literal one closes the element early, killing every control on
            the page and spilling the rest of the code out as text. HTML
            does not parse escapes inside a script, so the fix has to happen
            in the JSON: \\u003c is the same string to JavaScript and no
            longer a tag to the parser. Structural JSON never contains these
            characters, so only string contents are touched."""
            return (json.dumps(o).replace('<', '\\u003c').replace('>', '\\u003e')
                    .replace('&', '\\u0026'))
        A('<script>')
        A('var LINES=%s,AT=%s,SUGG=%s,TIERB=%s,CLUSTERS=%s,NAME=%s,CRLF=%s;'
          % (js(list_lines), js(editable_at), js(suggested_b or []),
             js(tier_b_all or []), js(clusters),
             js(list_name or 'duplicates-list.txt'),
             'true' if os.name == 'nt' else 'false'))
        A(r'''
var ORIG=LINES.slice(), tiles=[].slice.call(document.querySelectorAll('.it[data-p]')), cur=0;
function mark(p,on){var i=AT[p]; if(i===undefined)return;
  LINES[i]=(on?'X':'.')+LINES[i].slice(1);}
function isOn(p){var i=AT[p]; return i!==undefined && LINES[i].charAt(0)==='X';}
function paint(){var n=0;
  for(var k in AT){if(LINES[AT[k]].charAt(0)==='X')n++;}
  document.getElementById('nx').textContent=n;
  tiles.forEach(function(t,ix){var p=t.getAttribute('data-p'),on=isOn(p);
    t.classList.toggle('on',on); t.classList.toggle('sel',ix===cur);
    var b=t.querySelector('.tg'); if(b)b.textContent=on?'bin':'keep';});}
var CLOF={};                     // path -> the cluster it is counted in
for(var c in CLUSTERS){CLUSTERS[c].forEach(function(p){if(!(p in CLOF))CLOF[p]=c;});}
function survivors(cl){var m=CLUSTERS[cl]||[],n=0;
  for(var i=0;i<m.length;i++){if(!isOn(m[i]))n++;} return n;}
// The only rule: a cluster keeps at least one copy. WHICH one is yours to
// pick, the suggested keeper included. Refusing to mark the keeper was
// simpler and wrong - preferring the other copy could not be said at all.
function canBin(p){var cl=CLOF[p];
  return cl===undefined || survivors(cl)>1;}
function say(t,msg){var b=t&&t.querySelector('.tg'); if(!b)return;
  var old=b.textContent; b.textContent=msg; b.classList.add('no');
  setTimeout(function(){b.textContent=old;b.classList.remove('no');},1400);}
function toggle(p,t){
  if(!isOn(p)&&!canBin(p)){say(t,'last copy'); return;}
  mark(p,!isOn(p)); paint();}
tiles.forEach(function(t,ix){t.addEventListener('click',function(e){
  cur=ix; toggle(t.getAttribute('data-p'),t); e.preventDefault();});});
document.getElementById('ball').onclick=function(){
  SUGG.forEach(function(p){if(canBin(p))mark(p,true);});paint();};
document.getElementById('bnone').onclick=function(){TIERB.forEach(function(p){mark(p,false);});paint();};
document.getElementById('breset').onclick=function(){LINES=ORIG.slice();paint();};
document.getElementById('bdl').onclick=function(){
  var txt=LINES.join(CRLF?'\r\n':'\n'), blob=new Blob([CRLF?'﻿'+txt:txt],
      {type:'text/plain;charset=utf-8'}), u=URL.createObjectURL(blob),
      a=document.createElement('a'); a.href=u; a.download=NAME; a.click();
  setTimeout(function(){URL.revokeObjectURL(u);},2000);};
document.addEventListener('keydown',function(e){
  if(e.target.tagName==='INPUT')return;
  if(e.key==='ArrowRight'||e.key==='j'){cur=Math.min(cur+1,tiles.length-1);}
  else if(e.key==='ArrowLeft'||e.key==='k'){cur=Math.max(cur-1,0);}
  else if(e.key==='x'||e.key==='X'){if(tiles[cur])toggle(tiles[cur].getAttribute('data-p'),tiles[cur]);return;}
  else return;
  paint(); if(tiles[cur])tiles[cur].scrollIntoView({block:'nearest',inline:'center'});
  e.preventDefault();});
paint();
''')
        A('</script>')
    A('</html>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(P))


def write_list_and_script(list_path, py_path, bat_path, sh_path, recs,
                          tier_a, tier_b, root, info_b=None, b_edges=None):
    """Writes the list and the three recyclers.

    Returns (n_manifest_rows, lines, editable_at, suggested_b), the last
    three so the report can offer the same edits in a browser:
      lines         the list exactly as written, so the page rebuilds the
                    file by flipping one character rather than re-deriving
                    a format that would then be free to drift from this one
      editable_at   relative path -> index into `lines` of its mark line
      suggested_b   Tier B paths this scan would have marked, had Tier B
                    been the sort of match that may be acted on unreviewed
    """
    def info(i, keeper, linked=False):
        r = recs[i]
        t = '%dx%d, %.2f MB' % (shown_dims(r) + (r['b'] / 1048576.0,))
        if 'qsum' in r:
            t += ', q%d' % r['qsum']
        if keeper:
            t += ' - suggested keeper'
        elif linked:
            # It is in this cluster through another member, not through the
            # keeper. Worth saying in the file too - plenty of editing gets
            # done here rather than in the report.
            t += ' - linked via another file, not the keeper'
        return t

    plan, home = build_emission_plan(tier_a, tier_b, recs, info_b)
    check_emission(plan, home)
    # Real Tier B clusters carry suggested drops; info_b groups carry none,
    # and the plan cannot tell them apart because both use key 'B'. So take
    # the suggestion from where it was actually decided.
    b_sugg = set()
    for _k, _drops, _m in tier_b:
        b_sugg.update(_drops)
    editable_at, suggested_b, tier_b_all = {}, [], []

    L = ['# ' + '=' * 74,
         '#  DUPLICATE SELECTION LIST',
         '#  Root: ' + root,
         '#',
         '#  EVERY member of every cluster is an editable line. The first',
         '#  character decides what happens:',
         '#      X  = send this file to the OS trash (Recycle Bin / Trash)',
         '#      .  = keep this file',
         '#  Want to keep a different copy than suggested? Just MOVE the X.',
         '#  The Recycle script deletes a file only while at least one other',
         '#  member of the same cluster stays unmarked AND still verifies',
         '#  against its scan-time hash - so a cluster can never be wiped out,',
         '#  by edit or by accident.',
         '#',
         '#  A file can belong to two clusters (an exact duplicate that is also',
         '#  the uncropped original of something). It gets its editable line in',
         '#  ONE of them; the other shows it as a "also in cluster N" comment,',
         '#  so two lines can never disagree about the same file.',
         '#',
         '#  TIER A: suggested keeper pre-set to ".", the rest to X.',
         '#  TIER B: everything pre-set to "." - review in the HTML report first.',
         '# ' + '=' * 74]
    rows = []
    last_key = None
    for cl_id, key, keeper, editable, refs in plan:
        if key != last_key:
            head = ('TIER A  -  DUPLICATE' if key == 'A'
                    else 'TIER B  -  CROP / VARIANT  (review first)')
            L += ['', '# ' + '=' * 74, '#  ' + head, '# ' + '=' * 74]
            last_key = key
        members = editable + refs
        dims = set(shown_dims(recs[i]) for i in members)
        L += ['', '#  cluster %d  (%d files%s)'
              % (cl_id, len(members),
                 ', MIXED RESOLUTIONS' if len(dims) > 1 else '')]
        if not editable:
            L.append('#  (information only - every member of this cluster is'
                     ' editable in another cluster)')
        for i in refs:
            L.append('#  also in cluster %d (edit it there): %s   [%s]'
                     % (home[i], list_safe(recs[i]['p']), info(i, False)))
            if editable:
                # reference rows exist so the recycler can count them as
                # witnesses; a reference-only cluster has nothing to delete,
                # so it gets no manifest rows at all
                rows.append({'rel': recs[i]['p'], 'size': recs[i]['b'],
                             'sha': recs[i]['sha'], 'cl': cl_id, 'home': home[i],
                             't': key})
        for i in ([keeper] + [m for m in editable if m != keeper] if editable else []):
            is_keeper = (i == keeper)
            linked = (key == 'B' and not is_keeper and b_edges is not None
                      and keeper is not None and (keeper, i) not in b_edges)
            if list_safe(recs[i]['p']) != recs[i]['p']:
                # A control character in a filename (a newline above all)
                # would let the file's own name FORGE mark lines: writing it
                # raw splits it across lines, and anything after the newline
                # is parsed as a fresh mark that can override another file's
                # X or dot. Such files get no editable line - shown, kept in
                # the manifest as an unmarked member (so they still count as
                # witnesses and can never be deleted via the list), and said
                # out loud rather than silently skipped.
                L.append('#  [not editable - name contains control characters;'
                         ' always kept]')
                L.append('#    %s   [%s]' % (list_safe(recs[i]['p']),
                                             info(i, is_keeper, linked)))
                mark = None
            else:
                mark = '.' if (is_keeper or key == 'B') else 'X'
                L.append('%s  %s   [%s]'
                         % (mark, recs[i]['p'], info(i, is_keeper, linked)))
                editable_at[recs[i]['p']] = len(L) - 1
            # A file with control characters in its name gets NO editable
            # line, and the list says of it "always kept". Flagging it as a
            # suggestion anyway would let the recycler's bulk Tier B answer
            # mark it X and delete it - the one thing that branch promises
            # cannot happen. So the suggestion is conditional on the file
            # having a line to be suggested in.
            sd = 1 if (key == 'B' and not is_keeper and i in b_sugg
                       and recs[i]['p'] in editable_at) else 0
            if sd:
                suggested_b.append(recs[i]['p'])
            if key == 'B' and not is_keeper and recs[i]['p'] in editable_at:
                tier_b_all.append(recs[i]['p'])
            row = {'rel': recs[i]['p'], 'size': recs[i]['b'],
                   'sha': recs[i]['sha'], 'cl': cl_id, 'home': cl_id, 't': key}
            if sd:
                row['sd'] = 1
            if mark is not None:
                # the mark THIS run wrote, so the recycler can tell an
                # untouched list from a reviewed one
                row['m0'] = mark
            rows.append(row)
    L.append('')
    write_text_file(list_path, L)

    # One recycler, every OS. The safety rules live here once; the .bat and
    # .sh beside it only find a Python and run it. Two hand-written
    # implementations of these rules would drift, and this project has
    # already been bitten by exactly that (see "launcher fleet drift", v3).
    trash_src = _vendored_trash_source()
    py = (PY_RECYCLER_TEMPLATE
          .replace('__TRASH_MODULE__', trash_src)
          .replace('__ROOT_JSON__', json.dumps(root))
          .replace('__LIST_JSON__', json.dumps(os.path.basename(list_path)))
          .replace('__MANIFEST_JSON__', json.dumps(rows, ensure_ascii=False,
                                                   indent=1)))
    with open(py_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(py)
    try:
        os.chmod(py_path, 0o755)
    except OSError:
        pass

    stem = os.path.basename(py_path)
    bat = BAT_TEMPLATE.replace('__PY__', stem) \
                      .replace('__LIST__', os.path.basename(list_path))
    with open(bat_path, 'wb') as f:
        f.write(bat.replace('\n', '\r\n').replace('\r\r\n', '\r\n').encode('ascii'))
    sh = SH_TEMPLATE.replace('__PY__', stem) \
                    .replace('__LIST__', os.path.basename(list_path))
    with open(sh_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
    return len(rows), L, editable_at, suggested_b, tier_b_all


def write_text_file(path, lines):
    """The selection list is edited by hand, so it gets the conventions of
    the platform it was written on: CRLF plus a BOM on Windows (Notepad
    still needs both), plain UTF-8 with LF everywhere else. The recycler
    reads it with utf-8-sig and universal newlines, so either form works
    anywhere - this is only about not making the file look broken in the
    user's editor."""
    if os.name == 'nt':
        with open(path, 'wb') as f:
            f.write(b'\xef\xbb\xbf' + ('\r\n'.join(lines)).encode('utf-8'))
    else:
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines))


def _vendored_trash_source():
    """Inline _trash.py into the generated recycler so the recycler keeps
    working if it is moved away from this toolkit - it sits next to the
    user's images and may well outlive the folder it came from."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_trash.py')
    try:
        with open(here, encoding='utf-8') as f:
            src = f.read()
    except OSError:
        raise SystemExit('_trash.py is missing from %s - keep the toolkit '
                         'folder together.' % os.path.dirname(here))
    body = src[src.index('__all__'):]
    # Drop the module's own __main__ block: inlined into the recycler its
    # guard is true, and it would run its self-test on every launch.
    tail = body.find("\nif __name__ ==")
    if tail >= 0:
        body = body[:tail]
    return ('import errno\nimport os\nimport sys\n\n' + body.rstrip() + '\n')


PY_RECYCLER_TEMPLATE = r'''#!/usr/bin/env python3
# Recycle-Duplicates.py   (generated by analyze-inventory.py)
#
# Reads the selection list next to this script. EVERY cluster member is an
# editable line there: X = delete, . = keep. Moving the X to a different
# member is fully supported - the suggested keeper is only a suggestion.
#
# Safety, enforced per cluster at run time:
#   - a file is deleted only if at least one OTHER member of its cluster is
#     left unmarked, still exists, and still matches its scan-time SHA-256
#     (the "surviving witness");
#   - a cluster with every member marked X is refused outright;
#   - a file marked X is deletable only in the cluster that OWNS its line,
#     so one X can never become two deletions and a refusal cannot be
#     bypassed through a reference row;
#   - every file to be deleted must itself still match its scan-time size
#     and SHA-256.
# Anything failing a check is skipped, loudly. Files go to the OS trash
# (Recycle Bin / freedesktop Trash / ~/.Trash), never a permanent delete.
#
# Runs on Windows, Linux and macOS with nothing but a Python 3 install.
import hashlib
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass

ROOT = __ROOT_JSON__
LIST_NAME = __LIST_JSON__
MANIFEST = __MANIFEST_JSON__

HERE = os.path.dirname(os.path.abspath(__file__))
LIST_FILE = os.path.join(HERE, LIST_NAME)

__TRASH_MODULE__


def sha256_of(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def entry_ok(e):
    """Still the file the scan saw: exists, same size, same hash."""
    full = os.path.join(ROOT, *e['rel'].split('/'))
    try:
        if not os.path.isfile(full):
            return False
        if os.path.getsize(full) != e['size']:
            return False
    except OSError:
        return False
    return sha256_of(full) == e['sha']


def read_marks():
    """Returns (marks, unknown). Last mark on a path wins, matching the
    order a human reads the file in."""
    marks, unknown = {}, []
    by_rel = set(e['rel'] for e in MANIFEST)
    with open(LIST_FILE, encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not re.match(r'^[Xx.]\s', line):
                continue
            mark = line[0].upper()
            # strip only the TRAILING "   [WxH, size]" block, anchored and
            # with no ']' inside, so a name like "photo  [final].jpg" is safe
            rest = re.sub(r'\s{2,}\[[^\]]*\]\s*$', '', line[1:].strip())
            if rest in by_rel:
                marks[rest] = mark
            elif mark == 'X':
                unknown.append(rest)
    return marks, unknown


def tier_b_reviewed(marks):
    """True when the Tier B part of the list is not as this scan wrote it.

    Every editable row carries `m0`, the mark this run put there. If any
    Tier B row now reads differently, a human has been through it, and a
    file left on '.' means "I looked, and I am keeping this" - not "not yet
    read". Those two are opposite instructions and the old code could not
    tell them apart, so it kept offering to bin files that had just been
    deliberately spared."""
    for e in MANIFEST:
        if e.get('t') != 'B' or 'm0' not in e:
            continue
        if marks.get(e['rel'], e['m0']) != e['m0']:
            return True
    return False


def offer_tier_b(marks):
    """Tier B is written unmarked, so acting on it used to mean editing
    hundreds of lines by hand. Offer the bulk answer for that case only.

    It never overrides a decision. If the list has been edited at all in
    Tier B the list wins outright and nothing is asked. Otherwise it may
    only ADD marks, only to rows the scan itself nominated, and only after
    an explicit answer. Everything still passes the same survivor and hash
    checks below; this decides what to propose, not what is safe."""
    if tier_b_reviewed(marks):
        n = sum(1 for e in MANIFEST
                if e.get('t') == 'B' and marks.get(e['rel']) == 'X')
        print('')
        print('  Tier B: using your edits (%d marked X, rest untouched).' % n)
        return marks
    pend = [e for e in MANIFEST
            if e.get('sd') and marks.get(e['rel']) != 'X']
    if not pend:
        return marks
    try:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        interactive = False
    if not interactive:
        print('')
        print('  Tier B: untouched, %d flagged file%s kept. Using the list '
              'as it is.' % (len(pend), '' if len(pend) == 1 else 's'))
        print('          Mark them in the report, or run from a terminal to '
              'choose here.')
        return marks
    size = sum(e['size'] for e in pend) / 1048576.0
    noun = '%d Tier B file%s' % (len(pend), '' if len(pend) == 1 else 's')
    print('')
    print('  Tier B: %d file%s, %.1f MB, none marked.'
          % (len(pend), '' if len(pend) == 1 else 's', size))
    print('  Crop or variant matches - same subject, different pixels. The '
          'scan picked')
    print('  a copy to drop in each but left them unmarked; this tier is '
          'where it is')
    print('  least reliable.')
    print('')
    print('    [Enter]  use the list as it is        (default)')
    print('    b        also trash the %s' % noun)
    print('    q        exit without deleting')
    print('')
    print('  Marking per file in the report is more precise, and stops this '
          'prompt.')
    print('')
    try:
        a = input('  Choose: ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        print('')
        return marks
    if a == 'q':
        print('')
        print('  Nothing deleted. Mark them in the report, save the list over '
              'this one,')
        print('  and run again.')
        raise SystemExit(0)
    if a == 'b':
        for e in pend:
            marks[e['rel']] = 'X'
        print('  -> %s added. Same checks apply: every cluster keeps at least '
              'one copy,' % noun)
        print('     and files whose bytes changed since the scan are skipped.')
    return marks


def main():
    if not os.path.isfile(LIST_FILE):
        print('')
        print('  [FAIL] %s not found next to this script.' % LIST_NAME)
        print('')
        return 1
    if not os.path.isdir(ROOT):
        print('')
        print('  [FAIL] image folder not found: %s' % ROOT)
        print('')
        return 1

    marks, unknown = read_marks()
    marks = offer_tier_b(marks)
    nx = sum(1 for v in marks.values() if v == 'X')
    nk = sum(1 for v in marks.values() if v == '.')
    print('')
    print('  Duplicate cleanup -- preview')
    print('  ----------------------------')
    print('  %d marked X, %d marked . (kept)' % (nx, nk))
    print('  Deletions go to: %s' % trash_backend_name())
    print('')
    if unknown:
        print('  [WARN] %d line(s) marked X match no known entry - ignored:'
              % len(unknown))
        for u in unknown:
            print('           %s' % u)
        print('')
    if nx == 0:
        print('  Nothing marked for deletion. Done.')
        print('')
        return 0

    print('  Verifying hashes. This reads each affected cluster once - please wait.')
    clusters = {}
    for e in MANIFEST:
        clusters.setdefault(e['cl'], []).append(e)

    plan, skipped, total_bytes = [], 0, 0
    for cl in sorted(clusters):
        members = clusters[cl]
        # Deletable here only if this cluster OWNS the row. A file shared
        # with another cluster is deletable in its home cluster alone.
        xs = [e for e in members
              if marks.get(e['rel']) == 'X' and e['home'] == cl]
        keeps = [e for e in members if marks.get(e['rel']) != 'X']
        xs_all = [e for e in members if marks.get(e['rel']) == 'X']
        if not xs:
            continue

        if not keeps:
            print('  [SKIP] cluster %d: no copy would survive - refusing.' % cl)
            for e in xs_all:
                if e['home'] != cl:
                    print('         %s' % os.path.basename(e['rel']))
                    print('           is marked X in cluster %d, so it cannot be '
                          'the survivor here' % e['home'])
                else:
                    print('         %s  (marked X here)' % os.path.basename(e['rel']))
            print('         Unmark one of them - in the cluster where it is editable.')
            skipped += len(xs)
            continue

        witness = None
        for k in keeps:
            if entry_ok(k):
                witness = k
                break
        if witness is None:
            print('  [SKIP] cluster %d: no unmarked copy still verifies against the'
                  % cl)
            print('         scan - the surviving copy would be unproven. Re-scan first.')
            skipped += len(xs)
            continue

        for e in xs:
            full = os.path.join(ROOT, *e['rel'].split('/'))
            name = os.path.basename(e['rel'])
            if not os.path.lexists(full):
                skipped += 1
                continue
            if not entry_ok(e):
                print('  [SKIP] %s' % name)
                print('         contents changed since the scan - verify by hand')
                skipped += 1
                continue
            ok, why = precheck(full)
            if not ok:
                print('  [REFUSED] %s' % name)
                print('            %s' % why)
                print('            Left untouched on purpose. Move it somewhere')
                print('            shorter and re-scan if you still want it gone.')
                skipped += 1
                continue
            plan.append({'path': full, 'name': name, 'size': e['size']})
            total_bytes += e['size']

    print('')
    print('  %d verified, %d skipped, %.1f MB to reclaim.'
          % (len(plan), skipped, total_bytes / 1048576.0))
    print('')
    if not plan:
        print('  Nothing survived verification. Done.')
        print('')
        return 0
    try:
        answer = input('  Send these to the trash? (y/N) ')
    except EOFError:
        answer = ''
    if not answer.strip().lower().startswith('y'):
        print('')
        print('  Cancelled. Nothing was changed.')
        print('')
        return 0

    print('')
    done = fail = 0
    freed = 0
    for p in plan:
        ok, detail = send_to_trash(p['path'])
        if ok:
            done += 1
            freed += p['size']
        else:
            print('  [FAIL] %s' % p['name'])
            print('         %s' % detail)
            fail += 1
    print('')
    if fail == 0:
        print('  Done. %d file(s) in the trash, %.1f MB freed.'
              % (done, freed / 1048576.0))
    else:
        print('  Finished with %d failure(s). %d moved to the trash.' % (fail, done))
    print('  Restore anything from the trash if you change your mind.')
    print('')
    return fail


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('')
        print('  Aborted. Nothing further was changed.')
        sys.exit(130)
'''


BAT_TEMPLATE = r'''@echo off
setlocal EnableExtensions
cd /d "%~dp0"
rem ---------------------------------------------------------------
rem  Recycle-Duplicates.bat   (generated by analyze-inventory.py)
rem  Edit __LIST__ first: X = delete, . = keep.
rem  Keep this .bat, __PY__ and __LIST__ together.
rem  This only finds a Python; every rule lives in the .py.
rem ---------------------------------------------------------------
set "_PY=%~dp0__PY__"
set "_LIST=%~dp0__LIST__"

if not exist "%_PY%" (
    echo [FAIL] __PY__ not found next to this file.
    echo.
    pause
    exit /b 1
)
if not exist "%_LIST%" (
    echo [FAIL] __LIST__ not found next to this file.
    echo.
    pause
    exit /b 1
)

set "PYCMD="
if defined IMGDEDUP_PYTHON set "PYCMD="%IMGDEDUP_PYTHON%""
if not defined PYCMD (
    where py >nul 2>&1
    if not errorlevel 1 set "PYCMD=py -3"
)
if not defined PYCMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD (
    echo [FAIL] No Python found. Install Python 3 from python.org, or set
    echo        IMGDEDUP_PYTHON to a python.exe.
    echo.
    pause
    exit /b 1
)

%PYCMD% "%_PY%"
set "_RC=%ERRORLEVEL%"
echo.
pause
exit /b %_RC%
'''


SH_TEMPLATE = '''#!/bin/sh
# ---------------------------------------------------------------
#  Recycle-Duplicates.sh   (generated by analyze-inventory.py)
#  Edit __LIST__ first: X = delete, . = keep.
#  Keep this .sh, __PY__ and __LIST__ together.
#  This only finds a Python; every rule lives in the .py.
# ---------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1

if [ ! -f "__PY__" ]; then
    printf '[FAIL] %s not found next to this file.\\n' "__PY__" >&2
    exit 1
fi
if [ ! -f "__LIST__" ]; then
    printf '[FAIL] %s not found next to this file.\\n' "__LIST__" >&2
    exit 1
fi

PYCMD=""
if [ -n "${IMGDEDUP_PYTHON:-}" ] && [ -x "${IMGDEDUP_PYTHON}" ]; then
    PYCMD="${IMGDEDUP_PYTHON}"
else
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PYCMD="$c"; break; fi
    done
fi
if [ -z "$PYCMD" ]; then
    printf '[FAIL] No python3 found. Install Python 3, or set IMGDEDUP_PYTHON.\\n' >&2
    exit 1
fi

"$PYCMD" "__PY__"
'''


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description='Find duplicates in an image inventory.')
    ap.add_argument('inventory', nargs='?', help='inventory .jsonl, or a folder containing one')
    ap.add_argument('--tier-a-mad', type=float, default=4.0)
    ap.add_argument('--tier-a-cos', type=float, default=0.99)
    ap.add_argument('--tier-b-mad', type=float, default=4.0)
    # 0.90, lowered from 0.94 so this agrees with the floor used to nominate
    # neighbours in the first place. At 0.94 every pair between 0.90 and 0.94
    # was decoded, template-matched and then discarded on a criterion it was
    # never going to satisfy - wasted work and lost recall in one.
    #
    # It was the binding constraint on crops, not the pixel test. Measured on
    # 180 real images against their own crops, the share clearing 0.94:
    #
    #     crop keeps 95%   92%        crop keeps 70%   21%
    #     crop keeps 90%   76%        crop keeps 60%    5%
    #     crop keeps 80%   44%
    #
    # so 56% of 80% crops were thrown away AFTER the template match had
    # already accepted them. The NCC gate is what discriminates here: no
    # unrelated pair out of 296 reached even 0.85 there, topping out at
    # 0.816. On the 36,410-image library this takes Tier B from 323 to 592
    # candidates and leaves Tier A untouched at 3 clusters / 4 droppable.
    # Tier B is never deleted without review, so the cost is lines to look
    # at, not files.
    ap.add_argument('--tier-b-cos', type=float, default=0.90,
                    help='CLIP floor before a pair may enter the review tier '
                         '(default 0.90, matching the neighbour floor). Raise '
                         'it to 0.94 for a shorter review list that misses '
                         'about half of all moderate crops')
    ap.add_argument('--sig-cut', type=float, default=None,
                    help='signature prefilter ceiling (default: 8.0, raised to '
                         '2x --tier-a-mad when that is higher; an explicit '
                         'value is always used verbatim)')
    ap.add_argument('--no-orient', action='store_true',
                    help='skip rotation/mirror matching (faster sweep; '
                         'rotated and mirrored copies will be missed)')
    # 16, lowered from 48 in two steps after measuring what the cap cuts. On a
    # 36,410-image library the cosine of the LAST neighbour kept sits at
    # 0.9269 median / 0.9627 max with K=48, and moves all of 0.003 by
    # K=16 - the neighbours are packed in a narrow band, so a bigger K buys
    # near-identical matches rather than better ones. Tier A needs >= 0.99
    # and the cut line never approaches it at any K from 16 to 64, so this
    # cannot cost a deletion candidate. (Structurally too: an image with 32
    # neighbours above 0.99 is holding near-identical copies, and the exact
    # ones are already caught by SHA before CLIP is consulted.)
    #
    # 48 -> 32 on that library: candidate pairs 376,660 -> 304,953, crop
    # matches 76,770 -> 59,295, analyze 184 s -> 164 s, Tier A unchanged at
    # 3 clusters / 4 droppable, Tier B 328 -> 323 candidates.
    #
    # The cap earns its place regardless: median neighbour count above the
    # floor is 0, but p99 is 2,466 and the maximum 2,672. Those dense
    # clusters are overwhelmingly SCREENSHOTS - 21 of the 25 densest images
    # came from one Screenshots folder - which is CLIP working correctly and
    # being unhelpful, since every capture of the same app is semantically
    # near-identical whatever it shows. The pixel tiers do the real work
    # there, and without this cap those images alone would contribute
    # 2.7 million pairs.
    ap.add_argument('--clip-neighbors', type=int, default=16,
                    help='how many nearest neighbours each image contributes '
                         'from the embeddings (default 48). Bounds the '
                         'candidate set on large libraries, where a flat '
                         'cosine floor admits a fifth of all pairs. Cannot '
                         'cost a Tier A candidate; may trim review '
                         'candidates inside dense clusters')
    ap.add_argument('--no-embeddings', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    global _HAVE_CV2
    try:
        import cv2  # noqa: F401
        _HAVE_CV2 = True
    except Exception:       # see compute_nccs: importable-but-broken counts
        _HAVE_CV2 = False

    if args.self_test:
        return self_test()
    if not args.inventory:
        ap.error('give an inventory file or folder (or use --self-test)')

    inv = find_inventory(args.inventory)
    if not inv:
        print('No image-inventory*.jsonl found at: ' + args.inventory)
        return 2
    root, recs, errs, parts = load_inventory(inv)
    n = len(recs)
    print('Inventory: %s%s' % (os.path.basename(inv),
                               '  (+%d parts)' % (len(parts) - 1) if len(parts) > 1 else ''))
    print('Images: %d   unreadable: %d   root: %s' % (n, len(errs), root))
    if n < 2:
        print('Nothing to compare.')
        return 0

    vec, _ = (None, None) if args.no_embeddings else load_embeddings(inv, recs)
    workers = default_workers()

    phase('Reading thumbnail signatures ...')
    C = decode_signatures(recs, workers)
    TH = ThumbStore(recs, workers=workers)
    if not TH.preloaded:
        print('  (thumbnails decoded on demand - library too large to hold '
              'in memory)')

    bysha = {}
    for i, r in enumerate(recs):
        bysha.setdefault(r['sha'], []).append(i)
    exact = [v for v in bysha.values() if len(v) > 1]
    print('Exact (SHA) duplicate groups: %d' % len(exact))

    # The signature prefilter must stay ahead of the pixel ceiling it feeds:
    # the 8x8 signature underestimates thumbnail differences by roughly 2x,
    # so a raised --tier-a-mad silently starved of candidates at the default
    # cut. The default run is unchanged (max(8, 2 * 4.0) == 8); an explicit
    # --sig-cut is always honored verbatim.
    if args.sig_cut is not None:
        sig_cut = args.sig_cut
    else:
        sig_cut = max(8.0, 2.0 * args.tier_a_mad)
        if sig_cut > 8.0:
            print('(--sig-cut raised to %.1f to keep up with --tier-a-mad %.1f)'
                  % (sig_cut, args.tier_a_mad))
    phase('Sweeping all %d pairs ...' % (n * (n - 1) // 2))
    cand = set(sweep_candidates(C, sig_cut))
    phase('Signature sweep kept %d pair(s); scanning embeddings ...' % len(cand))
    orient_cand = set()
    # Rotated/mirrored copies have completely different signatures, so on
    # pixels alone they never reach the scorer - DISCOVERY needs a sweep of
    # the re-oriented signatures. When embeddings cover an image, CLIP has
    # already nominated its rotated pairs (it is largely orientation-
    # insensitive: a mirrored save scores ~0.98); the seven extra passes are
    # only needed for images the embeddings CANNOT vouch for. With no
    # embeddings that is everyone; with partial coverage it is exactly the
    # uncovered images - a subtlety that once silently skipped them.
    uncovered = ([] if vec is None else
                 [i for i, r in enumerate(recs) if r['sha'] not in vec])
    if not args.no_orient:
        if vec is None:
            orient_rows = None          # sweep every row
        elif uncovered:
            orient_rows = uncovered     # only rows CLIP cannot see
            print('Orientation: %d image(s) lack embeddings; sweeping their '
                  'rotations on pixels' % len(uncovered))
        else:
            orient_rows = []
            print('Orientation: relying on the embeddings to nominate rotated '
                  'and mirrored copies')
            print('             (they are semantically identical, so CLIP '
                  'already sees them).')
        if orient_rows is None or orient_rows:
            # Re-oriented signatures are pure grid permutations (see
            # oriented_signatures) - approximate for non-square thumbs, so
            # the full cut is used and every nominated pair is re-scored
            # exactly downstream.
            for k in range(1, 8):
                for (a, b) in sweep_candidates_cross(
                        C, oriented_signatures(C, k), sig_cut,
                        rows=orient_rows):
                    orient_cand.add((a, b) if a < b else (b, a))
            orient_cand -= cand
            cand |= orient_cand
            if orient_cand:
                print('Orientation sweep: %d extra candidate pair(s) to check '
                      'for rotated / mirrored saves' % len(orient_cand))
    used_clip = False
    has_vec = None
    if vec is not None:
        dim0 = next(iter(vec.values())).shape[0]
        zero = np.zeros(dim0, dtype=np.float32)
        V = np.stack([vec.get(r['sha'], zero) for r in recs])
        has_vec = np.array([r['sha'] in vec for r in recs], dtype=bool)
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.where(nrm == 0, 1, nrm)
        # The gram product is computed per row block - materializing the
        # full n x n float32 matrix is ~10 GB at 50k images, and this scan
        # only ever looks at one block of rows at a time anyway.
        #
        # Each image contributes only its K NEAREST neighbours, not every
        # partner above a fixed cosine. A flat threshold does not survive a
        # real library: CLIP's cosine has a high baseline that shifts with
        # the collection, and on a single-genre set a fifth of ALL pairs can
        # sit above 0.90 - at 36k images that is 130 million pairs, which is
        # tens of gigabytes of tuples before any of them is even scored.
        # A duplicate is always among its original's nearest neighbours, so
        # top-K finds the same pairs while the candidate count stays linear
        # in the library size and adapts to whatever the baseline happens
        # to be.
        kk = max(1, min(args.clip_neighbors, n - 1))
        block = max(64, min(2048, int(8_000_000 // max(1, n))))
        capped = 0
        # gather K+1 columns where possible: a row is CAPPED only if its
        # (K+1)-th nearest neighbour also clears the floor - a row with
        # exactly K qualifying neighbours lost nothing and must not be
        # counted, or the note overstates and sends users chasing a cap
        # that never bound.
        kk1 = min(kk + 1, n - 1)
        for a in range(0, n, block):
            b = min(n, a + block)
            Sb = V[a:b] @ V.T
            rows = np.arange(a, b)
            Sb[np.arange(b - a), rows] = -1.0        # never its own neighbour
            idx1 = np.argpartition(Sb, -kk1, axis=1)[:, -kk1:]
            vals1 = np.take_along_axis(Sb, idx1, axis=1)
            if kk1 > kk:
                drop = np.argmin(vals1, axis=1)      # the (K+1)-th nearest
                capped += int((vals1[np.arange(b - a), drop] >= 0.90).sum())
                keepm = np.ones_like(vals1, dtype=bool)
                keepm[np.arange(b - a), drop] = False
                idx = idx1[keepm].reshape(b - a, kk)
                vals = vals1[keepm].reshape(b - a, kk)
            else:
                idx, vals = idx1, vals1
            mask = vals >= 0.90
            ri, ci = np.nonzero(mask)
            if not len(ri):
                continue
            gi = rows[ri]
            gj = idx[ri, ci]
            lo = np.minimum(gi, gj)
            hi = np.maximum(gi, gj)
            cand.update(zip(lo.tolist(), hi.tolist()))
        used_clip = True
        if capped:
            # Never let a cap be silent - say how many images were limited
            # and how to lift it.
            print('  note: %d image(s) had more than %d neighbours above the '
                  'CLIP floor;' % (capped, kk))
            print('        only the closest %d were kept for each. Raise with '
                  '--clip-neighbors N' % kk)
    else:
        V = None
    cand = sorted(cand)
    print('Candidate pairs: %d%s' % (len(cand), '' if used_clip else '  (no embeddings)'))
    if not used_clip:
        # Saying "no embeddings" understates it. Crops move the 8x8
        # signature far more than a re-encode does, so without CLIP to
        # nominate them they never survive the sweep and never reach the
        # crop matcher. Measured on 200 real images: a copy trimmed to 90%
        # is missed 74% of the time, to 80% is missed 97% of the time. A
        # Tier B section still appears, which is exactly what makes the
        # silence misleading, so it is said out loud instead.
        print('   Cropped copies are mostly NOT found in this mode: a copy')
        print('   trimmed to 90% is missed ~74% of the time, to 80% ~97%.')
        print('   Everything else - exact, re-encoded, resized, rotated,')
        print('   mirrored - is unaffected. Run the embed stage to catch crops.')
    if not _HAVE_CV2:
        print('  NOTE: OpenCV is not installed, so crop detection is limited to what')
        print('        CLIP alone can see. For the full crop tier:')
        print('          ' + _hint('opencv-python-headless'))

    # score: exact pixel difference for every candidate pair (threaded), then
    # crop-containment only for the pairs whose tier B decision needs it.
    phase('Scoring %d candidate pairs on pixels ...' % len(cand))
    mads = compute_mads(TH, cand, workers)

    def cos_of(i, j):
        # "no embedding for this image" must read as UNKNOWN (None), not 0.0 -
        # a 0.0 silently fails every cosine floor and blocked even exact
        # duplicates from clustering when one sha was missing from the file.
        if V is None or not (has_vec[i] and has_vec[j]):
            return None
        # a direct dot of the two normalized rows - the full n x n cosine
        # matrix is no longer materialized (see the blocked scan above)
        return float(V[i] @ V[j])

    need_ncc = []
    for t, (i, j) in enumerate(cand):
        m = mads[t]
        c = cos_of(i, j)
        if (m <= args.tier_a_mad) and (c is None or c >= args.tier_a_cos):
            continue
        if m > args.tier_b_mad and not (c is not None and c >= 0.995) \
                and (c is None or c >= args.tier_b_cos):
            need_ncc.append((i, j))
    phase('Crop matching on %d pair(s) ...' % len(need_ncc))
    nccs = compute_nccs(TH, need_ncc, workers)

    # The colour and orientation questions are expensive per pair (a resample
    # each, eight for orientation), so they are asked ONLY of pairs that
    # every cheaper test has already rejected. Deciding first and computing
    # second keeps that set small: on a real library almost every candidate
    # exits above, and what remains is a rounding error next to the sweep.
    uf_a = UF()
    tierb_pairs = []
    fallback = []
    dead_zone = oriented = 0
    for t, (i, j) in enumerate(cand):
        m = mads[t]
        c = cos_of(i, j)
        is_dup = ((m <= args.tier_a_mad) and (c is None or c >= args.tier_a_cos)
                  and anim_compatible(recs[i], recs[j]))
        if is_dup:
            uf_a.union(i, j)
        elif m <= args.tier_b_mad:
            # Pixels agree but CLIP does not. This used to fall through every
            # branch and vanish - the pair was neither marked nor reviewed.
            # It is exactly where real duplicates hid: an aggressive downscale
            # scores mad ~3 (unrelated pairs never get below ~50) yet CLIP
            # reads 0.96, under the Tier A floor. CLIP's cosine has a high
            # baseline and is the weaker signal here, so it may ASK for review
            # but must not silently discard. Nothing is pre-marked.
            tierb_pairs.append((i, j))
            dead_zone += 1
        else:
            # Tier B must stay meaningful: with a shared genre, CLIP cosine has a
            # high baseline (random pairs here average ~0.76), so structural
            # containment alone admits same-prompt re-rolls that are simply
            # different pictures. Require decent semantic agreement too.
            if c is not None and c >= 0.995:
                tierb_pairs.append((i, j))
            # 0.90, not the old 0.92. Measured with the grid above on 150
            # real images: an 80% crop goes from 84% accepted to 91%, an 85%
            # crop 96% -> 98%, a 90% crop 97% -> 100%. Unrelated pairs top
            # out at 0.816 over 296 samples, so the margin is still 0.084 -
            # and this gate feeds Tier B, which is never deleted without
            # review, so its failure mode is an extra line to look at.
            # 0.85 was tried and rejected: it adds 399 review candidates on
            # a 36k library to catch crops the grid fix already gets.
            elif (c is None or c >= args.tier_b_cos) and nccs.get((i, j), 0.0) >= 0.90:
                tierb_pairs.append((i, j))
            else:
                fallback.append((i, j))

    # --- only now, and only for what is left, the expensive questions -----
    # A high colour difference with a tiny LUMA difference means the same
    # picture desaturated or recoloured: a grayscale copy scores mad ~30 in
    # colour and ~2 on brightness alone.
    # Ask the coarse question first. The 8x8 signature is already in hand, so
    # a brightness-normalised luma distance on it costs one vectorised pass
    # for the whole list, where the real check costs a resample per pair.
    # The margin is deliberately wide (3x the threshold): downsampling can
    # only smooth differences away, so a pair that is already far apart at
    # 8x8 cannot come back under the limit at full size. Verified against
    # the truth set - all 20 transformations still land.
    if fallback:
        CL = C.reshape(len(C), 64, 3) @ np.array([0.299, 0.587, 0.114],
                                                 dtype=np.float32)
        CL = CL - CL.mean(1, keepdims=True)
        fa = np.fromiter((p[0] for p in fallback), dtype=np.int64, count=len(fallback))
        fb = np.fromiter((p[1] for p in fallback), dtype=np.int64, count=len(fallback))
        coarse = np.abs(CL[fa] - CL[fb]).mean(1)
        keep_l = coarse <= args.tier_a_mad * 3.0
        luma_pairs = [fallback[t] for t in np.nonzero(keep_l)[0].tolist()]
    else:
        luma_pairs = []
    phase('Luma check on %d of %d pair(s) (coarse prefilter) ...'
          % (len(luma_pairs), len(fallback)))
    lmads = compute_luma_mads(TH, luma_pairs, workers)
    still = []
    for p in fallback:
        if lmads.get(p, 255.0) <= args.tier_a_mad:
            tierb_pairs.append(p)                   # same picture, desaturated
        else:
            still.append(p)
    # Rotated/mirrored saves: eight resamples per pair, the most expensive
    # question asked anywhere, so it gets the same coarse screen first. The
    # 8x8 signature grid can be rotated directly - it is a permutation, no
    # resampling - so seven vectorised passes over the list replace millions
    # of image rotations. Wide margin again (3x), and the truth set confirms
    # every rotated and mirrored case still lands.
    if still and not args.no_orient:
        sa = np.fromiter((p[0] for p in still), dtype=np.int64, count=len(still))
        sb = np.fromiter((p[1] for p in still), dtype=np.int64, count=len(still))
        best = None
        for k in range(1, 8):
            d = np.abs(oriented_signatures(C, k)[sa] - C[sb]).mean(1)
            best = d if best is None else np.minimum(best, d)
        keep_o = best <= args.tier_a_mad * 3.0
        orient_list = [still[t] for t in np.nonzero(keep_o)[0].tolist()]
    else:
        orient_list = []
    phase('Orientation check on %d of %d pair(s) (coarse prefilter) ...'
          % (len(orient_list), len(still)))
    omads = {} if args.no_orient else compute_oriented_mads(TH, orient_list, workers)
    for p in still:
        if omads.get(p, (255.0, ''))[0] <= args.tier_a_mad:
            tierb_pairs.append(p)                   # rotated / mirrored save
            oriented += 1

    if dead_zone:
        print('Pixel-identical but CLIP-vetoed: %d pair(s) sent to review '
              '(they used to be dropped)' % dead_zone)
    if oriented:
        print('Rotated / mirrored copies found: %d pair(s)' % oriented)

    # Byte-identical files are duplicates by definition - cluster them even
    # if the sweep or the embeddings could not vouch for them.
    for members in exact:
        for i in members[1:]:
            uf_a.union(members[0], i)

    tier_a = []
    for members in uf_a.groups():
        k = max(members, key=lambda i: quality_key(recs, i))
        tier_a.append((k, [i for i in members if i != k], members))
    a_drops = set(i for _, d, _ in tier_a for i in d)
    a_keeps = set(k for k, _, _ in tier_a)

    # Connected components, not cliques. Cliques were tried and reverted:
    # forcing every member to match every other splits a chain into groups
    # that each elect their own keeper, so the same picture is proposed for
    # keeping several times over and the relation between the pieces is
    # lost. The chaining is not the problem - not knowing WHY a file is in
    # the cluster is. So the cluster stays whole and the report says which
    # members match the keeper directly and which arrived through another
    # file.
    uf_b = UF()
    for i, j in tierb_pairs:
        uf_b.union(i, j)
    tier_b, info_b = build_tier_b(uf_b.groups(), recs, tier_a)
    b_edges = set()
    for i, j in tierb_pairs:
        b_edges.add((i, j))
        b_edges.add((j, i))

    phase('')
    try:
        check_invariants(tier_a, tier_b)
    except InvariantError as e:
        print('')
        print('  [ABORT] internal consistency check failed: %s' % e)
        print('  No files were written. This is a bug - please report it.')
        return 3
    print('Invariants: OK')

    dn = sum(len(d) for _, d, _ in tier_a)
    db = sum(recs[i]['b'] for _, d, _ in tier_a for i in d)
    tot = sum(r['b'] for r in recs)
    print('')
    print('Tier A duplicates : %d clusters, %d droppable, %.1f MB of %.1f MB (%.1f%%)'
          % (len(tier_a), dn, db / 1048576.0, tot / 1048576.0, 100.0 * db / max(1, tot)))
    print('Tier B review     : %d clusters, %d candidates%s'
          % (len(tier_b), sum(len(d) for _, d, _ in tier_b),
             '  (+%d with no suggested deletions, shown for review)' % len(info_b)
             if info_b else ''))

    outdir = os.path.dirname(os.path.abspath(inv))
    stem = os.path.basename(inv)[:-len('.jsonl')].replace('image-inventory', 'duplicates')
    if stem == os.path.basename(inv)[:-len('.jsonl')]:
        stem = stem + '-duplicates'
    rep = os.path.join(outdir, stem + '-report.html')
    lst = os.path.join(outdir, stem + '-list.txt')
    # The recycler carries the same version suffix as the list it reads; a
    # fixed name would silently rebind an old edited list to a new manifest
    # when two inventories share a folder.
    rec_name = 'Recycle-Duplicates' + (
        stem[len('duplicates'):] if stem.startswith('duplicates') else '-' + stem)
    rpy = os.path.join(outdir, rec_name + '.py')
    bat = os.path.join(outdir, rec_name + '.bat')
    sh = os.path.join(outdir, rec_name + '.sh')
    nothing = (dn == 0 and sum(len(d) for _, d, _ in tier_b) == 0)
    stats = {'n': n, 'exact': sum(len(v) - 1 for v in exact),
             'headline': ('%d exact duplicates, %d visual clusters.'
                          % (sum(len(v) - 1 for v in exact), len(tier_a))),
             'method': ('SHA-256 on every file, then all %d pairs swept on an 8x8 colour '
                        'signature%s; survivors re-scored at full thumbnail resolution by '
                        'mean absolute pixel difference. Tier A requires pixel difference '
                        '<= %.1f%s. Tier B is structural similarity with genuinely different '
                        'pixels, detected by multi-scale template matching.'
                        % (n * (n - 1) // 2,
                           ' and independently on CLIP embeddings' if used_clip else '',
                           args.tier_a_mad,
                           ' and CLIP cosine >= %.3f' % args.tier_a_cos if used_clip else ''))}
    plan, home = build_emission_plan(tier_a, tier_b, recs, info_b)
    check_emission(plan, home)

    if nothing:
        write_report(rep, recs, tier_a, tier_b, root, stats, plan, home)
        # Writing a selection list and a Recycle-Bin script with an empty
        # manifest is worse than writing nothing: it looks like a loaded tool
        # that silently does nothing, and a stale one from an earlier run is
        # genuinely misleading. So only the report is written here.
        print('')
        print('No duplicates found - this folder is already clean.')
        print('Wrote just the report:')
        print('   ' + rep)
        stale = [q for q in (lst, rpy, bat, sh) if os.path.exists(q)]
        # A versioned run should also flag fixed-name leftovers from earlier
        # runs - but only when no base inventory exists that would still
        # legitimately own them. .ps1 is included because every recycler
        # generated before v4.0 was PowerShell.
        legacy = ['Recycle-Duplicates' + e for e in ('.ps1', '.bat', '.py', '.sh')]
        if rec_name != 'Recycle-Duplicates' and not os.path.exists(
                os.path.join(outdir, 'image-inventory.jsonl')):
            stale += [q for q in (os.path.join(outdir, e) for e in legacy)
                      if os.path.exists(q)]
        if stale:
            print('')
            print('Left over from an earlier run of this tool, and now out of date.')
            print('This script never deletes anything, so remove these yourself:')
            for q in stale:
                print('   ' + q)
        print('')
        return 0

    # The list is written first so the report can carry a copy of it. The
    # page edits the very lines this wrote; rendering the format a second
    # time in JavaScript would leave two implementations free to drift, and
    # this project has been bitten by exactly that before.
    nrows, llines, editable_at, suggested_b, tier_b_all = write_list_and_script(
        lst, rpy, bat, sh, recs, tier_a, tier_b, root, info_b, b_edges)
    write_report(rep, recs, tier_a, tier_b, root, stats, plan, home,
                 list_lines=llines, editable_at=editable_at,
                 suggested_b=suggested_b, tier_b_all=tier_b_all,
                 list_name=os.path.basename(lst), b_edges=b_edges)
    print('')
    print('Wrote:')
    for q in (rep, lst, rpy, bat, sh):
        print('   ' + q)
    print('')
    print('Next: open the report, edit %s (X = delete, . = keep),'
          % os.path.basename(lst))
    runner = os.path.basename(bat if os.name == 'nt' else sh)
    print('then run %s. %d entries in the manifest.' % (runner, nrows))
    if os.name != 'nt':
        print('(chmod +x it first if your shell refuses, or run: python3 %s)'
              % os.path.basename(rpy))
    # A leftover PowerShell recycler from before v4.0 reads the same list but
    # carries the old, superseded logic - say so rather than let it sit there.
    old_ps = os.path.join(outdir, rec_name + '.ps1')
    if os.path.exists(old_ps):
        print('')
        print('NOTE: %s is from an older version and is now superseded by the'
              % os.path.basename(old_ps))
        print('      .py above. Delete it so it cannot be run by mistake.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('')
        print('Aborted.')
        sys.exit(130)
