# Working on this tool

Notes for whoever picks this up next, human or agent. CHANGES.md is the
history of what happened; this is the part that is easy to re-derive
badly — the standard a change is held to, and the things already tried
and measured so nobody spends another afternoon on them.

## The one rule

This tool moves your files to the trash. Nothing is ever deleted
automatically, and **a false delete is the cardinal sin**: a duplicate
left behind costs disk space, a wrongly-deleted original is gone. Every
guard in the codebase exists downstream of that asymmetry, and it is why
the verification bar below is higher than a normal refactor deserves.

Tier A is pre-marked `X` for deletion. Tier B is review-only, pre-marked
`.`. Anything that could move a pair from B to A, or widen A, needs the
decision-level evidence described below — not just a plausible argument.

## How to verify a change

**For anything that should change nothing** — a refactor, a
parallelisation, a cache — the bar is **byte-identical output on a real
library**. Run the stage before and after and compare the files:

- collect → the inventory JSONL
- embed → every vector line (the header carries a timestamp; ignore it)
- analyze → all five outputs: the list, the report, and the three
  recycler scripts

This is a much stronger condition than "the decisions did not change",
and that is the point. It is free to check, it has no false negatives,
and it catches the failure you did not think to test for. A change that
is genuinely work-preserving will pass it; if it does not, you have a
bug, not a rounding difference.

**For anything that should change something** — a threshold, a keeper
rule, a quality setting — byte-parity is the wrong bar and the project
does not use it. Those get measured on what actually reaches the
decision: how many pairs move tier, in which direction, and whether the
ones that moved were right to move. The v4.3.6 keeper-rule change is a
worked example: the A/B came out byte-identical on the real library, so
the harness was first validated on a corpus built to contain the case
the change was about, proving the experiment could detect a difference
before its null result was believed.

**Byte-parity is per-machine for the embed stage.** GPU floating-point
kernels differ across vendors and architectures, so vectors produced on
an AMD/ROCm box will not byte-match vectors from an NVIDIA/CUDA box.
That is expected and is not a bug. Compare embed against a baseline
produced on the *same* machine; across machines, compare tier decisions
instead. The collect and analyze stages are CPU-side and should match
anywhere, with the caveat that the signature sweep's prefilter runs
through BLAS — its bound is deliberately widened and every survivor is
re-checked exactly, so the candidate set is stable, but verify rather
than assume it.

**Self-tests**: `python analyze-inventory.py --self-test`. It spawns real
worker processes, holds both signature sweeps against brute-force
oracles, and exercises the OpenCV-absent fallbacks. Nothing ships
without it passing.

## Settled defaults — do not re-tune casually

All measured on a real 36,410-image library.

- Thumbnail JPEG quality **80**, from a seven-point curve. The curve is
  **not monotonic**: 85 is worse than 78 on every percentile. Do not
  "raise the quality" without re-measuring the whole curve.
- `--fast-thumbs` is the default; `--lossless-thumbs` opts back in.
- `--clip-neighbors` 16, `--tier-b-cos` 0.90, NCC gate 0.90 with
  0.88/0.92/0.97 in the scale grid, `FRAME_SAMPLES` 25 with a worst-frame
  veto at 60.0 beside the 6.0 mean.
- The NCC worker count has a **floor of four**, and that floor is
  load-bearing: two processes measured *slower* than the threads they
  replace (0.78x), because each chunk redoes per-image decode and
  grayscale work. A `cores // 2` default would hand exactly two workers
  to every four-thread machine. Below four logical cores it stays on
  threads.

## Measured and refuted — do not retry

Each of these was implemented or measured and rejected on evidence:

Single-matmul luma; cv2 for the post-matmul luma MAD; thread pinning; a
deeper signature window; WebP thumbnails; PNG `im.text`; `cv2.imread`
(returns None on non-ASCII Windows paths); `cv2.resize(INTER_AREA)` for
the signatures (a real 2.49x on an operation worth under 1% of the
pipeline, against a silent recall gate); raising `ONE_READ_LIMIT` (8
files of 36,420 exceed it); skipping PNG EXIF; `os.scandir` (stat is
0.03%); BGR buffers; cliques for review clusters (splits chains, so one
picture gets several keepers); worst-frame-only and segment-averaged
animation fingerprints.

**Process pools for the embed stage**: 4x slower, because each process
reloads transformers. More importantly the premise is wrong — the prep
threads already scale near-perfectly, since PIL releases the GIL during
decode and resize. Only *fewer CPU-ms per image* helps that stage.

From the v4.3.7 speed hunt, all implemented and timed rather than
argued about:

- Memoizing the shape-mismatch LANCZOS resize across pairs: 3m 33
  locked, 3m 10 lock-free, against 2m 58 unchanged. The resizes release
  the GIL, so the "redundant" work was already overlapping and removing
  it removed the overlap.
- Deepening collect's in-flight window: 163/173 s current against
  162/168 s deep. The spread inside one variant exceeds the gap.
- Packing NCC pairs by connected component: 147 s. The candidate graph
  puts ~98% of pairs in two components, so it collapses onto two workers.
- Shipping decoded grayscales to NCC workers instead of the stored
  base64: 98.6 s. Pickling hundreds of megabytes costs more than letting
  each worker decode its own slice.
- A size-keyed template cache plus `cv2.minMaxLoc` in the NCC inner
  loop: 102.8 s. The scale steps rarely collide at 64 px, so the cache
  bought nothing and paid dict overhead per call.

## The open opportunity: GPU preprocessing

The embed stage is CPU-bound, not GPU-bound. It runs at ~139 img/s
against a bare-GPU-forward ceiling of ~385, with the time going to
decode (~16.9 ms/img) and resize (~13.7 ms/img). **A faster GPU does not
speed this stage up; more CPU cores do.** Moving the resize and
normalise onto the idle GPU is worth roughly 40%, and it is the only
substantial lever left.

It is not free, and the numbers are measured rather than assumed
(200 real images, median downscale 5.7x):

- `F.interpolate(mode='bicubic')` at its **default `antialias=False` is
  disqualifying**. It keeps a fixed 4-tap window at any ratio, so it
  aliases badly on a 5.7x downscale. CLIP cosine between the same image
  prepped Pillow-vs-torch: median 0.9234, min 0.7307, with 196 of 200
  under the 0.99 Tier A gate — against 0.4986 for two entirely
  *different* images.
- `antialias=True` is a fair match: median 0.9997, min 0.9953, none
  under 0.99. Quality is not the objection.
- The kernels differ regardless: Pillow's cubic uses a = −0.5, torch's
  −0.75 (like OpenCV). The impulse-response negative lobe is −0.073
  against −0.110, and they sit up to 25.85/255 apart on an *upscale*,
  where antialiasing plays no part.

So it belongs behind an opt-in flag beside `--fp16`, never as a default:
the 0.9953 worst case sits close enough to the gate that a borderline
pair could flip, and `PRE_TAG` — which the embedder refuses to resume
across — means adopting it forces a full re-embed. Before trusting it,
A/B the **tier decisions** on a real library, not the cosines. Cosines
are not the quantity that reaches the gate.

## The lesson that has cost the most time

Measure the quantity that actually reaches the decision, and calibrate
on real data. This has gone wrong repeatedly and in both directions:

- Absolute JPEG noise has a median of 4.437 against a 4.0 gate, which
  says the tool cannot work. What reaches the gate is the *differential*
  between two similar images: max 1.66.
- Synthetic separation lied twice about animation fingerprints. Two
  fixes looked clean on generated GIFs and overlapped on 603 real pairs.
- A stall counter reported 156 s of head-of-line blocking in collect,
  which sounded decisive. It measured how long the writer waited on the
  head future, not whether workers went idle — they were busy on the
  other 31 in flight. The fix it motivated was worth nothing.
- An isolated replay favoured six NCC processes over four by 9%. End to
  end that shrank to 1%, because the replay did not have the parent's
  ~1.2 GB thumbnail store competing for memory bandwidth.

Isolated benchmarks flatter the change. A/B the whole stage before
believing one.

## User-visible text

Report what is happening, not why. Terse and literal — but terseness
must never drop the noun, because the eye lands on a menu line without
reading the paragraph above it, so any line that can be read alone has
to be self-contained. "also trash the 3 Tier B files", not "also trash
all 3". Reuse the project's existing term for a thing instead of
inventing a second name. Handle singular and plural rather than writing
"file(s)". Keep an explanation only where it is load-bearing.

## Platform split

Changes are verified where they can actually run. The `.bat` launchers
need a real Windows shell — `cmd.exe` semantics around delayed
expansion, drag-and-drop quoting and trailing backslashes have all
produced real bugs, and none of them reproduce anywhere else. The
install paths need a distro-managed Linux: on Arch, Debian 12+, Ubuntu
23.04+, Fedora 38+ and Homebrew, pip refuses every install into the
system interpreter, `--user` is not exempt, and `pip uninstall` is
blocked too. Reason about that case before calling install-path work
done, and check claims about it against the installed pip's own source
rather than recalling them.
