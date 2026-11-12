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
`.`. Tier C is weaker-evidence review: also all `.`, no suggested keeper,
no bulk-mark. C exists to unclutter B, not to promote anything. Anything
that could move a pair from B to A, or widen A, needs the decision-level
evidence described below — not just a plausible argument.

## How to verify a change

**For anything that should change nothing** — a refactor, a
parallelisation, a cache — the bar is **byte-identical output on a real
library**. Run the stage before and after and compare the files:

- collect → the inventory JSONL
- embed → every vector line (the header carries a timestamp; ignore it)
- analyze → all five outputs: the list, the report, and the three
  recycler scripts

Run both sides on one set of libraries. Pillow 13.0 resamples tall,
narrow pictures in a different order from 12.3: on the truth set 102 of
1,326 thumbnails differ, the candidate count moves from 12,530 to 12,525
and one review entry goes, with no change to the toolkit. Python 3.9 with
Pillow 10.0.0, numpy 1.24.0 and OpenCV 4.8.0 gives the same bytes as
Python 3.14 with Pillow 12.3.0, numpy 2.5.1 and OpenCV 5.0.0, and Python
3.15 the same as 3.14.

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

**When a release mixes both kinds and no real library is at hand**, one
check still separates them. Revert exactly the fixes meant to change
decisions, in a scratch copy, and run it on every fixture: its lists
must then match the previous release's byte for byte. That proves
nothing else moved a decision. It does not replace the A/B; it says the
A/B only has those fixes to judge. v4.4.2 was checked this way.

**Byte-parity is a within-machine comparison, and that is all it needs
to be.** Moving to a new machine does not weaken the gate: run the
shipped code there once, and that run is your baseline. Every comparison
after it is before-and-after on the same hardware, exactly as before.

Comparing outputs *across* machines is a different and one-time
question, and the answer is that they may legitimately differ. GPU
floating-point kernels differ across vendors, so vectors from an
AMD/ROCm box will not byte-match an NVIDIA/CUDA box — expected, not a
bug. The more consequential difference is CPU-side: a different Pillow
or libjpeg-turbo build can change the thumbnails, and thumbnails feed
every downstream decision in both collect and analyze.

**So re-collect on a new machine rather than carrying an inventory over
and resuming it.** `record_format()` gates resume on thumbnail size,
JPEG quality and frame count — it has no idea which Pillow produced the
pixels. That is the v4.3.6 bug class exactly (q78 thumbnails beside q80
ones, ~2.35 MAD apart against a 4.0 gate), and a library-version change
would slip past the same guard, leaving one inventory holding two
provenances. A fresh collect is under three minutes and removes the
question. Same for embed: re-run rather than carrying vectors.

If you want the cross-machine answer anyway, collect the same small
folder on both and compare the inventories. Byte-identical means the two
environments' Pillow and libjpeg agree, and the concern evaporates.

The analyze stage is CPU-side and should match anywhere, with the caveat
that the signature sweep's prefilter runs through BLAS — its bound is
deliberately widened and every survivor is re-checked exactly, so the
candidate set is stable, but verify rather than assume it.

**Built corpora, when no real library is at hand.** v4.4.3 was measured
on three, all made from 51 real pictures by a script outside the
repository:

- a truth set: each picture with 25 known variants (copies, re-encodes,
  resizes, six crops, rotations, a mirror, greyscale, brightness,
  contrast, hue, blur, a noise patch, a watermark), and a sparse one
  with 3 variants each. Scored per variant: in a cluster with its
  original or not, in which tier, and whether any file of another
  picture was pre-marked.
- 3,000 and 20,000 crops of the same pictures, each a random window
  with a fifth of them followed by a planted copy, re-save, resize,
  crop or rotation. The random draws can be replayed, so every file's
  window is known and any two files can be called the same window, a
  crop of one another, overlapping, or different pictures.

They answer "did this rule find what it was built for, and what else
did it let in". They do not replace a real library: 51 pictures cannot
show a genre pocket, and the two crop corpora are one dense pocket of
true crops, which is the opposite case. Treat a result there as
necessary, not sufficient. v4.4.3 then had its A/B on a real library
of 35,953 pictures (photos, art, screenshots, memes): both versions on
one inventory, every pair that changed tier listed, and the changed
pairs looked at on contact sheets. Three of the rules below were
changed by what that showed, and the corpora had said nothing about
any of them.

**Self-tests**: `python analyze-inventory.py --self-test`. It spawns real
worker processes, holds the signature sweeps against brute-force
oracles, and exercises the OpenCV-absent fallbacks. It also writes real
files to a temporary folder and runs the generated recycler on them with
the trash call stubbed (`_run_recycler`), so a recycler rule can be
tested as the user meets it. Nothing ships without it passing.

A new check is not done until it has failed once: disable the fix it
guards, run the self-test, and see that check report FAIL. A check that
passes either way tests nothing.

**Agent benchmark**: `.benchmarks/agent-control/` scores an AI agent
driving the toolkit end to end (`benchmark.py prepare | launch | validate
| score-exact | verify`). Its README carries the protocol; the baseline
runs it quotes live outside the repository.

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

## Set by reasoning, not yet measured

These arrived with the v4.4.2 bug fixes. Each has a stated reason, and
none has had the real-library A/B that the settled defaults above had:
the v4.4.3 A/B ran v4.4.2 against v4.4.3, so both sides carried these.
Measure before building on them:

- `FLAT_STD` **1.0**: a thumbnail whose luma standard deviation is below
  this is flat, and never counts as a crop or luma match.
- `DENSE_K` **16**: an image is in a dense pocket when 16 or more other
  pictures reach CLIP 0.90 with it. It equals the default
  `--clip-neighbors`. Since v4.4.3 copies count once: byte twins, and
  pairs that pass the Tier A test (within the gate, no local edit, CLIP
  0.99 or better), are one picture. Two wider groupings were tried and
  refused. The signature sweep's candidates found more planted crops on
  the 3,000-crop corpus (54 of 75 against 40), but those candidates
  chain, and a pocket of same-app screenshots - near-identical
  signatures, different pictures - would stop being dense, which is the
  case the rule is for. Any pair within the gate, without the edit and
  CLIP conditions, did the same on the real library: screenshots of one
  app score within the gate (same frame, a little other content), so
  they merged into one "picture", their pocket went sparse, and 147
  review pairs of screenshots appeared. The strict grouping costs
  planted crops on the crop corpora, 3 of 75 on the 3,000 and 32 of
  648 on the 20,000 (the window rule below takes 3 more there), where
  every picture has 59 or 390 windows and so is dense by
  construction; the loose one had been merging near-solid
  windows of one wallpaper into a "copy", thinning those pockets by
  accident.
- The crop matcher stops early at `C_NCC_GATE` (**0.95**), not the 0.90
  B gate, so a dense or dark pair is not capped below the score it needs.
- The keeper ranking: frames, colour, transparency, area, full colour
  over palette, lossless, size, `qsum`. It decides which file is proposed for deletion,
  and it is a heuristic. Its blind spot, a large upscale ranking over
  its smaller original, is covered from 1.8x up by the `UPSCALE_*` test
  below and stays open under that.
- `MIN_TORCH` **(2, 6)** and `LEGACY_CUDA_MAX` **cu126** in `_setup.py`
  follow upstream facts, the current transformers' torch floor (2.4 in
  5.10, 2.5 in 5.17, 2.6 in 5.22) and PyTorch's Turing cut-off from CUDA
  12.8, and move when those do. `SAFE_FLOOR` **pillow 13.0.0** is the
  same kind of number: the first Pillow with none of the decoder
  vulnerabilities the OSV database listed on 2026-11-12.

## Calibrated on built corpora, checked on one real library

These arrived with v4.4.3. Each was calibrated on the corpora described
under "How to verify a change", then run against v4.4.2 on the 35,953
real pictures: the three files v4.4.2 pre-marked there were all
pictures with a small edit, and none is pre-marked now. All of these
make Tier A narrower or add to review; none widens Tier A.

- `DUP_CORR` **0.90**: a pair is pre-marked only when the two luma planes
  correlate this well. On the 3,000-crop corpus true copies measured
  0.976 and up, and 177 pairs of different windows inside the gate
  measured under 0.90. The rotated-copy
  check uses it too. `GREY_CORR` **0.95** is the same idea for the luma
  check, and `GREY_SIG_CORR` **0.90** for the greyscale sweep's 8x8
  signatures.
- `LOCAL_EDIT_FINE` **20** on 8x8 tiles, for a pair inside the gate: no
  re-encode, resize, palette or upscale of the 51 pictures went over
  14.1 there, and 18 of 21 watermarked copies inside the gate reached
  20. `LOCAL_EDIT_SPOT` **12** with `LOCAL_EDIT_SPOT_RATIO` **5**: a
  smaller tile that still holds all the difference. The real library's
  three false pre-marks sat at 16-23 on their worst tile with means of
  0.25 to 1.7 (9 to 68 times the mean); over 407 re-encodes, resizes,
  palette and upscale copies the worst tile never exceeded 3.8 times
  the mean. `LOCAL_EDIT_MAD` **24** on 16x16 tiles, with
  `LOCAL_EDIT_SAME` **0.75**, `LOCAL_EDIT_TILES` **8** and
  `LOCAL_EDIT_TEXTURE` **4.0**, for a pair over it. A local edit is
  admitted to review only outside a dense pocket and at CLIP 0.90 or
  better, whichever side of the gate it is on: on the real library 247
  of 281 within-gate pairs of that shape, and 21 of 24 over it, were
  two screenshots of one app.
- `WINDOW_TILE` **8**, `WINDOW_TEXTURE` **4.0**, `WINDOW_TILE_CORR`
  **0.60**, `WINDOW_DISAGREE` **4** and `WINDOW_DISAGREE_SHARE` **0.25**:
  at the crop matcher's 64 px window, a pass whose textured 8x8 tiles
  (grey levels spread by 4 in either copy) fail to correlate at 0.60 in
  four tiles and a quarter of them is weaker evidence. Measured on the
  real library's crop-matcher review passes (67% of the same-app
  screenshot pairs, 29% of the same-view photographs) against the 479
  planted crops the 3,000-crop corpus admits (6%), the truth set's 97%
  and 90% crops (3%, 2%) and its brightness, contrast, hue and
  whole-frame copies (0%). A 128 px window catches 78% of the
  screenshots but 15% of the crops, since a resized crop no longer
  aligns tile by tile there; 0.70 and 0.80 correlation cost 11% and
  21% of the crops. The check runs only on passes headed for review
  (215 on the real library, 2,917 on the 3,000-crop corpus, about one
  per image on the 20,000-crop one) and stops each search at the
  window the matcher kept: 3 s, 4 s and 30 s. It is threads at the
  matcher's own per-pair cost, and the glue between the GIL-free calls
  means two workers do most of the work; a library with tens of
  thousands of review passes would want the process pool the matcher
  uses. A pass the 128 px retry admitted is left in review unjudged:
  its 64 px window is under the gate, the alignment the retry repairs.
- `CROP_BAND_COS` **0.80**, `CROP_BAND_K` **2** and `CROP_BAND_NCC`
  **0.95** (= `C_NCC_GATE`): among each image's `--clip-neighbors`
  nearest, its two nearest CLIP neighbours under the 0.90 floor and
  from 0.80 up are scored by the crop matcher
  alone, retried at 128 px in a window of their own ([0.90, 0.95), with
  a queue and cap of their own so they never spend the ordinary
  near-misses' retry budget, which binds on the crop corpora), and
  admitted to review at 0.95, to weaker evidence in a dense or dark
  pocket, and through the window rule like any pass - unless
  the pair is one framing twice: one aspect ratio within
  `CROP_BAND_ASPECT` **0.02** (either way up) and pixel counts within
  `CROP_BAND_AREA` **1.2** of each other. That test came from the real
  library: of the band's passes there, those of one aspect and much
  the same size were photographs of one view from one or two cameras
  and screenshots of one app, the two kinds of pair review knows too
  well, and those between files of two shapes or sizes were crops - a
  screenshot and the piece cut from it, a drawing and its gallery
  crop, a post and the picture inside it. A 90% aspect-locked crop is
  1.23x in area and still asked. On the
  truth set 27 of 51 crops keeping 70% sat under the floor, 25 of them
  in the band, 12 of those at 0.95; crops keeping 55% mostly do not
  reach 0.90 at all. Among 3,000 unrelated truth-set pairs in the band
  2 reached 0.95 and 9 reached 0.90. On the 29k library the band is
  19,770 extra matcher pairs and admits 38 passes to review and 2 to
  weaker evidence, some 35 of the 40 review pairs being genuine crops
  (contact sheets in the harness findings); its first form, without the
  framing test, admitted 61, and the 21 extra were photographs of one
  view from two cameras and screenshots of one app. It costs 26 s in
  the crop matcher there (177 s -> 206 s for the run), and the cost is
  per image touched rather than per pair: `CROP_BAND_K` 1 saves 4 s and
  loses 3 of the truth set's 70% crops and 2 of its 55% crops.
- `UPSCALE_SCALE` **1.8**, `UPSCALE_WIN` **512**, `UPSCALE_TILE` **16**,
  `UPSCALE_TEXTURE` **6.0**, `UPSCALE_FINE_MIN` **0.5**, `UPSCALE_TILES`
  **400**, `UPSCALE_RATIO` **1.20**, `UPSCALE_CAP` **64**: a keeper at
  least `UPSCALE_SCALE` times a directly matched member's width and
  height is compared with it on up to four 512 px windows at its own
  resolution,
  tile by tile: the 75th percentile, over tiles whose 3x3-blurred luma
  spreads by 6 or more and where the resized smaller copy carries 0.5
  of pixel-scale detail, of the larger copy's detail over the smaller
  copy's resized up. Under 1.20 with 400 such tiles in hand it is an
  upscale and the best of the other directly matched members keeps.
  Measured on 51 pictures at 2x: 34 of 51 LANCZOS upscales caught, the
  other 17 undecidable (under 400 textured tiles; every decidable
  upscale measured 1.17 or less), 0 of 102 genuine originals miscalled
  (the 66 decidable ones measured 1.32 and up; the two smooth seeds an
  unguarded test miscalls have 118-225 textured tiles). The cap is
  spent on the smallest size ratios first, a resample being 2-4x and
  a thumbnail 5x and more; a file that will not open is not charged to
  it. On the
  3,000-crop corpus 6 of 21 Tier A keepers 2x over a member were
  demoted, every one a window the corpus had itself upscaled 2.2x to
  3.2x, and none of the 7 genuine ones. 1.5x and 1.33x: see the
  refuted list.
- `ORIENT_NOISE_MAD` **2.0** and `ORIENT_LOWPASS_MAD` **0.75**, both
  times the Tier A gate: a rotated pair may score up to 8 at full
  thumbnail size when it scores 3 or less at a quarter of it. Pure
  rotations measured 2.8 or less there; rotations with an edit on top,
  3.5 and up at the same full-size score.
- `ACHROMA` **2.0**: a thumbnail is greyscale when its channels differ by
  no more than this on average.
- `RECORD_FIELDS` **2** in the collector: bump it whenever a record
  gains a field a later stage decides on, or a resumed scan will mix
  records that have it with records that do not.
- `SWEEP_MAX_PAIRS` **6,000,000**, `MAD_MP_MIN_PAIRS` **40,000** and
  `ORIENT_MP_MIN_PAIRS` **100,000** change no decision below the first
  and none at all for the other two.

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

From v4.4.3:

- `cv2.imdecode` for PNG in Collect: 5% faster alone and 22% slower once
  the pixels are handed back to Pillow, which already uses zlib-ng. An
  earlier "30% faster" was the harness charging its own copies to
  Pillow.
- An upscale test in the keeper rule that upscales the smaller copy to
  the larger one's size and compares there: it caught 51 of 51 upscales
  and 13 to 20 of 51 genuinely larger originals, because a smooth
  picture looks the same either way. The v4.4.4 test compares
  pixel-scale detail tile by tile instead (`UPSCALE_*` above), and only
  at 1.8x or more: at 1.5x the genuine originals' 75th-percentile ratio
  starts at 1.21 against 1.27 at most for an upscale, so the safest
  cut (1.12, 400 textured tiles) still miscalls 1 of 66 originals -
  a textured one, not a smooth one - for 91 of 153 upscales caught;
  at 1.33x the two overlap outright (0.92 to 2.05 against 1.04 to
  1.28). Grain in flat tiles, which an upscale cannot have, does not
  separate either: both sides are re-encoded, and JPEG grain is grain.
- A luma sweep over every pair, to nominate recoloured copies: on the
  3,000-crop corpus, which holds no recoloured copy of any of its
  files, 231 extra pairs then passed the luma check. The greyscale
  sweep is limited to a greyscale picture against a colour one, with
  correlated layouts.
- Threads for the pixel pass: 8 threads ran the score 1.5 times as fast
  as one and the score plus luma 1.0 times. It is worker processes now.
- Comparing embedding vectors as a gate for a decoder change while
  another process uses the GPU: two runs of the same code differed in
  930 of 1,272 vectors, at about 2e-6 cosine. Prove the pixels
  identical instead, or measure on a quiet machine.
- Three of the four measures tried at the crop matcher's window to
  tell a crop from the same view on another day, or from one app's
  frame with other content (the fourth, textured tiles that do not
  correlate, is the `WINDOW_*` rule above). Taken at the best window,
  both directions and every scale, on the real library's crop-matcher
  review edges (36 among 22 sunsets from one window, 94 among
  same-app screenshots) and on the 479 planted crops of the 3,000-crop
  corpus that clear the 0.90 gate: the tile-mean residual after an
  affine brightness fit (a third of the tiles off by 8 levels) marks
  64% of the sunset edges and 12% of the screenshot edges at 7% of the
  crops, and at the window's own resolution the screenshots' other
  content averages out (11% at 8 levels); a stricter gate for a
  full-frame match does not apply, since 32 of the 36 sunset edges
  match best on a sub-window; and the window's size does not separate
  either, the sunset windows being small (median a quarter of the
  frame) but so are a quarter of the genuine crops. On the pairs
  within the Tier A gate whose difference sits in one region (a new
  line in a chat), nothing at all separates a screenshot from a
  drawing with a retouched detail: the differing area (median 12% of
  the tiles), its shape, the neighbour count (median 5 against 1-5 for
  the drawings), CLIP (0.91-0.98 against 0.92-0.998) and flatness (a
  third of the truth set's pictures are as flat as a screenshot) all
  overlap. Those stay in review, and the README says so.

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

## GPU preprocessing: implemented, opt-in, and bounded

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

It now exists as `--gpu-preprocess`, behind an opt-in flag beside `--fp16`.
It is intentionally narrower than the first prototype:

- only opaque, non-animated images whose short edge is at least twice the
  model input use Torch;
- transparent images, animations, smaller resizes, and decoded RGB frames
  over 4 MP retain the byte-identical Pillow path;
- Torch clips **and rounds** its bicubic result to the uint8 samples Pillow
  would feed the processor before rescaling and normalization;
- queued raw data has a 768 MiB ceiling and an active raw batch has a separate
  256 MiB ceiling; and
- the embeddings header distinguishes Pillow/Torch and draft/full JPEG
  populations. Resume across them stops with an explanation rather than
  mixing them.

Those boundaries came from failed A/Bs, not taste. Sending upscales through
Torch produced a 0.9468 minimum cosine on tiny pixel art. Sending animations
through it moved which of two background-dominated but genuinely different
GIF scenes crossed Tier A. Sending transparent sprites through it made two
facial-expression variants newly pre-marked `X`. Each risky class now stays
on Pillow.

The final full-library A/B covered 35,910 unique images on an AMD R9700. Of
those, 26,118 vectors remained byte-identical and 9,792 took the GPU path.
GPU-vs-Pillow cosine for the changed vectors was 0.999945 median and 0.995336
minimum, with none below 0.995. More importantly, all 3,439 Tier A member
sets and all 9,335 editable Tier A `X` paths matched the Pillow baseline.
Tier B retained 1,812 review clusters and 5,821 candidates; one additional
information-only relationship changed reference rows, not deletions. After
the queue-memory bound and provenance hardening, the deterministic 2,623-image
large-JPEG subset produced the same vectors byte for byte at 226.4 img/s with
two decode threads, versus Pillow's 214.8–216.2 img/s.

Performance is conditional. On this 24-thread machine the hardened mixed
full-library run reached 277.4 img/s against Pillow's 310.3, 10.6% slower. On
a deterministic 2,623-unique-image, 512px-or-larger JPEG subset limited to
two decode threads, the hardened GPU path ran 226.4 img/s against Pillow's
214.8–216.2, about 5% faster. With eight decode threads it was slower. That
is why the flag is an explicit tool for CPU-constrained runs, not a new
default or a generic
“GPU = faster” claim.

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
produced real bugs, and none of them reproduce anywhere else. Three traps
from v4.4.2 are worth knowing before testing one:

- Explorer quotes a dropped path only when it contains a space, so a
  drop test from a folder like `C:\Users\First Last\...` is always
  quoted and cannot show a split. Use the 8.3 short form of the path,
  which has no spaces.
- A substitution on `!cmdcmdline!` rewrites `cmdcmdline` in place, so a
  second comparison against it reads the altered text. Copy it into a
  variable first and substitute on the copy.
- An `if defined X` does not guard a `%X:~-1%` or `%X:a=b%` on the same
  line. cmd expands and parses the whole line first, and an empty `X`
  leaves text it cannot parse, which ends the script. Put the guard on
  its own line above. Start every launcher once with no argument, too:
  every earlier test passed a folder, which is how a double-click crash
  lasted from v4.2f to v4.4.2.

And one from v4.4.3:

- A batch file run from a batch file without `call` never returns. A
  `python` that is a `.bat` shim, or an override naming one, ended every
  launcher at its first probe with no output. `call` is not the answer:
  it parses every argument a second time, so a `%` or `^` in a folder
  name is lost. The launchers refuse a batch-file Python by name. Test
  any change to the interpreter search with a `python.bat` first on PATH
  and nothing else, and with `IMGDEDUP_PYTHON` naming one.

The install paths need a distro-managed Linux: on Arch, Debian 12+,
Ubuntu 23.04+, Fedora 38+ and Homebrew, pip refuses every install into
the system interpreter, `--user` is not exempt, and `pip uninstall` is
blocked too. Reason about that case before calling install-path work
done, and check claims about it against the installed pip's own source
rather than recalling them.
