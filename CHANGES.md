# Image Inventorization — change log

Honest history, bugs included: each fix names what actually went wrong,
because half of these guards only exist since something broke for real.

## v4.3.2g — 2026-08-10 (current)

**An audit of yesterday's two releases, which found four defects, all four
of them introduced by those releases.** One could have deleted a file the
tool promises in writing to keep.

- **HIGH. The bulk Tier B answer could delete a file the list calls
  "always kept".** A name containing control characters gets no editable
  line on purpose, so it cannot be marked and cannot be deleted through the
  list. But the manifest row was flagged as a Tier B suggestion anyway, and
  the recycler's new `b` answer marks by manifest row, not by list line. So
  the one route added yesterday walked straight around a guarantee the file
  prints two lines above itself. Reachable on Linux, where a newline in a
  filename is legal; not on Windows, which forbids the characters. The flag
  is now conditional on the file actually having a line to be suggested in,
  and a self-test pins it.
- **MEDIUM. A path containing `</script>` broke the whole report.** One
  directory called `a<` and a file called `script>` is all it takes, and it
  is legal on Linux. The list is embedded in the page as JSON, and a
  literal `</script>` inside a string closes the element early: every
  control dies and the rest of the code spills onto the page as text. HTML
  does not parse escapes inside a script, so the fix is in the JSON -
  `<` is the same string to JavaScript and not a tag to the parser.
  Also pinned, by generating a real report from a hostile path.
- **MEDIUM. `build_fast_preprocess` crashed instead of declining.** A
  processor carrying `resample=None` reached `int(None)`, which raises -
  and the call sits outside the try that catches everything else, so it
  took the whole embedding stage down. That inverts the entire point of a
  function whose job is to refuse anything it does not recognise. Only
  reachable with a non-default `--model`.
- **LOW. "Clear every Tier B mark" cleared more than Tier B.** It unmarked
  every line that started as `.`, which includes Tier A keepers - so moving
  an `X` onto a keeper by hand, then clearing Tier B, silently undid the
  edit. It now clears exactly the Tier B lines, and a manual Tier A mark
  survives it.

Checked and found sound: the `.bat` interpreter dance in
`Find-Duplicates.bat` (a failed probe clears `PYTHON_CMD` and the restore
works), quote and non-ASCII handling in the tile attributes, and whether
bulk-marking can strand a cluster with no survivor (it cannot - the
existing per-cluster check catches it and refuses).

The selection list is still byte-identical to what v4.3.2 wrote for the
same inventory.

## v4.3.2f — 2026-08-10

**Tier B stopped being homework.** Reviewing it meant reading a few
hundred lines of text and moving characters by hand, which is why almost
nobody did. Two ways out, and neither one deletes anything on its own.

- **The report is now the review tool.** Click a thumbnail to mark or
  unmark it, or walk the strip with the arrow keys and toggle with
  <kbd>X</kbd>. A running count sits in the bar with **Mark all Tier B
  suggestions**, **Clear every Tier B mark** and **Reset to as-scanned**,
  and one button downloads the selection list with your marks in it. Drop
  that over the old file and run the recycler as before.
  The page holds the list *verbatim* and only ever rewrites the first
  character of a line it was told is editable. It cannot invent a line,
  reorder one, or touch a comment - rendering the format a second time in
  JavaScript would leave two implementations free to drift, which this
  project has already been bitten by. Verified: the downloaded bytes are
  identical to the file the run wrote, BOM and CRLF included.
  Keeper tiles get no toggle at all, so a cluster cannot be emptied here.
- **The recycler asks about Tier B instead of ignoring it.** When rows the
  scan nominated are still unmarked it offers, in plain words, `[Enter]`
  Tier A only, `b` to include Tier B, `q` to quit and review first. Enter
  is the default and changes nothing. Piped input or no terminal prints a
  note and continues with Tier A alone, so scripts behave exactly as
  before. Whatever is chosen still passes the same survivor and hash
  checks; this decides what to *propose*, never what is safe.
- **`Find-Duplicates.bat`**: drag a folder on, get all three stages and
  the report. Windows had a .bat per stage and no way to just run the
  thing; Linux has had `imgdedup.sh` for a while. The CLIP stage is
  treated as optional out loud - without PyTorch it says what is lost
  (recoloured and heavily cropped copies) and carries on with pixels,
  rather than stopping.
- **The Tier B count in the report was overstating the work.** It counted
  relations found, but a Tier B drop already editable in a Tier A cluster
  is shown as a reference and cannot be marked there - so a report saying
  "6 candidates" offered 3. It now counts what the page actually renders,
  which is also what the review bar can touch.

The selection list is byte-identical to the one the previous version
wrote for the same inventory; only the report and the recycler changed.

## v4.3.2 — 2026-08-10

**Embedding and analysis both got substantially faster, and neither
output moved by one byte.** On the 36,410-image reference library:
embedding 8.3 min to 4.8 min, analysis 242 s to 162 s. All 36,410 CLIP
vectors came out byte-identical, and `duplicates-list.txt` kept the md5 it
had before, down to every intermediate count.

### Embedding: the card was waiting on the CPU

Preprocessing costs 36 ms per image against 10.5 ms of wall time, so the
GPU was never the constraint - eight decode threads were.

- **Most of the image processor's time was not spent processing images.**
  Its four operations, called directly, cost 14.35 ms against 22.96 ms
  through `proc()`: about 8.6 ms per image of validation, list wrapping,
  dtype probing and channel-format negotiation. `build_fast_preprocess`
  does the same four operations itself, and throughput went from 95 to 126
  images per second.
- **Two details decide whether it is exact.** `rescale` upcasts to float64
  before multiplying and casts down to float32 only at the end, and
  `normalize` then runs in float32 with mean and std cast to the image's
  dtype. Multiplying in float32 directly - the obvious way to write it -
  is off by one ULP, ~5e-07. That is small enough to look like nothing and
  still move which pairs the analyzer nominates, so it is not tolerated:
  measured byte-identical across 1,500 images spanning 295 distinct sizes,
  then across all 36,410 vectors of a full run.
- **It refuses rather than guesses.** The fast path engages only when every
  processor setting matches what it replicates, and then only after being
  checked byte-for-byte at startup against the processor it stands in for,
  across five shapes: both orientations, already-square, an extreme aspect
  ratio, and an upscale. Any mismatch or exception and the real processor
  is used, with a printed note. A different model gets the slow path, not
  wrong vectors.

### Analysis

- **The hottest function in the tool was on the slow path.** v4.2f gave
  `absdiff_mean` a fast OpenCV route, but `mad_pair` carried its own copy
  of the old body and never got it. That is the single most-called
  function here, 376,660 pairs on the reference library and 99.8 s of a
  242 s run, so the one place it mattered most was the one place the
  speedup missed. It shares `absdiff_mean` now instead of duplicating it,
  which is also why the omission was possible in the first place.
- **`cv2.norm(NORM_L1)` replaced `cv2.absdiff().mean()`.** It sums the
  absolute differences in one pass with no intermediate image, so there is
  no uint8 difference buffer and no float32 pair. Dividing by `A.size`
  recovers the mean *exactly* rather than approximately, because the sum
  is over integers: measured 0.000e+00 apart over 400 trials, and again
  over 2,700 comparisons spanning 2-D grayscale, 1x1, and non-contiguous
  views. About 13x faster, 85.0 -> 6.3 us per pair.
- **Orientation got 2.2x faster for free**, 18.3 s to 8.4 s, because
  `oriented_mad` already called `absdiff_mean` and simply inherited it.
- Scoring itself is 2.8x rather than 13x faster, because what remains is
  no longer the arithmetic: only about a tenth of candidate pairs share a
  thumbnail shape, so the rest pay a Lanczos resize that now dominates the
  phase.
- Against the numpy fallback the values shift by ~1e-6, unchanged from
  v4.2f and in the more accurate direction. Zero verdict changes at the
  4.0 and 12.0 gates across 8,654 real pairs, and the full run is
  byte-identical.

### Collecting: measured, and left alone

Nothing here got faster, which is the honest result rather than a gap.

Almost all of it is one thing: the lossless-WebP attempt costs 135x the
JPEG encode on a real library (21.97 ms against 0.16 ms) and wins **2 times
in 2,000**. Skipping it looks like a 2x win for a rounding error's worth of
thumbnails, and it is still refused, because "which encoder wins" decides
the stored pixels: WebP is lossless where JPEG q78 is not.

The tempting fix is to predict the winner and only attempt WebP when it
might. It does not work, and the counterexample is the one already named in
`make_thumb`'s comment. A smooth RGB gradient has **every pixel unique** -
a distinct-colour ratio of 1.0000, indistinguishable by that measure from
photographic noise - and lossless WebP still beats JPEG on it by 23x. Any
threshold cheap enough to be worth computing skips that image. On real
content the two populations overlap directly: WebP wins up to a ratio of
0.0195 and starts losing at 0.0169.

Worker count is exhausted too. The default is already `min(8, cpu_count)`,
and 8 against 12 against 16 workers on 6,282 real images came out 57.1 s,
56.5 s, 56.4 s - 1.01x, inside the noise - with byte-identical inventories
at every setting.

So the only lever in this stage is the existing `--fast-thumbs`, which
trades those ~0.1% of thumbnails for roughly twice the speed. That stays a
choice rather than a default.

Rejected after measuring, recorded so they are not tried again: folding
the luma into one matmul (5% for 2.3e-05 of drift), cv2 for the
post-matmul luma MAD (faster but no more accurate), thread pinning (a win
under load that reverses on an idle machine), a deeper signature window
(noise), WebP thumbnails (64 of 36,410 qualify), and `cv2.imread` for
loading files, which returns `None` rather than raising on non-ASCII paths
on Windows and would have silently skipped part of a real library.

## v4.3.1 — 2026-08-09

**Four defects found by auditing this session's own changes**, three of
them introduced by it. Two were false DELETE recommendations, which is the
one outcome this tool exists to avoid.

- **Short animations got shorter fingerprints.** `frame_signature` built
  its sample indices with a *set* comprehension, so they deduplicated: a
  2-frame GIF emitted 384 bytes, 3-frame 576, 4-frame 768, ≥5 frames 960.
  Fingerprints of different lengths are positionally incomparable, and the
  comparison treated that as "nothing to say" and returned True, which
  reads as consent. Unrelated animations sharing a first frame have
  byte-identical thumbnails, so pixels and CLIP both agree and the
  fingerprint was the only thing standing between them and an automatic
  delete. Measured: **8 of 8 unrelated pairs allowed through, now 0 of 8**.
  Fixed in both directions, the collector always emits 5 tiles now
  (repeating an index when the animation is shorter, seeks still ascending
  for APNG), and a size mismatch falls back to the frame count instead of
  abstaining, because inventories written earlier today hold short ones.
- **`I;16B` was dropped from the inventory entirely.** `point()` is
  implemented for `I`, `I;16` and `F` only; the byte-order variants raise.
  `I;16B` is what a big-endian 16-bit TIFF opens as, ordinary scanner and
  Adobe output, and the raise took `make_thumb` down, so the file got no
  sha, no thumbnail, and was never compared against anything. Converts via
  `I` now.
- **A float image containing `Inf` or `NaN` turned solid black.** The span
  became infinite, the scale `0.0`, and every pixel mapped to 0, so two
  unrelated such images scored a perfect match and one was offered for
  deletion. Refused now, with a reason, so the file is reported rather than
  compared wrongly. Clamping in place is not available: Pillow's `point()`
  accepts only affine expressions for these modes, and Collect is
  deliberately Pillow-only with no numpy to fall back on.
- **`--fp16` checked the flag, not the precision actually used.** The flag
  is honoured only on a GPU, so a run that asks for fp16 and lands on the
  CPU writes fp32, and the guard compared against the flag, saw a match,
  and appended fp32 vectors to a file whose header said fp16. Exactly the
  silent mixing it exists to prevent, and invisible because the opposite
  direction was caught correctly. The device is resolved before the guard
  now, and the guard and the header stamp share one expression. When the
  flag was given but ignored, the message says so instead of advising a
  re-run with a flag that is already set.

Two self-test cases pin the fingerprint: the sample count is fixed for
every length, and a size-mismatched fingerprint is not consent.

## v4.3 — 2026-08-09

**Animations are compared as animations.** Until now the collector
thumbnailed frame 0 and nothing else ever looked further, which made every
animation that starts the same indistinguishable. Measured on real GIFs,
all three scoring a perfect MAD 0.0000:

| pair | before | now |
|---|---|---|
| two **different** animations, same first frame | reported as duplicates | not auto-deleted |
| a **still** lifted out of a GIF, vs the GIF | reported as duplicates | not auto-deleted |
| the same animation re-encoded | duplicates | duplicates *(unchanged)* |

The first two were false delete recommendations, and the second one is
worse than it looks: keeping the still and dropping the GIF discards every
frame after the first.

- **Five frames are sampled across the whole animation** and stored as an
  8x8 grey fingerprint each. Five rather than two because GIF seeking is a
  *replay from frame 0*, not random access, 2.6 ms to reach frame 1 and
  128 ms to reach frame 119 of the same file. Once the walk to the end is
  paid for, extra samples along the way are nearly free (K=4 and K=8 both
  measured 112 ms on a 120-frame GIF), so sampling generously costs nothing
  over sampling meanly.
- **Only animations pay.** A 30-frame GIF costs 33 ms including the scan, less than a 900×900 still at 38 ms. Stills store no fingerprint at all.
  The fingerprint is ~440 base64 characters against a thumbnail of several
  KB.
- **`anim` is finally read.** README has promised since the field existed
  that "the frame count is recorded so a still never silently 'duplicates'
  an animation". The field was recorded and never looked at, so the promise
  was not kept. It is now, and the README says what the code actually does.
- **The sampled frames decide, not the frame count.** The first rule
  refused any pair whose frame counts differed, and a real corpus of 2739
  GIFs disproved it immediately: four pixel-identical pairs were rejected
  purely on count, and the fingerprint scored them 0.97, 2.63, 2.94 and
  10.25, three of those sit *inside* the 0.00–1.93 range of the 45 pairs
  that were allowed through. They were re-encodes with a few frames
  trimmed, not different animations. Frame count is now only the fallback
  for records with no fingerprint.
- **`FRAME_CUT` is 6.0, set from that corpus rather than guessed.** Same
  animation with matching counts scored 0.00–1.93; same animation trimmed,
  0.97–2.94; the same base clip where one file is a longer loop, 10.25;
  genuinely different animations, 62–85. Six sits above every confirmed
  re-encode and below the loop variant, which is the right side to err on, a trimmed or extended clip goes to review instead of onto a delete list,
  and the margin to 62 keeps unrelated animations out regardless.
  Net effect on that library: 48 Tier A clusters instead of 45, with one
  pair held back for review.

Video is out of scope, and deliberately so; see the README.

## v4.2.7 — 2026-08-09

**Dual-camera phone photos are treated as the JPEGs they are.** A `.jpg`
from a phone with two lenses or a depth sensor carries a multi-picture
header, so Pillow returns an `MpoImageFile` with `format='MPO'` and
`is_animated=True`. Two consequences, on exactly the libraries this tool
is pointed at:

- **The libjpeg draft fast path was silently lost.** It was gated on
  `format == 'JPEG'`, so every MPO decoded at full resolution. Measured on
  a 3000px frame: 750x750 drafted versus 3000x3000 undrafted, **16x the
  pixels**, and 0.120s -> 0.061s per file. Fixed in both the collector and
  the embedder.
  This does move MPO thumbnails (UI content MAD 2.25), and that was worth
  checking rather than assuming, because a similar shift was rejected one
  version ago. Here it goes the right way: an MPO and an ordinary JPEG
  export of the *same photo* previously scored 1.87 against each other
  because only one of them was drafted, and now score **0.80**, the pair a
  deduper most needs to match got 2.3x tighter. The earlier rejection was
  of a change that moved thumbnails away from everything for no gain; this
  one moves them toward their own twin.
- **`anim` was set on ordinary still photos.** MPO frames are the second
  lens or a depth capture of the same instant, not animation, so recording
  `anim: 2` inverted what that field means. Excluded now.

`fmt` still reports `MPO`, which is what the file actually is. Plain JPEG
records are byte-identical.

## v4.2.6 — 2026-08-09

**Partial files are now named as partial**, and the obvious way to do
that was measured, found dangerous, and rejected.

- **Truncated files are identified by name.** An interrupted download, a
  bad copy or a half-synced cloud folder now reports as
  `TruncatedFile: incomplete download or copy`, distinct from a genuinely
  corrupt file, because the user can go and re-fetch one and not the
  other. Detection is a structural walk of the container's own length
  fields (JPEG segments, PNG chunks, RIFF size) over the bytes already
  read for the SHA-256, so it costs nothing. It touches no Pillow state
  and no globals, which matters at eight workers. Verified 12/12
  truncations caught across JPEG, PNG, WebP and EXIF-bearing JPEG, with
  **zero false positives** including the cases that defeat a naive tail
  check: a JPEG with an appended MP4 (every phone "motion photo"), a PNG
  with trailing junk, and a JPEG whose EXIF thumbnail carries its own
  end-of-image marker. Anything unrecognised is never accused.
- **`ImageFile.LOAD_TRUNCATED_IMAGES` was NOT enabled**, though it is the
  obvious fix and was the plan. Measured on this pipeline: Pillow fills
  the undecoded remainder with a *constant*, grey for JPEG, black for
  PNG, so two unrelated photos cut to 1200 bytes both become the same
  flat square and score **MAD 0.0710** against a Tier A gate of 4.0.
  Intact, that pair scores 97.21. It is the 16-bit white-square bug again,
  an order of magnitude worse (0.07 vs 0.71), and it would put real files
  on a delete list.
  It also buys nothing. A truncated file does not match *its own intact
  original*: self-MAD 8.4 / 37.8 / 67.3 with 90% / 50% / 10% of the bytes
  kept, every one outside the gate. So the feature would have added a
  false-delete risk in exchange for a duplicate match that does not
  happen. Partial pixels are never compared; the file is reported instead.

## v4.2.5 — 2026-08-09

**Large images no longer eat the machine, and PIL stops accusing your
photos of being an attack.**

- **Peak memory roughly halved on big non-JPEG images.** JPEG already
  escaped via `im.draft()`, which decodes at 1/8 inside libjpeg; PNG, TIFF,
  WebP and BMP have no equivalent and were decoded whole, then copied by
  `exif_transpose`, then copied again by `convert('RGB')`. Measured on this
  box: a 144 Mpx PNG peaked at **1135 MB → 590 MB** (−48%, and 1.5× faster);
  64 Mpx went **522 → 286 MB**. That peak is *per worker*, and the default
  fan-out is eight; which is how a folder of 12k textures became a
  `MemoryError`, reported to the user as nothing more than one more
  "unreadable" file.
  **Gated on size, deliberately.** `reduce()` followed by LANCZOS is not
  identical to LANCZOS alone, measured at up to MAD 1.18 on sharp/UI
  content, a third of the Tier A budget of 4.0. Applying it everywhere
  would shift every stored thumbnail to save memory on images that were
  never a problem. Verified: at 4, 36 and 80 Mpx the output is
  **byte-identical** to before; only above 80 Mpx does anything change, and
  there the alternative was running out of memory and being dropped from
  the report entirely. Palette modes are excluded, since `reduce()` would
  average palette *indices*.
- **The decompression-bomb warning is ours now.** Pillow wrote this
  straight to unbuffered stderr, so it appeared *above* the tool's own
  banner:
  `DecompressionBombWarning: Image size (324000000 pixels) exceeds limit
  ... could be decompression bomb DOS attack.`
  To someone scanning their own holiday photos that reads as a crash and as
  an accusation. The guard is aimed at untrusted uploads; here the user
  owns every file. Suppressed at module level, *not* with
  `catch_warnings()`, which is process-global state and not thread-safe
  across the eight workers, and replaced with a plain summary of very
  large images, their sizes, and the `--workers` lever. Sizes come from the
  lazy `Image.open` header, so they cost nothing and are known *before* the
  decode that might fail. `DecompressionBombError` (the >2× case) is a
  different class and still raises, and is still recorded as unreadable.

`pre` becomes `exif+pil+flat+big`.

## v4.2.4 — 2026-08-09

**Transparent images are composited, not flattened by accident.**
`convert('RGB')` discards the alpha band and keeps whatever RGB happens to
sit *under* transparent pixels, colour no human has ever seen, because
every viewer paints those pixels as background.

Measured before: two cut-outs a human sees as identical (same black
square, transparent background, junk RGB of `(255,0,0,0)` vs `(0,255,0,0)`
underneath) scored **MAD 146** against each other and were never reported
as duplicates. The same artwork saved once transparent and once flattened
onto white missed each other the same way, a headline use case for a
deduplicator, silently failing on the difference between two invisible
backgrounds.

After: both pairs score **0.0000**. Composited onto white, because that is
what viewers and file managers flatten onto.

Two properties worth stating, both verified rather than assumed:

- **Fully-opaque images are byte-unchanged.** Compositing an alpha=255
  image over anything returns that image, so ordinary screenshots and PNG
  exports keep the exact pixels they already had. An opaque RGBA and the
  same pixels as RGB thumbnail identically.
- **It subsumes the palette-transparency fix.** Going via RGBA is precisely
  what Pillow's "should be converted to RGBA images" warning asks for, so
  that branch collapsed into this one. Confirmed silent for byte-array
  tRNS, integer tRNS and LA.

`pre` becomes `exif+pil+flat`, and the resume warning is now cumulative, it names exactly which changes a given file predates, so an ancient file
is told all three and a nearly-current one only what actually differs.

**This changes stored thumbnails and vectors for any image with
transparency.** If you have inventories or embeddings covering such
images, re-run without `--resume` rather than mixing.

## v4.2.3 — 2026-08-09

**A false DELETE recommendation, which is the one thing this tool must
never produce.** Found while investigating two harmless-looking warnings.

- **16-bit and float images were destroyed, then called duplicates of each
  other.** Pillow's `I;16 -> RGB` path *clips* at 255 instead of rescaling,
  so every pixel above 1/257 of full scale became pure white and a 16-bit
  photo thumbnailed to a near-solid white square, Photoshop/Krita exports,
  depth maps, scientific TIFFs, AI-upscaler output.
  That is not merely an ugly thumbnail. Measured: two *visibly different*
  16-bit images both went 99.7% white and scored **MAD 0.71 against a Tier
  A gate of 4.0**, reported as automatic duplicates, one of them offered
  up for deletion. The identical pair as 8-bit scores 72.3 and is correctly
  rejected.
  Now rescaled before conversion. `1/257`, not `1/256`, because 257 is
  65535/255 and it makes a 16-bit image **byte-identical to its own 8-bit
  export** (verified, max diff 0). After the fix the two different images
  score 72.29, *exactly* the 8-bit number, and a 16-bit image against its
  own 8-bit twin scores 0.0000. So this removes a false positive **and**
  recovers a true positive the tool had been missing. 32-bit int and float
  carry no defined range, so those normalise by the image's own extrema.
- **`pre` is now a named constant, `exif+pil`.** Changing the image
  processor (below) altered what the model sees wherever torchvision was
  installed, and the header still said `exif`, so a resumed embeddings
  file could have quietly mixed two vector populations, the exact trap the
  model and precision guards exist to prevent. Kept a WARN rather than a
  STOP because both changes are *conditional*: no torchvision means no
  difference at all, and the rescale only touches high-bit-depth sources.
  That matches how the EXIF change was handled; the precision guard stops
  because it affects every vector.

Two warnings that fired on ordinary input and pointed at nothing useful:

- **Palette transparency.** A PNG whose tRNS is a byte array (per-entry
  alpha) made Pillow warn on every `convert()`. Going via RGBA is what it
  asks for and is pixel-identical, measured, since RGBA→RGB keeps the
  palette colours and drops only the alpha channel RGB has no room for.
  pngquant and TinyPNG output hits this constantly. Worth noting the first
  version of this fix's own comment claimed one warning *per image*; CPython
  dedupes by (message, category, module, lineno), so it is one line per run.
  The comment now says so.
- **torchvision.** transformers printed `CLIPImageProcessor requires
  torchvision (not installed); falling back to CLIPImageProcessorPil` and
  then returned that class regardless, verified, both paths construct the
  identical object. Asked for by name now. Deliberately unconditional
  rather than preferring the torchvision backend where it exists: the two
  resample differently, and vectors that depend on which optional package a
  machine happens to have are not comparable across machines.

### Windows launchers

- **`_pick-python.bat` now sees a `.venv`**, which `imgdedup.sh` already
  did. Setup offers the venv route whenever pip is missing, and that is not
  gated on platform, so on a Windows Python with broken pip it could
  install everything into a `.venv` every `.bat` then refused to look at,
  reporting those same packages missing while the fix sat on disk.
  Ordered *below* the `py` launcher, unlike Linux: a PEP 668 distro forces
  everything into the venv so it must win there, while on Windows `py` is
  the idiomatic entry point. Verified against real `cmd.exe`.
- **The doctor branches on platform too.** It was hard-coding the Linux
  order, which made its label "what the launchers use" false on Windows.

## v4.2.2 — 2026-08-08

**pip itself is now checked before anything tries to use it.** It is not
part of Python on most Linux distros, Arch splits it into `python-pip`,
Debian into `python3-pip`, so a base install genuinely has none, and
every command the toolkit printed would have died with `No module named
pip`, which reads like a broken toolkit rather than a missing system
package.

- **Setup reports pip and venv** beside the Python version, and stops with
  the distro's package name rather than letting the failure surface later.
  Where pip is missing but `venv` works, it offers the venv route instead:
  `ensurepip` carries a bundled pip wheel, so a fresh environment gets its
  own pip even where the system has none. That is exactly why Arch works.
- **The doctor reports pip per interpreter**, in the same `[ok]` / `[MISS]`
  style as everything else. An interpreter can be perfectly healthy and
  still have no way to install anything.
- **`requirements.txt` names both as prerequisites**, since they are what
  *runs* the rest of the file rather than anything in it.

Three defects found while verifying, each worth more than the feature:

- **A half-built venv is now removed.** `python -m venv` writes the
  interpreter and `pyvenv.cfg` *before* provisioning pip, so an abort left
  something that looked like a working environment and could install
  nothing. The launchers prefer a `.venv` beside them over any Python on
  PATH, so one failed setup would have captured every later run,
  including the setup meant to repair it. Only ever removes what that call
  created; a pre-existing `.venv` is never touched, and there is a test
  for both. `setup`'s launcher probe now requires `pip` as well, so a
  hand-made pip-less venv cannot trap it either.
- **The venv route is no longer recommended where it cannot be taken.**
  Debian 12 with `python3-pip` present but `python3-venv` absent is an
  ordinary state, and precisely the machine PEP 668 forces down that path.
  Setup recommended, and under `--yes` auto-selected, a route the
  machine could not follow. Note the split is narrower than usually told:
  the `venv` module is in the base `python3`; `python3-venv` adds only
  `ensurepip` and the wheels, so `python3 -m venv` imports fine and *then*
  fails.
- **openSUSE has no `python3-pip`.** The packages are version-flavoured
  (`python314-pip`). Every distro name here was checked against that
  distro's own package database rather than recalled, which is how this
  one was caught.

Also: the missing-pip explanation no longer tells a Windows user that
their situation is "normal on Linux", same symptom, three different
causes, and it now names the one that applies.

## v4.2.1 — 2026-08-07

**Installation was impossible on Arch, and the test suite could not see
the configuration it was breaking.** Both found by a tester on a real
machine rather than by anything here.

- **`setup` now works on distro-managed Python.** Arch, Debian 12+,
  Ubuntu 23.04+, Fedora 38+ and Homebrew mark their interpreter
  *externally managed* ([PEP 668](https://peps.python.org/pep-0668/)) and
  pip refuses to install into it. `pip_base()` was appending `--user` on
  exactly that path, on the assumption that `--user` is exempt. It is not, pip rejects it identically, and blocks `pip uninstall` too (verified
  against pip 26.2.1's own source, not from memory). Setup dead-ended
  with `pip exited 1` and no way forward.
  It now detects the marker and offers three routes, defaulting to the
  one that works: **a virtual environment beside the toolkit**, created
  and populated for you. `--break-system-packages` is offered but never
  taken silently. It is the thing the distro is actively preventing.
  Rather than guess package names, it quotes the marker's own `Error`
  text, so the advice is the distro's, and is right on distros this
  script has never heard of.
- **Every launcher now prefers that `.venv`.** Without this the fix would
  install into an environment no stage ever used.
- **`in_venv()` matches pip's own test.** The first version also accepted
  `$VIRTUAL_ENV`, which an activated venv exports even while the
  interpreter actually running is the system one. We would have believed
  ourselves exempt exactly where pip refuses.
- **A failed base install no longer reports success.** The return value of
  the first `pip install` was discarded, so a refusal fell through to
  `return 0` and the launcher printed "Setup finished" having installed
  nothing. That is *how* Arch failed silently.
- **The 17 printed `pip install --user ...` hints** across the four stage
  scripts were all wrong on those distros. They now route through one
  helper that knows the difference.
- **A `cv2` that imports and then fails** (numpy ABI mismatch, missing
  libGL) crashed `compute_nccs`, which caught only `ImportError` while
  the module-level guard caught `Exception`. Widened to match.

### Tests: the OpenCV-free path is now actually covered

The `absdiff_mean` recursion below shipped because **every test here runs
where OpenCV is installed**, so the fallback branch was unreachable, a
suite cannot cover a configuration it never enters. There are now 16
checks that take OpenCV away and compare against the real thing.

Two switches are needed and neither alone is enough: `_cv2 = None` reaches
`absdiff_mean` and `imdecode_rgb`, but `compute_nccs` imports `cv2`
*locally* and keeps scoring happily, only `sys.modules['cv2'] = None`
reaches it. A test setting just the first would have proved nothing about
the crop tier.

Validated by mutation, as everything else here is: reintroducing the
recursion produces 6 failures, swapping the Pillow fallback to BGR
produces 3, and making `compute_nccs` raise produces 1, with no
traceback in any case, because a check that raises is reported as a
failure rather than killing the run and hiding every check after it.

## v4.2f — 2026-08-07

**Faster and lighter, with the precision-affecting parts left off by
default.** Everything was measured before and after; nothing here is a
guess, and two promising ideas were *rejected* on the evidence.

Default (nothing to opt into, detection unchanged, recall still 20/20 with
embeddings, 15/20 without, zero false positives):

- **Orientation matching 4.4x faster**, the analyzer's single biggest cost
  at scale. It was resizing shape-changing rotations so it could compare
  them, but the collector thumbnails the *rotated* image, so a genuine
  rotated copy already has a transposed thumbnail and the orientation that
  matches it restores the original shape. Comparing a resized, distorted
  rotation is work that cannot succeed. Skipping those: 1846 -> 420 us per
  pair, **zero verdict changes** across thousands of measured pairs, and a
  real rotated copy still scores exactly 0.0000.
- **Thumbnail decoding ~2.6x faster** via `cv2.imdecode`, verified
  **pixel-identical** to Pillow across 240 JPEG and lossless-WebP
  thumbnails in both orientations, byte-for-byte, not merely close. Falls
  back to Pillow when OpenCV is absent.
- **Pixel scoring ~1.3x faster** using `cv2.absdiff` on uint8 instead of
  building float32 temporaries. Worth stating plainly: the values move by
  ~1e-6, and they move in the *more accurate* direction, absdiff's mean
  accumulates in float64 where the old path accumulated in float32. No tier
  verdict changed in measurement.
  **Fixed same day, after release:** this edit also replaced the
  OpenCV-free fallback with a call to the function itself, so Analyze died
  with `RecursionError` on any machine without OpenCV, a configuration
  `requirements.txt` explicitly calls optional. Nothing caught it because
  the self-test runs where OpenCV is installed, so the fallback branch was
  never entered. The numpy path is restored and now checked against
  `cv2.absdiff` on identical, random, off-by-one and full-range inputs
  (agreement to 1e-4); an AST sweep of every module confirmed no other
  function was left calling itself the same way.
- Luma comparison folded into one matmul and one mean (it was three
  multiply-adds per channel and two means).

Opt-in, because they change stored bytes or vectors:

- **`--fp16`** (Embed): float16 on the GPU, **2.9x faster**, 343 -> 999
  img/s measured on a batch of 64. Vectors shift: max pairwise-cosine change
  0.0006, enough to move a pair sitting exactly on the Tier A cosine floor
  into the review tier. The header records `"prec"`, and resuming a file
  built at the other precision now **stops** rather than silently mixing
  vectors, the same guard the model check has always had.
- **`--fast-thumbs`** (Collect): skips the lossless-WebP attempt, **~2.7x
  faster scanning** (35 ms -> 13 ms per image). Thumbnails become always-JPEG,
  so flat/UI content loses its smaller lossless copy and its stored pixels
  differ from a default run.

**Rejected on measurement**, and worth recording so nobody retries them:

- *Auto-gating the WebP attempt.* It costs 44x the JPEG encode and wins
  0/60 on photo and noise content, so a predictor looked obvious. Neither
  candidate works: JPEG size overlaps in both directions (WebP wins from
  818–4223 bytes and loses from 1660–4770), and colour-count fails outright
  because a smooth gradient has *every pixel unique* yet compresses 30x
  better losslessly (44 bytes vs 1478). Measuring the predictor also cost
  half a WebP encode. No cheap test predicts the winner, so the default
  behaviour stands and the speedup is offered as a flag instead.
- *Lower WebP effort.* Only 1.5x faster (20 -> 13 ms) and it **flipped the
  JPEG-vs-WebP winner** on 2 of 24 samples, which would change stored
  pixels. Not worth it.

## v4.2 — 2026-08-07

**AMD and Intel GPUs, and a setup helper that installs the right one.**

- **`_setup.py`, one guided installer, shared by every launcher.**
  `./imgdedup.sh setup`, `Check-Image-Tools.bat`, and each per-stage `.bat`
  all hand off to it rather than carrying their own install logic; the
  `.bat` side goes through `_offer-setup.bat` for the same reason. Code
  that *changes the user's machine* is the last place to let three
  launchers drift apart. It detects the GPU, asks which PyTorch build you
  want, prints the exact command and waits for a yes, never silent, and
  it installs `torch` only (never `torchvision`, which nothing here needs
  and a stale copy of which breaks transformers).
- **GPU detection without vendor toolchains.** `rocm-smi`/`nvidia-smi` only
  exist *after* a working install, which is precisely the case setup is
  fixing. It reads PCI vendor IDs instead, sysfs `/sys/class/drm/card*`
  on Linux, `Win32_VideoController` PNPDeviceID on Windows (`wmic` is
  deprecated). Same ID space both sides, so one vendor table serves both.
  Those tools are still used, but only to tell "card present" from
  "compute driver usable", a distinction the report now makes.
- **Fixed: the embedder told AMD users their GPU could never work.**
  `resolve_device` treated any build without `torch.version.cuda` as
  CPU-only and prescribed a CUDA wheel. But a ROCm build reports
  `torch.cuda.is_available() == True` (HIP reuses the `torch.cuda`
  namespace) and `torch.version.cuda` is **not** a reliable discriminator, PyTorch's own `collect_env.py` overrides it inside the HIP branch. Build
  detection now checks `torch.version.hip` first, and every fallback
  message names the vendor actually present. Intel XPU and Apple Metal are
  recognised too.
- **Index versions are discovered, not hardcoded.** The stable ROCm index
  moved 6.4 → 7.0 → 7.1 → 7.2 in a few releases, and CUDA is now `cu132`, the `cu128` this README had been repeating was already stale. Setup reads
  the live PEP-503 listing and sorts **numerically**: `rocm7.14` is newer
  than `rocm7.2`, which both string and float comparison get backwards.
  Pinned values remain only as an offline fallback.
- **AMD on Windows is explained instead of half-offered.** AMD ships
  ROCm-for-Windows only as full-ABI `cp312` wheels from `repo.radeon.com`;
  3.13/3.14 cannot load them, and no flag changes that. Setup detects the
  case and points at the fix that actually works, install Python 3.12
  alongside and aim *only* the Embed stage at it with `IMGDEDUP_PYTHON`,
  which the per-stage interpreter resolution already supports. `torch-directml`
  is documented as a dead end (maintenance mode, 2024, pins torch 2.4.1,
  also cp312-or-older, and not a drop-in `cuda` device).
- The doctor is fully platform-neutral now: GPU inventory for all three
  vendors with driver status, ROCm-aware verdict, POSIX-correct pip
  commands, and an offer to run setup when something is missing. The
  developer-machine venv fallback is gone from the launchers.

## v4.1 — 2026-08-07

**Analyze scales to tens of thousands of images.** A 36k library ran 40+
minutes with no end in sight. Profiled at that size, five causes fixed,
the 20/20 detection truth set re-checked after every change:

- **Thumbnails no longer all held in RAM** (1.24 GB at 36k, and a hard
  `MemoryError` past that). Preloaded when they fit a budget measured from
  actual free memory; decoded on demand with a bounded cache otherwise.
- **CLIP candidates bounded per image** (`--clip-neighbors`, default 48).
  A flat cosine-0.90 floor admitted a *fifth of all pairs* on a
  high-baseline library, 130M pairs at 36k. A duplicate is always among
  its original's nearest neighbours, so top-K finds the same pairs while
  the count stays linear. A binding cap is reported, never silent.
- **Expensive per-pair tests (luma, orientation) run last**, only on pairs
  every cheaper test rejected, each behind a coarse 8×8 screen with a wide
  3× margin. Measured: luma 1.65M → 546k pairs, orientation 1.65M → 235k.
- **Orientation sweep skipped when embeddings exist**: it is seven extra
  full O(n²) passes whose only job is discovering rotated pairs on pixels
  alone, which CLIP (rotation-insensitive) has already done. Still runs
  with `--no-embeddings`, where it is genuinely needed.
- **The all-pairs sweep is banded**: `mean|a-b| >= |mean(a)-mean(b)|`, so
  rows sorted by signature mean only need comparing within `cut`. Exact, `--self-test` proves set equality with brute force, including a pair
  placed deliberately just inside the band edge. 166s → 89s at 36k.
- Gram matrix in float32 with a widened bound (survivors re-checked
  exactly); the exact re-check no longer materialises ~800 MB of
  temporaries per chunk; every stage prints its elapsed time.

End to end on the 36k set: **18.7 min**, versus not finishing. That set is
deliberately adversarial (2.4M pairs survive to scoring); real libraries
are substantially faster.

**Release review** (adversarial pass over the final state; every finding
verified by reproduction before fixing):

- **Volumes without a Recycle Bin are refused on Windows.** On removable /
  FAT drives and network paths the shell delete call does not fail. It
  silently deletes PERMANENTLY and reports success, same family as the
  long-path case. The recycler now refuses UNC paths and queries the
  volume for a usable bin (`SHQueryRecycleBinW`) before touching anything.
- **Partial embedding coverage no longer disables rotation discovery for
  the uncovered images.** The v4.1 "skip the orientation sweep when
  embeddings exist" shortcut skipped it for *everyone*; images missing
  from the embeddings file (analyze accepts ≥50% coverage) had no path
  left that could discover their rotated copies. The sweep now runs for
  exactly the uncovered rows, using re-oriented signature grids (pure
  permutations, no decodes), so it costs |uncovered|×n, not n².
- **A filename containing a newline can no longer forge mark lines.**
  Written raw into the selection list, such a name splits across lines and
  everything after the newline parses as a fresh mark that can override
  another file's `X` or `.`. Control-character names now get a commented,
  non-editable entry (they stay in the manifest unmarked: still witnesses,
  never deletable via the list).
- The "cap is binding" note counted rows with *exactly* K qualifying
  neighbours; it now tests the (K+1)-th neighbour, so it fires only when
  something was actually dropped. `imgdedup.sh` help no longer truncates
  mid-sentence (the sed range now tracks the header's closing rule). The
  embedder docstring no longer claims an AI assistant consumes the
  embeddings; the analyzer does. The developer-machine venv path is gone
  from all four files that carried it, along with its fallback probes.
- Refuted by the same review, for the record: the ThumbStore preload was
  claimed to complete a doomed decode before falling back (it cancels),
  and the vendored-module splice was claimed fragile (it is anchored and
  the generated recycler is compile-verified in the suite).

## v4.0 — 2026-08-07

**Runs on Linux and macOS.** The Python stages were always portable; the
deletion stage existed only as PowerShell, so elsewhere the pipeline
stopped one step short of useful.

- **One recycler, every OS**: `Recycle-Duplicates.py` plus thin `.bat`/`.sh`
  launchers that only find a Python. Every safety rule lives in one place, the deciding argument, since duplicated launcher logic was a real v3 bug
  and v3.8 found a critical hole in the PowerShell survivor rule. The
  manifest is JSON, retiring the PowerShell quoting problem entirely.
  **Regenerate any `Recycle-Duplicates*.ps1` you still intend to run.**
- **Trash on every platform, never permanent deletion.** Windows:
  `SHFileOperationW` + `FOF_ALLOWUNDO` (Explorer's own call), with a
  correct `SHFILEOPSTRUCTW` declaration, the widely-copied one is wrong
  on 32-bit. Linux: the freedesktop.org Trash spec, per-volume trash
  dirs, `.trashinfo` with RFC 2396 byte-wise encoding, collisions won by
  `O_EXCL` (trashing six `IMG_1234.jpg` from six folders is this tool's
  normal case). macOS: `~/.Trash` ("Put Back" needs private metadata; said
  plainly rather than pretended). Cross-filesystem trashing is refused,
  not silently turned into a whole-file copy, glib refuses too.
- **`./imgdedup.sh collect|embed|analyze|doctor`**: one POSIX entry point
  with the same functional interpreter probing as the `.bat` fleet.
- **Fixed for Linux:** `--share`'s hardcoded `C:\Users\...` path would
  have silently created a directory *named* that on POSIX (now a
  per-platform data dir, `IMGDEDUP_SHARE_DIR` to override); inventories
  stored `sub\file.jpg` and could not move between OSes (now `/`,
  normalised on read); scans would have walked `~/.local/share/Trash`,
  `.Trash-*` and `lost+found`, offering deleted files back as duplicates.
- The doctor discovers interpreters per platform (`py` launcher on
  Windows; `python3.N`, `~/.local/bin`, Homebrew, `$VIRTUAL_ENV`
  elsewhere) and gives platform-correct advice.

Tested as far as this Windows machine allows: the freedesktop backend
against real files via `XDG_DATA_HOME` (24 checks, layout, header, date
format, encoding incl. non-ASCII and newlines, same-name collisions, stray
files never overwritten, symlink not target), generated `.sh` under a real
`sh`, the full pipeline through `imgdedup.sh`, and the 13-case recycler
safety suite with a verified Recycle-Bin round-trip. **Not verified:**
cross-filesystem trashing, desktop "Restore" integration,
undecodable-filename round-trips, those need a real Linux box.

## v3.10 — 2026-08-07

**One data-loss fix and a large precision gain**, both measured.

- **CRITICAL: long paths were deleted PERMANENTLY while reporting
  success.** At ≥260 resolved characters Windows' shell delete API
  destroys the file instead of recycling it, rc=0 either way (measured:
  259 recycles, 260 is gone). `\\?\` prefixes are rejected by the API and
  the 8.3 short-name trick is a no-op on modern Windows, so the script now
  **refuses** such files loudly, measuring the *expanded* path
  (`GetLongPathNameW`).
- **Recall 8/20 → 20/20** on a truth set of known same-picture
  transformations, no new false positives (unrelated pairs: mad ≥ 52.9;
  true duplicates: ≤ 3.2). Four causes: the CLIP veto silently discarded
  pixel-identical pairs (no branch matched, and CLIP is the weaker
  signal: an unrelated pair scored 0.982, above several real duplicates;
  such pairs now go to review); rotated/mirrored copies never became
  candidates (all eight orientations now swept; `--no-orient` opts out);
  grayscale copies missed (luma compared when colour rejects); gentle
  crops lost to scale-grid quantisation (0.9→1.0 jump; grid now finer).
  Everything routes to **Tier B (review)**, verified Tier A output
  byte-identical on a 3,050-image run.
- Cost at 3,050 images: 13.6s without orientation, 23.0s with. A cheaper
  luma gate was tried and **reverted**. It cost three truth-set cases and
  saved nothing. Without Embed the set scores 15/20 (CLIP is what
  nominates brightness/grayscale/crop pairs).

## v3.9 — 2026-08-07

**Review pass over v3.8, plus a dark report.** Every fix reproduced
against shipped code first.

- **CRITICAL: one filename could abort an entire scan.** v3.8's filename
  restore used `re.sub(..., repr(rel), ...)`. A string replacement is a
  regex *template*, and `repr()` of a non-printable (U+00A0 from a
  browser paste, soft hyphen, BOM) emits `\xNN`, which the template
  parser rejects. The error handler itself raised, truncating the scan
  with a footer claiming zero errors. Now a callable replacement, and the
  handler is wrapped so "never raises" is actually true.
- **Aspect-changing crops were still invisible**: v3.8 fixed only the
  exact-area tie, but thumbnails are capped on the long side, so a
  square crop of a 16:9 photo gets the *larger* thumbnail and the
  original was searched inside its own crop. Both directions always tried
  now; measured 0.406 → 0.963 on a real crop (gate 0.92). Strictly
  additive over 106 verified pairs.
- The cross-cluster refusal's explanatory branch was dead code (iterated
  the already-filtered list), the blocking file is now named. The
  embedder recorded the first path instead of the path actually read when
  twin fallback engaged. An explicit `--mirror-dir` beside the scan root
  silently copied nothing. `errors='replace'` narrowed to stdout only.
- **The last "appears nowhere" case**: a crop group reduced below two
  members by Tier A's drops was discarded whole, hiding any file whose
  only relation was to those drops. The drops' Tier A keeper now stands
  in (same pixels), nothing pre-marked; moved into `build_tier_b()` so
  `--self-test` proves all three cases.
- **The report is dark-themed**: read next to pictures, the thumbnails
  should be the brightest thing on screen. Same colour language.

## v3.8 — 2026-08-07

**Bug-audit pass.** Verified by a recycle-script harness, re-analysis of a
real 5,011-image scan, and new self-test cases.

- **CRITICAL: the Recycle script could delete a file its own safety rule
  had just refused.** A file shared by two clusters produced a deletion
  plan entry in both, at worst bypassing a home cluster's refusal through
  the reference row and recycling the last intact copy. Latent since v3.5.
  X-selection now requires the cluster to own the row; references still
  count as witnesses.
- Crop detection was direction-blind for same-size thumbnails (the
  equal-area tie made container choice arbitrary, filename order decided
  whether a crop was found). Equal-area pairs now try both directions.
- Tier B groups with no suggested deletions were silently discarded, including files with no other cluster, which then appeared nowhere.
  Now emitted as informational clusters, nothing pre-marked.
- The analyzer allocated a full n×n CLIP matrix (~10 GB at 50k images);
  now computed per row block, byte-identical output.
- The doctor dropped the `*`-marked DEFAULT Python from `py -0p` output.
  Ctrl+C could hang the embedder mid-decode (untimed future waits).
  The embedder errored on a missing file while a byte-identical twin sat
  on disk (now tries every known path per sha). Collector errors said
  `<_io.BytesIO object at 0x...>` instead of the filename. Console output
  survives redirection under a legacy codepage.

## v3.7 — 2026-08-06

**Speed pass over all three stages, same results, proved.** Inventory
records/thumbnails byte-identical, analyzer list/report byte-identical on
identical inputs, embedding cosines 1.0000 (two deliberate exceptions).

- **Collect ~5.6× faster** (240 full-HD JPEGs: 11.4s → 2.0s): thread pool,
  one disk read per file, deterministic output order, WebP effort 75.
- **Embed ~3.2× more GPU throughput** (19 → 60 img/s): threaded decode
  ahead of the model, JPEG draft decode (vector effect ≤ 0.00003 cosine),
  `inference_mode`, pinned transfers, GPU batch 64. A poisoned batch is
  retried one image at a time.
- **Analyze ~2.8× faster at 3,050 images**: BLAS signature sweep with a
  provably lossless bound (self-test asserts set equality), uint8 thumbs,
  threaded scoring, per-image grayscale/template caches.
- **Detection fixes:** a sha missing from the embeddings file read as
  cosine 0.0 and silently blocked every tier, even byte-identical copies
  (missing now means "judge by pixels"; SHA-equal files force-clustered).
  The embedder ignored EXIF orientation while thumbnails corrected it, so
  the CLIP veto blocked genuinely identical rotated pairs (header records
  `"pre": "exif"`). The embedder never registered pillow-heif (HEICs
  failed to embed); failed embeds were never retried; `--resume` could
  reuse wrong-size thumbnails and preferred the *oldest* inventory.
- **Safety/robustness:** versioned recycler names (a second analysis
  can't rebind an edited list); `[brackets]` in paths no longer break
  discovery; `.part` in a folder name no longer truncates stems; orphaned
  `.partN` never overwritten; aborted scans record `done: false`; the
  list parser strips only the trailing info block (`photo  [final].jpg`
  stays matchable); drive-root drag-and-drop fixed; systemic embed
  failures abort loudly after 8 in a row; duplicate error records written
  once; Python gate raised to 3.9 (the stale 3.8 gate and an `--out`
  truncation hazard were caught by this release's own review).

## v3.6 — 2026-08-06

- **Cluster numbers in the report**, matching the list exactly, with
  anchor links. Report and list generate from the same emission plan so
  numbering cannot drift. Files editable elsewhere show as grey
  `IN CLUSTER N` tiles instead of being invisible.

## v3.5 — 2026-08-06

- **Fixed: one file could get an editable line in two clusters**, and the
  lines could disagree, the Recycle script keys marks by filename, last
  line wins, which turned one real cluster's `.`+`X` into "every copy is
  marked X". First cluster to claim a file now owns its line (Tier A
  first); others show a comment. A new `check_emission` invariant refuses
  double lines; refusals name the cluster where the file is editable.

## v3.4 — 2026-08-06

- **Survives a stale torchvision after a torch reinstall** ("DLL load
  failed" from `_C.pyd` linked against a replaced torch). The embedder
  hides a present-but-unloadable torchvision, transformers then uses its
  Pillow path, same vectors, and prints both cleanup options. Nothing in
  this toolkit needs torchvision; it is probed only because a broken one
  takes transformers down. The old error blamed the wrong causes.

## v3.3 — 2026-08-06

- **Keeper choice is yours to edit.** Every cluster member is an editable
  line (the old `#  KEEP` comment made an `X` on the keeper a silent
  no-op). The Recycle script enforces the surviving-witness rule per
  cluster; an all-`X` cluster is refused. Verified against five scenarios.

## v3.2 — 2026-08-05

- **The GPU actually gets used, and silence about it is gone.** The
  launcher probes for a CUDA-capable Python first; the embedder explains
  any CPU fallback (CPU-only wheel vs driver), checks `nvidia-smi`, and
  prints the exact reinstall command; `--device` gives manual control.

## v3.1 — 2026-08-05

- **Sharing is opt-in** (`--share` / `--mirror-dir`); by default every
  stage is fully local and says so.

## v3 — 2026-08-05

**New: `analyze-inventory.py`**, the duplicate analysis is a tested
script, not logic re-derived per run.

- **Keeper/candidate collision fixed**: crop relationships chain, and
  Tier B built from raw pairs could mark one file KEEP in one entry and
  delete-candidate in another (three real files hit this). Tier B is
  clustered before keeper election; four invariants asserted before
  anything is written; violation aborts. `--self-test` added, every
  check mutation-tested.
- Clean folders get only the report, no empty-manifest deletion script;
  stale outputs are named for manual removal.
- **Launcher fleet drift fixed**: three launchers with three different
  interpreter orders became one shared `_pick-python.bat` with functional
  probes. The doctor no longer drops interpreters with spaces in their
  paths. Footer/record key collision fixed (schema `img-inv/3`).
- Embedder output named after its inventory; model/width mismatch stops
  instead of silently mixing vector spaces.

## v2 — 2026-08-05

**New: CLIP semantic stage and doctor.** v2 collector: resume, `.partN`
splitting, adaptive-lossless 128px thumbnails, `qsum`, PNG AI-parameter
capture, HEIC/AVIF awareness, mirror folder.

- **Hollow-venv false positive fixed**: a gutted install still imports as
  an empty namespace package, so presence probes lied. All probes became
  functional; the doctor reports `[EMPTY]`.
- transformers 4.x/5.x `get_image_features` incompatibility handled.
- cu128 (not cu121) guidance for torch on Python 3.14.

## v1 — 2026-08-05

First release: SHA-256 + 96px thumbnails collector. Proved on a real
4,854-image folder (2.95 GB in 66 s, zero errors); that scan's analysis
set the calibration baselines the analyzer still uses.
