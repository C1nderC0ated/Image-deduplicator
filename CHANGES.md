# Image Inventorization — change log

Honest history, bugs included: each fix names what actually went wrong,
because half of these guards only exist since something broke for real.

## v4.3.8 — 2026-08-13 (current)

**Opt-in GPU preprocessing, with the unsafe surface cut away.**
`embed-images.py --gpu-preprocess` moves antialiased bicubic resize and CLIP
normalization to the accelerator only for opaque still images being
downscaled by at least 2x. Transparent images, animations, smaller resizes,
and decoded RGB frames over 4 MP stay on the existing byte-identical Pillow
path. (JPEG draft decoding can bring a larger source below that decoded
ceiling.)
Torch's fractional bicubic result is clipped and rounded to match the uint8
samples Pillow feeds into normalization. Raw source batches and GPU working
sets are bounded: decoded futures have a 768 MiB ceiling and the active model
batch has a separate 256 MiB ceiling.

The separate `exif+pil+flat+torch-aa-down2x-opaque+jpeg-draft` provenance tag
is load-bearing: adding or removing the flag on a resumed embeddings file
stops before model load, because mixing two resize kernels would make cosine
comparisons inconsistent. `--no-draft` has a distinct `jpeg-full` tag for the
same reason.

The accepted boundary took several rejected whole-library A/Bs to find.
Torch upscales moved tiny pixel art badly; GPU-processed animations moved an
unsafe GIF deletion; GPU-processed transparent sprites introduced two new
`X` marks on facial-expression variants. All three classes now stay on
Pillow. On the final 35,910-image A/B, 26,118 vectors were byte-identical and
the 9,792 GPU vectors had median cosine 0.999945, minimum 0.995336, none below
0.995. The decision gate then matched all 3,439 Tier A member sets and every
one of the 9,335 editable Tier A `X` paths. Tier B kept 1,812 clusters and
5,821 candidates; one information-only relation changed references only.

It is not automatically faster. The hardened mixed-library run reached
277.4 img/s against Pillow's 310.3 on a 24-thread Ryzen, 10.6% slower. A
deterministic 2,623-image large-JPEG run limited to two decode threads reached
226.4 img/s against 214.8–216.2, about 5% faster after bounding decoded
run-ahead. The option therefore remains off by default and is documented for
CPU-constrained preprocessing, not as a universal accelerator.

## v4.3.7 — 2026-08-12

**The whole pipeline is about 25% faster, and every byte of output is
identical.** Measured end to end on the same 36,410-image library, same
machine, nothing else running:

| stage | v4.3.6 | v4.3.7 |
| --- | --- | --- |
| collect | ~2m 45 | ~2m 45 |
| embed | 4m 55  (123 img/s) | 4m 18  (139 img/s) |
| analyze (with CLIP) | 4m 03 | ~3m 00 |
| analyze (`--no-embeddings`) | 5m 56 | 3m 03 |

Every accepted change was gated the same way: run it on the real
library and compare the output files byte for byte against the run
before it. The inventory, all 36,234 stored vectors, and all five
analyze outputs (list, report, and the three recycler scripts) are
identical to what v4.3.6 produced. Nothing here trades a duplicate for
a second.

**The crop matcher got its own interpreters.** It was the one stage
still cheated of its parallelism. Its pair loop makes about 23 tiny
GIL-released C calls per pair — a template resize, a `matchTemplate`, a
max — with Python glue between them, and at 168,120 pairs the glue is
what runs. Eight threads measured 28% efficient: 127 s of wall clock
for roughly 200 core-seconds of work. Large pair lists now slice across
worker processes, and the worker *is* `compute_nccs` itself, so the
scores cannot disagree with the threaded path — which is still what
small lists and any machine that cannot spawn will use. Crop matching
127 → 88 s; the stage 243 → ~180 s.

The worker count has a floor of four, and that floor is the part worth
reading. Replaying the same 168,120 pairs at two, four, six and eight
workers gives 163.6, 92.5, 84.1 and 83.3 s — **two processes are slower
than the threads they replace**, because each chunk redoes the decode
and grayscale for every image its pairs touch and at two workers that
duplicated work outweighs the second interpreter. The obvious
`cores // 2` default would have handed exactly two workers, and a
measured 0.78x, to every four-thread machine. Below four logical cores
there is no count that absorbs the duplication, so those machines stay
on threads. Above the floor the exact number matters far less than the
isolated replay implies: in a real run the parent is also holding a
~1.2 GB thumbnail store, and six workers finished in 87.7 s against
four workers' 88.9 s, inside the run-to-run spread.

Three plausible refinements on top of it were each implemented and each
measured slower, so none of them shipped: packing pairs by connected
component (147 s — the candidate graph puts 98% of pairs in two
components, so packing collapses onto two busy workers), shipping
decoded grayscales instead of the stored base64 (98.6 s — pickling
hundreds of megabytes through the parent costs more than letting each
worker decode its own slice), and tuning the inner loop with a
size-keyed template cache and `cv2.minMaxLoc` (102.8 s — the 64 px
scale steps rarely collide, so the cache bought nothing and paid dict
overhead on every call).

**The orientation cross-sweep got the band the main sweep has always
had.** `mean|a-b| >= |mean(a)-mean(b)|` means rows whose signature
means differ by more than the cut cannot pair, so sorting by mean turns
every-pair into a band. The main sweep has done this since it existed;
the cross-sweep — image *i* as stored against image *j* re-oriented —
ran seven full n×n grams instead, because nothing about the bound cares
that B is a permuted copy. That was 256 s of a 356 s `--no-embeddings`
run, and it is 81 s now. With full CLIP coverage the cross-sweep never
runs, so that path is untouched; partial coverage gets the same
banding.

One wrinkle the main sweep does not have: a re-oriented row sums in a
different order, so the two float32 means of a *true* pair can differ
by rounding. The band is widened by 0.01 — mean error over 192 values
in 0..255 stays under 1e-3 — and every survivor is still re-checked
exactly, so the widening can cost extra checks and never a pair. The
self-test now holds the cross-sweep to a brute-force oracle, full sweep
and `rows=` subset both.

**Two hot paths stopped copying frames that were already RGB.** Pillow's
`convert('RGB')` on an image that is already RGB is documented to
return `self.copy()` — a full-frame allocate and memcpy.

The embedder ran it on every drafted frame. JPEG decodes straight to
RGB, and the draft lands larger than "4x the model input" suggests: a
4000x3000 JPEG drafts to 2000x1500, because both edges must stay at or
above 896 and 1/2 is then the deepest scale libjpeg will pick — so the
wasted copy was about 9 MB per photo. The rescale-and-normalize chain
was also computing per pixel what only 256 values can produce; it is
now three per-channel table gathers, built in the exact scalar op order
the processor uses. That order is not incidental: rescale upcasts to
float64 before multiplying and normalize runs in float32, and getting
either backwards shifts the vectors by ~5e-07, which is small enough to
look like nothing and still change which pairs get nominated. The
startup probe that proves the fast path byte-for-byte against the real
processor is what makes that safe to assert, and it still runs every
time. Together: 123 → 139 img/s, and a 250-file A/B across JPEG, PNG,
WebP, GIF and BMP, alpha and grayscale included, matched tensors
exactly.

The collector's `frame_signature` ran the same copy on every sampled
animation frame — and Pillow 9.1 and later already composite GIF frames
after the first to RGB during the seek, so at `FRAME_SAMPLES` 25 that
was up to 24 wasted full-frame copies per animation. Fingerprints
byte-identical on 40 of 40 real animated files.

**Two more candidates were measured and rejected**, recorded here
because the measurements are the useful part.

Memoizing the shape-mismatch resize across pairs looked obvious: nine
of ten candidate pairs mix thumbnail shapes, and the LANCZOS resize in
front of the comparison costs ~470 us against ~6 us for the comparison
itself. A locked cache serialized the scorers that had been running
eight wide and made the run *slower* (3m 33). Lock-free it reached 3m
10, still worse than the 2m 58 it was trying to beat: the resizes
release the GIL, so what looked like redundant work was already
overlapping, and removing it removed the overlap.

Deepening the collector's in-flight window was tested against a stall
counter that reported 156 s of head-of-line blocking on the real
library — which sounded decisive and was not. Alternating A/B/A/B: the
current window ran 163 s and 173 s, the deep window 162 s and 168 s.
The spread *within* one variant is larger than the gap between them.
The counter measured how long the writer waited on the head future,
which is not the same as workers going idle — they were busy on the
other 31 in flight the whole time. A metric that is not the quantity
reaching the decision, again.

## v4.3.6 — 2026-08-12

A bug audit, and then the bugs. Ten agents read the three stages and the
launchers looking for defects rather than for improvements; twenty
findings survived, nineteen of them after a second pass whose job was to
refute the first. Nothing here came from a user report, which is the
point: these are the failures that had not happened yet.

### The five that could lose a file

**A Windows junction made one folder look like two, and the copies
inside it look like duplicates.** `os.walk` does not stop at a reparse
point and `os.path.islink()` returns False for one, so a junction was
followed and every file behind it inventoried a second time under a
second path. Two paths, one file, identical SHA - which is exactly what
an exact duplicate looks like, and the recycler would have deleted a
"copy" that was the original seen twice. The walk now resolves each
directory and visits each physical one once, whatever route reaches it,
and says so when it skips a second route rather than quietly finding
fewer files than you expect.

**`--resume` reused records written in a format the run no longer
uses.** The guard compared the thumbnail size and nothing else, so both
of v4.3.5's changes went straight past it: thumbnails at quality 78 sat
beside quality 80 ones (~2.35 MAD apart, against a 4.0 gate), and
5-frame animation fingerprints beside 25-frame ones. `frames_agree`
cannot compare two lengths and falls back to bare frame-count equality -
two unrelated 40-frame GIFs sharing only their first frame passed it and
landed in one Tier A cluster with one of them pre-marked X. The guard is
now the whole record format, and a record with no format stamp counts as
a mismatch.

**A mark could move to the wrong file.** The recycler read the list back
by stripping whitespace from each line, so a filename beginning with a
space matched the line above it. The mark you put on ` photo.jpg` was
read as a mark on the previous entry. The separator is now taken by
width, because two spaces is the format and one leading space is a
filename.

**Tier A pre-marked files its keeper had never been compared with.** A
cluster is a connected component: A matches B, B matches C, and C
arrives with an X on it without anything ever having tested C against A.
Those are still shown, still in the cluster, but no longer pre-marked -
the report labels them LINKED and says which claim is which.

**`Find-Duplicates.bat` ate exclamation marks and accepted a gutted
PyTorch.** Delayed expansion was on in that one file and nowhere else in
the fleet, so `D:\Wow! Photos` arrived as `D:\Wow Photos` and a folder
named `!New` - the kind people create to sort first - became whatever
`$New` happened to be. The torch probe checked only that the name
imported, which an emptied install passes as a namespace package. Both
now match what every other launcher does.

### The rest

**The keeper rule preferred a big JPEG over the PNG it was exported
from.** After pixel count, `quality_key` ranked on file size - a fair
proxy for detail within one encoding and no proxy at all across two,
since a JPEG re-save of a PNG is routinely larger than its original.
Demonstrated end to end: a gradient saved as a 10 KB PNG and re-saved as
a 105 KB JPEG at the same size put the X on the PNG. A lossless format
(PNG, BMP, TIFF - not GIF, whose 256-colour cap makes it the worse copy
of a photo) now ranks between pixels and size. On the real library this
flips nothing today - the two selection lists are byte-identical - so it
is purely the guard for the day a re-save shows up.

**Four promises the report page was not keeping.** A control-character
filename was drawn red, labelled DROP and styled as already marked, with
no button to change it - a deletion the list refuses to perform, offered
with no way to decline. "Clear Tier B marks" skipped the keepers, so a
mark you put on a Tier B keeper survived the button that says it clears
them. The last-copy guard looked a file up in one cluster when a file
can be in several, and emptying any of them is the thing it exists to
prevent. Ctrl+X marked a file instead of copying, because the key
handler ignored modifiers.

**An Intel Arc owner was told they had no GPU.** Two places knew only
`torch.cuda`. AMD was covered by accident - a ROCm build reuses the
`torch.cuda` namespace - but XPU and Apple Metal set neither, so
`Embed-Images.bat` sent an Arc machine down its CPU path with a message
saying so, and the doctor offered a reinstall for a setup already using
the GPU.

**`./imgdedup.sh doctor` died before it started on macOS.** /bin/sh
there is bash 3.2, which under `set -u` treats an empty `"$@"` as an
unbound variable.

**The setup menu counted backwards.** `usable[int(raw) - 1]` is a Python
index, so typing `0` at a three-way PyTorch menu installed the last
entry and `-1` the second-to-last, both silently. Verified against the
old code: `0` gave the CPU build, `-1` gave ROCm.

**Three ways the collector's output path misbehaved.** `--out` into a
folder that does not exist was not discovered until the scan was over,
because that is when the file is first opened - minutes of reading
thrown away and then a traceback. `--split-mb` is a float and `int()`
came before the multiply, so `0.5` meant 1 MB and `1.7` meant 1. And a
new part was opened the instant the previous one filled, so a scan whose
last record filled a part ended with a `.partN` holding a header, a
footer and nothing else.

## v4.3.5 — 2026-08-12

**Thumbnail quality 78 -> 80, which found real crop and letterbox pairs
the old setting was losing.** Confirmed on the 36,410-image library. The
whole pipeline now runs in about **12 minutes**: collect ~3, embed ~5,
analyze ~4.

The quality curve is not monotonic, which two sample points had hidden.
Mapped on 420 real thumbnails, differential noise reaching the 4.0 gate:

| q | median | p95 | p99 | bytes/img |
| --- | --- | --- | --- | --- |
| 74 | 0.983 | 2.796 | 3.151 | 3334 |
| 78 | 0.610 | 1.484 | 1.905 | 3588 |
| **80** | **0.351** | **1.146** | **1.520** | **3732** |
| 82 | 0.399 | 1.239 | 1.946 | 3919 |
| 85 | 0.863 | 4.363 | 5.402 | 4250 |
| 92 | 0.235 | 0.895 | 1.829 | 5565 |

42% less median noise and 20% less at p99 for 4% more storage, and better
than q92 at p99 for a third of the extra. 85 is worse than 78 on every
percentile. An earlier reading of only 78 and 92 concluded "quality buys
more than size"; the full curve does not support it, and the comment now
carries the curve instead of that conclusion.

**Why it surfaced letterbox pairs specifically**, which was not the
intent: at q78 the ringing around a hard black-to-content edge smears the
bar boundary, so `trim_bars` cuts in slightly different places on two
copies of the same framed image and the trimmed regions no longer line
up. Cleaner edges make the trim land consistently. The letterbox fix from
v4.3.4 was being partly undone by the quality of the thumbnail feeding
it.

Existing inventories store q78 thumbnails, so a `--resume` scan mixes the
two - the same caveat `--lossless-thumbs` carries, and for the same
reason. Re-collect for a uniform set.

**`_pick-python.bat` no longer expires.** It hardcoded `py -3.14` down to
`-3.9` with nothing saying the list has a shelf life. When 3.15 ships it
is never tried, and the file falls through to `.venv` and then bare
`python` - often a Store stub or absent on Windows - so a machine whose
only Python was 3.15 could be told there is none while `py -3.15` sat
there working. `py -3` is tried after the explicit list now: it means
"newest Python 3 the launcher knows about" and never needs editing. It
sits below the list rather than replacing it, because each entry is
probed functionally and an older interpreter should still win when the
newest cannot import what a stage needs. `imgdedup.sh` carries the same
list and degrades safely, since `python3` almost always is the newest,
which is now stated rather than assumed.

**Two constants gained the measurements they never had.** `gray_small`'s
64 px cap looked like a cheapness with no accuracy check on record; it
costs nothing (94% average crop acceptance against 93% at full 128 px,
and half the time). `quality_key` - the rule that decides which of your
files is proposed for deletion - had no docstring at all, and now names
the blind spot it cannot see: area wins first, so a heavily-compressed
upscale beats a pristine original and the ORIGINAL is the copy marked X.
Still unmeasured, but written down as a heuristic rather than left
looking like a result.

## v4.3.4f — 2026-08-11

**Two animations that differ in the middle were being called the same, and
on a real corpus that happened 36% of the time.** Examining the last two
unexamined constants found one sound and one hiding a false-delete path.

`FRAME_SAMPLES` was 5. Five samples of a 60-frame animation look at frames
0, 15, 30, 44 and 59 - eight per cent of the timeline - so a change
anywhere else is invisible. Six of eight planted differences scored
**0.0000**: byte-identical fingerprints for animations that genuinely
differ. Sharing frame 0 they also share a thumbnail and a CLIP vector, so
nothing else would have caught it and one of them was a Tier A delete.

Sampling is now 25 frames, which costs nothing - GIF seeking replays from
frame 0, so once the walk to the last frame is paid for the samples along
the way are free, and K=5 and K=25 both measure ~15 ms on a 120-frame GIF.

**But more samples alone made it worse**, which is the part worth
remembering. The comparison took the mean over the whole fingerprint, so
one bad frame was divided by however many agreed: with the mean, 5 samples
missed 6 of 8 and 17 samples missed 7 of 8. Sampling harder diluted the
evidence.

So the fingerprint is now judged on two questions, neither asked to do the
other's job. The **mean** answers "do these look alike overall" and keeps
its 6.0 cut. A new **worst-frame** test asks whether any single sampled
frame disagrees badly, at 60.0. Both must pass.

Calibrated on 603 real animation pairs from a live library, not on
synthetic ones - which mattered, because two plausible fixes died there:

| | mean | worst frame |
| --- | --- | --- |
| same clip, re-encoded (70 pairs) | up to 6.65 | up to 47.09 |
| same clip, stretch replaced (39) | up to 13.51 | up to 127.13 |
| genuinely different (494) | never below 22.82 | never below 24.89 |

Planted differences missed: **14 of 39 before, 1 of 39 now**. Different
clips wrongly allowed: 0 of 494. The two re-encodes it blocks already fail
the mean test today, so the new check costs nothing.

Rejected after measuring, both of which looked right on synthetic data:

- **Worst-frame alone**, dropping the mean. On real animations the
  distributions overlap - re-encoded copies reach 47.09 while genuinely
  different clips start at 34.40 - because palette quantisation moves
  single frames a lot. That is exactly the noise the mean absorbs.
- **Segment averaging**, folding every frame into one of K buckets so
  nothing goes unsampled. It fixes coverage and breaks trim tolerance: the
  buckets cover different frames when the length changes, so a genuinely
  trimmed copy scored 13.9 against a weakest real difference of 12.4.
  Point sampling picks frames by FRACTION of the timeline, which is what
  survives a trim.

Fingerprints are now 4,800 bytes rather than 960, so an inventory written
by an older version is a different size and falls through the existing
size-mismatch path rather than being misread.

`HUGE_PX` (80 M pixels) was examined and left alone: it only annotates a
record and prints a summary, gating nothing, and 80 M warns just below
Pillow's own 89.5 M bomb threshold, which is the right side to err on.

## v4.3.4 — 2026-08-11

**Crops are found far more often, review clusters explain themselves, and
the suggested keeper stopped being a fixture.** Scanning is 2.16x faster
as well. Every number below was measured on the same 36,410-image library,
and Tier A came out at 3 clusters / 4 droppable in every single run - no
change here made anything newly deletable.

### The crop tier was failing in three places at once

- **A hole in the scale grid.** Acceptance was not monotonic in crop size:
  a 90% crop was accepted 79% of the time while an 80% crop managed 84%.
  That is what a gap looks like, not a threshold set wrong -
  `NCC_SCALES` stepped 0.85 -> 0.9 -> 0.95 and a 90% crop fell between the
  teeth. Filling it with 0.88/0.92/0.97 took 90% crops to 97%.
- **The gate was set for a grid that had holes.** 0.92 -> 0.90, which with
  the filled grid gives 90% crops 100%, 85% 98%, 80% 91%, 70% 94%.
  Unrelated pairs top out at 0.816 over 296 samples, so the margin is
  0.084, and this gate feeds Tier B - never deleted without review.
- **The CLIP floor was throwing away what the pixels had accepted.**
  `--tier-b-cos` was 0.94 while neighbours are nominated at 0.90, so every
  pair in that band was decoded, template-matched and then discarded on a
  criterion it could never satisfy. It was the binding constraint: 56% of
  80% crops died there *after* passing the template match. Now 0.90.

**Letterbox bars are trimmed before matching.** Phone screenshots of
vertical artwork carry big black bars, identical between unrelated
pictures and most of the frame, so template matching correlated on them:
with 30% bars at each end 2% of unrelated pairs cleared the gate, with 40%
bars 11% did, against 1% bare. Removing them is also net *additive* on
real data, because it aligns genuinely letterboxed pairs that previously
failed to match.

### Review clusters now say why each file is in them

A review cluster is a connected component, so a member can be there
because it matches the keeper or because it matches something that
matches the keeper. The report drew both the same way, which is how
eighteen screenshots read as a bug rather than as a chain. **REVIEW**
matched the keeper; **LINKED**, dashed violet, arrived through another
file. On the reference library that is 257 of 601 review tiles - 43% -
which measures how much the old display was hiding. The list says it too.

Cliques were tried first and rejected after testing: forcing every member
to match every other splits a chain into groups that each elect their own
keeper, so the same picture gets proposed for keeping several times and
the relation between the pieces disappears.

### The suggested keeper can be overruled

The keeper tile had no toggle, so preferring the other copy could not be
expressed in the report at all - the one decision the list exists to let
you make. Every tile is toggleable now; what the page refuses is binning
the *last remaining copy* in a cluster, which is the rule the recycler
enforces anyway. References count as members for that check, because the
recycler treats an unmarked reference as a surviving copy.

### Scanning

**`--fast-thumbs` is the default; `--lossless-thumbs` brings the old
behaviour back.** The lossless-WebP attempt cost 135x the JPEG encode and
won 2 times in 2,000 - most of the scan for a tenth of a percent of the
thumbnails. Settled by running the whole pipeline both ways: collect
382.1 s -> 176.5 s, same 350 list lines, same marks, same clusters. Three
extra candidate *pairs* appeared, which is the noise reaching the
prefilter and stopping short of a verdict.

Two attempts to reason about that were wrong before it was measured
properly, and both are worth remembering. Absolute JPEG noise has a
median of 4.437, *above* the 4.0 gate, which would say the tool cannot
work at all; it works because JPEG is deterministic, so two copies of one
picture thumbnail to identical bytes and the error cancels. What reaches
the gate is the *differential* between two similar images: max 1.66 over
478 real pairs, zero verdict flips.

### Smaller

- **`--clip-neighbors` 48 -> 16.** Going from 48 to 16 moves the cut line
  by 0.003 - the neighbours sit in a narrow cosine band, so a bigger K
  buys near-identical matches rather than better ones, and the cut never
  approaches the 0.99 Tier A needs at any K from 16 to 64. Confirmed on
  the real library: no visible difference, and faster.
- **The dense clusters that make the cap necessary are screenshots**: 21
  of the 25 densest images, and 2,047 of the 3,070 with 500+ neighbours,
  came from Screenshots folders. Without the cap they alone contribute
  2.7 million pairs.
- **Skipping the embed stage costs crops, and now says so.** Without
  embeddings a copy trimmed to 90% never reaches the crop matcher 74% of
  the time, 80% -> 97%. A Tier B section still appeared, which is what
  made the silence misleading.

### Measured and declined

`cv2.resize(INTER_AREA)` for the signatures (2.49x on that step, but under
1% of the pipeline against a silent recall gate); raising `ONE_READ_LIMIT`
(8 files of 36,420 exceed it); skipping PNG EXIF (147 ms looked decisive,
but end to end it is 125.55 vs 126.21 ms - it was only paying the decode
`make_thumb` needs anyway); `os.scandir` (`stat` is 0.03%); more workers,
deeper queues, process pools and thread pinning (all inside the noise, and
processes are 4x *slower* - each reloads transformers); and keeping
OpenCV's BGR buffers (would shift every crop score for ~1 ms a thumbnail).

Embedding is at the hardware ceiling on the reference machine: 34.97 ms of
CPU per image across 4 physical cores is 114 img/s, and it measures
114-125. The GPU sustains ~390, waiting.

## v4.3.3 — 2026-08-11

**Scanning is 2.16x faster, and the duplicate decisions did not move.**
382.1 s to 176.5 s on the 36,410-image library. The lossless-WebP
thumbnail attempt is no longer the default; `--lossless-thumbs` brings it
back.

That attempt cost 135x the JPEG encode (21.97 ms against 0.16 ms) and won
2 times in 2,000, so it was most of the scan for a tenth of a percent of
the thumbnails. Dropping it is not free in principle - those thumbnails
become lossy - so it was settled by running the whole pipeline both ways:

| | default | `--lossless-thumbs` |
| --- | --- | --- |
| collect | **176.5 s** | 382.1 s |
| list lines | 350 | 350 |
| marks that differ | \- | **0** |
| Tier A clusters / droppable | 3 / 4 | 3 / 4 |
| Tier B clusters | 95 | 95 |
| candidate pairs | 376,663 | 376,660 |

Three extra candidate *pairs* appeared. That is the thumbnail noise
showing up in the prefilter and stopping short of any verdict, and it is
recorded here rather than rounded away.

**Why it is safe is worth stating, because two earlier attempts to reason
about it were wrong.** The obvious measurement - how much noise JPEG
injects into one thumbnail - gives a median of 4.437, which is *above* the
4.0 Tier A gate and would suggest the tool cannot work at all. It does
work, because JPEG is deterministic: two copies of one picture thumbnail
to byte-identical output, so the error cancels rather than accumulating.
What reaches the gate is the *differential* between two similar images,
and measured on real pairs that tops out at **1.66** against a gate of
4.0, with zero verdict flips in 478 pairs.

`--fast-thumbs` still runs, and now does nothing, so existing commands and
scripts keep working.

Also measured and declined, from a set of externally suggested
optimisations:

- **`cv2.resize(INTER_AREA)` for the 8x8 signatures** is real - 2.49x on
  that step, and zero prefilter changes across 4,295 pairs. It is declined
  anyway: `decode_signatures` is ~12.5 s of a 164 s analyze, so it buys
  under 1% of the pipeline, while the pair distances it produces differ by
  up to 4.786 against a sweep cut of 8.0. Perturbing a recall gate that
  fails *silently* to buy 1% is the wrong trade.
- **Raising `ONE_READ_LIMIT`** to avoid double-reading large files: 8 of
  36,420 images exceed the current 32 MB. Median file is 0.14 MB.
- **Skipping EXIF on PNG**: `getexif()` costs 147 ms on a PNG against
  0.0037 ms on a JPEG, which looks decisive until measured end to end -
  125.55 ms with it, 126.21 ms without. It was only paying the decode
  `make_thumb` needs anyway.
- **`os.scandir` to avoid a `stat` per file**: `os.stat` is 0.013 ms,
  0.03% of per-image cost.
- **More workers**: 8/12/16 gave 57.1/56.5/56.4 s, and a deeper in-flight
  window 64.3 s against 64.6 s. Both inside the noise on an 8-core box.
- **Keeping OpenCV's native BGR buffers**: saves a `cvtColor`, but
  `gray_small` feeds `convert('L')`, which would apply the red weight to
  the blue channel. Crop-matching scores shift, so the 0.92 NCC gate would
  need recalibrating for ~1 ms per thumbnail.

## v4.3.2h — 2026-08-11

**The recycler could not tell "not read yet" from "read, and keeping
it".** Reported from real use, and correct: after marking some Tier B
files in the report and putting the list back, it still offered to bin
every Tier B file left on `.` - which is exactly the set just spared on
purpose. The edits were read correctly, and a file marked `X` was
excluded, but nothing distinguished a deliberate keep from an unreviewed
one, so a reviewed list looked ignored.

Every editable row now records the mark this run wrote. If any Tier B row
reads differently, the list has been through a human and it wins outright:
no question is asked, and it says so. The bulk answer is offered only for
a genuinely untouched Tier B, which is the case it was added for.

- **A mislabel in the dangerous direction, found while fixing the above.**
  The default read `[Enter] Tier A only, and leave Tier B alone`, but it
  never did that - it proceeded with the list as written, Tier B `X` marks
  included. Marking Tier B files and then pressing a key that promises to
  leave them alone would have deleted them. It now reads `go with the list
  exactly as it stands`, which is what it has always done.
- The prompt points at the report as the more precise route, since marking
  per file is what stops it asking.
- Wording throughout the prompt, the review bar and the launcher was cut
  back to what is happening, without the commentary around it. Each choice
  names what it acts on rather than a bare count, since the eye lands on
  the options without reading the header above them: "also trash the 3
  Tier B files", not "also trash all 3".

The selection list is unchanged - the new field lives in the recycler's
manifest, and the list is still byte-identical to v4.3.2's for the same
inventory.

## v4.3.2g — 2026-08-10

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
