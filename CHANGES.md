# Image Inventorization — change log

Honest history, bugs included: each fix names what actually went wrong,
because half of these guards only exist since something broke for real.

## v4.2f — 2026-08-07 (current)

**Faster and lighter, with the precision-affecting parts left off by
default.** Everything was measured before and after; nothing here is a
guess, and two promising ideas were *rejected* on the evidence.

Default (nothing to opt into, detection unchanged — recall still 20/20 with
embeddings, 15/20 without, zero false positives):

- **Orientation matching 4.4x faster**, the analyzer's single biggest cost
  at scale. It was resizing shape-changing rotations so it could compare
  them — but the collector thumbnails the *rotated* image, so a genuine
  rotated copy already has a transposed thumbnail and the orientation that
  matches it restores the original shape. Comparing a resized, distorted
  rotation is work that cannot succeed. Skipping those: 1846 -> 420 us per
  pair, **zero verdict changes** across thousands of measured pairs, and a
  real rotated copy still scores exactly 0.0000.
- **Thumbnail decoding ~2.6x faster** via `cv2.imdecode`, verified
  **pixel-identical** to Pillow across 240 JPEG and lossless-WebP
  thumbnails in both orientations — byte-for-byte, not merely close. Falls
  back to Pillow when OpenCV is absent.
- **Pixel scoring ~1.3x faster** using `cv2.absdiff` on uint8 instead of
  building float32 temporaries. Worth stating plainly: the values move by
  ~1e-6, and they move in the *more accurate* direction — absdiff's mean
  accumulates in float64 where the old path accumulated in float32. No tier
  verdict changed in measurement.
  **Fixed same day, after release:** this edit also replaced the
  OpenCV-free fallback with a call to the function itself, so Analyze died
  with `RecursionError` on any machine without OpenCV — a configuration
  `requirements.txt` explicitly calls optional. Nothing caught it because
  the self-test runs where OpenCV is installed, so the fallback branch was
  never entered. The numpy path is restored and now checked against
  `cv2.absdiff` on identical, random, off-by-one and full-range inputs
  (agreement to 1e-4); an AST sweep of every module confirmed no other
  function was left calling itself the same way.
- Luma comparison folded into one matmul and one mean (it was three
  multiply-adds per channel and two means).

Opt-in, because they change stored bytes or vectors:

- **`--fp16`** (Embed): float16 on the GPU, **2.9x faster** — 343 -> 999
  img/s measured on a batch of 64. Vectors shift: max pairwise-cosine change
  0.0006, enough to move a pair sitting exactly on the Tier A cosine floor
  into the review tier. The header records `"prec"`, and resuming a file
  built at the other precision now **stops** rather than silently mixing
  vectors — the same guard the model check has always had.
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

- **`_setup.py` — one guided installer, shared by every launcher.**
  `./imgdedup.sh setup`, `Check-Image-Tools.bat`, and each per-stage `.bat`
  all hand off to it rather than carrying their own install logic; the
  `.bat` side goes through `_offer-setup.bat` for the same reason. Code
  that *changes the user's machine* is the last place to let three
  launchers drift apart. It detects the GPU, asks which PyTorch build you
  want, prints the exact command and waits for a yes — never silent, and
  it installs `torch` only (never `torchvision`, which nothing here needs
  and a stale copy of which breaks transformers).
- **GPU detection without vendor toolchains.** `rocm-smi`/`nvidia-smi` only
  exist *after* a working install, which is precisely the case setup is
  fixing. It reads PCI vendor IDs instead — sysfs `/sys/class/drm/card*`
  on Linux, `Win32_VideoController` PNPDeviceID on Windows (`wmic` is
  deprecated). Same ID space both sides, so one vendor table serves both.
  Those tools are still used, but only to tell "card present" from
  "compute driver usable" — a distinction the report now makes.
- **Fixed: the embedder told AMD users their GPU could never work.**
  `resolve_device` treated any build without `torch.version.cuda` as
  CPU-only and prescribed a CUDA wheel. But a ROCm build reports
  `torch.cuda.is_available() == True` (HIP reuses the `torch.cuda`
  namespace) and `torch.version.cuda` is **not** a reliable discriminator —
  PyTorch's own `collect_env.py` overrides it inside the HIP branch. Build
  detection now checks `torch.version.hip` first, and every fallback
  message names the vendor actually present. Intel XPU and Apple Metal are
  recognised too.
- **Index versions are discovered, not hardcoded.** The stable ROCm index
  moved 6.4 → 7.0 → 7.1 → 7.2 in a few releases, and CUDA is now `cu132` —
  the `cu128` this README had been repeating was already stale. Setup reads
  the live PEP-503 listing and sorts **numerically**: `rocm7.14` is newer
  than `rocm7.2`, which both string and float comparison get backwards.
  Pinned values remain only as an offline fallback.
- **AMD on Windows is explained instead of half-offered.** AMD ships
  ROCm-for-Windows only as full-ABI `cp312` wheels from `repo.radeon.com`;
  3.13/3.14 cannot load them, and no flag changes that. Setup detects the
  case and points at the fix that actually works — install Python 3.12
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
  high-baseline library — 130M pairs at 36k. A duplicate is always among
  its original's nearest neighbours, so top-K finds the same pairs while
  the count stays linear. A binding cap is reported, never silent.
- **Expensive per-pair tests (luma, orientation) run last**, only on pairs
  every cheaper test rejected, each behind a coarse 8×8 screen with a wide
  3× margin. Measured: luma 1.65M → 546k pairs, orientation 1.65M → 235k.
- **Orientation sweep skipped when embeddings exist** — it is seven extra
  full O(n²) passes whose only job is discovering rotated pairs on pixels
  alone, which CLIP (rotation-insensitive) has already done. Still runs
  with `--no-embeddings`, where it is genuinely needed.
- **The all-pairs sweep is banded**: `mean|a-b| >= |mean(a)-mean(b)|`, so
  rows sorted by signature mean only need comparing within `cut`. Exact —
  `--self-test` proves set equality with brute force, including a pair
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
  FAT drives and network paths the shell delete call does not fail — it
  silently deletes PERMANENTLY and reports success, same family as the
  long-path case. The recycler now refuses UNC paths and queries the
  volume for a usable bin (`SHQueryRecycleBinW`) before touching anything.
- **Partial embedding coverage no longer disables rotation discovery for
  the uncovered images.** The v4.1 "skip the orientation sweep when
  embeddings exist" shortcut skipped it for *everyone*; images missing
  from the embeddings file (analyze accepts ≥50% coverage) had no path
  left that could discover their rotated copies. The sweep now runs for
  exactly the uncovered rows, using re-oriented signature grids (pure
  permutations — no decodes), so it costs |uncovered|×n, not n².
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
  embeddings — the analyzer does. The developer-machine venv path is gone
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
  launchers that only find a Python. Every safety rule lives in one place —
  the deciding argument, since duplicated launcher logic was a real v3 bug
  and v3.8 found a critical hole in the PowerShell survivor rule. The
  manifest is JSON, retiring the PowerShell quoting problem entirely.
  **Regenerate any `Recycle-Duplicates*.ps1` you still intend to run.**
- **Trash on every platform, never permanent deletion.** Windows:
  `SHFileOperationW` + `FOF_ALLOWUNDO` (Explorer's own call), with a
  correct `SHFILEOPSTRUCTW` declaration — the widely-copied one is wrong
  on 32-bit. Linux: the freedesktop.org Trash spec — per-volume trash
  dirs, `.trashinfo` with RFC 2396 byte-wise encoding, collisions won by
  `O_EXCL` (trashing six `IMG_1234.jpg` from six folders is this tool's
  normal case). macOS: `~/.Trash` ("Put Back" needs private metadata; said
  plainly rather than pretended). Cross-filesystem trashing is refused,
  not silently turned into a whole-file copy — glib refuses too.
- **`./imgdedup.sh collect|embed|analyze|doctor`** — one POSIX entry point
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
against real files via `XDG_DATA_HOME` (24 checks — layout, header, date
format, encoding incl. non-ASCII and newlines, same-name collisions, stray
files never overwritten, symlink not target), generated `.sh` under a real
`sh`, the full pipeline through `imgdedup.sh`, and the 13-case recycler
safety suite with a verified Recycle-Bin round-trip. **Not verified:**
cross-filesystem trashing, desktop "Restore" integration,
undecodable-filename round-trips — those need a real Linux box.

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
  pixel-identical pairs (no branch matched — and CLIP is the weaker
  signal: an unrelated pair scored 0.982, above several real duplicates;
  such pairs now go to review); rotated/mirrored copies never became
  candidates (all eight orientations now swept; `--no-orient` opts out);
  grayscale copies missed (luma compared when colour rejects); gentle
  crops lost to scale-grid quantisation (0.9→1.0 jump; grid now finer).
  Everything routes to **Tier B (review)** — verified Tier A output
  byte-identical on a 3,050-image run.
- Cost at 3,050 images: 13.6s without orientation, 23.0s with. A cheaper
  luma gate was tried and **reverted** — it cost three truth-set cases and
  saved nothing. Without Embed the set scores 15/20 (CLIP is what
  nominates brightness/grayscale/crop pairs).

## v3.9 — 2026-08-07

**Review pass over v3.8, plus a dark report.** Every fix reproduced
against shipped code first.

- **CRITICAL: one filename could abort an entire scan.** v3.8's filename
  restore used `re.sub(..., repr(rel), ...)` — a string replacement is a
  regex *template*, and `repr()` of a non-printable (U+00A0 from a
  browser paste, soft hyphen, BOM) emits `\xNN`, which the template
  parser rejects. The error handler itself raised, truncating the scan
  with a footer claiming zero errors. Now a callable replacement, and the
  handler is wrapped so "never raises" is actually true.
- **Aspect-changing crops were still invisible** — v3.8 fixed only the
  exact-area tie, but thumbnails are capped on the long side, so a
  square crop of a 16:9 photo gets the *larger* thumbnail and the
  original was searched inside its own crop. Both directions always tried
  now; measured 0.406 → 0.963 on a real crop (gate 0.92). Strictly
  additive over 106 verified pairs.
- The cross-cluster refusal's explanatory branch was dead code (iterated
  the already-filtered list) — the blocking file is now named. The
  embedder recorded the first path instead of the path actually read when
  twin fallback engaged. An explicit `--mirror-dir` beside the scan root
  silently copied nothing. `errors='replace'` narrowed to stdout only.
- **The last "appears nowhere" case**: a crop group reduced below two
  members by Tier A's drops was discarded whole, hiding any file whose
  only relation was to those drops. The drops' Tier A keeper now stands
  in (same pixels), nothing pre-marked; moved into `build_tier_b()` so
  `--self-test` proves all three cases.
- **The report is dark-themed** — read next to pictures, the thumbnails
  should be the brightest thing on screen. Same colour language.

## v3.8 — 2026-08-07

**Bug-audit pass.** Verified by a recycle-script harness, re-analysis of a
real 5,011-image scan, and new self-test cases.

- **CRITICAL: the Recycle script could delete a file its own safety rule
  had just refused.** A file shared by two clusters produced a deletion
  plan entry in both — at worst bypassing a home cluster's refusal through
  the reference row and recycling the last intact copy. Latent since v3.5.
  X-selection now requires the cluster to own the row; references still
  count as witnesses.
- Crop detection was direction-blind for same-size thumbnails (the
  equal-area tie made container choice arbitrary — filename order decided
  whether a crop was found). Equal-area pairs now try both directions.
- Tier B groups with no suggested deletions were silently discarded —
  including files with no other cluster, which then appeared nowhere.
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

**Speed pass over all three stages — same results, proved.** Inventory
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
  cosine 0.0 and silently blocked every tier — even byte-identical copies
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
  lines could disagree — the Recycle script keys marks by filename, last
  line wins, which turned one real cluster's `.`+`X` into "every copy is
  marked X". First cluster to claim a file now owns its line (Tier A
  first); others show a comment. A new `check_emission` invariant refuses
  double lines; refusals name the cluster where the file is editable.

## v3.4 — 2026-08-06

- **Survives a stale torchvision after a torch reinstall** ("DLL load
  failed" from `_C.pyd` linked against a replaced torch). The embedder
  hides a present-but-unloadable torchvision — transformers then uses its
  Pillow path, same vectors — and prints both cleanup options. Nothing in
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

**New: `analyze-inventory.py`** — the duplicate analysis is a tested
script, not logic re-derived per run.

- **Keeper/candidate collision fixed**: crop relationships chain, and
  Tier B built from raw pairs could mark one file KEEP in one entry and
  delete-candidate in another (three real files hit this). Tier B is
  clustered before keeper election; four invariants asserted before
  anything is written; violation aborts. `--self-test` added, every
  check mutation-tested.
- Clean folders get only the report — no empty-manifest deletion script;
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
