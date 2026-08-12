# Image Inventorization

A three-stage pipeline that finds duplicate, re-encoded, resized, rotated,
mirrored and cropped images in large libraries. Tested from 1,500 to 36,000
images. Everything runs on your own machine: nothing is uploaded, and there
is no account to make. Runs on Windows, Linux and macOS.

Nothing is ever deleted automatically. The pipeline produces a visual
report, a selection list you edit yourself, and a recycler script that
re-checks every file at the moment of deletion and asks before doing
anything. There is no "clean everything" button. You read the report, mark
what should go, and whatever the recycler removes lands in the Recycle Bin
or Trash rather than being erased.

```
your image folder
   │
   ▼  Collect-Image-Inventory.bat        (hash + thumbnail + metadata scan)
image-inventory.jsonl
   │
   ▼  Embed-Images.bat                   (CLIP vectors — skippable, but see below)
image-embeddings.jsonl
   │
   ▼  Analyze-Inventory.bat              (pair sweep, tiers, safety invariants)
report.html  +  selection list  +  Recycle-Duplicates.py (+ .bat / .sh)
   │
   ▼  you edit the list (X = delete, . = keep), run the Recycle script
Recycle Bin / Trash
```

On Linux and macOS the same stages run through
`./imgdedup.sh collect|embed|analyze`, with `setup` and `doctor` alongside.

Each stage writes plain files next to your images, so you can stop at any
point, inspect everything in a text editor, and re-run stages independently.

---

## Release contents

| Path | Role |
|------|------|
| `Find-Duplicates.bat` | All three stages on one folder, then opens the report |
| `Collect-Image-Inventory.bat` + `collect-image-inventory.py` | Stage 1 — scan a folder into `image-inventory.jsonl` |
| `Embed-Images.bat` + `embed-images.py` | Stage 2 — CLIP embeddings. Skippable, but **cropped copies are only found with it** (see [Skipping the embed stage](#skipping-the-embed-stage)) |
| `Analyze-Inventory.bat` + `analyze-inventory.py` | Stage 3 — find duplicates, write report / list / recycler |
| `Check-Image-Tools.bat` + `check-image-tools.py` | Doctor — every Python on the machine and what each can actually do |
| `imgdedup.sh` | POSIX launcher: `setup` / `collect` / `embed` / `analyze` / `doctor` |
| `_setup.py` | GPU detection + guided install, shared by every launcher |
| `_offer-setup.bat` | Shared "something is missing" handler for the .bat launchers |
| `_trash.py` | The only code that can make a file disappear — one trash backend per OS |
| `_pick-python.bat` | Shared interpreter chooser used by every `.bat` launcher |
| `_why-no-python.bat` | Shared failure report — prints what each Python candidate *actually said* |
| `requirements.txt` | Package list (see [Requirements](#requirements)) |
| `CHANGES.md` | Version history, including the bugs and what they taught the tool |

Generated at run time, next to your images:

| File | Written by | What it is |
|------|-----------|------------|
| `image-inventory.jsonl` (+ `.partN` above ~200 MB) | Collect | One JSON line per image: path, size, SHA-256, dimensions, EXIF, JPEG quality fingerprint, AI-generation text chunks, 128 px thumbnail |
| `image-embeddings.jsonl` | Embed | One CLIP vector per unique image, keyed by SHA-256 |
| `<name>-report.html` | Analyze | Every cluster as pictures, **numbered to match the list** — keeper green, drops red, review amber, **linked violet-dashed** (in the group via another member, not the keeper), grey = editable in another cluster. Click any tile to mark it, including the keeper; the last surviving copy refuses. Dark-themed, so the thumbnails stay the brightest thing on screen |
| `<name>-list.txt` | Analyze | The selection list you edit: first character `X` = delete, `.` = keep |
| `Recycle-Duplicates.py` + `.bat` / `.sh` | Analyze | The only thing that deletes — after verification and your y/N. The `.py` holds every rule; the `.bat`/`.sh` only find a Python. Versioned (`Recycle-Duplicates-2.*`) when the inventory is, so each stays bound to its own list |

---

## Requirements

- **Windows 10/11, Linux, or macOS.** Windows gets a `.bat` per stage to
  drag folders onto; elsewhere use `./imgdedup.sh`. GPU acceleration (Embed
  only) works with **NVIDIA, AMD and Intel**; see [AMD GPUs](#amd-gpus)
  for the one platform that needs care. Deleted files go to the
  Recycle Bin, the freedesktop Trash, or `~/.Trash` respectively, never a
  permanent delete on any of them.
- **Python 3.9+** (tested on 3.14). No admin rights, no installer.
- Packages, by stage:

| Stage | Needs |
|-------|-------|
| Collect | Pillow |
| Collect (HEIC/HEIF phone photos) | pillow-heif *(optional)* |
| Analyze | Pillow + numpy |
| Analyze (crop detection) | OpenCV *(recommended)* |
| Embed | torch + transformers |

  By hand on Windows that is
  `py -3 -m pip install --user pillow numpy opencv-python-headless`, and
  the same with `python3` on a Linux distro that permits it.

  **Several no longer permit it, and Arch is one of them.** Arch, Debian
  12+, Ubuntu 23.04+, Fedora 38+ and Homebrew all mark their Python as
  *externally managed* ([PEP 668](https://peps.python.org/pep-0668/)), and
  pip refuses to install into it at all. `--user` is refused too. That is
  the part that catches people out, and so is `pip uninstall`. There, the
  install **must** go into a virtual environment. `./imgdedup.sh setup`
  detects this, offers to create one beside the toolkit, and installs into
  it; every launcher then prefers that `.venv` over the system Python
  automatically, so nothing else changes.

  **`pip` may not be there at all**, which is a different problem with a
  different fix. It is packaged separately from Python on most distros, `python-pip` on Arch, `python3-pip` on Debian/Ubuntu, `python314-pip`
  and friends on openSUSE, so a base install has none, and every pip
  command simply reports `No module named pip`. Setup and
  `./imgdedup.sh doctor` both check and name the package. Where pip is
  missing but `venv` works, a virtual environment supplies its own, so the
  venv route fixes both problems at once. Debian and Ubuntu are the
  exception worth knowing: they split `ensurepip` into `python3-venv`, so
  `python3 -m venv` imports and *then* fails until that package is added.

**Easiest: let the toolkit install it.** Setup detects your GPU
(NVIDIA / AMD / Intel), asks which PyTorch build you want, shows the exact
pip command and waits for a yes. It never installs anything silently:

```
Windows        Check-Image-Tools.bat      (offers setup when something is missing)
Linux / macOS  ./imgdedup.sh setup
```

The per-stage launchers do the same: if a stage cannot run, they offer
setup instead of just failing.

By hand, PyTorch is installed separately because the right build depends on
your hardware; pick one, then `pip install transformers` either way:

| Hardware | Command |
|---|---|
| CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA (CUDA) | `pip install torch --index-url https://download.pytorch.org/whl/cu132` |
| AMD on **Linux** (ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm7.2` |
| Intel (XPU) | `pip install torch --index-url https://download.pytorch.org/whl/xpu` |
| AMD on **Windows** | not from PyPI — see [AMD GPUs](#amd-gpus) |

> Those suffixes **move**: `cu128` became `cu132`, `rocm6.4` became
> `rocm7.2`. Setup reads the current list from download.pytorch.org rather
> than trusting a number printed in a README, so prefer it to copying these.
> (It also sorts them numerically — `rocm7.14` is *newer* than `rocm7.2`,
> which string and float comparison both get backwards.)

> **Python 3.14 note:** the old `cu121` index has no 3.14 wheels at all.

### AMD GPUs

**Linux is straightforward.** The ROCm index above carries cp310–cp315
wheels, so any supported Python works. You need the amdgpu/ROCm kernel
driver, `rocm-smi` running is the sign it is there. One thing that looks
wrong but isn't: a ROCm build still reports its device as `cuda`, because
HIP deliberately reuses that namespace. `Device: cuda (Radeon …)` is
correct.

**Windows is awkward, and not because of this tool.** AMD publishes
ROCm-for-Windows wheels only at `repo.radeon.com`, only for **Python 3.12**,
and only as full-ABI `cp312` builds. Python 3.13/3.14 cannot load them. That is a wheel-ABI fact, not a flag: `--ignore-requires-python` skips the
check and the import then fails anyway. Upstream PyTorch ships no ROCm
wheels for Windows at all.

Three honest options:

1. **Install Python 3.12 alongside** your main Python, run setup with it,
   and point *only* the Embed stage at it, `set IMGDEDUP_PYTHON=C:\Path\To\Python312\python.exe`. The toolkit already
   resolves an interpreter per stage, so Collect and Analyze stay on 3.14.
   You also need AMD's stated graphics driver (26.2.2 for ROCm 7.2.1).
2. **Use the CPU build.** Embedding a few thousand images is a coffee, and
   it is resumable, a perfectly reasonable choice.
3. Not recommended: `torch-directml`. In maintenance mode, last release
   2024, hard-pins `torch==2.4.1`, is *also* cp312-or-older, and is not a
   drop-in `cuda` device, so it costs the same Python constraint while
   adding an incompatible device API.

Setup detects which case you are in and says so, instead of printing a
command that cannot work.

Not sure what you have? **Run the doctor first** (`Check-Image-Tools.bat` /
`./imgdedup.sh doctor`). It lists every Python on the machine, tests each
one *functionally*, reports your GPUs and whether a compute driver is
actually present, and offers to install what is missing. torchvision is
probed but **not needed**. It is only reported because a broken install of
it takes transformers down (see [Troubleshooting](#troubleshooting)).

---

## How to run

**The short way: drag your folder onto `Find-Duplicates.bat`.** It runs
all three stages in order and opens the report at the end. If PyTorch is
not installed it says so and carries on with pixel comparison alone,
rather than stopping — you lose the pass that catches recoloured and
heavily cropped copies, not the tool. On Linux/macOS the same thing is
`./imgdedup.sh collect <folder>` then `embed` then `analyze`.

Then open the report, click the pictures you want gone, press **Download
duplicates-list.txt**, save it over the old one, and run the recycler.
Arrow keys move, <kbd>X</kbd> toggles, and the keeper of a group cannot be
marked at all, so no group can be emptied by accident.

The stage-by-stage route below still works and is worth knowing when a run
goes wrong, since it lets you re-run one stage without redoing the others.

1. Copy the whole toolkit folder anywhere (or leave it where it is, the
   launchers find their scripts next to themselves).
2. **Drag the folder you want scanned onto `Collect-Image-Inventory.bat`**
   (or double-click it to scan the folder the .bat sits in). The scan runs
   on a thread pool; ~1,600 images take well under a minute on an SSD.
3. *(Optional but worth it)* drag the new `image-inventory.jsonl`, or its
   folder, onto `Embed-Images.bat`. First run downloads the CLIP model
   (~600 MB, cached forever after). **Skip this and cropped copies are not
   found** — see [Skipping the embed stage](#skipping-the-embed-stage).
4. Drag the same folder onto `Analyze-Inventory.bat`.
5. Open `<name>-report.html`, look at the pictures. Each group is labelled
   **`cluster N`**, the same number the list uses, search the .txt for
   `cluster 91` to land on the same group (clicking the number in the
   report links to it). Then edit `<name>-list.txt` if you disagree with
   any decision, the first character of a line is the whole interface:
   `X` deletes, `.` keeps. **Every cluster member is an editable line, the
   suggested keeper included, prefer a different copy? Just move the `X`.**
6. Run `Recycle-Duplicates.bat` (Windows) or `./Recycle-Duplicates.sh`
   (Linux/macOS), read the preview, answer `y`. Both just run
   `Recycle-Duplicates.py`, which is where every safety rule lives.
   If Tier B is untouched it asks first (once you have marked any Tier B
   file yourself, it stops asking and just uses your list):
   `Enter` for Tier A only (the default, and what happens with piped input
   or no terminal), `b` to include them, `q` to stop and review. Either way
   they go through the same survivor and hash checks as everything else.

On Linux/macOS, steps 2–4 are `./imgdedup.sh collect <folder>`,
`./imgdedup.sh embed <folder>`, `./imgdedup.sh analyze <folder>`.

Deleted something you regret? It is in the **Recycle Bin / Trash**.
Restore it.

---

## The pipeline in detail

### Stage 1 — Collect

Walks the folder tree, 17 extensions, the ones that actually turn up in an
image gallery: `.jpg .jpeg .jfif .jpe .png .apng .webp .gif .tif .tiff .bmp
.tga .qoi`, plus `.heic .heif .hif .avif` when pillow-heif is present. Icons,
cursors, Photoshop files, game textures and the legacy encodings are
deliberately excluded (an asset folder reuses the same texture on purpose,
so every "duplicate" found there is intended); `collect-image-inventory.py`
lists each exclusion and the reason for it. Camera RAW is not supported at
all, Pillow cannot decode it. It writes one JSON line per
image: relative path, byte size, mtime, **SHA-256**, format, dimensions,
key EXIF fields (timestamp, camera, software, orientation), a **JPEG
quantization fingerprint** (`qsum`, lower means less recompressed), any
**AI-generation text chunks** found in PNGs (Stable Diffusion / ComfyUI
`parameters`), and a 128 px thumbnail, rotation-corrected, stored losslessly
when that is smaller than JPEG.

- **Read-only.** It never touches an image. Its own output files are the
  only thing it writes.
- **Parallel.** Hashing, decoding and thumbnailing run on a thread pool
  (`--workers`, default min(CPU cores, 8)); each file is read from disk
  once. Records stream out in scan order, so the output is deterministic.
- **Resume.** Re-running offers to reuse the previous inventory: files with
  unchanged size+mtime are carried over without re-reading, so a re-scan
  after a cleanup takes seconds. Unreadable files are always retried, installing `pillow-heif` and re-running with resume fills in exactly the
  HEICs. The newest previous inventory wins, records from a different
  `--thumb` are never reused, and superseded inventory files are listed as
  safe to delete (it will not delete them itself).
- **Portable output.** Paths are stored with forward slashes, so an
  inventory made on one OS resolves on another.
- **Big libraries.** Output rolls into `.part2`, `.part3`… at ~200 MB;
  every later stage understands parts automatically.
- **Skipped folders:** anything starting with `.`, plus `venv`, `.venv`,
  `__pycache__`, `node_modules`, `$RECYCLE.BIN`, `System Volume
  Information`, `_inventory`, and the Linux trash/system dirs (`.Trash-*`,
  `lost+found`, `.cache`, `.thumbnails`).
- Corrupt files are logged as unreadable, counted by extension at the end,
  and never crash the scan. `Ctrl+C` aborts safely; a partial inventory is
  still usable.

### Stage 2 — Embed

Opens each **original** image and computes a CLIP embedding
(`openai/clip-vit-base-patch32` by default, `--model` to change). Output is
keyed by the file's SHA-256, so renaming or moving images does not
invalidate it, and byte-identical files share one vector.

Why bother: pixel comparison cannot see that a cropped image is "the same
picture", the pixels genuinely differ. CLIP can. In testing this stage
caught a crop whose pixel difference was **nine times** the duplicate
threshold; no pixel method would ever have flagged it.

#### Skipping the embed stage

Everything still runs without it, and exact, re-encoded, resized, rotated
and mirrored duplicates are all still found — those are pixel work.
**Cropped copies are not.** That is not a degradation, it is close to a
total loss, and the report gives no sign of it: a Tier B section still
appears, built from whatever crops happened to survive.

The reason is the prefilter. Before any pixel comparison, all pairs are
swept on an 8x8 colour signature and only those within `--sig-cut` (8.0 by
default) go further; with embeddings, CLIP nominates candidates as well.
A crop moves the signature far more than a re-encode does, so it is the
CLIP path that carries crops through. Measured over 200 real images, the
share of cropped copies whose signature exceeds the cut:

| crop keeps | median signature distance | never reaches the crop matcher |
| ---------- | -------------------------: | -----------------------------: |
| 97%        | 3.15                       | 0%                             |
| 90%        | 10.53                      | **74%**                        |
| 80%        | 20.66                      | 97%                            |
| 70%        | 30.20                      | 98%                            |

So anything trimmed by more than a few percent needs CLIP. Raising
`--sig-cut` is not a substitute: catching 90% crops needs roughly 30, and
the sweep already keeps 96,255 pairs out of 662 million at 8.0.

- **Pipelined.** Images are decoded and preprocessed on a thread pool that
  runs ahead of the model, so the GPU never waits for the disk; JPEGs
  decode via libjpeg draft mode at ~4x the model input (`--no-draft` opts
  out). EXIF orientation is applied before embedding, the same correction
  Stage 1 applies to thumbnails.
- **GPU-first, honestly.** The launcher probes for a Python whose torch can
  actually drive a CUDA GPU before settling for a CPU build, and when the
  embedder still lands on CPU it prints *why*, the usual culprit is the
  CPU-only torch wheel (`+cpu` in the version), which can never use a GPU
  regardless of code. `--device cuda` forces the matter (clean error if
  impossible), `--device cpu` opts out of the lecture.
- **Resumable.** Re-running skips every SHA already embedded and appends;
  failed files are retried every run, like the collector. One pathological
  image cannot kill the run, a failed batch is retried one image at a
  time and only the offender is logged. If a byte-identical twin of a
  missing file exists, the twin is read instead.
- **Refuses to mix models.** A file built with a different model (or
  vector width) stops the run with instructions instead of silently
  corrupting the results.
- GPU is used automatically when available; CPU works and is just slower
  (roughly 5–15 images/second).

### Stage 3 — Analyze

Sweeps **every possible pair** on an 8×8 colour signature through BLAS with
a provably lossless bound (`--self-test` asserts set equality against brute
force), adds each image's nearest CLIP neighbours, re-scores survivors at
full thumbnail resolution on a thread pool, and sorts findings into two
tiers. Rotated/mirrored, grayscale/recoloured and cropped copies get their
own detectors, each applied only to pairs every cheaper test rejected.
Images whose SHA-256 is missing from the embeddings file are judged by
pixels alone, and byte-identical files are always clustered:

| Tier | Meaning | Evidence required | Pre-set |
|------|---------|-------------------|---------|
| **A — duplicate** | Same picture: re-encoded, resized, format-converted | mean pixel difference ≤ 4/255 **and** (when embeddings exist) CLIP cosine ≥ 0.99 — two independent measures must agree | `X` |
| **B — crop / variant** | Structurally the same, genuinely different pixels: crops, rotations, recolours, inpaints, re-rolls — plus pixel-identical pairs that CLIP disputes | CLIP ≥ 0.995, or containment ≥ 0.92 with CLIP ≥ 0.94, or luma/orientation match at duplicate level | `.` always |

Within a cluster the **keeper** is chosen by highest resolution, then
largest file, then finest JPEG quantization (`qsum`), the
least-recompressed copy, not just the biggest. It is only a *suggestion*:
every member of every cluster is an editable line in the list.

Numbers you will see in the report:

- **mad**: mean absolute pixel difference, 0–255. 0 is identical; a JPEG
  re-save lands around 1–3; unrelated images land in the tens.
- **cos**: CLIP semantic similarity. Beware its high baseline: in a
  single-genre library *random* pairs average ~0.76 and can exceed 0.98,
  which is why Tier A demands 0.99 *and* pixel agreement, not either alone.
- **ncc**: how well the smaller image matches somewhere *inside* the
  larger one (1.0 = perfect containment). This is the crop detector; it
  needs OpenCV, and the run tells you if that is missing.

Long runs print per-stage elapsed times, so you can see where the time
goes. If the folder is clean, Analyze writes **only the report** (which
says so), no selection list and no recycler. Stale outputs from earlier
runs are named for manual removal; it deletes nothing itself.

### Stage 4 — Recycle

The generated `Recycle-Duplicates.py` (run via its `.bat` on Windows or
`.sh` elsewhere. Both only locate a Python) is deliberately paranoid, and
it enforces safety **per cluster**, not per line:

1. a file marked `X` is deleted only while at least one *other* member of
   its cluster stays unmarked, still exists, and still matches its
   scan-time SHA-256, the *surviving witness*;
2. a cluster with every member marked `X` is refused outright ("unmark at
   least one");
3. the file itself must still match its scan-time size and SHA-256;
4. a file shared between two clusters is deletable only in the cluster
   that owns its line, so one `X` can never become two deletions and a
   refusal cannot be bypassed through a reference row;
5. lines whose path matches nothing in the manifest are warned about and
   ignored rather than guessed at;
6. on Windows, paths too long to recycle are **refused**, past ~260
   resolved characters the OS deletes permanently while reporting success,
   so the recycler will not attempt them at all.

A file that belongs to two clusters, an exact duplicate that is also the
uncropped original of something, gets its editable line in exactly one;
the other shows it as an `also in cluster N (edit it there)` comment, so
two lines can never disagree about the same file.

This is what makes keeper-swapping safe: whichever copy you leave unmarked
becomes the witness, and no edit, deliberate or accidental, can make a
cluster lose its last verified copy. The script shows the count and total
size, asks `y/N`, and moves files to the OS trash, never a permanent
delete. The exit code is the number of failures.

Trash backends: Windows uses the same Recycle-Bin call Explorer makes;
Linux implements the freedesktop.org Trash specification (per-volume trash
directories, `.trashinfo` records, atomic name claiming); macOS moves to
`~/.Trash` (restorable by hand; Finder's "Put Back" needs private metadata
no dependency-free tool can write). A file on a different filesystem from
its trash directory is refused, not silently copied.

---

## Safety model

Two layers, because they protect against different things.

**The generated recycler protects against the world changing**, files
edited, moved, or replaced between scan and delete. That is the re-hashing
above.

**The analyzer protects against itself.** Before writing anything, it
asserts four invariants over its own output:

1. every file appears **at most once** as a deletion candidate;
2. a file chosen as a **keeper never appears as a candidate** anywhere;
3. every candidate's keeper exists and is not itself a candidate;
4. no cluster has fewer than two members.

A violation aborts the run with nothing written. This is not theoretical:
crop relationships chain (A is a crop of B, B of C), and an earlier
revision built Tier B from raw pairs, leaving files marked KEEP in one
entry and delete-candidate in another. The invariants exist because that
happened, and `--self-test` re-proves all of them in a second.

---

## Command line

Every `.bat` accepts one dragged-and-dropped argument; `imgdedup.sh` passes
everything after the subcommand straight through. The Python scripts offer
more:

**collect-image-inventory.py** `[folder]`

| Option | Default | Meaning |
|---|---|---|
| `--thumb N` | 128 | thumbnail max side, px |
| `--workers N` | auto (≤ 8) | parallel hash/decode/thumbnail threads |
| `--lossless-thumbs` | off | also try a lossless WebP thumbnail and keep it when smaller. ~2.2× slower. Across 36,410 images it changed no duplicate decision, which is why it is no longer the default — but a collection that is mostly screenshots, UI captures or pixel art has a far larger share of qualifying thumbnails, and this keeps their pixels exact |
| `--split-mb N` | 200 | roll output to a new `.partN` past this size |
| `--resume` / `--no-resume` | ask | reuse previous inventory for unchanged files |
| `--out FILE` | `<folder>/image-inventory.jsonl` | output path |
| `--share` | off | also copy the output into the shared folder for an AI assistant |
| `--mirror-dir DIR` | off | copy the output to a custom folder instead |

**embed-images.py** `<inventory.jsonl | folder>`

| Option | Default | Meaning |
|---|---|---|
| `--model NAME` | `openai/clip-vit-base-patch32` | HF model id (~600 MB first download) |
| `--root DIR` | from inventory header | override the image folder |
| `--batch N` | 64 GPU / 8 CPU | batch size |
| `--workers N` | auto (≤ 8) | decode/preprocess threads feeding the model |
| `--no-draft` | off | decode JPEGs at full resolution (slower) |
| `--fp16` | off | float16 on the GPU (~2.9× faster). Vectors shift by up to 0.0006 pairwise cosine — enough to move a pair sitting exactly on the Tier A floor into review. Recorded in the header; a resumed file of the other precision is refused |
| `--device D` | auto | `auto` / `cuda` / `cpu` — auto prefers GPU and explains any fallback |
| `--share` / `--mirror-dir` | off | as in Collect |

**analyze-inventory.py** `<inventory.jsonl | folder>`

| Option | Default | Meaning |
|---|---|---|
| `--tier-a-mad N` | 4.0 | pixel-difference ceiling for Tier A |
| `--tier-a-cos N` | 0.99 | CLIP floor for Tier A |
| `--tier-b-mad N` | 4.0 | pixel floor for the review tier |
| `--tier-b-cos N` | 0.90 | CLIP floor before a pair may enter Tier B. Matches the floor that nominates neighbours, so no pair is scored and then discarded on it. Raise to 0.94 for a shorter review list that misses about half of all moderate crops |
| `--sig-cut N` | 8.0 | signature prefilter ceiling (default auto-raises to 2× `--tier-a-mad`; an explicit value is used verbatim) |
| `--clip-neighbors N` | 16 | nearest neighbours each image contributes from the embeddings; bounds the candidate set on large libraries (a binding cap is reported). Cannot cost a Tier A candidate — the cut line never approaches the 0.99 that tier needs |
| `--no-orient` | off | skip rotation/mirror matching — faster, but rotated and mirrored copies are missed |
| `--no-embeddings` | off | ignore embeddings even if present |
| `--self-test` | — | run the invariant + sweep-equality tests and exit |

**Interpreter choice.** The `.bat` launchers call `_pick-python.bat`;
`imgdedup.sh` probes `python3.14` … `python3.9`, then `python3`/`python`.
Both try `IMGDEDUP_PYTHON` first (always wins) and probe *functionally*, the candidate must actually import and call into the packages the stage
needs. To pin an interpreter for everything:

```
set IMGDEDUP_PYTHON=C:\path\to\python.exe        (Windows)
export IMGDEDUP_PYTHON=/usr/bin/python3.12       (Linux/macOS)
```

---

## Working with an AI assistant (optional)

The toolkit is fully standalone, no stage needs an AI assistant, a network
connection (after the one-time model download), or anything outside your
machine. **By default nothing is copied anywhere.**

If you *want* a second pair of eyes, asking your AI assistant (e.g.
Claude) to sanity-check a borderline cluster or tune thresholds, pass
`--share` to Collect or Embed. That drops a copy of the output into a
shared data folder the assistant can be pointed at (per-platform default;
`IMGDEDUP_SHARE_DIR` or `--mirror-dir <path>` to choose your own).

The shared copy only ever contains inventories and embeddings, hashes,
metadata and small thumbnails, never your original image files.

---

## Troubleshooting

Newest lessons first. Most were discovered the hard way, on this very
machine.

- **After reinstalling torch, Embed dies with "entry point not found" /
  "DLL load failed while importing _C" pointing at `torchvision\_C.pyd`.**
  torchvision's compiled extension is linked against one exact torch
  build, so replacing torch leaves it stale. It still *looks* installed,
  which is why transformers picks it up and dies. The embedder detects
  this, ignores the broken torchvision and carries on via Pillow, since
  nothing in this toolkit needs torchvision. Clean it up anyway:
  `pip uninstall torchvision`, or reinstall the matched pair with
  `pip install --force-reinstall torch torchvision --index-url
  https://download.pytorch.org/whl/cu132`. Rule of thumb: torch and
  torchvision must be reinstalled together, always.
- **Embedding runs on the CPU although I have an Nvidia GPU.** Almost
  always the installed torch is the CPU-only wheel. The version says so
  (`2.x.y+cpu`), and no setting can route a `+cpu` build to a GPU. The
  doctor and the embedder both name the fix:
  `pip uninstall torch` then
  `pip install torch --index-url https://download.pytorch.org/whl/cu132`.
  Only the Embed stage uses the GPU at all; Collect and Analyze on CPU is
  correct, not a bug.
- **A tool picked a Python that "has" a package, yet imports fail, or the
  doctor shows `[EMPTY]`.** A package whose files were deleted but whose
  *folder* survived still imports as an empty namespace package, so plain
  `pip list` / `import` checks lie. Every probe in this toolkit is
  functional, and the doctor reports such corpses as `[EMPTY]`. On
  Windows, if a folder under `Downloads` keeps losing files, check whether
  **Storage Sense** is set to auto-clean Downloads.
- **Analyze found 0 duplicates and wrote "only" a report.** Intended on a
  clean folder. If the run lists stale `Recycle-Duplicates.*` / list files
  from an earlier pass, delete them; they no longer describe reality.
- **`pip install torch` says "no matching distribution" on Python 3.14.**
  You are on the `cu121` index; it has no 3.14 wheels. Use `cu132` (or
  `/cpu`). These suffixes move, `cu128` became `cu132`, so prefer the
  setup helper, which reads the current list from the index instead of
  trusting a number written here.
- **Embed refuses to run: "built with a different model".** Deliberate, vectors from different models are not comparable. Re-run with the old
  `--model`, or move/delete the embeddings file to start fresh.
- **The run says OpenCV is missing.** Crop detection degrades to what CLIP
  alone can see. `pip install opencv-python-headless` fixes it (on a
  distro-managed Python, `./imgdedup.sh setup`; see PEP 668 above).
- **HEIC/HEIF files show as unreadable.** Install `pillow-heif`, then
  re-run Collect with `--resume`, only those files are re-read.
- **Analyze is slow on a huge library.** It prints per-stage times, so you
  can see where it is. `--no-orient` skips rotation matching; on 36k
  images the full default run measured ~19 minutes on an adversarial
  synthetic set, and real libraries are faster. Since v4.3.7 the
  rotation sweep is banded like the main one, so `--no-orient` buys much
  less than it used to — try it only if the printed orientation stage is
  actually where your run is spending its time. Embeddings skip that
  stage outright: CLIP already nominates rotated and mirrored copies.
- **Two inventories ended up in one folder.** Collect never overwrites: a
  second run writes `image-inventory-2.jsonl`. Embed and Analyze pick the
  newest automatically; the recycler and list carry the same `-2` suffix
  so they stay bound to each other.
- **A cluster was refused with "no copy would survive", but its own lines
  show a `.`.** One of its members is marked `X` in a different cluster, the same file can appear in two, and the refusal names the file and
  the cluster where it is editable. Unmark it there.
- **A file I marked `X` was skipped or refused.** Read the printed reason: its bytes (or its keeper's) changed since the scan, the keeper is gone,
  or on Windows its path is too long to recycle safely. That is the guard
  doing its job.

---

## Notes & caveats

Comparisons run on 128 px thumbnails, which is what makes a million-pair
sweep fast. The thresholds were calibrated against real libraries with
contact-sheet review: a JPEG re-save scores mad ≈ 1–3, genuinely different
images 10 or more. Borderline pairs get surfaced in Tier B rather than
decided.

Tier B is a review queue, not a verdict. Same-prompt AI re-rolls, inpaint
variants and crops are *structurally* alike, and only you know which of
them you count as "the same picture". That is why every line there starts
pre-set to `.`.

What it will miss: crops retaining under about a third of the original,
and, without the Embed stage, brightness or colour edits, because nothing
nominates those pairs for scoring in the first place. Against a 20-case
truth set of known same-picture transformations the pipeline finds
**20/20 with embeddings and 15/20 without**, so run the semantic stage.

Animated GIF, WebP and APNG are compared by five frames sampled across the
whole animation rather than by the first one alone. Frame 0 is not enough:
two completely different animations that happen to start the same way
score a perfect pixel match, and so does a still lifted out of a GIF. Both
used to be reported as automatic duplicates. An animation and a still are
now never auto-deleted against each other, and neither are two animations
whose frames diverge; those go to the review tier instead. Only animations
pay for the scan, and not much: a 30-frame GIF costs about what one
900×900 photo does, and the frame fingerprint adds roughly 440 characters
beside a thumbnail of several KB.

Paths in Cyrillic, Japanese, emoji, typographic quotes and non-printable
characters all work. Each of those became a test case after breaking
something once.

### Why video is out of scope

Not mainly because decoding it is expensive. A GIF is an image file Pillow
already opens, so extending to it cost one seek loop; video needs a decoder
this toolkit does not have. OpenCV's wheels bundle FFmpeg, but OpenCV is
*optional* here, so video support would either become a hard dependency or
work only sometimes, and working only sometimes is worse than not working.

The deeper problem is that the pixel-plus-CLIP model does not describe
video at all. The same clip in two containers, at two frame rates, from a
resolution ladder, or trimmed a second differently is the same video, and
every one of those pairs would score as unrelated.

And the safety model does not survive the trip. This whole design rests on
you *seeing* a thumbnail of each file before approving a deletion, which
one still frame of a ten-minute video cannot give you. Deduplicating video
properly is a different tool, not a flag on this one.

---

## Tests

`analyze-inventory.py --self-test` runs the built-in suite: the four
invariants, emission-plan rules, the pair-chaining case that caused the
original keeper/candidate bug, and exact set-equality of the BLAS sweep
against brute force (including pairs placed deliberately at the band
edge). Every guard was validated by *mutation testing*, deliberately
re-breaking the code and confirming the check fires.

The stages are tested end-to-end on synthetic image sets (exact copies,
re-encodes, resizes, crops, rotations, mirrors, grayscale copies, EXIF
rotations, corrupt files, Unicode filenames) and the recycler against a
13-scenario safety harness, including a verified Recycle-Bin round-trip
and the freedesktop trash layout exercised against real files.

---

## Version

**v4.3.7** (2026-08-12). Full history, including every bug and what it
taught the tool, lives in [CHANGES.md](CHANGES.md).

---

## License

[MIT](LICENSE). Use it, change it, ship it; just keep the copyright
notice.

One clause is worth reading rather than skimming, because this tool moves
your files: the software is provided **as is, without warranty**. It is
built to be careful. Nothing is deleted without a list you edited
yourself, and everything goes to the recycle bin or trash rather than
being erased, but the guarantees end at the license text. Keep backups
of anything you cannot replace.

Third-party dependencies, Pillow and NumPy, plus optionally OpenCV,
PyTorch, Hugging Face Transformers and pillow-heif, are installed from
PyPI under their own licenses, and none of their code is bundled here.

One nuance if you ever ship a *frozen bundle* rather than source: a few
of those wheels embed binaries under copyleft terms
(`opencv-python-headless` carries LGPL-2.1 FFmpeg, `pillow-heif` carries
GPL-2.0 x265). That constrains a redistributed binary. It does not
constrain this source release, which ships no one else's code.

---

*The pipeline never deletes on its own, the human holds the list, the
script holds the trash, and the trash holds everything else. Built with
Claude, 2026.*
