# Human + frontier-agent comparison

This is not an autonomy benchmark and not a contest between one prompt and a
finished pipeline. It asks a product-level question:

> Can a reasonably experienced LLM user, working interactively with a
> frontier coding agent, reach a similarly useful and safe image-deduplication
> result without this project?

Clarifying questions, iterative inspection, threshold changes, scripts written
by the agent, and human review are all valid. They are also measured. A run is
not penalized merely for needing input; it is penalized when the required
input is unusually expert, expensive, unsafe, or fails to produce a good
result.

## Fixed setup

- Host corpus: `/home/solteris/Apps/Asset Pack V2` (visible to the agent only
  as `/corpus`)
- Initial corpus: 17,579 recognized image-extension files, 1.645 GiB
- Primary agent: Codex CLI with `gpt-5.6-sol`, high reasoning, through the
  user's subscription rather than Cursor/API credits.
- Cost-aware replication: `gpt-5.6-terra`, high reasoning, only after the
  primary protocol is stable.
- Optional outside adjudicator: Claude Opus (maximum-capability Opus only),
  used occasionally on a disagreement sample rather than on every step.
- Operator: a human familiar with ordinary LLM collaboration. The repository
  owner should perform at least one canonical run because operator skill is
  part of the question being measured.

The agent may use the shell, Python, installed general-purpose libraries, and
new scripts it writes. It may not inspect this repository, its inventories,
its thresholds, or its output before submitting its own result.

The current project-side reference on this exact corpus is recorded locally
under `.baseline/asset-pack-project/` and is not mounted into the agent run:
17,579 recognized paths, 291 exact groups / 401 redundant exact files, 734
Tier A clusters / 1,235 suggested drops (102.9 MB), and 910 Tier B clusters /
1,616 candidates. Collection took 17 s, default fp32 embedding ran at 294.3
img/s, and analysis took about 10.5 s after loading. These are candidate-oracle
figures; disagreement adjudication still determines comparative quality.

## Safety and isolation

`benchmark.py launch` uses Bubblewrap to mount a minimal Linux filesystem.
The agent gets a writable `/work`, read-only `/corpus`, system tools, and
private Codex session state. The project repository, prior runs, and the rest
of the user's home directory are not mounted. The harness stores workspaces
outside the repository under `~/Benchmarks/image-agent-control/` by default;
`IMAGE_AGENT_BENCH_RUNS` can override that location.
The environment is cleared before launch, so editor project labels and API
credential variables cannot leak in; ChatGPT authentication is mounted
read-only into the isolated Codex state.

Before and after every run, the harness hashes a sorted stream containing
every entry's relative path, type, size, nanosecond mtime, mode, and SHA-256
file content (or symlink target). An unreadable entry prevents preparation;
a mismatch invalidates the run until explained.

No deletion trial is necessary to score a recommendation. The deliverable is
a manifest and report. If recoverability itself is later tested, use a copied
fixture and trash rather than permanent deletion.

## One run

Run these commands from `.benchmarks/agent-control/` (or prefix
`benchmark.py` with that path from the repository root).

1. Run `python benchmark.py prepare <date>-<model>`. It creates the external
   run workspace, copies the three templates, and records the corpus content
   fingerprint in public and private `run.json` state.
2. Record machine information and operator start time. The launcher records
   the fixed model, reasoning level, CLI version, and start/end timestamps in
   `run.json` automatically.
3. Run `python benchmark.py launch <date>-<model>`. This fixes the primary to
   subscription-backed `gpt-5.6-sol` at high reasoning, opens one persistent
   interactive session, and captures `terminal.typescript`; Codex's session
   JSONL lives in private harness state. Resume later with
   `python benchmark.py launch <date>-<model> --resume <session-id>`.
4. Collaborate normally until the operator considers the result ready for
   review. The operator may answer questions, ask for evidence, reject unsafe
   calls, request a sample, or suggest a generic technique they would have
   thought of without this repository.
5. Record every human intervention in `operator-log.md`. Do not retrospectively
   clean up awkward turns; interaction cost is part of the result.
6. Require `manifest.jsonl` and `report.md` before scoring.
7. Run `python benchmark.py validate <date>-<model>` to check the manifest
   contract, `python benchmark.py score-exact <date>-<model>` to generate
   byte-identical ground truth and recall, then
   `python benchmark.py verify <date>-<model>` to fingerprint the corpus
   again. A changed corpus invalidates the run until explained. Score
   transformed matches through stratified human review.

Do not impose a fixed turn limit. Record time-to-useful-result and
time-to-operator-confidence instead. The project itself requires review, so a
one-shot completion requirement would be an invalid comparison.

## Context and prompt-cache discipline

- Keep the stable task, safety rules, output contract, and corpus description
  in the unchanged initial prompt so repeated prefixes remain cacheable.
- Resume the same Codex session for related follow-ups. Do not paste prior
  manifests or terminal output back into the conversation when the agent can
  read the artifact by path.
- Put changing instructions at the end of a turn and ask for deltas rather
  than restating the task.
- Start a fresh compact session for independent adjudication or when earlier
  implementation history is irrelevant. Give it only the disputed sample and
  scoring rubric.
- Capture JSONL usage fields, including cached-input tokens when exposed.
  Record subscription usage separately from API-dollar cost; do not pretend
  one is the other.
- Prefer one well-scoped Sol run plus targeted resumed turns over many fresh
  frontier invocations. Terra is the normal lower-cost replication. Claude
  Opus is reserved for occasional cross-vendor adjudication.

This follows the current OpenAI guidance: stable reusable prefixes benefit
from caching, while lean prompts and relevant-only context improve both cost
and task quality.

## Required manifest

`manifest.jsonl` contains one object per proposed action:

```json
{"group":"g1","keeper":"relative/path.png","candidate":"relative/copy.jpg","action":"delete","relation":"reencoded","confidence":0.98,"evidence":"same composition; 0.997 visual score; keeper is lossless and larger"}
```

Required fields are `group`, `keeper`, `candidate`, `action`, `relation`,
`confidence`, and `evidence`. `action` is `delete` or `review`. `relation` is
one of `exact`, `reencoded`, `resized`, `cropped`, `rotated`, `mirrored`,
`animation`, or `other`. Paths are corpus-relative. The human-readable report
must state failures, blind spots, thresholds, runtime, and how to reproduce or
resume the analysis. Validation also requires one action per candidate, one
keeper per group, and no delete candidate that is used as a keeper elsewhere.

## Scoring

Keep separate numbers rather than collapsing safety into one headline score:

- Exact recall: byte-identical duplicate files found / byte-identical
  duplicate files present. SHA-256 supplies ground truth.
- Transformed recall: confirmed transformed duplicates found / confirmed
  transformed duplicates in a stratified union of both methods' candidates.
- Delete precision: confirmed safe deletions / all suggested deletions.
- Review precision: true relationships / review suggestions.
- Keeper quality: fraction whose suggested keeper survives human review.
- Coverage by relation: re-encode, resize, crop, rotation, mirror, animation.
- Human burden: turns, active operator minutes, clarification, correction,
  technique suggestion, evidence request, and manual-adjudication counts.
- Compute burden: wall time, CPU/GPU time where available, tokens, cached
  tokens, subscription usage, and API cost if any.
- Operational quality: auditability, resumability, reproducibility,
  recoverability, and source-tree safety.

The project output is a candidate oracle, not unquestioned truth. Audit all
method disagreements plus a random sample of agreements, with extra weight on
anything either method would pre-mark for deletion.

The files in `runs/pilot-2026-08-12/` are discarded protocol pilots. They used
Cursor Auto and incorrectly treated clarification as failure; they are kept
locally only so that mistake remains visible.
