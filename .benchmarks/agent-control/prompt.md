I want to reclaim space by deduplicating the images under
`/home/solteris/Apps/Asset Pack V2` and all of its subdirectories. Work with
me interactively to find exact copies and visually equivalent copies that may
have been re-encoded, resized, cropped, rotated, mirrored, or animated. Help
me choose the best-quality copy to keep.

Do not use or search for an existing image-deduplication application or
project. Approach this as you normally would with the shell, Python, installed
general-purpose libraries, and scripts you write yourself. You may ask useful
clarifying questions and show me samples when human judgment would improve the
result. Explain important tradeoffs plainly enough that I can make them.

The source tree is read-only: never modify, move, rename, or delete a source
file. Write scripts, caches, and results only inside your current benchmark
workspace. Before we finish, produce:

1. `manifest.jsonl`, one proposed action per line, with corpus-relative fields
   `group`, `keeper`, `candidate`, `action`, `relation`, `confidence`, and
   `evidence`. `action` is `delete` or `review`; `relation` is `exact`,
   `reencoded`, `resized`, `cropped`, `rotated`, `mirrored`, `animation`, or
   `other`.
2. `report.md`, explaining methods, thresholds, failures, blind spots,
   elapsed work, and how to reproduce or resume it.

Do not stop at a plan. Once we have resolved genuinely useful questions,
carry the analysis through to reviewable results.
