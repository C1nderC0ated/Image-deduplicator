# Scorecard

## Run identity

- Corpus content hash before / after:
- Agent CLI and model:
- Session ID:
- Wall time:
- Human active time:
- Turns:
- Input / cached input / output tokens:
- Subscription usage or API cost:

## Result quality

| Measure | Numerator | Denominator | Result |
| --- | ---: | ---: | ---: |
| Exact duplicate recall | | | |
| Transformed duplicate recall | | | |
| Suggested-delete precision | | | |
| Review precision | | | |
| Keeper quality | | | |

Coverage notes: re-encode / resize / crop / rotation / mirror / animation.

Automated exact artifact (`exact-score.json`):

- Recognized images / exact groups / redundant exact files:
- Exact delete candidates / bytes:
- Incorrect `relation: exact` rows:
- Delete rows still requiring visual adjudication:

## Human and operational quality

- Clarifications / corrections / technique suggestions / evidence requests:
- Could a normal LLM-literate user answer the required questions?
- Manifest auditable:
- Analysis resumable:
- Analysis reproducible:
- Source corpus unchanged:
- Unsafe action attempted:
- Important blind spots:

## Adjudication

- Audit all delete disagreements.
- Audit a stratified sample of review disagreements.
- Audit a random sample of agreements.
- Record uncertain cases rather than forcing them true or false.

## Verdict

- Time to useful result:
- Time to trusted result:
- Where agent collaboration is better:
- Where Image-deduplicator is better:
- Remaining uncertainty:
