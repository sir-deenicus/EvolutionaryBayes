# Repository instructions

## Staged implementation documentation

When completing a stage from any staged implementation plan in this repository,
update that plan's primary documentation in the same change as the code:

1. Edit the existing explanatory sections in place so they describe how the
   implementation actually works. Replace stale future tense and planned
   behavior rather than adding a second, conflicting explanation.
2. Edit the applicable Future work, improvements, roadmap, or staged-plan
   section in place. Mark the completed stage accurately, update assumptions or
   later stages affected by the work, and remove future-work items that the
   completed stage resolved.
3. Append a dated entry to an `Implementation history` section in the primary
   document. Create that section if it does not exist. Use an ISO date
   (`YYYY-MM-DD`), name the stage, and record:

   - the behavior and API implemented;
   - important correctness, memory, or performance decisions;
   - tests, allocation checks, and benchmarks run;
   - material deviations from the plan and the reason for them.

Do not describe a stage as complete or add its completion-history entry until
all of that stage's exit criteria pass. Keep the history append-only; correct a
past entry with a new dated note rather than silently rewriting the record.

## Benchmark result history

When a stage or material optimization runs benchmarks, keep the complete
results in one consolidated benchmark-results Markdown document at the
repository root. Use a stable filename such as `BENCHMARK_RESULTS.md`; inside
it, group runs under ISO-dated headings and give every stage its own subheading.
Record the source revision and working-tree qualification, build configuration,
runtime, processor, seed, workload parameters, elapsed time, throughput,
retained memory where relevant, and allocation measurements that apply. Link
the consolidated result from the primary plan or design document.

Also record material execution conditions such as CPU or power throttling,
background CPU load, thermal state when known, and whether timings came from a
shared or otherwise busy machine. Under unstable load, treat absolute
throughput as provisional, use interleaved before/after runs with multiple
samples, and base optimization claims on medians rather than one timing.

Do not overwrite earlier results. Append a new dated run section, or a clearly
labelled additional run under the same date, so performance history and changed
methodology remain auditable.

## Commit protocol

Before creating a commit:

1. Export the complete staged diff to `changes.txt` (for example,
   `git diff --cached --output=changes.txt`). Do not include unstaged changes.
2. Read and summarize `changes.txt`, identifying the meaningful behavior,
   design decisions, and motivation represented by the staged changes.
3. Generate the commit message from that summary. The message must cover the
   actions in `changes.txt` and explain why they matter to a reader unfamiliar
   with the project, without becoming either shallow or overlong.
4. Commit only the staged changes, using the generated message.

## Write Commit Messages for Future Readers

- A commit message must let a reader understand the meaningful change and the relevant design choice
  without reopening the diff or reconstructing the author's reasoning.
- The subject names the outcome, not the file operation. Add a short body when it is needed to explain
  what changed and why that shape was chosen.
- Be concrete rather than vague: name the important structure or behavior introduced, preserved, or
  changed.
- Let length follow missing context. Use the shortest message that makes the intent clear; do not force
  a one-line summary when a few brief body lines are necessary, and do not turn the message into a
  diff narration or mini-specification.
- Keep the subject scannable (roughly 50–72 characters) and wrap body lines around 72 characters.
