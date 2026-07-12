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
