# CFR benchmark results

## 2026-07-12

### Environment qualification

This machine was operating with approximately 65% throttling and typically
greater than 50% total CPU load from other work during these runs. Absolute
elapsed time and throughput therefore represent a constrained, noisy snapshot,
not peak hardware performance. They remain useful for allocation checks,
storage accounting, algorithmic scaling, and same-session comparisons, but
small timing differences should not be treated as optimization wins.

Future before/after performance decisions should use interleaved runs under
the same throttling and background-load conditions, report multiple samples,
and prefer medians. A clean-machine rerun should be recorded as a new dated
section rather than silently replacing these results.

### Stage 0 — legacy retained memory

The Stage 0 implementation is still present unchanged, so its storage could be
measured retrospectively. Each case constructs one exhaustive tree with unique
history-based information-set keys. Depth is varied to keep the number of
retained information sets useful across action counts.

`Numeric array payload` is exact: every `StrategyNode` owns two `double[]`
arrays, so it is $16KA$ bytes for $K$ retained information sets and $A$
actions. `Managed retained bytes` is the forced-GC heap delta while the legacy
dictionary remains live and therefore includes dictionary, string, node, and
array overhead; it is runtime-dependent. `Construction allocated bytes` also
includes temporary histories, masks, and traversal arrays and is not retained
memory.

| Actions | Depth | Information sets | Numeric array payload | Managed retained bytes | Construction allocated bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 14 | 16,383 | 524,256 | 3,225,560 | 8,779,872 |
| 3 | 9 | 9,841 | 472,368 | 2,213,832 | 6,362,624 |
| 4 | 7 | 5,461 | 349,504 | 1,240,520 | 3,979,208 |
| 8 | 5 | 4,681 | 599,168 | 1,387,944 | 5,131,976 |
| 16 | 4 | 4,369 | 1,118,464 | 1,876,424 | 8,243,672 |
| 32 | 3 | 1,057 | 541,184 | 705,544 | 3,670,784 |

### Baseline — Stage 1 legacy solver

- Source revision: `c465dff134caea1bf5c8222a47f061b4e0bfd4b0` with the Stage 1 benchmark harness in the working tree
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`
- Seed: `1729`

Reproduce with:

```powershell
dotnet run --project .\EvolutionaryBayes.CFR.Benchmarks\EvolutionaryBayes.CFR.Benchmarks.fsproj -c Release -- --revision c465dff134caea1bf5c8222a47f061b4e0bfd4b0
```

| Traversal | Actions | Depth | Iterations | Nodes | Elapsed ms | Nodes/s | Allocated bytes | Bytes/iteration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exhaustive | 2 | 2 | 5000 | 35000 | 41.790 | 837519 | 4160576 | 832.1 |
| sampled | 2 | 2 | 10000 | 50000 | 205.642 | 243141 | 20080544 | 2008.1 |
| exhaustive | 3 | 2 | 2222 | 28886 | 34.262 | 843091 | 3094056 | 1392.5 |
| sampled | 3 | 2 | 4444 | 31108 | 76.892 | 404568 | 13190792 | 2968.2 |
| exhaustive | 4 | 2 | 1250 | 26250 | 11.157 | 2352762 | 2641272 | 2113.0 |
| sampled | 4 | 2 | 2500 | 22500 | 29.588 | 760438 | 9541176 | 3816.5 |
| exhaustive | 8 | 2 | 312 | 22776 | 7.050 | 3230592 | 2059784 | 6601.9 |
| sampled | 8 | 2 | 624 | 10608 | 22.110 | 479781 | 5479080 | 8780.6 |
| exhaustive | 16 | 2 | 78 | 21294 | 6.200 | 3434295 | 1830632 | 23469.6 |
| sampled | 16 | 2 | 156 | 5148 | 15.718 | 327514 | 3593456 | 23035.0 |
| exhaustive | 32 | 2 | 19 | 20083 | 3.834 | 5237859 | 1697912 | 89363.8 |
| sampled | 32 | 2 | 38 | 2470 | 8.815 | 280211 | 2642144 | 69530.1 |
| exhaustive | 2 | 3 | 2500 | 37500 | 16.477 | 2275872 | 4961336 | 1984.5 |
| sampled | 2 | 3 | 5000 | 45000 | 47.176 | 953883 | 14921336 | 2984.3 |
| exhaustive | 3 | 3 | 740 | 29600 | 11.647 | 2541427 | 3448344 | 4659.9 |
| sampled | 3 | 3 | 1480 | 23680 | 22.806 | 1038319 | 6846520 | 4626.0 |
| exhaustive | 4 | 3 | 312 | 26520 | 10.431 | 2542349 | 2861000 | 9169.9 |
| sampled | 4 | 3 | 624 | 15600 | 12.718 | 1226589 | 3964296 | 6353.0 |
| exhaustive | 8 | 3 | 39 | 22815 | 8.879 | 2569430 | 2165880 | 55535.4 |
| sampled | 8 | 3 | 78 | 6318 | 3.998 | 1580172 | 1300456 | 16672.5 |
| exhaustive | 16 | 3 | 4 | 17476 | 4.642 | 3765000 | 1653904 | 413476.0 |
| sampled | 16 | 3 | 8 | 2312 | 1.178 | 1962482 | 443448 | 55431.0 |
| exhaustive | 32 | 3 | 1 | 33825 | 16.376 | 2065535 | 3669632 | 3669632.0 |
| sampled | 32 | 3 | 2 | 2178 | 1.809 | 1203780 | 395968 | 197984.0 |

### Stage 2 — flat scalar core

This is a separate benchmark of the Stage 2 storage and kernels; it does not
use a game traversal. The memory case contains 250,000 information sets,
1,000,000 legal action slots, maximum depth 64, maximum action count 32, and a
sampled delta capacity of 65,536. Values are managed-array payloads and exclude
object headers and alignment. `InfoSetMeta` is measured at 12 bytes per struct.

| Component | Payload bytes |
| --- | ---: |
| Persistent regrets + strategy sums | 16,000,000 |
| Information-set metadata | 3,000,000 |
| Common traversal scratch | 1,032,768 |
| Exhaustive-only workspace | 10,000,000 |
| Sampled-only workspace | 2,786,432 |
| Exhaustive total including persistent + metadata | 30,032,768 |
| Sampled total including persistent + metadata | 22,819,200 |

Each timed row cycle runs regret matching, mandatory average-strategy
accumulation, clipped aggregate-delta application, and average-strategy
normalization. `Logical action-slots/s` counts one action slot for each of those
four kernels; it is a workload-normalized throughput measure, not a CPU
instruction count.

| Actions | Iterations | Elapsed ms | Row cycles/s | Logical action-slots/s | Allocated bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20,000,000 | 1583.324 | 12,631,653 | 50,526,610 | 0 |
| 2 | 10,000,000 | 976.328 | 10,242,460 | 81,939,676 | 0 |
| 3 | 6,666,666 | 743.908 | 8,961,681 | 107,540,169 | 0 |
| 4 | 5,000,000 | 815.669 | 6,129,940 | 98,079,036 | 0 |
| 8 | 2,500,000 | 713.999 | 3,501,408 | 112,045,053 | 0 |
| 16 | 1,250,000 | 636.009 | 1,965,382 | 125,784,443 | 0 |
| 32 | 625,000 | 611.603 | 1,021,905 | 130,803,870 | 0 |

### Stage 3 - exhaustive two-player general-sum solver

- Source revision: `385bfe3` plus the Stage 3 working-tree implementation
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`

The representative fixture is a complete perfect-information tree with four
legal local actions and terminal depth five. It has 341 information sets,
1,364 stored legal action slots, and 1,365 visited nodes per target traversal.
Each sample measures 500 iterations after warm-up. Stage 3 performs both target
traversals, or 1,365,000 node visits per sample; legacy performs one traversal,
or 682,500 visits. Seven runs were interleaved in alternating order, and the
table reports medians. Throughput counts every visited node, so the comparison
does not hide Stage 3's correct two-target update schedule.

| Solver | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: |
| Legacy exhaustive | 205.774 | 3,316,737 | 76,256,000 |
| Stage 3 CFR | 85.710 | 15,925,852 | 0 |

Stage 3 is approximately $4.80\times$ faster by visited-node throughput and
uses about $0.42\times$ the elapsed time despite performing twice as many node
visits. As noted in the environment qualification, the machine was throttled
and busy, so the large same-session interleaved difference is meaningful while
small absolute timing differences would not be.

For this fixture, exact managed-array payload is:

| Component | Payload bytes |
| --- | ---: |
| Persistent regrets + strategy sums | 21,824 |
| Information-set metadata | 4,092 |
| Depth strategy + utility scratch | 320 |
| Average-strategy epochs | 1,364 |
| Exhaustive regret deltas | 10,912 |
| Touched rows + touched epochs | 2,728 |
| Total | 41,240 |

Thus the only numerical workspace beyond the two persistent tables is the
$8M$-byte regret-delta array and the $16DA$-byte depth scratch for configured
depth $D$ and maximum action count $A$. Integer epoch and touched-row arrays
are shown separately. The warmed value-state allocation test also measured
zero bytes over 10,000 exhaustive iterations.

### Stage 3a - exhaustive hot-path optimization

- Source revision: `385bfe3` plus the Stage 3 and Stage 3a working-tree implementation
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`

The representative fixture and warm-up are unchanged from Stage 3: four legal
actions, terminal depth five, 341 information sets, 1,364 legal slots, and
1,365 nodes per complete tree. Each sample measures 500 iterations. Seven runs
were interleaved in alternating order and the table reports medians. The
two-target reference and the one-pass specialization start from separate,
identically initialized tables.

| Solver | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| Stage 3 two target passes | 1,365,000 | 98.644 | 13,837,568 | 0 |
| Stage 3a one pass | 682,500 | 97.402 | 7,007,014 | 0 |

The specialization halves logical tree visits and reduced median elapsed time
by $1.26\%$ in the final conservative run. Two earlier same-session
interleaved runs reduced elapsed time by $19.7\%$ and $8.5\%$. The large spread
is consistent with the machine qualification above, so the final smaller gain
is used for the exit gate and no peak-performance claim is made. Node
throughput is reported for transparency but is not directly comparable between
the rows: a one-pass node propagates both players' utilities.

The fused regret boundary was measured separately over 341 rows, 1,364 slots,
and 20,000 boundaries. Both variants restored the same deltas for the next
boundary and seven runs were interleaved.

| Boundary variant | Median elapsed ms | Median allocated bytes |
| --- | ---: | ---: |
| Apply, then clear touched rows | 296.555 | 0 |
| Fused apply-and-clear | 291.498 | 0 |

The fused loop reduced this final median by $1.71\%$ and removes the separate
slot-clearing pass. Earlier sessions measured a $21.1\%$ improvement and a
$0.14\%$ improvement; none showed a regression.

An unchecked touched-row helper was also tested separately and rejected. In
two interleaved sessions, checked versus unchecked medians were respectively
245.044 versus 343.229 ms and 224.193 versus 383.904 ms across 341 validated
IDs and 100,000 epochs. The unchecked helper was removed, leaving the simpler
and faster checked implementation.

Stage 3a added no persistent or workspace arrays, so the exact fixture payload
remains 41,240 bytes. The warmed allocation test again measured zero bytes over
10,000 exhaustive iterations.

#### Same-day Stage 3a rerun

The unchanged Release binaries were rerun under the same documented constrained
environment. This is an additional interleaved seven-run median, not a
replacement for the conservative result above.

| Solver | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| Stage 3 two target passes | 1,365,000 | 113.248 | 12,053,214 | 0 |
| Stage 3a one pass | 682,500 | 84.111 | 8,114,326 | 0 |

The one-pass traversal reduced elapsed time by $25.73\%$ in this session. The
separately measured fused boundary was also faster:

| Boundary variant | Median elapsed ms | Median allocated bytes |
| --- | ---: | ---: |
| Apply, then clear touched rows | 291.472 | 0 |
| Fused apply-and-clear | 287.678 | 0 |

That is a $1.30\%$ boundary reduction. These results strengthen the evidence
for retaining the one-pass traversal, while the variation between sessions
continues to justify reporting all interleaved medians rather than selecting a
single best run.

## 2026-07-13

### Stage 4 - two-player public CFR, CFR+, MCCFR, and MCCFR+

- Source revision: `eff7135` plus the Stage 4 working-tree implementation
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`
- Seed: `1729`

The environment qualification at the top of this document still applies: the
machine is approximately 65% throttled and typically carries more than 50% CPU
load from other work. The four modes were therefore run in rotating interleaved
order for seven repetitions, and medians are reported. Absolute times are a
constrained snapshot; zero-allocation results, exact workspace payloads, node
counts, and the large same-session traversal differences are the reliable
signals.

The fixture has four legal actions, terminal depth five, 341 nonterminal
information sets, and 1,364 legal information-set/action slots. Each measured
sample ran 500 iterations after 100 warm-up iterations. `Node visits` counts
nonterminal actor queries, so vanilla CFR's one-pass specialization visits 341
nodes per iteration, while CFR+ makes two exhaustive target passes. The paired
sampled traversals visit 67 nonterminal nodes per iteration.

| Mode | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| CFR | 170,500 | 79.797 | 2,136,677 | 0 |
| CFR+ | 341,000 | 99.229 | 3,436,482 | 0 |
| MCCFR | 33,500 | 14.882 | 2,251,102 | 0 |
| MCCFR+ | 33,500 | 13.919 | 2,406,851 | 0 |

On this fixed-shape fixture, MCCFR visits $80.35\%$ fewer nonterminal nodes than
CFR and reduces median elapsed time by $81.35\%$. MCCFR+ visits $90.18\%$ fewer
nonterminal nodes than CFR+ and reduces median elapsed time by $85.97\%$. These
are algorithmic traversal comparisons, not claims that sampled and exhaustive
iterations provide equal statistical progress. All four warmed paths allocate
zero bytes.

The sampled log capacity was set to the exact 104 target action slots touched by
the paired passes. Persistent regrets, strategy sums, and metadata are shared
by both traversal kinds and are excluded from this workspace-only comparison.

| Workspace | Exact managed-array payload bytes |
| --- | ---: |
| Exhaustive | 15,324 |
| Sampled | 5,660 |

The sampled workspace is 9,664 bytes, or $63.06\%$, smaller for this fixture.
It replaces the dense exhaustive regret-delta array and touched-row arrays with
two information-set sample-cache arrays and a sparse integer/double delta log.
Both workspaces retain the same depth-local strategy/utility scratch and
average-strategy epoch array.

Reproduce with:

```powershell
dotnet run --project .\EvolutionaryBayes.CFR.Benchmarks\EvolutionaryBayes.CFR.Benchmarks.fsproj -c Release -- --revision working-tree
```

### ValueOption terminal-contract verification

- Source revision: `eff7135` plus the Stage 4, training-surface, exact-profile,
  and `ValueOption` working-tree changes
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`
- Seed: `1729`

This additional run checks the terminal hot path after replacing the
output-`byref` callback with `double voption`. The machine remained
approximately 65% throttled and typically above 50% background CPU load. The
four modes were run in rotating interleaved order for seven repetitions, and
the table reports medians. The fixture, seed, warm-up, and workload match the
Stage 4 run: four actions, terminal depth five, 341 information sets, 1,364
slots, 100 warm-up iterations, and 500 measured iterations.

| Mode | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| CFR | 170,500 | 85.035 | 2,005,052 | 0 |
| CFR+ | 341,000 | 100.608 | 3,389,403 | 0 |
| MCCFR | 33,500 | 12.427 | 2,695,743 | 0 |
| MCCFR+ | 33,500 | 16.734 | 2,001,876 | 0 |

All four modes remain allocation-free. Relative to the earlier constrained
Stage 4 session, elapsed-time changes range from a 16.5% reduction to a 20.2%
increase depending on mode. Because this is not an isolated before/after binary
comparison and the machine load is unstable, those timing differences are
provisional and do not establish either a speedup or regression. The reliable
result is that the cleaner `ValueOption` contract preserves node counts and
zero steady-state allocation. An isolated interleaved callback benchmark would
be required before claiming a throughput difference.

### Stage 5 - $N$-player general-sum and chance completion

- Source revision: `eff7135b61a5c4053a9a19fc541ab0859e4ef099` plus the
  Stage 5 and preceding uncommitted CFR working-tree changes
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`
- Seed: `1729`

The machine remained approximately 65% throttled and normally above 50%
background CPU load. All timing tables therefore report medians from seven
rotating or paired-interleaved repetitions. Absolute throughput remains
provisional. Exact node counts, array payloads, and zero-allocation results are
the stronger evidence.

The common fixture has four actions, terminal depth five, 341 information
sets, and 1,364 legal slots. Every sample runs 500 measured iterations after
100 warm-up iterations. `Node visits` counts nonterminal actor queries. The
unchanged two-player workload verifies that Stage 5 still selects one
exhaustive traversal for CFR, two direct target traversals for CFR+, and the
paired external-sampling path for both MCCFR modes.

| Two-player mode | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| CFR | 170,500 | 69.836 | 2,441,441 | 0 |
| CFR+ | 341,000 | 87.007 | 3,919,216 | 0 |
| MCCFR | 33,500 | 8.890 | 3,768,406 | 0 |
| MCCFR+ | 33,500 | 10.071 | 3,326,515 | 0 |

Because separate-day throughput varies substantially on this machine, Stage 5
also adds a same-session paired comparison. `Direct` calls the selected
internal two-player solver and performs scalar utility accounting, approximating
the pre-multiplayer boundary. `Public` calls the Stage 5 `Solver.RunIteration`
tuple path. Pair order alternates by repetition.

| Mode | Direct median ms | Public median ms | Public/direct | Direct allocated bytes | Public allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFR | 92.255 | 87.776 | 0.951 | 0 | 0 |
| CFR+ | 103.202 | 110.502 | 1.071 | 0 | 0 |
| MCCFR | 13.011 | 11.298 | 0.868 | 0 | 0 |
| MCCFR+ | 15.682 | 14.964 | 0.954 | 0 | 0 |

The public/direct ratios range from 0.868 to 1.071, with three public medians
lower and one 7.1% higher. Under the known unstable load, this does not support
a throughput improvement claim, but it shows no consistent material boundary
regression. Both sides allocate zero bytes. More importantly, the public node
counts exactly select the established two-player schedules; multiplayer logic
does not enter the recursive hot paths.

The three-player fixture assigns actors by depth modulo three. Exhaustive modes
make one complete target traversal per player. Sampled modes include the exact
full-tree average-strategy sweep once per iteration, followed by three external-
sampling regret passes.

| Three-player mode | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| CFR | 511,500 | 212.548 | 2,406,511 | 0 |
| CFR+ | 511,500 | 212.162 | 2,410,898 | 0 |
| MCCFR | 197,500 | 56.127 | 3,518,780 | 0 |
| MCCFR+ | 197,500 | 50.788 | 3,888,706 | 0 |

All four multiplayer modes allocate zero steady-state bytes. The public solver
owns two reusable `double[3]` utility arrays, an exact 48-byte numeric payload.
Only sampled multiplayer modes own the additional `double[3]` exact-average
reach vector, adding 24 bytes. Two-player solvers do not allocate that vector.
Persistent regret/strategy tables and packed metadata remain unchanged at
$16M$ numeric bytes plus metadata for $M$ legal slots.

## 2026-07-14

### Stage 6 - minimal production API cutover

- Source revision: `98c99dc` plus the Stage 6 working-tree implementation
- Working-tree qualification: includes the opaque production API, test-only
  legacy move, growable sampled-delta log, migrated examples, tests, benchmark,
  and documentation; unrelated repository edits remained present but were not
  part of either measured executable
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`
- Seed: `1729`

The machine remained approximately 65% CPU/power throttled and normally above
50% background CPU load. Thermal state was not available. Timings are therefore
provisional even though each comparison alternated order and reports the median
of seven interleaved repetitions. Allocation counts, node schedules, and exact
array payloads are the stronger evidence.

The common fixture has four actions, terminal depth five, 341 information
sets, and 1,364 legal slots. Every timing sample ran 500 measured iterations
after 100 warm-up iterations. The legacy comparison uses its allocating
dictionary, string-history, global-action-mask CFR+ traversal. Production CFR+
uses the final opaque `Solver.runIteration` path and performs two target-player
traversals, whereas the legacy comparison performs one recursive traversal.

| Variant | Median elapsed ms | Median allocated bytes | Allocated bytes/iteration |
| --- | ---: | ---: | ---: |
| Legacy dictionary CFR+ | 273.334 | 76,256,000 | 152,512 |
| Production packed CFR+ | 135.523 | 0 | 0 |

The production median is 50.4% lower despite performing two target traversals,
and it eliminates all 76.3 MB of measured steady-state allocation. This is a
same-session structural comparison, but absolute throughput remains
provisional under the recorded machine conditions.

The opaque-boundary check compares each direct internal specialized solver
with the final public wrapper on identical tables and traversal schedules.
Both sides perform 500 measured iterations after 100 warm-up iterations, with
seven paired alternating-order samples.

| Mode | Direct median ms | Public median ms | Public/direct | Direct allocated bytes | Public allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFR | 96.119 | 91.119 | 0.948 | 0 | 0 |
| CFR+ | 125.242 | 124.536 | 0.994 | 0 | 0 |
| MCCFR | 12.146 | 11.813 | 0.973 | 0 | 0 |
| MCCFR+ | 14.621 | 14.316 | 0.979 | 0 | 0 |

The public medians happen to be lower in this run, but the wrapper cannot make
the shared traversal faster; these ratios demonstrate timing variability, not
a speedup. The reliable result is that one operation-level opaque dispatch
introduces no observed allocation and no consistent measurable regression.

Exact payloads below come from actual allocated array lengths after the sampled
solver's 100-iteration warm-up. Its delta log began at 20 entries
(`maxDepth * maxActionCount`) and reached a reusable 160-entry high-water
capacity. A live solver owns the persistent tables and metadata plus only the
workspace selected by its mode.

| Component | Exact payload bytes |
| --- | ---: |
| Persistent regrets + strategy sums | 21,824 |
| Information-set metadata | 4,092 |
| Exhaustive workspace | 15,324 |
| Sampled workspace after warm-up | 6,332 |
| Exhaustive solver total | 41,240 |
| Sampled solver total | 32,248 |

All 77 tests passed. Separate public-path allocation checks ran 10,000 warmed
iterations of CFR, CFR+, MCCFR, and MCCFR+ and measured zero bytes for every
mode. Existing exhaustive, sampled, and three-player allocation checks also
remained zero.

### Stage 7 - profile-guided single-thread optimization

- Source revision: `99db8fd` plus the Stage 7 benchmark harness and
  documentation working tree
- Production qualification: `EvolutionaryBayes/CFRCore.fs` is unchanged from
  `99db8fd`; every candidate below failed the merge gate and was removed
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`
- Seed: `1729`

The machine remained approximately 65% CPU/power throttled and normally above
50% background CPU load. Thermal state was unavailable. Absolute timings are
provisional. Candidate comparisons alternate order and report seven-run
medians and observed min--max ranges. `COMPlus_TieredCompilation=0` was used
for the generic-loop, SIMD, and validation experiments so tier promotion could
not differ between variants. The small-action experiment used the ordinary
tiered runtime. Allocation counts, exact node counts, generated-code evidence,
and retained-memory calculations are more reliable than small timing changes.

#### Final Release snapshot

The Stage 1 action/depth matrix was rerun unchanged. It remains a legacy
dictionary/string baseline, retained to preserve the original longitudinal
workload. These are one-run snapshots rather than candidate comparisons.

| Traversal | Actions | Depth | Iterations | Nodes | Elapsed ms | Nodes/s | Allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exhaustive | 2 | 2 | 5,000 | 35,000 | 40.318 | 868,092 | 4,160,576 |
| sampled | 2 | 2 | 10,000 | 50,000 | 172.959 | 289,086 | 16,720,544 |
| exhaustive | 3 | 2 | 2,222 | 28,886 | 30.471 | 947,986 | 3,094,056 |
| sampled | 3 | 2 | 4,444 | 31,108 | 103.839 | 299,578 | 11,910,920 |
| exhaustive | 4 | 2 | 1,250 | 26,250 | 28.545 | 919,591 | 2,641,272 |
| sampled | 4 | 2 | 2,500 | 22,500 | 52.958 | 424,864 | 8,981,176 |
| exhaustive | 8 | 2 | 312 | 22,776 | 10.603 | 2,148,092 | 2,059,784 |
| sampled | 8 | 2 | 624 | 10,608 | 30.498 | 347,825 | 5,279,400 |
| exhaustive | 16 | 2 | 78 | 21,294 | 6.918 | 3,078,057 | 1,830,632 |
| sampled | 16 | 2 | 156 | 5,148 | 7.430 | 692,895 | 3,573,488 |
| exhaustive | 32 | 2 | 19 | 20,083 | 4.163 | 4,824,281 | 1,697,912 |
| sampled | 32 | 2 | 38 | 2,470 | 6.982 | 353,757 | 2,710,240 |
| exhaustive | 2 | 3 | 2,500 | 37,500 | 12.509 | 2,997,889 | 4,961,336 |
| sampled | 2 | 3 | 5,000 | 45,000 | 40.414 | 1,113,481 | 13,241,336 |
| exhaustive | 3 | 3 | 740 | 29,600 | 9.012 | 3,284,582 | 3,448,344 |
| sampled | 3 | 3 | 1,480 | 23,680 | 17.635 | 1,342,769 | 6,420,280 |
| exhaustive | 4 | 3 | 312 | 26,520 | 7.078 | 3,746,874 | 2,861,000 |
| sampled | 4 | 3 | 624 | 15,600 | 11.234 | 1,388,654 | 3,824,520 |
| exhaustive | 8 | 3 | 39 | 22,815 | 5.415 | 4,213,608 | 2,165,880 |
| sampled | 8 | 3 | 78 | 6,318 | 4.159 | 1,519,006 | 1,275,496 |
| exhaustive | 16 | 3 | 4 | 17,476 | 3.639 | 4,802,418 | 1,653,904 |
| sampled | 16 | 3 | 8 | 2,312 | 0.908 | 2,547,097 | 442,424 |
| exhaustive | 32 | 3 | 1 | 33,825 | 10.209 | 3,313,123 | 3,669,632 |
| sampled | 32 | 3 | 2 | 2,178 | 1.692 | 1,287,082 | 399,552 |

The final four-action, depth-five production snapshot used 500 measured
iterations after 100 warm-up iterations and seven-run medians. Node visits
count nonterminal actor queries.

| Mode | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| CFR | 170,500 | 85.468 | 1,994,910 | 0 |
| CFR+ | 341,000 | 95.573 | 3,567,972 | 0 |
| MCCFR | 33,500 | 10.545 | 3,176,741 | 0 |
| MCCFR+ | 33,500 | 12.860 | 2,604,956 | 0 |

The scalar row-cycle profile runs regret matching, average accumulation,
clipped regret application, and average normalization. It shows that rows are
real work but does not by itself justify changing an end-to-end traversal.

| Actions | Iterations | Elapsed ms | Row cycles/s | Logical slots/s | Allocated bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20,000,000 | 1,504.679 | 13,291,874 | 53,167,497 | 0 |
| 2 | 10,000,000 | 981.792 | 10,185,458 | 81,483,663 | 0 |
| 3 | 6,666,666 | 770.346 | 8,654,120 | 103,849,442 | 0 |
| 4 | 5,000,000 | 729.828 | 6,850,930 | 109,614,883 | 0 |
| 8 | 2,500,000 | 623.415 | 4,010,168 | 128,325,372 | 0 |
| 16 | 1,250,000 | 635.851 | 1,965,869 | 125,815,619 | 0 |
| 32 | 625,000 | 614.658 | 1,016,825 | 130,153,615 | 0 |

#### Generated-code and candidate results

The baseline `regretMatchUnchecked` JIT body was 302 bytes and contained range
checks in both loops. Explicit two- and three-action kernels did not simplify
the generated code: the two-action body was 300 bytes, the corrected
three-action body was 333 bytes, and the action-count dispatcher was another
87 bytes. Using F# generic `max` had initially expanded the three-action body
to 597 bytes. A Span-based bounds-check experiment retained element checks and
expanded the generic body to 382 bytes. Both forms were removed.

The corrected small-action candidate also specialized average accumulation.
Negative change is faster; every row allocated zero bytes. The candidate did
not clear the 10% affected-workload gate consistently and regressed one common
three-action workload by 21.5%.

| Mode | Actions/depth | Nodes | Baseline median (range) ms | Candidate median (range) ms | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| CFR | 2 / 8 | 1,275,000 | 390.154 (248.875--539.533) | 396.660 (278.069--505.929) | +1.7% |
| CFR+ | 2 / 8 | 1,275,000 | 207.215 (159.937--524.125) | 198.787 (136.381--355.310) | -4.1% |
| CFR | 3 / 6 | 728,000 | 335.621 (257.535--459.096) | 297.366 (241.856--387.422) | -11.4% |
| CFR+ | 3 / 6 | 728,000 | 137.510 (123.555--268.258) | 167.007 (138.474--232.538) | +21.5% |

A generic-loop candidate computed the reciprocal of the scaled regret total
once, replacing one division per positive action with multiplication. It was
mathematically stable but failed the end-to-end gate and was removed. Both
variants allocated zero bytes.

| Mode | Actions/depth | Nodes | Baseline median (range) ms | Candidate median (range) ms | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| CFR | 2 / 8 | 1,275,000 | 396.963 (374.927--522.124) | 445.947 (369.409--493.496) | +12.3% |
| CFR+ | 2 / 8 | 1,275,000 | 353.190 (312.903--437.383) | 372.806 (313.536--425.698) | +5.6% |
| CFR | 3 / 6 | 728,000 | 342.339 (253.649--460.593) | 325.511 (281.313--391.205) | -4.9% |
| CFR+ | 3 / 6 | 728,000 | 172.845 (153.624--212.195) | 175.626 (162.028--230.121) | +1.6% |

An AVX `Vector<double>` apply/clip/clear kernel used four doubles per vector on
this processor. The isolated boundary microbenchmark processed 4,096 rows and
included restoring deltas for the next cycle. It showed a real kernel-level
win and zero allocation, so it advanced to end-to-end testing.

| Actions | Cycles | Scalar median (range) ms | SIMD median (range) ms | Change |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1,200 | 1,791.141 (1,607.123--2,045.238) | 220.527 (184.609--276.867) | -87.7% |
| 8 | 600 | 1,037.976 (897.891--1,102.797) | 174.409 (152.612--180.250) | -83.2% |
| 16 | 300 | 775.972 (678.359--803.611) | 162.818 (117.091--253.679) | -79.0% |
| 32 | 150 | 443.797 (369.509--651.942) | 134.058 (121.833--181.713) | -69.8% |

The same SIMD kernel was then selected only for rows of at least two vectors.
It did not clear the end-to-end gate and was removed; isolated kernel speed did
not dominate recursive game and regret-matching work.

| Mode | Actions/depth | Nodes | Scalar median (range) ms | SIMD median (range) ms | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| CFR | 8 / 3 | 730,000 | 817.593 (669.992--921.481) | 838.238 (745.133--978.531) | +2.5% |
| CFR+ | 8 / 3 | 730,000 | 463.462 (364.486--705.479) | 428.652 (353.833--542.330) | -7.5% |
| CFR+ | 16 / 2 | 1,360,000 | 1,785.752 (1,656.903--2,066.782) | 1,788.351 (1,699.937--2,003.877) | +0.1% |
| CFR+ | 32 / 2 | 1,320,000 | 3,688.651 (3,347.326--4,357.546) | 3,425.591 (3,207.712--3,716.455) | -7.1% |

Finally, a temporary duplicate two-player CFR traversal removed the hot-path
depth, action-count, actor, information-set, metadata, probability, and
normalization checks without adding a per-node mode branch. The trusted path
was slower in every paired median, so both it and the proposed public trusted
mode were removed. The equal 40-byte readings are a benchmark-call artifact,
not steady-state solver allocation.

| Actions/depth | Nodes | Checked median (range) ms | Trusted median (range) ms | Change | Allocated bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 / 8 | 1,275,000 | 472.457 (334.658--509.244) | 481.202 (394.898--723.558) | +1.9% | 40 / 40 |
| 3 / 6 | 1,456,000 | 750.260 (665.123--838.422) | 763.122 (640.977--883.906) | +1.7% | 40 / 40 |
| 4 / 5 | 1,705,000 | 1,040.746 (953.580--1,188.890) | 1,104.863 (913.092--1,172.869) | +6.2% | 40 / 40 |

#### Exit result

No specialization was retained. The production scalar source, persistent
tables, and workspaces are unchanged: exact payload remains 21,824 bytes for
regrets plus strategy sums, 4,092 bytes for metadata, 15,324 bytes for the
exhaustive workspace, and 6,332 bytes for the warmed sampled workspace on the
341-information-set fixture. Totals remain 41,240 and 32,248 bytes. All four
production modes measured zero steady-state allocation, the complete Release
solution built with zero warnings, and all 77 tests passed. The benchmark
executable retains only a compact `--stage7-sample` command for future
one-shot production profiling; experimental candidate implementations were
removed.

## 2026-07-15

### Stage 7 addendum - clipped-row SIMD confirmation

- Source revision: `99db8fd` plus the uncommitted Stage 7 harness and
  documentation working tree
- Compared builds: a preserved scalar Release build and a temporary build with
  only the clipped fused apply/clear SIMD candidate
- Final production qualification: `EvolutionaryBayes/CFRCore.fs` and its tests
  were restored byte-for-byte to `99db8fd` after the candidate failed the gate
- Build configuration: `Release`
- Runtime: `.NET 8.0.23`, with `COMPlus_TieredCompilation=0`
- OS: `Microsoft Windows 10.0.19045`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Logical processors: `12`
- Seed: `1729`

This run reconsiders the earlier SIMD no-go because the first Stage 7 run had
shown 7.5% and 7.1% end-to-end improvements at 8 and 32 actions. For this
confirmation, a repeatable $5\%$ affected-workload improvement was sufficient.
The machine remained permanently about 65% CPU/power throttled, normally above
50% CPU load, and subject to load spikes. Thermal state was unavailable.
Absolute times remain provisional; the decision therefore uses paired ratios
and a second drift-correcting ABBA experiment rather than selecting the best
independent median.

The temporary candidate used `Vector<double>` only in
`applyRegretDeltaAndClearUnchecked` when the update was clipped and the row had
at least eight actions. It fused regret addition, zero clipping, and delta
clearing across four doubles per AVX vector, then used the unchanged scalar
loop for the tail. Signed CFR, sampled modes, traversals, tables, workspaces,
and public APIs were unchanged. The candidate introduced no persistent or
workspace memory.

Before timing, a randomized scalar-reference test ran 200 trials at each of 7,
8, 9, 10, 15, 16, 17, 31, 32, and 33 actions. It covered the SIMD threshold,
vector tails, nonzero source offsets, large finite values, exact delta clearing,
and untouched sentinels. All 78 tests passed, including the existing warmed
allocation suite, and the candidate measured zero steady-state allocation.

#### Workloads and method

Every benchmark process constructed the public opaque solver, ran 100 warm-up
iterations, forced collection, then timed only the requested training budget.
Exact nonterminal node visits and allocation were identical for scalar and SIMD:

| Actions / depth | Iterations | Node visits per sample |
| --- | ---: | ---: |
| 8 / 3 | 20,000 | 2,920,000 |
| 16 / 2 | 40,000 | 1,360,000 |
| 32 / 2 | 20,000 | 1,320,000 |

A 730,000-node smoke pair at 8 actions measured 528.244 ms scalar and 504.626
ms SIMD, a 4.5% candidate improvement. The longer experiment first ran 11
adjacent scalar/SIMD pairs, reversing order on every pair. It then ran five
ABBA blocks, alternating `scalar, SIMD, SIMD, scalar` with the reverse order and
averaging both placements of each build inside the block. Negative change below
means the SIMD candidate was faster. Every timed sample allocated zero bytes.

#### Eleven alternating pairs

Eight actions, depth three:

| Pair | Scalar ms | SIMD ms | Paired change |
| ---: | ---: | ---: | ---: |
| 1 | 2,130.411 | 2,069.116 | -2.88% |
| 2 | 2,147.218 | 2,144.042 | -0.15% |
| 3 | 2,237.387 | 2,070.808 | -7.45% |
| 4 | 2,045.972 | 1,968.501 | -3.79% |
| 5 | 2,039.661 | 2,197.112 | +7.72% |
| 6 | 2,151.592 | 2,037.569 | -5.30% |
| 7 | 1,945.194 | 1,961.887 | +0.86% |
| 8 | 1,901.958 | 2,004.130 | +5.37% |
| 9 | 1,860.297 | 1,767.008 | -5.01% |
| 10 | 2,154.521 | 1,957.325 | -9.15% |
| 11 | 1,971.620 | 2,242.971 | +13.76% |

Sixteen actions, depth two:

| Pair | Scalar ms | SIMD ms | Paired change |
| ---: | ---: | ---: | ---: |
| 1 | 1,788.530 | 1,838.724 | +2.81% |
| 2 | 2,037.351 | 1,870.272 | -8.20% |
| 3 | 2,124.424 | 1,875.419 | -11.72% |
| 4 | 2,093.204 | 2,095.109 | +0.09% |
| 5 | 1,870.186 | 1,760.861 | -5.85% |
| 6 | 2,025.652 | 2,034.792 | +0.45% |
| 7 | 1,849.034 | 1,994.324 | +7.86% |
| 8 | 2,154.984 | 1,964.527 | -8.84% |
| 9 | 2,115.145 | 1,872.022 | -11.49% |
| 10 | 1,958.982 | 1,857.725 | -5.17% |
| 11 | 1,869.780 | 1,625.610 | -13.06% |

Thirty-two actions, depth two:

| Pair | Scalar ms | SIMD ms | Paired change |
| ---: | ---: | ---: | ---: |
| 1 | 3,318.726 | 3,536.143 | +6.55% |
| 2 | 3,218.321 | 3,440.093 | +6.89% |
| 3 | 3,595.690 | 3,421.687 | -4.84% |
| 4 | 4,075.092 | 3,349.639 | -17.80% |
| 5 | 3,881.617 | 3,639.564 | -6.24% |
| 6 | 3,456.148 | 3,476.406 | +0.59% |
| 7 | 3,569.486 | 3,818.860 | +6.99% |
| 8 | 3,350.036 | 3,359.447 | +0.28% |
| 9 | 3,966.769 | 3,136.771 | -20.92% |
| 10 | 3,381.884 | 3,442.520 | +1.79% |
| 11 | 3,260.809 | 3,691.111 | +13.20% |

| Actions | Scalar median (range) ms | SIMD median (range) ms | Median paired change |
| ---: | ---: | ---: | ---: |
| 8 | 2,130.411 (1,860.297--2,237.387) | 2,069.116 (1,767.008--2,242.971) | -0.15% |
| 16 | 2,037.351 (1,788.530--2,154.984) | 1,875.419 (1,625.610--2,095.109) | -5.17% |
| 32 | 3,569.486 (3,218.321--4,075.092) | 3,476.406 (3,136.771--3,818.860) | +1.79% |

#### Drift-correcting ABBA blocks

Eight actions, depth three:

| Block | Scalar mean ms | SIMD mean ms | Paired change |
| ---: | ---: | ---: | ---: |
| 1 | 2,288.726 | 1,930.237 | -15.66% |
| 2 | 2,324.649 | 2,320.460 | -0.18% |
| 3 | 2,236.816 | 2,337.247 | +4.49% |
| 4 | 2,151.762 | 2,038.885 | -5.25% |
| 5 | 2,043.982 | 2,132.132 | +4.31% |

Sixteen actions, depth two:

| Block | Scalar mean ms | SIMD mean ms | Paired change |
| ---: | ---: | ---: | ---: |
| 1 | 1,872.666 | 1,779.108 | -5.00% |
| 2 | 1,968.565 | 1,913.276 | -2.81% |
| 3 | 1,998.362 | 1,934.835 | -3.18% |
| 4 | 1,957.626 | 1,812.946 | -7.39% |
| 5 | 1,937.839 | 2,062.809 | +6.45% |

Thirty-two actions, depth two:

| Block | Scalar mean ms | SIMD mean ms | Paired change |
| ---: | ---: | ---: | ---: |
| 1 | 3,488.956 | 3,435.506 | -1.53% |
| 2 | 3,481.434 | 3,502.124 | +0.59% |
| 3 | 3,549.581 | 3,640.056 | +2.55% |
| 4 | 3,697.730 | 3,425.814 | -7.35% |
| 5 | 3,646.548 | 4,070.627 | +11.63% |

| Actions | Scalar block median ms | SIMD block median ms | Median paired change | Paired range |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2,236.816 | 2,132.132 | -0.18% | -15.66%--+4.49% |
| 16 | 1,957.626 | 1,913.276 | -3.18% | -7.39%--+6.45% |
| 32 | 3,549.581 | 3,502.124 | +0.59% | -7.35%--+11.63% |

#### Decision and final verification

The longer adjacent pairs did not repeat the original 8- or 32-action wins.
The ABBA experiment then failed the practical $5\%$ gate at every possible
threshold: 8 actions was effectively neutral, 16 actions favored SIMD by only
3.18%, and 32 actions was 0.59% slower. Large sign-changing ranges are expected
on this throttled, loaded machine, but they do not justify selecting a favorable
isolated result. The candidate and its temporary correctness test were removed.

The final production `CFRCore.fs` and 77-test suite are identical to revision
`99db8fd`. The complete Release solution rebuilt for `net5.0`,
`netstandard2.1`, and both .NET 8 executables with zero warnings, and all 77
tests passed. Persistent memory, workspace memory, and warmed allocation remain
the Stage 7 scalar values: 41,240 bytes total for the exhaustive fixture,
32,248 bytes for the sampled fixture, and zero steady-state allocation in all
four public modes.
