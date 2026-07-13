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
