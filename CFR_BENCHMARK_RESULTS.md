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
