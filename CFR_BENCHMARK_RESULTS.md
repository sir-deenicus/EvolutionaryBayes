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
