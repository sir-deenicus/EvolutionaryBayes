# Review: `RegretMinimization.fs`

## Summary

The non-contextual code is a full-information, linear-factor multiplicative
weights learner with distribution averaging. It is not regret matching: it
updates action weights directly and never accumulates each action's advantage
over the learner's expected utility. The names `Regrets`, `SumRegrets`,
`RegretLearner`, and `NormalizedRegret` therefore describe behavior the code
does not implement.

The update

```fsharp
w.[i] * (1. + rate * boundedReward.[i])
```

is a legitimate multiplicative-weights update. It should not be described as
an additive algorithm merely because its multiplier is linear rather than
exponential. It is the linear-factor MW variant; Hedge commonly uses the
related exponential multiplier `exp (rate * reward)`. The linear variant needs
an explicit bounded-feedback and learning-rate contract to keep every
multiplier nonnegative and to support the usual regret guarantee.

`ContextualExpert` is an incomplete exact-context wrapper with a disconnected
nearest-context fallback. Its miss path currently drops the first observation.

The replacement stack for the first five implementation items now lives in
[`EvolutionaryBayes/MWUA.fs`](EvolutionaryBayes/MWUA.fs), which contains the
full-information learners and EXP3, and
[`EvolutionaryBayes/ContextualRegret.fs`](EvolutionaryBayes/ContextualRegret.fs),
which contains the exact-context wrappers and EXP4.
Full-information MWUA and regret matching share
`ExternalRegretMinimizer<'Action>`; sampled-action EXP3 uses
`BanditRegretMinimizer<'Action>`. Exact-context variants create one of these
learners per context behind algorithm-specific constructors. EXP4 learns over
contextual policies and shares policy evidence across contexts. The old types
reviewed below remain unchanged for compatibility and should be treated as the
legacy surface until a separate retirement decision is made.

## Bugs and correctness issues

### 1. `NormalizedRegret` divides by zero and is not regret

Before the first `Learn` call, `expert.Total` is zero, so dividing every entry
of `SumRegrets` by it produces `NaN` values. Returning a zero vector would avoid
the exception but would not be a probability distribution. The replacement API
should define the no-history result explicitly, for example by returning the
current distribution, returning `None`, or exposing the round count alongside
the average.

More fundamentally, `SumRegrets` contains accumulated weights, so this member
is an average distribution and should be named accordingly.

### 2. The average records the post-update distribution

`learningStep` aliases `e.Weights` as `w`, copies the newly calculated weights
into that same array, and only then adds `w` to `SumRegrets`. Consequently the
average contains the distribution produced after observing the round's
feedback.

In the usual online-learning order, the learner selects from `p_t`, receives
feedback, and produces `p_(t+1)`. If the average is intended to describe played
strategies, it should accumulate `p_t` before the update. If post-update
averaging is intentional, that less common convention needs to be documented.

### 3. Feedback is clipped only above the declared range

The code applies `min r maxAmount` but does not clamp values below `minAmount`.
`scaleTo -1. 1. minAmount maxAmount` can therefore return values below `-1`,
making a multiplier negative even when the nominal learning rate would be safe
for bounded feedback.

The implementation should either reject out-of-range rewards or clamp at both
bounds. Rejection is preferable when the declared bounds are part of the
algorithm's correctness contract; symmetric clamping is appropriate only when
lossy saturation is explicitly desired.

### 4. Learning-rate, bounds, and finite-value invariants are unchecked

The linear-factor update needs constraints that guarantee
`1 + rate * scaledReward >= 0`. With scaled rewards in `[-1, 1]`, a simple
strictly positive-normalization contract is a finite `rate` in `[0, 1)`.
Allowing `rate = 1` can make all weights zero after an all-minimum feedback
vector. Negative rates, rates above one, invalid bounds, `NaN`, and infinities
are currently accepted until they
produce negative or non-finite weights or make normalization fail.

Validate at least:

- a finite learning rate in the supported range;
- finite bounds with `minAmount < maxAmount`;
- finite rewards;
- a finite, nonnegative initial distribution with positive total mass;
- a finite, positive normalization total after each update.

An exponential Hedge implementation removes the nonnegative-multiplier
restriction but still needs finite inputs and a numerically stable log-weight
or max-shifted normalization implementation.

### 5. Empty and mismatched action state is not rejected

`newExpert 0` attempts to normalize an empty array, and `Sample` cannot operate
without actions. A supplied `prevweights` value can also have arrays whose
lengths differ from `possibleActions.Length`, or whose `Weights` and
`SumRegrets` lengths differ from each other. These cases should fail at
construction with clear argument errors.

The constructor should copy caller-owned action and initial-weight arrays, or
document and enforce ownership. The public `Expert` and `Actions` members
currently expose mutable arrays that callers can use to violate invariants.

### 6. `Reset` does not restore a supplied initial distribution

Even when construction receives `prevweights`, `Reset` always installs a
uniform distribution. If the new API accepts a prior/initial distribution,
reset should restore that distribution. If `prevweights` is intended as a
complete resumable mutable state rather than an initial distribution, it should
be named and validated as such.

### 7. The feedback model is full information but undocumented

Every `Learn` call evaluates `reward (action, observation)` for every possible
action. This is valid full-information MWUA only when the observation lets the
caller evaluate counterfactual rewards for actions that were not selected.

When only the sampled action's reward is observable, this update is not valid;
an importance-weighted bandit algorithm such as EXP3 is required. The public
API and documentation should make that distinction explicit.

## `ContextualExpert` issues

### 8. `PredictWith` ignores the query context

On a dictionary miss, `PredictWith(context)` passes only the existing context
keys to `distancefn`; it never passes `context`. A conventional nearest-context
selector needs both the query and the candidates. As written, the result cannot
normally depend on the requested context unless `distancefn` relies on hidden
mutable state.

### 9. Prediction on an empty dictionary delegates an empty candidate set

The first unseen context calls `distancefn [||]`. There is no defined fallback
learner and most candidate selectors cannot choose from an empty set.

### 10. The first observation for every new context is dropped

On a context miss, `Learn` constructs and stores a new learner but never calls
`expert.Learn(observation)`. Learning starts only with the second observation
for that exact context.

### 11. Prediction and learning use incompatible miss semantics

Prediction attempts to borrow some learner selected from existing contexts.
Learning never consults that selection; it always creates an independent
learner for the new key. Thus the learner used to make a prediction need not be
the learner updated from the resulting observation.

There are two coherent designs that should remain distinct:

1. An exact-context table creates one independent learner per repeating context
   and has no nearest-neighbor behavior.
2. A contextual policy learner competes over policies of type
   `'Context -> 'Action`, providing a genuine contextual-regret objective.

Nearest-neighbor sharing can be offered as an explicit heuristic policy, but it
does not by itself provide a contextual-regret guarantee without assumptions
about the metric and reward smoothness.

## Design and maintainability

### Mutable arrays are not themselves a record-field bug

The immutable record fields contain mutable array objects. Copying new values
into `Weights` with `Array.Copy` preserves array identity and is a coherent
in-place design; making the field itself mutable and replacing the array is not
automatically an improvement and could invalidate retained references.

The real problems are unclear ownership, public exposure through `Expert`, and
the opaque update sequence. A private state type with explicit in-place kernels
would preserve the allocation behavior while protecting invariants.

### Terminology should separate the algorithms

The implemented surface uses the mathematical umbrella
`ExternalRegretMinimizer<'Action>` while keeping algorithm-specific factories:

- `Mwua.create` and `Mwua.createWith` construct full-information exponential
  multiplicative-weights learners;
- `RegretMatching.create` and `RegretMatching.createWith` construct standard
  cumulative external-regret matching;
- `RegretMatching.createPlus` and `RegretMatching.createPlusWith` construct the
  per-round clipped plus variant;
- `Exp3.create` and `Exp3.createWith` construct sampled-action adversarial
  bandit learners with explicit exploration;
- `ContextualMwua`, `ContextualRegretMatching`, and `ContextualExp3` construct
  exact-context tables without exposing a `(unit -> learner)` factory;
- `Exp4.create` and `Exp4.createWith` construct policy-based contextual
  bandits over deterministic policies of type `'Context -> 'Action`.

EXP3 keeps a max-shifted log-weight vector. If `q` is its normalized
exploitation distribution and `gamma` is the configured exploration fraction,
the sampling distribution is

```text
p_i = (1 - gamma) q_i + gamma / K.
```

For the sampled action `I`, reward `r_I` is importance weighted and only its
log weight is updated:

```text
log w_I <- log w_I + eta r_I / p_I.
```

Full-information MWUA and finite Replicator share the internal max-shifted
exponential reweighting primitive in `MWUA.fs`. MWUA wraps it in persistent
learner state, sampling, and averaging; Replicator supplies the current
population shares and population-dependent fitness for one pure step.

Rewards are required to be finite values in `[0, 1]`. `Choose` returns an
opaque choice containing the action and `p_I`; `Learn(choice, reward)` accepts
that choice once. Keeping one outstanding choice prevents accidental updates
with stale, foreign, or mismatched sampling probabilities. `LearnBatch`
accepts a sequence of per-round reward functions and performs the same
`Choose`/evaluate-selected-action/`Learn` loop in enumeration order.

The exact-context layer deliberately provides memorization rather than
generalization. An unseen context gets an independent underlying learner, its
first `Learn` call is applied immediately, and an opaque contextual bandit
choice is routed back to the exact learner that issued it. `LearnBatch`
accepts `(context, feedback)` pairs. Reset clears each initialized learner's
history while retaining the context table. Contexts are generic values with
structural equality, so callers can already use integers, compact structs, or
discriminated unions instead of a heavyweight object representation.

EXP4 provides the policy-generalizing contextual objective. If policy `j`
recommends action `a_j(c)` in context `c`, its current policy probability is
`q_j`, and there are `K` actions, the sampled action distribution is

```text
p_a(c) = (1 - gamma) sum_j q_j 1[a_j(c) = a] + gamma / K.
```

After sampling action `I` and observing `r_I`, every policy that recommended
`I` receives the importance-weighted estimate `r_I / p_I`; other policies
receive zero for that round. Policy log weights use the exponential update
with the configured learning rate. `PolicyStrategy` and
`AveragePolicyStrategy` expose current and pre-update-average policy mixtures,
while `Strategy(context)` reports the induced explored action distribution.

### The legacy public behavior is undocumented and untested

`RegretMinimization.fs` has no XML documentation explaining feedback timing,
reward bounds, learning-rate constraints, averaging, state ownership, or the
difference between full-information and bandit feedback. There are also no
dedicated tests for that legacy module. Its replacement in `MWUA.fs` documents
the public contract and has focused tests described below.

## Suggested implementation order

1. **Completed 2026-07-15:** specify and test a standalone full-information
   exponential MWUA learner with log-space weights.
2. **Completed 2026-07-15:** add separate standard and plus non-tree
   regret-matching learners behind the same external-regret surface.
3. **Completed 2026-07-15:** add EXP3 with explicit exploration and
   importance-weighted sampled-action feedback.
4. **Completed 2026-07-15:** add exact-context MWUA, regret-matching, and EXP3
   wrappers that apply the first observation and route feedback to the same
   learner used for prediction.
5. **Completed 2026-07-15:** add EXP4 policy-expert contextual bandits for
   generalization across contexts.
6. **Completed 2026-07-15:** add the finite deterministic `Replicator` stage
   specified in [`EvolutionaryDynamics.md`](EvolutionaryDynamics.md).
7. Retire or obsolete the misleading `RegretLearner`, `Regrets`, and
   `ContextualExpert` surface after defining the desired compatibility policy.

### Carry-forward: exact-context representation optimizations

The completed exact-context API is representation-generic, but its current
implementation intentionally favors one uniform dictionary-backed path. The
following performance work remains recorded; it is not a correctness gap and
is independent of the completed finite `Replicator` stage:

- remove the per-lookup `box` used by null validation for value-type contexts,
  while continuing to reject null reference contexts;
- add comparer-aware construction for workloads that need a specialized
  `IEqualityComparer<'Context>`, while preserving structural equality as the
  default;
- consider a separate array-indexed exact-context implementation for bounded,
  dense context identifiers, avoiding dictionary hashing and one learner-table
  lookup per operation.

These paths should be added only with workload evidence. Their exit criteria
are semantic equivalence with the dictionary-backed API, focused tests for
context isolation and choice-token routing, and allocation/throughput
measurements showing that the specialization pays for its extra surface area.

At minimum, tests should cover empty inputs, invalid and non-finite values,
equal-feedback invariance, concentration on a consistently better action,
pre-update averaging, reset-to-prior behavior, long-run numeric stability,
first-observation contextual learning, and deterministic seeded sampling.

The completed online-learning stack now covers construction and feedback
validation, common-utility-shift invariance, exact one-step updates, pre-update
averaging, reset-to-prior behavior, sequential `LearnBatch`, long-run MWUA and
EXP3 stability, standard-versus-plus regret updates, fallback strategies,
importance weighting and exploration floors, deterministic seeded sampling,
choice-token ownership, read-only state snapshots, first-observation exact-
context learning, context isolation, policy-advice aggregation, and transfer
to unseen contexts through learned EXP4 policies.

## Implementation history

### 2026-07-15 — Exponential MWUA and standalone regret matching

- Added `ExternalRegretMinimizer<'Action>` with `Strategy`,
  `AverageStrategy`, `Choose`, single-round `Learn`, sequential `LearnMany`,
  `Rounds`, and `Reset`. Added simple and advanced factories for exponential
  MWUA, standard regret matching, and regret matching+.
- Used reusable flat arrays for learner state and scratch storage. MWUA keeps
  max-shifted log weights so long runs do not overflow or permanently lose a
  small positive weight merely because its reported probability underflows.
  Regret matching updates cumulative advantage over the current strategy's
  expected utility; the plus variant clips each cumulative regret after every
  complete round. Averages record the strategy in force before feedback.
- Validated nonempty actions, initial strategy mass and shape, finite learning
  rates and utilities, caller-supplied randomness, numeric update range, and
  round overflow. Feedback is evaluated completely before state mutation, and
  strategy access returns copies rather than mutable internal arrays.
- Built the library in Release for `net5.0` and `netstandard2.1`, built the
  `net8.0` test executable, and ran all 86 tests with zero failures. Ten tests
  cover the new online-learning surface. No dedicated allocation measurement
  or performance benchmark was run; the update kernels reuse their arrays but
  do not yet have an allocation or throughput gate.
- Deviations: the new MWUA is exponential Hedge rather than a repaired version
  of the legacy linear-factor update. Standalone regret matching uses its own
  stable normalization kernel rather than depending on CFR internals, keeping
  the public learner independent of the game-tree implementation and allowing
  a configured fallback strategy.

### 2026-07-15 — Batch API naming correction

- Renamed `LearnMany` to `LearnBatch` across the public API, examples, and
  tests. The behavior is unchanged: feedback is still applied sequentially in
  enumeration order and is equivalent to repeated calls to `Learn`.

### 2026-07-15 — EXP3 sampled-action regret minimization

- Added `BanditRegretMinimizer<'Action>`, opaque `BanditChoice<'Action>`
  tokens, and `Exp3.create`/`Exp3.createWith`. The single-round API is
  `Choose` followed by `Learn(choice, observedReward)`; `LearnBatch` accepts
  per-round reward functions and runs that same loop sequentially while
  evaluating only the sampled action.
- Implemented the explicit exploration mixture
  `(1 - gamma) q_i + gamma / K` and the importance-weighted update
  `eta * reward / p_i` in reusable max-shifted log-weight arrays. Averages
  record the sampling distribution before feedback. Initial strategies must
  give every action positive mass, and all observed rewards must be finite and
  in `[0, 1]`.
- Enforced exactly one outstanding choice per learner. Stale, foreign,
  duplicate, and mismatched choices are rejected; invalid rewards are atomic
  and may be retried with the same choice. A failed batch callback cancels its
  inaccessible local choice while preserving earlier successful rounds.
- Built the full solution in Release for the library's `net5.0` and
  `netstandard2.1` targets and the `net8.0` test and benchmark projects. Ran all
  95 tests with zero failures; nine EXP3 tests cover its exact update,
  averaging, token lifecycle, atomic validation, sequential batching, reset,
  exploration floor, long-run finite state, and fixed-seed reproducibility.
  No dedicated allocation measurement or performance benchmark was run.
- Deviation: batch input is a sequence of reward functions rather than a
  materialized sequence of `(choice, reward)` pairs. EXP3 choices depend on all
  preceding updates, so generating them inside the sequential loop preserves
  the intended sampling probabilities and makes the safe bulk API useful.

### 2026-07-15 — Exact-context regret minimization

- Added `ExactContextExternalRegretMinimizer<'Context, 'Action>` and
  `ExactContextBanditRegretMinimizer<'Context, 'Action>`, with public
  `ContextualMwua`, `ContextualRegretMatching`, and `ContextualExp3`
  constructors. The internal learner factory is hidden; invalid underlying
  options are validated eagerly at contextual construction.
- Stored one independent learner per structurally equal, non-null context.
  First observations are learned immediately. Full-information feedback is
  routed by its explicit context, while opaque `ContextualBanditChoice` tokens
  bind sampled rewards to the exact EXP3 learner that issued them.
- Kept single-step and sequential `LearnBatch` surfaces. Invalid rewards are
  atomic and retriable; failed local batch callbacks cancel their inaccessible
  choice without losing earlier rounds. Reset clears histories and invalidates
  a pending choice while retaining initialized contexts.
- Built the full solution in Release for `net5.0`, `netstandard2.1`, and the
  `net8.0` test and benchmark projects. Ran all 101 tests with zero failures;
  six exact-context tests cover first-observation learning, isolation,
  sequential batching, reset, token routing, retry, failed-batch recovery, and
  eager validation. No allocation measurement or benchmark was run.
- Deviation: the exact-context bandit wrapper permits one outstanding choice
  across the whole table, matching the simple `Choose`/`Learn` contract rather
  than supporting concurrent outstanding rounds for different contexts.

### 2026-07-15 — EXP4 policy-contextual bandits

- Added `ContextualPolicy<'Context, 'Action>`,
  `ContextualBanditRegretMinimizer<'Context, 'Action>`, and
  `Exp4.create`/`Exp4.createWith`. Policies are deterministic functions from a
  context to one configured action; their learned weights generalize across
  contexts.
- Implemented the EXP4 action mixture from aggregated policy probability plus
  explicit uniform exploration. Sampled rewards in `[0, 1]` are importance-
  weighted back to every policy that recommended the chosen action. Policy
  log weights are max-shifted, averages record pre-update policy mixtures, and
  contextual action strategies are available without reserving a choice.
- Enforced distinct nonempty actions, nonempty non-null policies, valid policy
  recommendations, finite parameters and rewards, opaque token ownership,
  atomic retryable failures, sequential batching, and reset-to-policy-prior.
- Built the full solution in Release for both library targets and the `net8.0`
  test and benchmark projects. Ran all 109 tests with zero failures; eight
  EXP4 tests cover its exact update, policy aggregation, cross-context
  generalization, token lifecycle, atomic validation, batch equivalence and
  recovery, reset, and invalid inputs. No allocation measurement or benchmark
  was run.
- Deviation: this first policy layer supports deterministic advice only.
  Randomized policies or full expert advice distributions remain a possible
  extension; learning rate and exploration are independent explicit inputs
  rather than being tied by a preset regret-bound schedule.

### 2026-07-15 — Exact-context configuration ownership validation

- Added constructor-time copies of action arrays and optional initial-policy
  arrays used by the hidden per-context factory. Later contexts therefore see
  the same configuration as the eagerly validated first learner even if a
  caller subsequently mutates its input arrays.
- Ran the full Release suite with 110 tests and zero failures. The added
  regression test mutates both caller-owned inputs before creating a later
  context and verifies that its actions and prior remain unchanged. No
  allocation measurement or benchmark was run.

### 2026-07-15 — Finite deterministic Replicator roadmap stage

- Completed roadmap item 6 with the separate pure evolutionary API documented
  in [`EvolutionaryDynamics.md`](EvolutionaryDynamics.md). Added immutable
  finite populations, population-dependent fitness, the continuous-time
  derivative, a positivity-preserving exponential step, and a sequential
  repeated-step runner; no `Choose`/`Learn` interaction or average strategy is
  shared with the regret learners.
- Built the full Release solution and ran all 118 tests with zero failures,
  including eight Replicator tests, then ran the standalone `replicator.fsx`
  example. No allocation measurement or performance benchmark was run. The
  particle-based continuous-trait stages remain unimplemented.

### 2026-07-15 — Shared MWUA and Replicator reweighting correction

- Moved the common max-shifted exponential reweighting calculation into an
  internal module in `MWUA.fs`. Both exponential MWUA and finite Replicator now
  delegate to that primitive, while their public state and feedback models
  remain separate.
- Added a direct one-step equivalence regression test. The full Release suite
  now passes all 119 tests, and `replicator.fsx` runs with `MWUA.fs` loaded
  before `Replicator.fs`. No allocation measurement or benchmark was run.

### 2026-07-15 — Algorithm-family compile-unit consolidation

- Merged EXP3 into `MWUA.fs`, alongside the other non-contextual
  external-regret algorithms, and merged EXP4 into `ContextualRegret.fs`,
  alongside the exact-context wrappers. Public types, modules, feedback
  contracts, and behavior are unchanged; only source ownership and load order
  changed.
- Removed the standalone `EXP3.fs` and `EXP4.fs` project entries and simplified
  `minregret.fsx` to load the two consolidated compile units. Rebuilt the full
  Release solution, ran all 119 tests with zero failures, and ran both example
  scripts successfully. No allocation measurement or benchmark was run.
