# How `cfr.fs` Works

[`cfr.fs`](EvolutionaryBayes/cfr.fs) implements Counterfactual Regret Minimization (CFR) for
finite, two-player, zero-sum games with imperfect information. It provides two
tree traversals:

- `cfr`, which expands every legal player action;
- `cfrSampled`, which implements external-sampling Monte Carlo CFR (MCCFR).

Both use CFR+ regret clipping, legal-action masks, and linearly weighted
strategy averaging.

## 1. Assumed game model

The implementation assumes:

1. exactly two players, numbered 0 and 1;
2. strictly alternating turns, with the player at depth `d` given by `d % 2`;
3. zero-sum terminal utilities;
4. a finite, fixed global array of actions;
5. at least one legal action at every non-terminal state.

The traversal does not contain explicit chance nodes. Private or random state
is supplied through the `contexts` array before a traversal begins.

Let:

- $h$ denote a public action history;
- $I$ denote an information set;
- $A(I)$ denote the legal actions at $I$;
- $\sigma_i(I,a)$ denote the probability that player $i$ selects action $a$
  at $I$;
- $u_i(z)$ denote player $i$'s utility at terminal history $z$.

For a two-player zero-sum game,

$$
u_0(z)=-u_1(z).
$$

This identity is why the recursive traversal can switch player perspective by
negating a child value.

## 2. Information sets

An information set groups decision states that the acting player cannot
distinguish. The code constructs its key as

```fsharp
let infoset = $"{stringifyContext contexts.[player]}|{history}"
```

Thus the information available to the player consists of:

- that player's entry in `contexts`, converted to a string;
- the public `history` string.

The key must contain exactly the information visible to the acting player.
Including hidden information creates strategies that can improperly condition
on it. Omitting visible information merges decisions that should be learned
separately.

The `|` delimiter reduces accidental collisions between the private-context
text and public-history text. The caller's `stringifyContext` function must
still produce stable, unambiguous representations.

## 3. Information-set state

Each information set has a `StrategyNode`:

```fsharp
type StrategyNode =
    { regretSum: double []
      strategySum: double [] }
```

For action index $a$:

- `regretSum[a]` stores cumulative counterfactual regret;
- `strategySum[a]` stores a weighted sum of training strategies.

`newStrategyNode n` creates both arrays with $n$ zeroes. Although the F# record
fields cannot be replaced, the arrays themselves are mutable. Traversals
therefore update nodes in place.

The module receives `lookup` as a callback. It must retrieve the node belonging
to an information-set key or create one if it does not exist. Both arrays in
the returned node must have the same length as the global action array.

## 4. Regret matching

CFR does not optimize a strategy directly. It tracks how much better each
action would have performed relative to the strategy actually used.

Let $R_i(I,a)$ be player $i$'s cumulative regret for action $a$ at information
set $I$. Define positive regret by

$$
R_i^+(I,a)=\max\!\left(R_i(I,a),0\right).
$$

Regret matching selects actions in proportion to positive cumulative regret:

$$
\sigma_i(I,a)=
\begin{cases}
\displaystyle
\frac{R_i^+(I,a)}
     {\sum_{b\in A(I)}R_i^+(I,b)},
& \text{if }\sum_{b\in A(I)}R_i^+(I,b)>0, \\
\displaystyle
\frac{1}{|A(I)|},
& \text{otherwise.}
\end{cases}
$$

`getStrategyMasked` implements this formula. It first replaces negative legal
regrets with zero and ignores illegal actions. If the remaining sum is zero,
it returns a uniform distribution over legal actions. Illegal actions always
receive probability zero.

For example, suppose

$$
R(I)=(-1,3,2)
$$

and the mask declares only the first two actions legal. The positive legal
regrets are $(0,3,0)$, so

$$
\sigma(I)=(0,1,0).
$$

If every legal regret is non-positive, the first two actions instead receive
probability $1/2$ each.

`ensureLength` verifies that masks, regrets, strategies, and strategy sums use
the same indexing scheme. A mismatch raises an argument error immediately.

## 5. Reach probabilities

For history $h$, the reach probability under strategy $\sigma$ factors into
the contribution of player 0, player 1, and chance:

$$
\pi^\sigma(h)
=\pi_0^\sigma(h)\,\pi_1^\sigma(h)\,\pi_c(h).
$$

The exhaustive traversal carries the two player contributions as `p0` and
`p1`. If player 0 chooses action $a$ at information set $I$, the updated reach
values are

$$
(p_0',p_1')
=\left(p_0\,\sigma_0(I,a),p_1\right).
$$

For a player 1 action they are

$$
(p_0',p_1')
=\left(p_0,p_1\,\sigma_1(I,a)\right).
$$

There is no explicit $\pi_c$ argument because `cfr.fs` does not traverse
chance outcomes. Chance weighting must be handled outside this traversal or by
extending the implementation.

## 6. Counterfactual regret

At information set $I$ belonging to player $i$, let $v_i(I,a)$ be the expected
utility obtained by selecting action $a$ and then following the current
strategy. The value of the mixed strategy is

$$
v_i(I)
=\sum_{a\in A(I)}\sigma_i(I,a)\,v_i(I,a).
$$

The instantaneous regret for action $a$ is

$$
r_i(I,a)=v_i(I,a)-v_i(I).
$$

A positive value means action $a$ would have performed better than the current
mixture. A negative value means it would have performed worse.

The exhaustive `cfr` traversal applies the counterfactual update

$$
R_i(I,a)\leftarrow R_i(I,a)
+\pi_{-i}^{\sigma}(I)
 \left(v_i(I,a)-v_i(I)\right),
$$

where $\pi_{-i}^{\sigma}(I)$ is the opponent's contribution to reaching $I$.
In the code, `oppReach` is `p1` at a player 0 node and `p0` at a player 1 node.

The acting player's own reach probability does not multiply the regret. This
is what makes the value counterfactual: the algorithm asks how well an action
would perform if the player deliberately reached $I$, while retaining the
probability that the opponent's choices lead there.

For a numerical example, suppose

$$
\sigma(I)=\left(\tfrac12,\tfrac12\right),\qquad
v(I,\cdot)=(1,-1),\qquad
\pi_{-i}(I)=0.4.
$$

Then

$$
v(I)=\tfrac12(1)+\tfrac12(-1)=0,
$$

so the regret increments are

$$
0.4\,(1-0,\,-1-0)=(0.4,-0.4).
$$

## 7. CFR+ clipping

Ordinary CFR permits cumulative regret to remain negative. CFR+ truncates it
after every update:

$$
R_i^t(I,a)
=\max\!\left(
0,
R_i^{t-1}(I,a)
+\pi_{-i}^{\sigma^t}(I)r_i^t(I,a)
\right).
$$

`clipRegretsPlus` performs this operation for every legal action. Illegal
action entries are left unchanged, although they are ignored while their mask
entry remains false.

Returning to the previous example, if regret initially equals $(0,0)$, the
unclipped result is $(0.4,-0.4)$ and the CFR+ result is

$$
R^+(I)=(0.4,0).
$$

The next regret-matched strategy therefore selects the first action with
probability one.

## 8. Why recursive utilities are negated

The action evaluation in `cfr` is

```fsharp
util.[a] <- -cfr (...)
```

At depth $d$, the value is expressed from the perspective of player
$d\bmod 2$. The child call is at depth $d+1$, so its return value is expressed
from the other player's perspective. Zero-sum utility gives

$$
v_i(h,a)=-v_{1-i}(h'),
$$

which explains the negation.

The terminal callback must consequently obey this contract:

```fsharp
reward contexts player history = Some utility
```

where `utility` is measured from the supplied `player`'s perspective. Returning
all terminal values from a fixed player's perspective would make the recursive
negations incorrect.

## 9. Exhaustive traversal: `cfr`

A normal root call starts with `d = 0` and `p0 = p1 = 1`. At each node, `cfr`:

1. computes `player = d % 2`;
2. returns immediately if `reward` identifies a terminal history;
3. obtains the legal-action mask;
4. constructs the information-set key and retrieves its node;
5. derives the current strategy with `getStrategyMasked`;
6. accumulates the current strategy for later averaging;
7. recursively evaluates every legal action;
8. calculates the expected node utility;
9. updates regrets using opponent reach;
10. clips legal regrets to zero.

The expected node utility is accumulated as

```fsharp
nodeUtil <- nodeUtil + strategy.[a] * util.[a]
```

which is the direct implementation of

$$
v_i(I)=\sum_{a\in A(I)}\sigma_i(I,a)v_i(I,a).
$$

Although `cfr` expands every legal action, it only does so for the supplied
`contexts`. It neither enumerates nor samples contexts or chance outcomes.

## 10. Average-strategy accumulation

The current regret-matched strategy can oscillate even while CFR converges.
The policy normally reported after training is therefore the reach-weighted
average strategy.

For iteration $t$, `avgWeight` defines a linear averaging weight with burn-in
$b$:

$$
w_t=
\begin{cases}
0, & t\le b, \\
t-b, & t>b.
\end{cases}
$$

`accumulateStrategyMasked` updates the exhaustive traversal's strategy sum as

$$
S_i(I,a)\leftarrow S_i(I,a)
+w_t\,\pi_i^{\sigma^t}(I)\,\sigma_i^t(I,a).
$$

Here $\pi_i^{\sigma^t}(I)$ is `ownReach`. Weighting by own reach prevents
rarely reached information sets from contributing as much as frequently
reached ones. Linear weighting gives later, generally better strategies more
influence, while burn-in discards early strategies entirely.

After training, the average policy is obtained by normalizing `strategySum` at
each information set:

$$
\bar\sigma_i(I,a)
=\frac{S_i(I,a)}{\sum_{b\in A(I)}S_i(I,b)}.
$$

If the denominator is zero, the reporting code should return a uniform
distribution over legal actions. `getStrategyMasked` is not an average-policy
function: it derives the current policy from `regretSum`.

## 11. Sampling a legal action

`sampleIndex` is used by the sampled traversal. It:

1. gathers legal action indices;
2. clamps their supplied strategy weights to non-negative values;
3. renormalizes those weights over legal actions;
4. falls back to a uniform legal distribution if the sum is zero;
5. draws one action using a cumulative-probability scan.

It returns the global action index and the normalized probability of the
selected action. The function fails if the mask contains no legal action.

## 12. External-sampling MCCFR: `cfrSampled`

External sampling reduces traversal cost by designating one `targetPlayer`, or
traverser.

At a target-player node, `cfrSampled` expands every legal action. It computes
sampled action values $\widetilde v_i(I,a)$, their strategy-weighted mean, and
updates the target player's regrets.

At a non-target node, it samples exactly one action from the current strategy
and follows only that branch. No regret update is performed at that node during
this traversal.

The external-sampling regret update is

$$
R_i(I,a)\leftarrow R_i(I,a)
+\left(
\widetilde v_i(I,a)-\widetilde v_i(I)
\right).
$$

There is no additional `oppReach` multiplier. Sampling opponent actions from
their current strategy already makes the sampled continuation an unbiased
estimator of the opponent-reach-weighted counterfactual value. Multiplying by
opponent reach again would count that probability twice.

Similarly, average strategies are accumulated only at non-target nodes:

```fsharp
if player <> targetPlayer then
    accumulateStrategyMasked (avgWeight iter burnIn)
        actionsMask node.strategySum strategy
```

Those nodes are visited by sampling that player's previous actions, so visit
frequency supplies own-reach weighting in expectation. This is why the sampled
version does not explicitly multiply the averaging weight by `ownReach`.

The `p0` and `p1` arguments are propagated through `cfrSampled`, but its current
regret and averaging updates do not read them.

## 13. `cfrSampledIteration`

One external-sampling traversal updates only its designated target player's
regrets. `cfrSampledIteration` therefore performs two traversals:

```fsharp
cfrSampled rnd 0 ...
cfrSampled rnd 1 ...
```

The first updates player 0's regrets and accumulates player 1's average
strategy. The second updates player 1's regrets and accumulates player 0's
average strategy.

Both traversals begin at depth zero. Consequently, both returned utility
estimates are expressed from root player 0's perspective. The local names
`player0Util` and `player1Util` identify the traversal target, not two
opposite-perspective utilities.

## 14. Callback contract

The traversal is adapted to a concrete game with these arguments:

- `reward contexts player history -> double option` returns `Some utility` for
  a terminal history and `None` otherwise. Utility must be from `player`'s
  perspective.
- `lookup actionCount nodeMap infoset -> StrategyNode` retrieves or creates an
  information-set node with correctly sized arrays.
- `adjustState contexts player action history -> string` applies an action and
  returns the next public history.
- `getActionsMask depth contexts history -> bool[]` identifies legal entries
  in the global action array.
- `stringifyContext contexts[player] -> string` encodes the acting player's
  private information.
- `nodeMap` stores and shares mutable information-set nodes across iterations.
- `contexts` supplies per-player private state; the code indexes players 0 and
  1.
- `actions` is the fixed global action array used by every node.

The same information-set key must always use compatible action meanings and
array lengths. Action ordering must remain stable for as long as a `nodeMap` is
reused.

## 15. Limitations, issues, and invariants

### Known correctness issues

#### Information-set keys are not player-scoped

Both traversals construct a key with

```fsharp
$"{stringifyContext contexts.[player]}|{history}"
```

Information sets belong to a particular player, but the key does not contain
`player`. If the two players produce the same context and history strings, they
will retrieve the same mutable `StrategyNode`. Their regrets and average
strategies will then be silently merged.

This can occur in a valid imperfect-information game when an action changes
the acting player without changing the public history. A robust key should
contain at least `(player, privateContext, publicHistory)`, preferably as a
structured value rather than a concatenated string.

#### A repeated information set can change strategy during one traversal

`cfr` reads a node's strategy and then immediately updates and clips that
node's regrets. If another history belonging to the same information set is
visited later in the same depth-first traversal, the later visit derives its
strategy from already-modified regrets. The traversal therefore no longer
evaluates every history under one fixed iteration strategy.

Applying CFR+ clipping separately to the contributions is also not generally
equivalent to aggregating them first:

$$
\max\!\left(0,
    \max\!\left(0,R+r_1\right)+r_2
\right)
\ne
\max\!\left(0,R+r_1+r_2\right).
$$

The current implementation is safe from this issue only if every information
set is reached at most once during a traversal. Supporting repeated visits
requires holding the iteration strategy fixed, accumulating all regret
contributions for each information set, and applying the update and clipping
after aggregation.

#### External samples are not reused by information set

At every non-target node, `cfrSampled` independently calls `sampleIndex`.
Standard external-sampling MCCFR instead samples a deterministic opponent and
chance strategy for the traversal: when the same sampled information set is
encountered again, it reuses the action previously selected for that
information set.

The independent samples may still produce unbiased individual branch values
in some games, but this is not precisely the published external-sampling
estimator and its convergence guarantee does not directly cover the
implementation. A traversal-local map from information-set key to selected
action would implement the standard sampling rule. See
[Lanctot et al., *Monte Carlo Sampling for Regret Minimization in Extensive
Games*](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).

#### `player1Util` is not player 1's utility

Both calls made by `cfrSampledIteration` begin at depth zero. The recursive
negations therefore express both return values from root player 0's
perspective. `player1Util` is the player-0-perspective estimate produced by the
traversal whose target is player 1; it is not an opposite-perspective player 1
utility.

Code consuming this pair as `(utilityForPlayer0, utilityForPlayer1)` will be
wrong. The values should be named after their traversal targets, or the API
should explicitly convert and document their perspectives.

#### Invalid target players fail silently

`cfrSampled` does not require `targetPlayer` to be 0 or 1. For any other value,
every node is treated as a non-target node: actions are sampled and average
strategies are accumulated, but no regrets are updated. Public entry points
should reject invalid target players.

### Mathematical preconditions

CFR's convergence result requires a finite extensive-form game with perfect
recall. Perfect recall means that a player never forgets information they
previously observed or actions they previously selected. The current API
cannot verify this property; it is the caller's responsibility to encode
information-set keys that preserve it.

Every history assigned to the same information set must also have the same
legal actions. Consequently, `getActionsMask` must return the same mask and
action meanings whenever lookup produces the same information-set key.

The regret update in `cfr` has no explicit chance-reach factor. It is suitable
when `contexts` are sampled according to their chance distribution. Exact
enumeration of non-uniform chance outcomes would require chance-weighted regret
updates or an explicit chance-node implementation; merely calling `cfr` once
for every context would weight all contexts equally.

### CFR+ sampling caveat

`cfrSampled` clips noisy sampled regrets using regret matching+. This is a
recognizable MCCFR+ variant, but the convergence theorem for ordinary
external-sampling MCCFR should not automatically be treated as a theorem for
this clipped estimator. Plain MCCFR+ is also known to behave poorly when
sampling variance is high; variance-reduced MCCFR+ was introduced to address
that problem. See [Schmid et al., *Variance Reduction in Monte Carlo
Counterfactual Regret Minimization Using Baselines*](https://ojs.aaai.org/index.php/AAAI/article/view/4048).

### Structural limitations

The implementation does not directly support:

- explicit chance nodes;
- more than two players;
- general-sum utilities;
- skipped or non-alternating turns;
- simultaneous actions;
- information sets whose identity cannot be represented by private context and
  public history;
- safe concurrent mutation of one shared `nodeMap`.

### Usage invariants

For correct use:

- start a complete game traversal at `d = 0` and `p0 = p1 = 1`;
- pass a monotonically increasing, one-based `iter` for linear averaging;
- return exactly `actions.Length` mask entries;
- ensure at least one action is legal at every non-terminal node;
- keep action ordering stable;
- ensure terminal rewards use the requested player's perspective;
- normalize `strategySum`, rather than `regretSum`, to report the learned
  average policy.

## 16. Future work and improvements

The redesign is a small, allocation-free CFR/MCCFR core rather than a large
framework. Stages 1 through 3a are implemented: the legacy traversals have a
reproducible safety net, an internal flat scalar storage/kernel layer exists
beside them, and the new exhaustive two-player general-sum CFR/CFR+ solver is
the deterministic oracle for later sampled modes. Vanilla two-player CFR now
returns both utilities from one exhaustive pass, while CFR+ retains its
alternating target passes. Later traversal stages will support $N$-player, general-sum games, while treating
two-player general-sum games as the primary optimized path. No special storage,
solver, or scheduling support will be added for graphical, polymatrix, or
hypergraphical games.

The future public API will expose CFR, CFR+, MCCFR, and MCCFR+ as distinct
solver modes. The Stage 2 internal core already centralizes these four modes,
but the unchanged public `cfr` and `cfrSampled` functions still always apply
CFR+ regret clipping and linear averaging.

### Stage summary

The redesign is divided into six required milestones and two performance-gated
extensions. Each stage ends with a compiling, tested, measurable result; no
stage requires an unfinished public mode to remain exposed.

| Stage | Milestone | Result at the end of the stage |
| --- | --- | --- |
| 1 | Correctness and performance baseline — completed 2026-07-12 | The current solver is characterized by deterministic tests, analytical fixtures, and repeatable allocation and throughput measurements. |
| 2 | Flat scalar core — completed 2026-07-12 | Dense player-scoped information sets, packed SoA tables, scalar kernels, and reusable workspaces exist beside the unchanged public solver. |
| 3 | Exhaustive two-player solver - completed 2026-07-12 | An internal target-utility, two-player general-sum implementation of `CFR` and `CFRPlus` is correct, allocation-free after warm-up, and is the oracle for small games. |
| 3a | Immediate exhaustive hot-path optimization - completed 2026-07-12 | Vanilla two-player CFR returns both utilities from one traversal and exhaustive regret application clears deltas in the same slot pass. A proposed unchecked touch helper was benchmarked and rejected. |
| 4 | Sampled two-player solver | `MCCFR` and `MCCFRPlus` share the same tables and kernels; all four modes are then exposed through one public API. |
| 5 | $N$-player and chance completion | Every mode supports finite sequential $N$-player general-sum games and explicit chance nodes without slowing the selected two-player path. |
| 6 | Production cutover | The legacy recursive implementation is removed from production, the minimal API is frozen, and memory, allocation, convergence, and migration checks pass. |
| 7 | Profile-guided single-thread optimization | Small-action scalar fast paths and, only where profitable, isolated SIMD kernels improve measured throughput without changing the design. |
| 8 | Optional deterministic parallel batches | Parallel MCCFR is added only if representative workloads justify its complexity and it passes reproducibility and memory gates. |

Stage 3a passed before Stage 4 began. Stages 1 through 6 define the required
production core. Stage 7 may legitimately
end with scalar code if SIMD does not win. Stage 8 may legitimately end with a
documented no-go result; parallel execution is not required to consider the
core complete.

### Target game model

The current implementation infers the player from `depth % 2` and changes
utility perspective by negating recursive results. Those operations restrict it
to two-player, alternating, zero-sum games.

The exhaustive solver receives the acting player from the game state
and always evaluates utility for one designated target player. Its minimal
internal contract is:

```fsharp
type IExhaustiveGame<'State> =
    abstract TryGetTerminalUtility:
        state:'State * targetPlayer:int * utility:byref<double> -> bool
    abstract Actor: state:'State -> int
    abstract InformationSetId: state:'State -> int
    abstract ActionCount: state:'State -> int
    abstract NextState: state:'State * localAction:int -> 'State
    abstract ChanceProbability: state:'State * localAction:int -> double
```

`Actor` currently returns player zero, player one, or the reserved
`ChanceActor = -1`; every other value is rejected. Stage 5 will generalize the
same rule to a player in $0,\ldots,N-1$ or the reserved integer for chance.
`InformationSetId` is read only at player nodes. Every integer in
`0 .. actionCount state - 1` is a legal local action, so the core does not need
to allocate or scan a Boolean action mask. At a chance node,
`chanceProbability` supplies a normalized fixed distribution; chance has no
regret or average-strategy row.

The recursive function is conceptually

```fsharp
traverse targetPlayer state
```

and returns that target player's utility at every depth. Child values are
propagated directly rather than negated. Stage 3 supports:

- two players now, with the same target-utility structure intended for $N$ players in Stage 5;
- general-sum terminal utilities;
- consecutive turns by the same player;
- arbitrary player order determined by the game state.

Information-set IDs must be unique across players. Private observations,
public history, and player identity are the game's responsibility when mapping
a state to a dense integer ID.

For an eventual $N$-player game, one traversal is performed for each target
player whose regrets are being updated. The optimized vanilla two-player CFR
path instead returns both utilities in one value tuple and updates the acting
player's regret while retaining the deferred, simultaneous application
boundary. CFR+, sampled modes, and the $N$-player fallback retain target-player
traversal semantics.

Supporting general-sum and multiplayer games does not extend the two-player
zero-sum equilibrium theorem. The solver will minimize counterfactual regret,
but its documentation and API must not claim that average independent
strategies necessarily converge to a Nash equilibrium outside game classes
where that result is established.

### Necessary production core

The optimized production implementation should contain only:

1. flat regret and average-strategy storage;
2. one reusable traversal workspace;
3. allocation-free regret matching;
4. allocation-free legal-action sampling;
5. one target-player exhaustive traversal (implemented in Stage 3);
6. one target-player external-sampling traversal;
7. a small iteration function that invokes the selected traversal for each
   player.

The production core may additionally contain one short, two-player-only
vanilla CFR traversal. This is a specialization of item 5, not a second table
model or a general utility-vector framework. It returns
`struct (utility0, utility1)` and reuses the same packed tables, regret-delta
buffer, strategy scratch, utility scratch, and update kernels.

Exhaustive CFR will remain a supported public option for small games, not only
a test implementation. The public API should expose the four intended
algorithm choices directly:

```fsharp
type SolverMode =
    | CFR
    | CFRPlus
    | MCCFR
    | MCCFRPlus
```

These modes represent two independent internal choices:

| Public mode | Traversal | Regret update | Default averaging |
| --- | --- | --- | --- |
| `CFR` | exhaustive | signed cumulative regret | uniform |
| `CFRPlus` | exhaustive | cumulative regret clipped at zero | linear |
| `MCCFR` | external sampling | signed sampled regret | uniform |
| `MCCFRPlus` | external sampling | sampled regret clipped at zero | linear |

All four modes will share tables, workspace, terminal utility semantics,
information-set IDs, and the underlying scalar kernels. Exhaustive and sampled
traversals should remain separate and short so that the hot path does not branch
on traversal type at every node. The signed and clipped regret-update loops may
also be specialized if profiling shows that an invariant branch is material;
the public design does not require four unrelated implementations.

`CFRPlus` uses target-player traversals in sequence, giving it alternating
player updates. `MCCFRPlus` remains explicitly identified as a sampled CFR+
variant, with the variance and convergence caveats described earlier in this
document. Exhaustive CFR also serves as the deterministic correctness oracle
against which both sampled implementations can be tested.

For a target-player exhaustive traversal, only two reach products are needed
even in an $N$-player game:

- the target player's own reach;
- the combined reach of every other player and chance.

An action by the target player multiplies own reach. An action by any other
player multiplies external reach. This avoids allocating or updating an
$N$-element reach vector in the recursive hot path.

### Data layout

The Stage 2 core stores large numeric state in two flat structure-of-arrays
fields. Information sets use dense global IDs, and every row records its owning
player:

```fsharp
type SolverTables =
    { Regrets: double[]
      StrategySums: double[] }

[<Struct>]
type InfoSetMeta =
    { Owner: int
      Offset: int
      ActionCount: int }

type PackedTable =
    { PlayerCount: int
      InfoSets: InfoSetMeta[]
      Tables: SolverTables
      SlotCount: int }
```

`PackedTable.create` accepts dense `InfoSetDefinition` values, canonicalizes
them by ID, validates ownership and positive action counts, and constructs all
offsets. `PackedTable.ofMetadata` is the lower-level checked boundary for
already constructed metadata; it rejects gaps, overlaps, and non-contiguous
rows. Scalar kernels receive these validated slices without rescanning
metadata. The Stage 3 traversal still checks that each dynamic game state has a
valid actor and information-set ID and agrees with the row's owner and length.

For information set $I$ and local action slot $a$, the numerical index is

$$
j=\operatorname{meta}[I].\operatorname{Offset}+a,
\qquad 0\le a<\operatorname{meta}[I].\operatorname{ActionCount}.
$$

Rows contain legal actions only. `nextState` interprets the local action slot,
so the solver does not need a Boolean mask or a second array mapping slots to
game-specific action values. A rectangular game is simply the special case
where every row has the same `ActionCount`; it uses the same traversal and does
not introduce a representation branch.

This hybrid uses SoA for vectorizable numerical fields and compact AoS metadata
for fields read together. It replaces per-information-set heap arrays, removes
pointer chasing, keeps numerical rows contiguous, and makes $M$ exactly the
number of legal information-set/action pairs.

The dated benchmark history also contains a retrospective Stage 0 memory run
for the unchanged dictionary/string/array representation. It distinguishes the
exact legacy numeric payload from measured managed retained memory and total
construction allocation; see
[`CFR_BENCHMARK_RESULTS.md`](CFR_BENCHMARK_RESULTS.md).

### Allocation-free workspace

Stage 2 provides separate `ExhaustiveWorkspace` and `SampledWorkspace` records,
so construction allocates only the storage required by the selected traversal.
Both own this common depth-local portion:

```fsharp
type TraversalScratch =
    { Strategies: double[]
      Utilities: double[]
      AverageEpochs: int[]
      mutable Epoch: int }
```

`Strategies` and `Utilities` provide one slice per active recursion depth. The
strategy calculation writes directly into its slice instead of allocating
temporary and result arrays. `AverageEpochs` ensures that one information set
contributes to the average at most once in the averaging pass, even if several
histories map to it.

The exhaustive workspace additionally owns one reusable `double[M]` regret-
delta array plus touched-row bookkeeping. It keeps the strategy profile fixed
while all histories in an information set contribute their regret, then applies
and clears only the rows touched by the relevant update boundary. Vanilla CFR
applies every touched row after its one complete two-utility traversal; CFR+
applies and clips after each complete target pass. Application writes the new
regret and zeroes the corresponding delta in the same slot loop.

The sampled workspace instead owns `SampledActions` and `SampleEpochs`, each
indexed by information-set ID. `SampledActions` caches the non-target action
selected for an information set during an external-sampling traversal.
`SampleEpochs` avoids clearing the cache before every traversal: an entry is
valid only when its epoch matches the current traversal epoch. This implements
consistent external sampling without a traversal-local dictionary.

It also owns reusable append-only `DeltaIndices` and `DeltaValues` buffers.
They hold only target slots touched by the sampled passes, allowing vanilla
MCCFR to defer both players' updates until the simultaneous iteration boundary
without allocating the exhaustive `double[M]` delta array. MCCFR+ drains and
reuses the same buffers after each alternating target pass.

All caches use integer epochs. Advancing past `Int32.MaxValue` explicitly
clears the associated epoch array and restarts at epoch one, preventing a stale
entry from becoming valid after wraparound. Sampled delta capacity is fixed at
construction; exceeding it fails explicitly instead of resizing inside a
traversal.

An exhaustive solver does not allocate the sampled cache, and a sampled solver
does not allocate the dense exhaustive delta array. Mode selection occurs at
the iteration entry point, not at every node.

The hot path should allocate no arrays, strings, tuples, closures, or dictionary
entries after solver initialization and initial depth-capacity warm-up.

### Memory efficiency

Let $M$ be the total number of stored information-set/action slots. With
double-precision regrets and average-strategy sums, the essential persistent
numeric storage is

$$
8M+8M=16M\ \text{bytes}.
$$

The following measures reduce memory without complicating the mathematical
core.

#### Store only legal action slots

The packed metadata layout above is the one production representation. It makes
$M$ equal to the number of legal information-set/action pairs rather than
`informationSetCount * globalActionCount`. Rectangular games use the same
representation with equal row lengths, so no layout tag or per-node branch is
needed.

#### Pay once for correct exhaustive aggregation

The $16M$ figure is persistent policy state, not total solver memory. Correct
exhaustive traversal must hold the iteration strategy fixed while repeated
histories in an information set are aggregated. Its reusable regret-delta array
therefore costs another

$$
8M\ \text{bytes}.
$$

Touched-row IDs and epochs add $O(K)$ integers for $K$ information sets and
avoid clearing all $M$ entries after every target pass. This extra dense buffer
is acceptable because exhaustive CFR is intended for small games and as a
correctness oracle. External-sampling solvers do not allocate it.

#### Do not store current strategies

Current strategies are derived from regrets into the depth-local scratch
buffer. Persisting them for every information set would add another $8M$ bytes
without providing information that cannot be recomputed cheaply.

#### Always retain average-strategy sums

`StrategySums` account for half of the persistent numerical storage, but they
are mandatory solver state. CFR's equilibrium guarantees generally apply to
averaged play; the current regret-matched strategy can oscillate and must not be
silently presented as the solver's equilibrium output.

The production API will therefore always track average strategies for all four
solver modes. There will be no configuration for disabling average-strategy
tracking. This keeps the output contract unambiguous and avoids another mode in
the hot traversal.

#### Allocate scratch space by observed depth

The workspace should grow geometrically when a deeper state is first reached
and then be reused. It should not allocate for an excessively conservative
theoretical maximum depth. For maximum observed depth $D$ and maximum action
count $A$, the reusable strategy and utility scratch arrays require roughly

$$
16DA\ \text{bytes}.
$$

This is normally small compared with the persistent information-set tables.
The common average-epoch array adds approximately $4K$ bytes for $K$
information sets.

#### Keep sampled-action caching compact

The epoch-based sampling cache uses two integer arrays, approximately eight
bytes per information set. It exists only for external sampling; exhaustive
workspaces do not need it. The sampled delta log uses one integer index and one
double value per touched slot, approximately $12T$ bytes at capacity $T$. If
action count is known to fit in a smaller integer, the cached action component
may be narrowed, but a generic numeric-storage framework is not warranted
solely for this saving.

#### Use sparse parallel deltas

Future parallel workers must not each allocate a complete copy of the regret
and strategy tables. Worker-local buffers should contain only touched action
indices and their deltas, followed by deterministic reduction into the shared
tables. This keeps parallel memory proportional to sampled work rather than

$$
\text{workerCount}\times M.
$$

#### Avoid retaining the game tree

The solver should traverse a compact value-type state or integer state ID. It
should not retain history strings or allocate tree nodes merely to perform
recursion. Precomputed transition tables are a game-level time/memory tradeoff
and are not required by the CFR core.

#### Retain double precision by default

Using `float32` would halve regret and average storage and double SIMD lane
width, but cumulative regrets and linearly weighted strategy sums can run for
many iterations. Precision loss must therefore be evaluated through
exploitability and convergence benchmarks. The core should use `double` by
default and should not become generic over numeric type. A separate specialized
single-precision implementation is justified only by measured memory pressure.

### Allocation-free scalar kernels

Regret matching needs two simple passes over legal actions:

1. sum positive regrets and count legal actions;
2. write either normalized positive regrets or a uniform legal strategy into
   the reusable strategy buffer.

Action sampling likewise needs only two passes:

1. obtain the total legal probability;
2. scan cumulative probability until it reaches the random draw.

It does not require temporary `legal`, `raw`, or `probs` arrays.

The first optimized implementation should use clear scalar loops. Games with
two or three actions should receive small scalar fast paths if benchmarks show
a benefit. SIMD should be confined to isolated regret-update, clipping, and
strategy-accumulation kernels, and added only when typical action counts or
batched traversal widths are large enough to amortize vector setup and tail
handling.

### Two-player fast path

Two-player general-sum games are the main performance target. Their terminal
state can provide two utilities without heap allocation, for example as a
struct tuple, and the selected target utility can be propagated unchanged
through the traversal:

```fsharp
ValueSome(struct (utility0, utility1))
```

The optimized path must not use zero-sum negation. It selects `utility0` or
`utility1` according to the target player. The generic $N$-player interface may
still expose target-player utility directly so that neither path needs to
allocate a utility array.

Specialization should occur when the solver is constructed or when its entry
point is selected, not through a game-kind branch at every tree node. The
general and two-player paths should share storage and mathematical kernels
instead of becoming two unrelated implementations.

### Correctness before parallelism

Parallel traversal will not mutate shared regret arrays directly. A later
parallel implementation should:

1. freeze the strategy used by a batch;
2. accumulate worker-local regret and strategy deltas;
3. reduce those deltas deterministically;
4. apply regret updates and CFR+ clipping after aggregation.

This avoids races and also prevents repeated histories in one information set
from changing the strategy halfway through an iteration. Locks around every
information-set update are not part of the intended design.

### Explicit non-goals

The CFR module will not add:

- graph, edge, or hyperedge data structures;
- polymatrix-specific payoff reduction;
- graph coloring or edge scheduling;
- independent per-edge regret learners;
- GPU graph-processing abstractions;
- a general-purpose entity/component framework.

Graph-structured simultaneous games are better handled by a separate normal-
form regret-matching implementation. A game may still use a graph internally to
calculate state transitions or terminal utility, but the CFR core will treat it
like any other game state.

### Detailed staged implementation plan

The stages below are intentionally sequential. A later stage may be developed
on a branch, but it is not considered started for planning purposes until the
previous stage's exit criteria pass. At every boundary, the repository builds,
all enabled tests pass, benchmark inputs and seeds are recorded, and the
production API contains no placeholder or silently degraded mode.

#### Stage 1: establish the correctness and performance baseline — completed 2026-07-12

**Milestone outcome.** The current implementation remains behaviorally
unchanged, but its valid behavior, known limitations, convergence, allocations,
and traversal cost are reproducible. This is a complete safety-net milestone;
it does not depend on any redesigned solver code.

**Implemented result.** `EvolutionaryBayes.CFR.Tests` is a dependency-free
executable test project containing analytical fixtures, deterministic legacy
tests, and named acceptance invariants for every Section 15 issue.
`EvolutionaryBayes.CFR.Benchmarks` records node visits, elapsed time, and
thread-local allocation for the required action-count and depth matrix. The
checked-in result is
[`CFR_BENCHMARK_RESULTS.md`](CFR_BENCHMARK_RESULTS.md).

**Implementation work.**

1. Add one F# test project and one small benchmark executable to the solution.
   Keep the benchmark harness dependency-light: it needs deterministic warm-up,
   elapsed time, node counts, and
   `GC.GetAllocatedBytesForCurrentThread`, not a benchmarking framework in the
   runtime library.
2. Instrument test games with a node counter outside the solver. Record:

   $$
   \text{nodes per second}
   =\frac{\text{visited nodes}}{\text{elapsed seconds}},
   \qquad
   \text{bytes per iteration}
   =\frac{\text{allocated bytes}}{\text{iterations}}.
   $$

3. Preserve the current `cfr` implementation as a reference only for games
   where its assumptions hold: two players, alternating turns, zero-sum
   utilities, unique player-scoped keys supplied by the fixture, and no repeated
   information-set visit in one traversal. Analytical calculations, not the
   legacy function, are authoritative outside that subset.
4. Add small, deterministic `.fs` fixtures for:

   - one information set with hand-computed strategy and regret updates;
   - an alternating two-player zero-sum tree with unique information sets;
   - a compact imperfect-information zero-sum game with a known equilibrium;
   - a repeated-information-set tree whose correct aggregate update is known;
   - a two-player general-sum terminal tree used later to catch negamax.

5. Characterize the current four helper behaviors separately: masked regret
   matching, CFR+ clipping, linear averaging with burn-in, and legal-action
   sampling. The tests should state that the two current traversals are
   plus-style implementations rather than tests for unavailable vanilla modes.
6. Record a baseline matrix covering action counts $2,3,4,8,16,$ and $32$,
   shallow and deep trees, and exhaustive and sampled traversal. Build both
   library target frameworks, but run performance measurements only through a
   runnable benchmark target. Framework upgrades are outside this CFR
   milestone.

**Verification.**

- Hand-computed regret and average-strategy values match exactly or within a
  documented floating-point tolerance.
- Fixed random seeds reproduce the same sampled action sequence.
- The compact zero-sum fixture shows decreasing exploitability under averaged
  play.
- Known-bug fixtures have independently checked analytical expectations. They
  are clearly labelled as Stage 3 solver acceptance cases rather than being
  forced through the legacy traversal.
- The benchmark report includes source revision, build configuration, runtime,
  processor, iteration count, seed, nodes visited, elapsed time, and allocated
  bytes.

**Exit criteria.** The solution and test project build cleanly; all baseline
tests pass; one checked-in benchmark result can be reproduced; and every known
correctness issue in Section 15 has a named regression fixture or invariant.

#### Stage 2: build the flat allocation-free scalar core — completed 2026-07-12

**Milestone outcome.** A complete internal storage and kernel layer exists
beside the unchanged public solver. It can be tested and benchmarked without a
game traversal. No public mode is added yet, so the stage is independently
shippable without exposing unfinished algorithms.

**Implemented result.** `CFRCore.fs` contains the internal four-mode semantics,
canonical packed table builder, separate exhaustive and sampled workspaces,
wrap-safe epoch caches, sparse sampled delta log, and checked and unchecked
scalar kernels. The unchecked training kernels operate on caller-provided array
slices and allocate zero steady-state bytes after warm-up. The two persistent
numeric arrays occupy exactly $16M$ bytes for $M$ legal action slots.
The separate Stage 2 memory, allocation, and scalar-throughput results are in
[`CFR_BENCHMARK_RESULTS.md`](CFR_BENCHMARK_RESULTS.md).

**Implementation work.**

1. Add the four-case `SolverMode` internally and centralize its semantics:

   | Mode | Traversal | Regret transform | Averaging weight | Update schedule |
   | --- | --- | --- | --- | --- |
   | `CFR` | exhaustive | signed | uniform | simultaneous |
   | `CFRPlus` | exhaustive | clipped | linear | alternating |
   | `MCCFR` | external sampling | signed | uniform | simultaneous |
   | `MCCFRPlus` | external sampling | clipped | linear | alternating |

   For burn-in $b$ and one-based iteration $t$, use

   $$
   w_t^{\mathrm{vanilla}}
   =\mathbf 1[t>b],
   \qquad
   w_t^{+}
   =\max(0,t-b).
   $$

2. Implement the canonical packed table builder. It validates dense IDs,
   metadata ownership, positive row lengths, non-overlapping contiguous
   offsets, and total slot count once, before training.
3. Implement the two persistent `double[]` arrays, compact `InfoSetMeta[]`,
   and mode-specific reusable workspaces described above. Epoch wraparound must
   perform one explicit cache clear and restart at epoch one; it must never make
   stale entries valid.
4. Implement allocation-free scalar kernels over an
   `(offset, actionCount)` row:

   - regret matching;
   - uniform fallback;
   - probability sampling;
   - mandatory average-strategy accumulation;
   - signed regret application;
   - aggregate-then-clip regret application;
   - average-strategy normalization for reporting.

5. Define the update transaction explicitly. Traversal code contributes
   $\Delta R_j$ values, and the boundary kernel applies

   $$
   R_j\leftarrow
   \begin{cases}
   R_j+\Delta R_j, & \text{CFR or MCCFR},\\[2mm]
   \max(0,R_j+\Delta R_j), & \text{CFR+ or MCCFR+}.
   \end{cases}
   $$

   CFR+ clipping never occurs separately for two histories contributing to the
   same row.
6. Use `while` loops and caller-provided scratch slices from the start. Do not
   introduce sequences, list comprehensions, per-row closures, numeric
   generics, a table interface hierarchy, or a stored current-strategy table.

**Verification.**

- Kernel tests cover zero, positive, negative, non-finite, and very large
  regrets; one, two, and many actions; uniform fallback; and probability
  normalization. Checked entry points reject non-finite values; the unchecked
  training kernel relies on the finite-utility invariant and adds no hot-path
  scan solely for validation.
- Signed modes retain negative cumulative regret; plus modes clip only after
  the complete supplied delta has been added.
- Table construction rejects duplicate, out-of-range, cross-player, gapped, and
  zero-action metadata.
- Reporting an untouched average row returns a uniform legal distribution.
- After workspace warm-up, repeated kernel benchmarks allocate zero bytes.
- The measured persistent numeric storage is $16M$ bytes, excluding documented
  metadata and mode-specific workspace.

**Exit criteria.** The internal kernels pass all unit and property-style tests,
their allocation benchmark is zero after warm-up, and the legacy public solver
still produces the Stage 1 baseline results.

#### Stage 3: implement exhaustive two-player general-sum CFR and CFR+ - completed 2026-07-12

**Milestone outcome.** A complete two-player exhaustive solver exists for small
games. It supports both signed-regret CFR and alternating-update CFR+, explicit
chance, consecutive turns, and two independent terminal utilities. It becomes
the deterministic correctness oracle for later stages.

**Implemented result.** `CFRCore.fs` now contains the generic internal
`IExhaustiveGame<'State>` contract, reserved `ChanceActor`, and
`ExhaustiveSolver<'State,'Game>`. The solver uses direct target utilities,
explicit actors and chance, legal local action slots, reusable depth slices,
epoch-guarded average accumulation, and the Stage 2 packed tables. CFR defers
both players' aggregate regret deltas to a simultaneous boundary; CFR+ applies
and clips one complete target pass before traversing the other player. The
legacy public API remains unchanged until Stage 4 supplies the sampled modes.
Correctness, allocation, memory, and interleaved throughput evidence is in
[`CFR_BENCHMARK_RESULTS.md`](CFR_BENCHMARK_RESULTS.md).

**Implementation work.**

1. Introduce the minimal game contract shown in the target game model:
   terminal utility for a requested player, current actor, dense information-set
   ID, local action count, state transition, and chance probability. Terminal
   detection occurs before any actor or information-set lookup.
2. Reserve one actor integer for chance and reject every other actor outside
   $0,\ldots,N-1$. At player nodes, verify that `InfoSetMeta.Owner` matches the
   actor and the row length matches the state's action count. Stage 3 retains
   these contract checks in Release builds.
3. Replace depth parity and recursive negation with
   `traverse targetPlayer state ownReach externalReach`. A terminal returns
   $u_{\text{target}}(z)$ directly. For a target action $a$,

   $$
   \pi_i\leftarrow\pi_i\sigma_i(I,a);
   $$

   for another player's action or a chance action,

   $$
   \pi_{-i}\leftarrow\pi_{-i}p(a).
   $$

4. At a target information set, evaluate every action and accumulate

   $$
   \Delta R_i(I,a)
   \mathrel{+}=
   \pi_{-i}(I)\left(v_i(I,a)-v_i(I)\right)
   $$

   into the exhaustive delta buffer. Apply the row only after every history
   contribution in the pass has been aggregated.
5. Accumulate the target player's average strategy once per information set per
   target pass:

   $$
   S_i(I,a)\mathrel{+}=
   w_t\,\pi_i(I)\,\sigma_i(I,a).
   $$

   An epoch check prevents duplicate accumulation when multiple histories share
   $I$.
6. Give vanilla CFR simultaneous iteration semantics: run both target-player
   traversals against unchanged regrets, retain all deltas, and apply them after
   both finish. Give CFR+ alternating semantics: run one target pass, apply and
   clip its complete deltas, then run the other target against the updated
   profile.
7. Keep the new implementation internal until sampled modes are complete. The
   legacy API remains the public fallback during this stage, while exhaustive
   tests and benchmarks invoke the new solver through the test assembly.

**Verification.**

- A one-iteration hand calculation checks both players' action values, node
  values, regret deltas, and strategy sums.
- A terminal payoff such as `struct (3.0, 1.0)` returns $3$ for target zero and
  $1$ for target one; no negation reconstructs either value.
- Consecutive turns by one player and non-alternating player order produce the
  analytical result.
- Exact chance enumeration matches a hand-computed expected value and includes
  chance in external reach.
- Two histories in one information set use the same pre-update strategy,
  aggregate their deltas, and clip once.
- Two players with otherwise identical private and public observations cannot
  collide because metadata ownership and dense IDs differ.
- On a compact two-player zero-sum fixture, averaged CFR and CFR+ approach the
  known value and CFR+ does not contain negative regrets.
- The warmed exhaustive iteration allocates zero bytes for a value-type game
  state.

**Exit criteria.** Every exhaustive correctness fixture passes, the new solver
beats or matches the legacy node throughput on the representative small-game
baseline, and its only per-solver numerical memory beyond `Regrets` and
`StrategySums` is the documented exhaustive delta workspace and depth scratch.

#### Stage 3a: optimize the exhaustive hot path - completed 2026-07-12

**Milestone outcome.** Two-player vanilla CFR visits the exhaustive game tree
once per iteration rather than once per target player. The optimization remains
strictly two-player-specific. The target-player traversal stays intact as the
simple fallback for $N$-player CFR and as the required basis for CFR+, MCCFR,
and MCCFR+. Independently, every exhaustive mode applies and clears touched
regret deltas in one pass. The proposed unchecked touched-row registration did
not pass its benchmark gate and is not part of the implementation.

**Implemented result.** `ExhaustiveSolver` selects the short
`traverseTwoPlayerCfr` recursion for vanilla CFR. It carries player-zero,
player-one, and chance reach as scalars, returns `struct (utility0, utility1)`,
stores only the acting player's child utilities in the existing depth scratch,
and applies all regret deltas only after the traversal. CFR+ continues to use
two alternating target-player passes. `applyRegretDeltaAndClearUnchecked`
combines application and clearing without changing table or workspace size.
The retained target-pair path is also the direct correctness oracle used by
the tests and benchmark.

**Correctness basis.** The specialized recursion returns both players' values:

$$
T(s)=\operatorname{struct}(v_0(s),v_1(s)).
$$

At a terminal $z$ it requests both independent utilities,

$$
T(z)=\operatorname{struct}(u_0(z),u_1(z)),
$$

so general-sum behavior does not use negation. At a player node owned by $i$,
the same pre-iteration strategy produces both node values,

$$
v_j(I)=\sum_{a\in A(I)}\sigma_i(I,a)v_j(I,a),
\qquad j\in\{0,1\},
$$

and only the acting player's counterfactual regret is accumulated:

$$
\Delta R_i(I,a)
\mathrel{+}=
\pi_{-i}(I)\left(v_i(I,a)-v_i(I)\right).
$$

Because no regret row is applied until the complete traversal finishes, every
information set observes the unchanged pre-iteration profile and vanilla CFR's
simultaneous-update semantics are preserved.

**Implemented work.**

1. Added one short `traverseTwoPlayerCfr` recursion returning
   `struct (double * double)`. It carries player-zero reach, player-one reach,
   and chance reach as three scalar doubles.
2. At a player node, both node values accumulate in scalar locals. The solver reuses the
   existing depth utility slice for only the acting player's per-action values,
   which form that row's regret delta after enumeration. No $N$-utility vector,
   additional $M$-sized table, or additional depth utility array was added.
3. The acting player's mandatory average strategy accumulates once per
   information set using its own reach and the existing epoch guard.
4. Chance nodes enumerate the fixed distribution once, weight both returned
   utilities by the same chance probability, and multiply chance reach for
   descendant counterfactual updates.
5. The solver selects the specialization only for `CFR` with exactly two players.
   `CFRPlus` remains on two alternating target passes. When Stage 5 enables more than
   two players, vanilla CFR uses one target pass per player rather than growing
   a dynamic utility-vector hot path.
6. Regret application and delta clearing are fused for every exhaustive mode. For
   each touched action slot perform

   $$
   R_j\leftarrow
   \begin{cases}
   R_j+\Delta_j, & \text{signed mode},\\
   \max(0,R_j+\Delta_j), & \text{plus mode},
   \end{cases}
   \qquad
   \Delta_j\leftarrow0,
   $$

   in one loop, then the touched-row count resets. Rows are no longer revisited
   solely to call `Array.Clear`.
7. The planned unchecked touched-row helper was implemented and measured, then
   removed because it was materially slower than the existing checked helper
   in two interleaved benchmark sessions. The simpler checked helper remains.

**Verification performed.**

- Compared one-pass and existing two-pass CFR from identical tables on every
  Stage 3 analytical fixture. Root utilities, regret deltas, and strategy sums
  agree within the existing floating-point tolerance.
- Retained exact tests for general-sum terminal utilities, chance reach,
  consecutive turns, repeated information sets, and matching-pennies
  convergence.
- Confirmed that CFR+ behavior and its two-traversal count are unchanged.
- Confirmed zero steady-state allocation for a value-type state and no increase
  in persistent or workspace array payload.
- Compared fused apply-and-clear against the existing apply-then-clear sequence
  for signed and clipped rows, including repeated-information-set aggregates;
  every touched delta is zero immediately after the boundary.
- Retained deduplication, visit-order, and rejection tests on checked touched-row
  registration. The discarded unchecked variant is recorded as a benchmarked
  no-go rather than retained as dead optimization code.
- Added an interleaved repeated-median benchmark comparing the completed Stage 3
  two-pass CFR implementation with the single-pass specialization on the same
  four-action, depth-five fixture. Record node visits as well as elapsed time.
  Record the fused-boundary result and the rejected unchecked-touch experiment
  as separate sub-results within the Stage 3a benchmark section so their
  effects are not attributed silently to the single-pass traversal.

**Exit result.** The specialization matches the two-pass oracle on all
correctness fixtures, reduces two-player vanilla CFR visits from two complete
trees to one, allocates zero steady-state bytes, adds no solver-sized storage,
and improves median elapsed time on the representative workload. Fused clearing
matches its scalar reference and did not regress median throughput. The
unchecked touching path did not beat the checked path and was removed.

**Recorded later optimizations.** Repeated dynamic game-contract validation is
not removed in Stage 3a because doing so changes failure behavior. Stage 7 must
measure actor, action-count, information-set ownership/length, terminal, and
chance-normalization checks before considering one construction-time selection
between checked and trusted traversal. The hot loop must not branch on that
selection, the checked path must remain available, and the trusted path is kept
only if its end-to-end benefit clears the Stage 7 merge gate. Small-action
unrolling, generic-loop tuning, and SIMD also remain explicitly in Stage 7.

#### Stage 4: add two-player MCCFR and MCCFR+

**Milestone outcome.** All four modes work for two-player general-sum games
through one table layout and one public solver API. External sampling is the
published pure-strategy sampling scheme, not independent resampling at repeated
information sets.

**Implementation work.**

1. Add a second short target-player traversal. At the target player's nodes it
   expands every local action; at the other player's nodes and chance nodes it
   samples one action.
2. Cache the sampled action by non-target information-set ID and traversal
   epoch. Reuse it whenever that information set is encountered again in the
   same traversal. Chance is sampled from its fixed distribution.
3. Use the sampled counterfactual regret update

   $$
   \Delta R_i(I,a)
   =\widetilde v_i(I,a)-\widetilde v_i(I)
   $$

   without multiplying by external reach again. Apply signed or clipped
   accumulation according to the mode, at the same pass boundaries defined in
   Stage 2.
4. For the optimized two-player path, accumulate the other player's average
   strategy at sampled non-target nodes. The paired traversal then accumulates
   the first player's average. Keep this standard two-player estimator separate
   from the generic $N$-player averaging rule introduced in Stage 5.
5. Select exhaustive versus sampled traversal once in the public iteration
   entry point. Regret matching, row metadata, terminal semantics, strategy
   sums, and reporting remain shared.
6. Accept a deterministic seed or caller-owned random source at construction.
   Reject an invalid target, a non-normalized chance row, an empty action row,
   or a sampled probability outside $[0,1]$ at the checked boundary.
7. Promote the final public `SolverMode` with exactly `CFR`, `CFRPlus`,
   `MCCFR`, and `MCCFRPlus`. At this stage construction explicitly requires
   `PlayerCount = 2`; Stage 5 removes that temporary restriction.

**Verification.**

- With a fixed strategy profile, the sample mean of MCCFR regret deltas
  approaches the exhaustive delta:

  $$
  \frac{1}{S}\sum_{s=1}^{S}\Delta\widetilde R_s
  \longrightarrow \Delta R
  $$

  within a tolerance chosen from the observed standard error, not an arbitrary
  single-seed comparison.
- A repeated non-target information set receives one cached action throughout a
  traversal.
- Fixed seeds give reproducible samples and table values.
- `MCCFR` can retain negative cumulative regret; `MCCFRPlus` cannot.
- Uniform and linear average weights differ exactly as the mode table states,
  and every average row normalizes to one.
- Sampled traversal touches fewer nodes than exhaustive traversal on a fixture
  with opponent or chance branching.
- Steady-state sampled iterations allocate zero bytes after workspace warm-up.

**Exit criteria.** All four public modes pass their semantic tests; sampled
regret estimates pass the statistical comparison with exhaustive CFR; the
public API has no Boolean “plus” or “sampled” switches; and no traversal
contains a per-node branch over the four modes.

#### Stage 5: complete $N$-player, general-sum, and chance support

**Milestone outcome.** The same four modes accept any finite number of players
in a sequential perfect-recall extensive-form game. The construction-time
two-player specialization remains the primary fast path, and graph structure
remains entirely outside the solver.

**Implementation work.**

1. Generalize the target loop to `0 .. playerCount - 1` and validate
   `playerCount >= 2`. Each target traversal still carries only
   `ownReach` and combined `externalReach`; it never allocates an
   $N$-element reach vector in the regret hot path.
2. Request terminal utility only for the active target player. This permits
   arbitrary general-sum payoff vectors without constructing a payoff array at
   every terminal state.
3. Preserve a construction-selected two-player entry point that makes exactly
   two direct target calls and selects one component of a struct terminal
   payoff. The generic player loop is not executed by that path.
4. Keep exhaustive averaging target-local and exact. For two-player external
   sampling, retain the fast paired-pass estimator from Stage 4.
5. For $N>2$ external sampling, perform one exact average-strategy sweep before
   the first target-player regret pass. All players therefore contribute the
   same start-of-iteration profile. The sweep reuses one mutable $N$-reach array
   with
   multiply/restore backtracking and an information-set epoch; it does not
   allocate a reach vector per node. This deliberately favors a simple,
   unambiguous average over a more complex stochastic weighting estimator.
   Document that this exact sweep can dominate the cost of sampled regret
   updates in multiplayer games. Multiplayer is supported correctly, but is
   not the primary MCCFR performance target.
6. Validate the mechanically checkable finite-game invariants: every visited
   nonterminal state has exactly one actor, information-set ownership and row
   length are stable, and chance probabilities are non-negative and sum to one.
   Perfect recall remains a documented caller obligation; an exhaustive
   validator may check small test games, but the solver cannot prove it cheaply
   for a large procedural game.
7. State the result contract precisely: CFR minimizes counterfactual regret,
   but outside two-player zero-sum classes the reported average independent
   strategies are not advertised as converging to a Nash equilibrium.

**Verification.**

- A three-player general-sum fixture has hand-computed target utilities and
  regret deltas for every player.
- A fixture contains consecutive turns, a skipped player, and player order
  unrelated to depth; all values remain correct.
- Player-scoped information sets with identical local observations remain
  distinct.
- Exhaustive chance values match direct enumeration; sampled chance values are
  unbiased within statistical tolerance.
- The $N$-player exact averaging sweep matches a direct calculation of
  reach-weighted strategy sums.
- The Stage 4 two-player benchmarks select the specialized path and show no
  material throughput or allocation regression.

**Exit criteria.** Every mode passes two-player and three-player tests, all
general-sum utilities retain their requested perspective, explicit chance works
in exhaustive and sampled traversal, and the documented two-player fast path is
still selected without a per-node game-kind branch.

#### Stage 6: cut over to the minimal production API

**Milestone outcome.** There is one production implementation, one unambiguous
average-policy contract, and no dictionary/string/array-allocation legacy path
inside `cfr.fs`. The redesign is now complete even if Stages 7 and 8 are never
performed.

**Implementation work.**

1. Freeze a small conceptual public surface:

   ```fsharp
   create          : SolverMode -> Game<'state> -> InfoSetMeta[] -> seed:int -> Solver<'state>
   runIteration    : Solver<'state> -> root:'state -> unit
   run             : iterations:int -> Solver<'state> -> root:'state -> unit
   averageStrategy : Solver<'state> -> infoSetId:int -> double[]
   ```

   The concrete F# signatures may group construction data into one record, but
   they should not add inheritance, plugins, optional average tracking, a
   numeric-type parameter, or graph-specific configuration. Reporting may
   allocate a result array because it is outside training; add a destination-
   buffer overload only if reporting is itself measured as a bottleneck.
2. Keep `Solver<'state>` opaque. Expose mode, iteration count, metadata, and
   normalized average strategy through functions rather than writable table
   fields.
3. Move the old recursive implementation into the test project as a restricted
   reference, then remove `StrategyNode`, dynamic string keys, action masks,
   `p0`/`p1`, negamax, and the old public recursion functions from production.
   Do not retain a second legacy engine or a permanent compatibility adapter.
4. Treat this as a documented breaking API change. Add a concise migration
   example mapping old callbacks to the dense metadata and local-action game
   contract.
5. Run an allocation audit through complete iterations. For a value-type state
   and non-allocating game callbacks, both exhaustive and sampled training must
   allocate zero bytes after initial depth growth. Reference-type game states
   may allocate in caller code; report those bytes separately from solver
   allocations.
6. Verify memory from actual array lengths:

   $$
   \text{persistent numeric bytes}=16M,
   $$

   plus metadata, depth scratch, and exactly one mode-specific workspace.
   Ensure exhaustive solvers do not own sample caches and sampled solvers do not
   own the dense exhaustive delta array.
7. Update `CFR.md` and the README to describe the final API, mode semantics,
   memory formulas, convergence boundaries, and benchmark command.

**Verification.**

- The complete solution builds for the repository's target frameworks.
- All Stage 1 through Stage 5 tests pass against the production API.
- Public API inspection shows exactly four solver modes and no way to disable
  average-strategy tracking.
- Steady-state allocation tests pass for every mode.
- Benchmark results include legacy-versus-production comparisons and show that
  traversal work, rather than allocation or string hashing, dominates.

**Exit criteria.** The legacy implementation is absent from the production
assembly, the public API and documentation agree, all correctness and
allocation gates pass, and a user can train and retrieve an average strategy
in any of the four modes without knowing the internal table representation.

#### Stage 7: apply profile-guided single-thread optimization

**Milestone outcome.** The scalar production solver is tuned for the workloads
that actually matter. SIMD is an implementation detail of isolated kernels, not
a second architecture. A well-supported decision to keep scalar code also
completes this milestone.

**Implementation work.**

1. Re-run the Stage 1 benchmark matrix in Release mode and profile inclusive
   time, branch behavior, and allocation before changing code. Rank hotspots by
   total solver time, not by microbenchmark attractiveness.
2. Optimize in this order:

   - measure repeated game-contract validation; only if material, consider a
     checked/trusted traversal selected once outside the hot loop, with checked
     behavior retained for untrusted games;
   - remove remaining bounds checks and non-inlined helper overhead where the
     generated code demonstrates a cost;
   - add explicit two-action and three-action scalar kernels if they improve the
     primary games;
   - tune the generic scalar loop;
   - consider SIMD only for rows large enough to amortize vector setup,
     horizontal reduction, and the scalar tail.

3. Limit `System.Numerics.Vector<double>` or hardware intrinsics to regret
   application, clipping, strategy accumulation, or positive-regret scans over
   contiguous rows. Do not vectorize recursive control flow or restructure the
   game into an ECS, tree batch, or graph engine.
4. Select a kernel by action count inside the isolated row operation. Do not add
   four SIMD traversals or a strategy-object dispatch at each node.
5. Use a merge gate: a specialization must improve median throughput by at
   least $10\%$ on a representative affected workload and must not regress the
   common two- and three-action workloads by more than $3\%$. Record variance
   and generated-code evidence with the result.
6. Retain `double`. A `float32` specialization is a separate future proposal
   requiring convergence and exploitability evidence; it is not part of this
   stage.

**Verification.**

- Every optimized kernel is compared with the scalar reference over randomized
  rows, including non-vector-width tails and extreme finite values.
- Floating-point reordering is accepted only within a documented tolerance and
  does not measurably degrade convergence on the analytical fixtures.
- End-to-end nodes per second, not only kernel operations per second, clears the
  merge gate.
- Steady-state allocation remains zero and persistent memory remains unchanged.

**Exit criteria.** The benchmark report identifies every retained
specialization and its measured benefit. Unprofitable SIMD code is removed.
The remaining implementation stays short enough to compare directly with the
scalar CFR equations.

#### Stage 8: optionally add deterministic parallel MCCFR batches

**Milestone outcome.** If profiling justifies it, sampled traversals can run in
parallel batches with bounded additional memory and deterministic reduction.
This does not add a fifth algorithm mode. If the gate fails, the documented
decision not to ship parallelism is the completed milestone.

**Implementation work.**

1. Limit the first parallel implementation to batches of independent
   `MCCFR` or `MCCFRPlus` samples. Keep exhaustive CFR single-threaded; a
   general parallel tree scheduler is outside the necessary core.
2. Freeze shared regrets for the batch. Each worker receives its own traversal
   scratch, sample cache, deterministic random stream, and append-only sparse
   regret and strategy-delta buffers. Workers never mutate shared numerical
   tables.
3. Derive random streams from the master seed, batch number, and worker index so
   the same configuration is reproducible.
4. Sort or otherwise canonicalize worker deltas by numerical slot, reduce them
   in fixed worker order, then apply signed accumulation or one CFR+ clip after
   the complete batch contribution:

   $$
   R_j\leftarrow
   \max\!\left(0,R_j+\sum_w\Delta R_{w,j}\right)
   $$

   for a plus mode. Average-strategy deltas are mandatory and use the same
   deterministic reduction discipline.
5. Grow sparse buffers geometrically during warm-up and reuse them. Never
   allocate a complete `Regrets` or `StrategySums` copy per worker.
6. Keep worker count and batch size as execution settings, not new
   `SolverMode` cases. A one-worker, one-sample batch must use the same
   mathematical update boundary as the scalar sampled implementation. A batch
   size above one changes the regret-application cadence; document and evaluate
   that algorithmic choice rather than presenting it as scheduling alone.
7. Use a go/no-go gate on representative medium and large sampled games: four
   workers should deliver at least $1.5\times$ end-to-end throughput over one
   worker without superlinear memory growth or degraded convergence per visited
   node. Record hardware and variance.

**Verification.**

- Repeated runs with the same seed, worker count, and batch size are bitwise
  reproducible.
- A one-worker batch matches the corresponding scalar batch.
- Thread-safety tests find no shared writes during traversal.
- Plus clipping occurs after deterministic aggregation, never in workers.
- Additional memory is proportional to touched sampled slots and worker count,
  not $\text{workerCount}\times M$.
- Convergence is compared by nodes visited as well as wall-clock time so a
  throughput win cannot hide a worse update schedule.

**Exit criteria.** Either the parallel implementation passes correctness,
determinism, memory, and speed gates and remains an optional execution setting,
or it is removed and the benchmarked no-go decision is recorded. In both cases,
the four scalar modes and their public contract remain unchanged.

The desired endpoint is compact enough to audit against the CFR equations,
fast enough that traversal work rather than allocation and hashing dominates,
and structured so optional SIMD or parallel kernels do not complicate the core
algorithm.

## 17. Implementation history

### 2026-07-12 — Stage 1: correctness and performance baseline

- Added dependency-free test and benchmark executables to the solution while
  leaving the legacy public solver behavior unchanged.
- Added exact helper tests, a hand-computed regret fixture, a compact zero-sum
  convergence fixture, fixed-seed sampling, general-sum and repeated-information
  analytical cases, and named regression or acceptance invariants for every
  issue and contract in Section 15.
- Built both library target frameworks in Release and ran 33 total Stage 1 and
  Stage 2 tests with zero failures. Captured the $2,3,4,8,16,32$ action-count
  matrix for exhaustive and sampled traversal at depths 2 and 3 in
  [`CFR_BENCHMARK_RESULTS.md`](CFR_BENCHMARK_RESULTS.md),
  including revision,
  runtime, processor, seed, node count, elapsed time, and allocation.
- Deviation: the test project is an executable assertion harness rather than a
  package-based test framework. This keeps the old repository dependency-light
  and gives it a single deterministic command-line entry point.

### 2026-07-12 — Stage 2: flat allocation-free scalar core

- Added internal `SolverMode` semantics for CFR, CFR+, MCCFR, and MCCFR+;
  dense globally unique information-set definitions with player ownership;
  canonical packed metadata; flat regret and mandatory strategy-sum arrays;
  and separate exhaustive and sampled reusable workspaces.
- Implemented checked reporting/construction boundaries and unchecked scalar
  hot-path kernels for stable regret matching, uniform fallback, sampling,
  average accumulation and normalization, signed regret updates, and
  aggregate-then-clip plus updates. Very large finite rows are normalized with
  scaled sums to avoid overflow.
- Verified invalid metadata, non-finite inputs, one through many actions,
  property-style normalization, epoch wraparound, sample reuse, touched-row
  clearing, exact $16M$ persistent numeric storage, and zero allocated bytes
  across 10,000 warmed-up kernel cycles. The complete Release test run passed
  all 33 tests, and the legacy baseline fixtures remained unchanged.
- Deviation: Stage 1 and Stage 2 were requested in one working session. Stage 2
  source was added beside the legacy solver while the harness was being built,
  but completion was recorded only after Stage 1 passed first and the full
  Stage 2 exit criteria then passed.

### 2026-07-12 — Stage 2 benchmark addendum

- Extended the benchmark executable with a traversal-independent Stage 2
  section rather than mixing kernel numbers into the Stage 1 legacy baseline.
- Recorded actual array-payload memory for 250,000 information sets and
  1,000,000 legal slots: 16,000,000 persistent numeric bytes, 30,032,768 total
  bytes for the exhaustive configuration, and 22,819,200 for the sampled
  configuration under the documented workspace capacities.
- Measured the combined scalar row cycle separately for 1, 2, 3, 4, 8, 16,
  and 32 actions. Every warmed-up run allocated zero bytes; detailed elapsed
  time and throughput remain in the consolidated dated benchmark document.

### 2026-07-12 — Stage 0 memory benchmark addendum

- Retrospectively measured the unchanged legacy representation, so the
  optimization baseline now includes memory as well as traversal allocation
  and throughput.
- For six action/depth cases, recorded retained information-set count, exact
  `StrategyNode` numeric-array payload, forced-GC managed retained bytes, and
  total construction allocation. The managed figure intentionally includes
  dictionary, string, node, and array overhead, while construction allocation
  also includes temporary traversal objects.
- Corrected the first probe before documentation: warm-up dictionaries were
  explicitly released before measuring. The discarded probe produced invalid
  zero retained-byte results and is not part of the benchmark history.

### 2026-07-12 - Stage 3: exhaustive two-player general-sum CFR and CFR+

- Added the minimal generic game contract and internal exhaustive solver for
  two-player general-sum games. Terminal detection precedes actor and
  information-set lookup; terminal utilities are requested independently for
  each target player; explicit actors support consecutive turns; and exact
  chance enumeration contributes chance probability to external reach.
- Implemented simultaneous signed-regret CFR and alternating CFR+. Repeated
  histories accumulate into the dense exhaustive delta workspace and each row
  is applied only at its mode's update boundary, so CFR+ clips the complete
  aggregate once. Mandatory average strategies use target own reach, mode
  weight, and a per-pass epoch guard.
- Added analytical tests for one-iteration general-sum values and deltas,
  independent terminal payoffs $(3,1)$, chance-weighted counterfactual reach,
  repeated-information-set clipping, consecutive turns, invalid metadata and
  mode rejection, matching-pennies convergence, nonnegative CFR+ regrets, and
  value-state steady-state allocation. The Release solution built with zero
  warnings and all 42 tests passed.
- Added a distinct Stage 3 benchmark with seven interleaved runs and medians.
  On the constrained machine, the new solver measured 15,925,852 nodes/s and
  zero steady-state bytes versus 3,316,737 nodes/s and 76,256,000 bytes for the
  legacy traversal on the representative four-action, depth-five workload.
- Memory remained within the planned core: beyond the two persistent packed
  arrays, the exhaustive solver owns only depth-local strategy/utility scratch,
  one dense regret-delta array, and integer epoch/touched-row bookkeeping. No
  current-strategy table, action mask, sampled workspace, dictionary, or
  traversal-local collection was added.
- Deviation: state/metadata contract checks remain active in Release rather
  than being compiled out. Their benchmarked cost still clears the throughput
  gate by a wide margin, and retaining explicit failures is simpler and safer;
  Stage 7 may reconsider only if profiling shows a material win.

### 2026-07-12 — Benchmark environment qualification

- Recorded that the benchmark machine was operating with approximately 65%
  throttling and typically more than 50% total CPU load from other work.
- Reclassified the recorded absolute throughput as a constrained, noisy
  snapshot. Allocation results, exact payload calculations, broad scaling, and
  same-session comparisons remain useful; small timing differences do not
  establish an optimization win.
- Future performance gates require comparable load conditions, interleaved
  before/after samples, and median results, with a clean-machine rerun appended
  under a new date rather than replacing this run.

### 2026-07-12 - Stage 3a: exhaustive hot-path optimization

- Added a two-player-only vanilla CFR recursion that returns both independent
  utilities in a struct tuple, carries both player reaches and chance reach as
  scalars, updates only the acting player's regret, and preserves simultaneous
  application. CFR+ remains on two alternating target-player traversals.
- Reused the existing strategy and acting-player utility scratch; no utility
  vector, second depth buffer, current-strategy table, or other solver-sized
  storage was added. Fused regret application and delta clearing now write and
  zero each touched slot in one loop.
- Added lockstep tests against the retained two-target oracle across the
  general-sum matrix, burn-in, chance/repeated-information-set, and
  consecutive-turn fixtures. Added traversal-count, fused-boundary, touched-row
  validation, and allocation checks. The Release solution built with zero
  warnings and all 45 tests passed; 10,000 warmed iterations allocated zero
  bytes.
- On the four-action, depth-five benchmark, the retained two-target reference
  visited 1,365,000 nodes in 98.644 ms and the one-pass path visited 682,500 in
  97.402 ms. The fused boundary measured 291.498 ms versus 296.555 ms for
  apply-then-clear. Both retained paths allocated zero bytes and exact managed
  array payload remained 41,240 bytes for the fixture.
- Deviation: the planned unchecked touched-row helper was removed after two
  interleaved benchmark sessions showed large regressions (245.044 versus
  343.229 ms and 224.193 versus 383.904 ms, checked versus unchecked). Keeping
  the checked helper is both faster on this runtime and simpler. Timing gains
  varied materially under the documented throttling and background load, so
  the final conservative interleaved median is recorded rather than the best
  observed run.

### 2026-07-12 - Stage 3a benchmark rerun addendum

- Reran the unchanged Release implementation with seven interleaved repetitions.
  The two-target reference measured 113.248 ms and the one-pass path measured
  84.111 ms, a $25.73\%$ elapsed-time reduction; both allocated zero bytes.
- The separate fused-boundary comparison measured 287.678 ms versus 291.472 ms
  for apply-then-clear, a $1.30\%$ reduction with zero allocation.
- Appended the rerun rather than replacing the earlier conservative result,
  preserving the observed timing variability on the throttled, busy machine.
