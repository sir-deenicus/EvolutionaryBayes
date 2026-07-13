# CFR Large-Game Extension Proposal

**Status:** future design, not yet implemented

**Last updated:** 2026-07-13

This document proposes the post-core path for information-set abstraction,
lazy discovery, neural function approximation, outcome sampling, and larger
validation games. It does not renumber or replace the production-core stages
in [CFR.md](CFR.md). The packed tabular solver remains the small-game oracle
and the preferred implementation whenever the game fits in memory.

## Executive summary

The design has four independent library extensions:

1. **Explicit coarse graining.** Most callers provide a pure F# function that
   maps an observable information state to an abstract key. The extension
   lazily interns keys to dense integer information-set identifiers.
2. **Nearest-neighbor abstraction.** A stateful resolver assigns a newly seen
   information state to a compatible representative when its caller-defined
   distance is within a threshold, otherwise it creates a new representative.
   This is a second abstraction strategy, not a second meaning of coarse
   graining.
3. **Neural CFR.** A separate library extension implements external-sampling
   Linear Deep CFR with either a history transformer or a small spatial
   convolution-transformer hybrid. It does not add machine-learning
   dependencies to the packed core.
4. **Outcome sampling.** A later, explicitly selected MCCFR traversal samples
   one terminal trajectory for games whose action branching makes external
   sampling impractical. It is performance-gated because its lower work per
   iteration comes with importance weighting and greater estimator variance.

Before data enters any of these extensions, a game adapter may apply an exact,
lossless canonicalization under its own game automorphisms. That is part of the
game's information-state and action representation, not a fifth generic CFR
feature. The CFR core is unaware of geometry, symmetry groups, and action
permutations.

The validation ladder is:

1. hidden-information $3\times3$ tic-tac-toe;
2. hidden-information $4\times4$ tic-tac-toe;
3. hidden-information $3\times3\times3$ tic-tac-toe;
4. Mini Hold'em;
5. Mini-Jotto;
6. five-letter Jotto;
7. Mini-Scrabble; and
8. full two-player Scrabble.

Each game adds one material difficulty before the next combines them.

## 1. Scope and design boundaries

### Goals

- Preserve the existing exact and sampled packed solvers as small, fast,
  independently usable implementations.
- Allow concrete information states to be grouped explicitly or by a
  caller-defined similarity rule.
- Let game adapters remove exact representational duplication before any lossy
  grouping while preserving observable history and action semantics.
- Discover information sets on demand instead of requiring complete game-tree
  enumeration before solving.
- Approximate counterfactual advantages and the mandatory average strategy
  without storing a permanent row for every concrete information set.
- Support ordered histories, private observations, spatial boards, and
  variable action candidates without leaking hidden information.
- Keep two-player games as the optimized path while building on the core's
  implemented support for finite sequential $N$-player general-sum games.
- Make every approximation choice visible and measurable.

### Non-goals

- Replacing the packed tabular core with a generic model-backend abstraction.
- Adding tensor, accelerator, or training dependencies to the core library.
- Claiming that function approximation preserves exact tabular convergence.
- Claiming Nash convergence for general-sum or $N$-player self-play.
- Adding special graph-game, polymatrix, or hypergraph scheduling.
- Making one universal network architecture serve every game.
- Implementing full Scrabble before the smaller games validate information
  encoding, variable actions, sampling, and average-policy extraction.

## 2. Relationship to the packed core

The current game contract already asks a game for a dense
`InformationSetId`. A caller can therefore perform precomputed coarse graining
today by returning the same identifier for several concrete states. The future
generic abstraction work adds two missing capabilities:

- accepting stable abstract keys and assigning dense identifiers lazily; and
- providing reusable assignment strategies, including nearest-neighbor
  assignment.

Exact, lossless canonicalization is outside that generic abstraction layer.
The game adapter owns its geometry, observable-history encoding, deterministic
canonical choice, and the mapping between concrete and canonical action order.
It supplies an already canonical state or key and an already aligned local
action row to tabular abstraction. A neural adapter likewise encodes canonical
inputs and maps model outputs back to concrete actions. Neither path requires
the CFR core to define or execute a symmetry transform.

The packed mathematics should continue to receive dense legal actions and dense
information-set rows. It should not know whether an identifier came from an
exact precomputed table, a coarse-graining function, or a nearest-neighbor
index. The current fixed `PackedTable` remains unchanged. A future lazy tabular
solver may use a caller-supplied fixed capacity or separate chunked storage; it
must not insert row-growth checks into the existing fixed-table hot path.

Neural traversal has different performance requirements: inference must be
batched and training produces variable-length encoded examples. It therefore
belongs in a separate extension, tentatively named
`EvolutionaryBayes.CFR.Neural`, which references the game semantics and
mathematical kernels without inserting a model callback into the scalar hot
loop.

## 3. Information-state representation and abstraction

This proposal uses *lossless canonicalization* for what is sometimes called a
lossless abstraction. The distinct name is deliberate: it identifies exact
equivalence without conflating it with the approximate mappings below.

| Layer | Owner | Effect |
| --- | --- | --- |
| Lossless canonicalization | Game adapter | Chooses one representation from a proven exact equivalence class and permutes actions consistently. |
| Explicit coarse graining | Generic abstraction extension | Deliberately merges strategically distinguishable states using a caller key. |
| Nearest-neighbor assignment | Generic abstraction extension | Deliberately merges nearby compatible states under caller features and distance. |
| Neural encoding and regression | Separate neural extension | Replaces permanent tabular rows with approximate learned predictions. |

Only the first row is lossless. It is optional, precedes every approximate
layer, and requires no change to the CFR core.

### 3.1 Lossless game-specific canonicalization

Lossless canonicalization is a representation step performed by a game
adapter, not a generic information-set abstraction implemented by CFR. It may
quotient states only by an exact game automorphism. Let a finite group $G$ act
on observable histories and legal actions. A transformation $g\in G$ is usable
only when it preserves the acting player, the player's observations, legal
transitions, chance probabilities, and every player's terminal utility.
Strategies then obey

$$
\sigma(gI,ga)=\sigma(I,a).
$$

The game adapter selects a deterministic representative and maps concrete
legal actions to a stable canonical order. The inverse mapping must also be
available so regret targets, average-policy values, sampled actions, and
deployed policy outputs all refer to the same moves. A game with no useful
automorphisms simply uses the identity representation.

The generic coarse-graining resolver receives the result of this step; it does
not define group actions or invoke a game-specific permutation. The packed core
continues to receive only dense information-set rows and dense local actions.
The Incomplete Tic-Tac-Toe adapters and their concrete transforms are specified
in Sections 10.1 through 10.3.

The processing order is

$$
\text{observable history}
\longrightarrow \text{exact symmetry canonicalization}
\longrightarrow \text{optional lossy abstraction or neural encoding}.
$$

### 3.2 Lossy primary API: a pure coarse-graining function

Explicit coarse graining is naturally a function in F#:

```fsharp
// Illustrative shape; this is not a frozen public signature.
type CoarseGrain<'State, 'Key when 'Key : equality> =
    int -> 'State -> 'Key
```

The arguments are the acting player and game state. The returned key must be
computed only from information observable to that player. Typical mappings
can:

- bucket numeric observations;
- discard parts of history judged strategically unimportant;
- combine public history with the acting player's private observation; or
- merge several states whose distinctions the caller deliberately accepts
  losing into one immutable record or value tuple.

The extension maintains a lazy key-to-row interner. On the first occurrence of
a key it allocates the next dense row; later occurrences reuse that row. The
caller describes *which states share an approximate row*, while the library
owns dense storage identifiers and row lifetime.

This function is the normal public entry point because it is easy to test,
compose, and reason about. Callers should not need to implement a large
abstraction object merely to provide a pure mapping.

### 3.3 Common boundary: a small resolver interface

Nearest-neighbor lookup and custom indexes are stateful. Internally, and as an
advanced extension point, both strategies fit behind a small resolver:

```fsharp
// Illustrative shape; row allocation details remain an implementation choice.
type IInformationSetResolver<'State> =
    abstract Resolve:
        player:int * state:'State * actionCount:int -> int
```

The function-based mapper is wrapped by a resolver that performs lazy
interning. A nearest-neighbor resolver implements the same operation with a
feature index. The packed solver sees only the resulting integer.

This split gives the simple case a function and the stateful case an interface,
without forcing stateful concerns into user game definitions.

### 3.4 Correctness contract for grouped states

Every concrete information state assigned to one abstract row must have:

- the same acting player;
- the same number of legal actions;
- the same action meanings in the same canonical order; and
- no distinction that the acting player could use unless losing that
  distinction is an intentional approximation.

Matching only the action count is insufficient. If local action zero means
"upper-left" in one state and "lower-right" in another, their regrets cannot
share a slot without remapping.

The first implementation should require already action-aligned mappings. Any
lossless action permutation has been applied by the game adapter before this
boundary. Per-visit closure allocation or arbitrary action-remapping callbacks
do not belong in the generic resolver or packed hot path.

### 3.5 Lazy discovery

Lazy discovery creates a row only when traversal first reaches an abstract
information set. It should obey these rules:

- identifiers never change during a solve;
- a row's player and action schema are validated at first reuse;
- when storage is growable, growth is chunked so adding a row does not copy
  every existing row; a fixed-capacity implementation instead fails explicitly
  when its declared capacity is exhausted;
- memory limits and overflow behavior are explicit;
- no row containing accumulated regret or average strategy is silently
  evicted; and
- seeded traversal plus deterministic key equality produces reproducible row
  assignment.

An explicit memory cap may stop the solve or direct *new* states to a configured
fallback abstraction. It must not silently discard learned rows and continue
as if the result were unchanged.

### 3.6 Nearest-neighbor abstraction

For a concrete information state $I$, the caller supplies a feature projection

$$
\phi(I)\in\mathbb{R}^{d},
$$

a distance function $D$, and an acceptance radius $\tau$. The resolver selects

$$
j^* = \operatorname*{arg\,min}_{j\in C(I)}
      D\!\left(\phi(I),c_j\right),
$$

where $C(I)$ contains only representatives with the same player and action
schema. It assigns $I$ to $j^*$ when

$$
D\!\left(\phi(I),c_{j^*}\right)\leq\tau;
$$

otherwise it creates a new representative.

The initial implementation should use a simple exact scan and deterministic
tie-breaking. Approximate indexes are justified only after profiling. Existing
assignments must remain stable during a solve; periodically reclustering old
states would change the meaning of accumulated regrets and requires an explicit
regret-transfer algorithm that is outside this proposal.

Nearest-neighbor assignment is useful when the caller has a meaningful metric
but no natural bucket boundary. Explicit coarse graining remains preferable
when domain rules define useful approximate buckets. Exact equivalences belong
in the game adapter's lossless canonicalization step before either option.

## 4. Neural CFR algorithm

### 4.1 Selected variant

The initial neural solver will use alternating-player external-sampling Linear
Deep CFR:

- external sampling at chance and non-traverser nodes;
- all legal actions evaluated at traverser nodes;
- one counterfactual-advantage model per player;
- linear iteration weights $w_t=t$;
- regret matching over predicted advantages; and
- a separately trained average-policy model.

This follows the data construction of
[Deep CFR](https://proceedings.mlr.press/v97/brown19b) while changing the model
architecture and information-state encoding. A transformer is a different
function approximator, not a different CFR update.

Neural CFR+ is not part of the initial design. CFR+ applies the nonlinear
recurrence

$$
R_t^+(I,a)=\max\left(0,R_{t-1}^+(I,a)+r_t(I,a)\right),
$$

which cannot be recovered merely by regressing the average of independently
stored instantaneous regret samples. The existing tabular CFR+ and MCCFR+
modes remain available.

### 4.2 Advantage targets

At a traverser information state $I$, traversal evaluates every legal action:

$$
v_t(I,a).
$$

The current strategy value is

$$
v_t(I)=\sum_{b\in A(I)}\sigma_t(I,b)v_t(I,b),
$$

and the instantaneous sampled counterfactual advantage is

$$
\widetilde r_t(I,a)=v_t(I,a)-v_t(I).
$$

The advantage memory stores an encoded observable information state, legal
action description, action mask, iteration, and full legal-action target
vector. A bounded reservoir prevents storage from growing with the total
number of traversals.

The masked, linearly weighted loss is

$$
\mathcal L_A(\theta)=
\mathbb E\!\left[
t\,
\frac{
\sum_a m_a
\left(\widehat A_\theta(I,a)-\widetilde r_t(I,a)\right)^2
}{
\sum_a m_a
}
\right].
$$

For a fixed information state, the MSE optimum is a weighted average:

$$
\widehat A(I,a)=
\frac{\sum_t t\,\widetilde r_t(I,a)}{\sum_t t}.
$$

That is proportional across actions to linearly weighted cumulative regret.
Regret matching is invariant to positive common scaling, so it produces

$$
\sigma_{t+1}(I,a)=
\frac{[\widehat A(I,a)]_+}
     {\sum_b[\widehat A(I,b)]_+},
\qquad [x]_+=\max(x,0),
$$

with a uniform fallback when no predicted advantage is positive. Advantage
outputs remain signed and receive no output softmax or ReLU.

### 4.3 Initial data and training loop

No pre-existing regret table is required:

1. initialize the advantage output layers to zero;
2. zero advantages produce uniform regret-matching strategies;
3. run a fixed batch of external-sampling traversals;
4. add the resulting instantaneous advantage vectors to each player's
   reservoir;
5. train the player's advantage model on the accumulated reservoir; and
6. repeat with the newly predicted strategy.

The first correctness implementation should follow Deep CFR and retrain an
advantage model from its reservoir at each outer iteration. Warm-starting from
the preceding weights is a later measured optimization, not an assumed
equivalent behavior.

For development, the packed exhaustive solver supplies exact small-game
strategies and traversal targets. It is an oracle for encoding, loss, masking,
and exploitability tests, not a production source of neural labels.

### 4.4 Mandatory average strategy

The final regret-matched strategy is not CFR's reach-weighted average strategy.
During opponent traversals, the extension stores observed current strategies in
a separate bounded reservoir. The target is

$$
\bar\sigma_T(I,a)=
\frac{
\sum_{t=1}^{T}t\,\pi_i^{\sigma_t}(I)\sigma_t(I,a)
}{
\sum_{t=1}^{T}t\,\pi_i^{\sigma_t}(I)
}.
$$

External-sampling visitation supplies the player-reach sampling, while the
sample weight supplies the linear iteration factor. A masked softmax model is
trained from these strategy samples, initially with weighted MSE to stay close
to the Deep CFR formulation.

The average-policy model is the returned, deployable result. Avoiding it by
storing every historical advantage model, as proposed by
[Single Deep CFR](https://arxiv.org/abs/1901.07621), remains a possible research
comparison but is not the initial library design because it complicates model
storage and inference.

## 5. Supported neural model families

Both model families implement the same advantage and average-policy roles.
They share training, sampling, reservoir, masking, and evaluation code but not
necessarily weights.

### 5.1 History transformer

The transformer consumes only the acting player's information history:

- public events and actions;
- that player's private observations;
- player identity, turn, phase, and game-rule tokens; and
- candidate-action descriptions when actions do not have fixed global slots.

It must never receive hidden opponent observations or an omniscient serialized
state.

The minimal initial architecture is:

- encoder only;
- one pre-layer-normalized transformer block;
- model width 64;
- four attention heads;
- feed-forward width 128;
- learned event, actor, observation, and position embeddings; and
- one decision token or contextual action tokens.

There is no decoder, autoregressive generation loop, cross-attention stack, or
language-model objective. A second encoder block is the first capacity increase
if the one-block model demonstrably underfits.

Fixed small action sets may use a linear vector head. Jotto and Scrabble require
an action-conditioned scorer:

$$
\widehat A(I,a)=q_\theta\!\left(h_\theta(I),e_\theta(a)\right),
$$

so a word, placement, or other candidate has stable semantics independent of
its position in a generated list.

### 5.2 Spatial convolution-transformer hybrid

Board occupancy has strong local structure, while imperfect-information play
also depends on ordered actions and private observations. The supported hybrid
therefore uses:

1. two shallow 2D or 3D convolution layers, initially 32 then 64 channels;
2. no spatial downsampling, because every cell may correspond to an action;
3. one feature token per board cell;
4. appended move-history, player, private-card, turn, and rule tokens;
5. one width-64, four-head transformer block; and
6. a shared linear action score from each contextualized cell token.

For $3\times3\times3$ tic-tac-toe, only the convolution stem changes from 2D
to 3D. The attention, advantage, policy, and training components stay the same.

The convolution supplies spatial weight sharing and local pattern extraction;
attention supplies global interaction and belief-relevant history. This is a
small domain-specific form of the convolution/attention combination explored
by [CvT](https://arxiv.org/abs/2103.15808) and
[CoAtNet](https://arxiv.org/abs/2106.04803), not an attempt to reproduce those
large vision architectures.

A pure temporal CNN may be retained as an experimental baseline, but it is not
one of the two promised extension models. If measured, it must encode move
order, such as through per-cell move-time planes; occupancy alone would merge
observable histories and introduce an unacknowledged imperfect-recall
abstraction.

## 6. Separate library and model boundary

The neural extension should expose game-specific encoding as small functions
or records and keep the trained model opaque. An illustrative encoding shape is:

```fsharp
// Illustrative only; tensor and batch representations are not frozen here.
type NeuralEncoding<'State, 'Observation, 'Action> =
    { Observe : int -> 'State -> 'Observation
      EncodeAction : 'State -> int -> 'Action }
```

`Observe` receives the target player explicitly to prevent accidental use of
another player's private view. `EncodeAction` gives each local legal action a
stable description for action-conditioned scoring.

The internal model boundary must be batch-oriented. A per-node virtual tensor
callback in the existing recursive scalar traversal would make accelerator
inference inefficient and encourage temporary allocations. The neural engine
may begin with batch size one for correctness, but its data representation and
model API must allow several independent traversal queries to be evaluated in
one tensor batch without redesign.

Dependencies flow in one direction:

$$
\text{Neural extension}
\longrightarrow
\text{CFR core},
$$

never from the core to a particular tensor library.

## 7. Outcome sampling as future work

External sampling remains the initial neural traversal because it evaluates all
traverser actions and produces lower-variance full advantage vectors. Its cost
is unacceptable when a player has a very large word or token action set.

Outcome sampling instead follows one complete terminal trajectory per
iteration. It therefore reduces traversal work but requires:

- a sampling policy with nonzero exploration probability for every legal
  action;
- correct reach and sampling probabilities;
- importance-weighted regret estimators;
- careful average-strategy sampling; and
- variance and numerical-stability controls.

Both outcome and external sampling are established MCCFR schemes in
[Monte Carlo Sampling for Regret Minimization in Extensive Games](https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html).
If outcome sampling is implemented here, it must be an explicit public choice;
it must not silently replace the existing meaning of `MCCFR` or `MCCFRPlus`.

Outcome sampling is accepted only after it:

1. reproduces unbiased regret estimates statistically on exact small games;
2. passes fixed-seed legality and reach-probability tests;
3. extracts the correct average strategy;
4. compares convergence, time, and memory against external sampling with
   interleaved repeated runs; and
5. wins on a representative high-branching game such as five-letter Jotto or
   Mini-Scrabble.

## 8. Memory and performance policy

- Retain `double` for recursive utility and reach arithmetic until error tests
  justify otherwise.
- Store model weights, encoded numeric features, regret targets, and strategy
  targets as `float32`.
- Store token identifiers and offsets as the smallest practical integer types.
- Use bounded reservoir sampling rather than retaining every traversal sample.
- Pack variable-length histories and candidate lists into contiguous buffers
  with offset/length metadata rather than nested F# arrays or object graphs.
- Reuse legal masks, traversal workspaces, and tensor staging buffers.
- Do not store complete mutable game states in replay memory when a compact
  information-state encoding is sufficient.
- Batch independent model queries; measure the point at which batching
  complexity pays for itself.
- Record model parameters, buffer capacity, batch size, seed, traversal count,
  wall time, throughput, retained memory, and allocations in the consolidated
  benchmark history.

For variable-action games, a sample must preserve the candidate identity and
target alignment. Any heuristic that removes legal candidates changes the
solved game and must be named as an action abstraction, not presented as an
exact optimization.

## 9. Correctness requirements

### Information safety

An encoded information state contains exactly what the acting player observes.
Two histories in one exact information set must encode identically unless a
documented perfect-recall representation intentionally retains observable
ordering. Hidden opponent cards, racks, secret words, or chance outcomes must
not leak through derived features, padding lengths, candidate generation, or
cache keys.

### Action alignment

Legal masks, action encodings, advantage targets, regret-matching outputs, and
average-policy targets must use one stable ordering. Symmetry transforms must
apply the same permutation in both directions.

### Lossless-canonicalization proof obligation

A mapping is called lossless only when the game adapter can establish that
every allowed transform preserves player observations, actor identity, legal
transitions, chance probabilities, and every player's terminal utility. Tests
must include the identity transform, deterministic canonical tie-breaking,
action-map round trips, transform/inverse transition checks, and hidden-data
noninterference. On a tractable configuration, solving with identity and
canonical representations must produce the same game value and exploitability
within numerical tolerance. Any grouping that does not meet this obligation is
reported and tested as a lossy abstraction.

### Average strategy

Every solver result must expose the average strategy. Advantage-model loss or
the final current strategy is not an acceptable substitute.

### Approximation disclosure

Coarse graining, nearest-neighbor assignment, bounded reservoirs, neural
regression, and candidate pruning each introduce different approximation
errors. Tests and reports must identify which are active. Low supervised loss
alone is not evidence of a good strategy; small games must measure policy error
and exploitability against the packed exact solver.

### General-sum and multiplayer behavior

The data structures may support one advantage model and utility perspective per
player. The established equilibrium guarantee for Deep CFR remains the
two-player zero-sum setting. In general-sum or $N$-player games, the extension
may report learned average strategies and regret diagnostics but must not claim
Nash convergence without a separate theorem.

## 10. Validation-game ladder

### 10.1 Hidden-information $3\times3$ tic-tac-toe

Each player independently receives one private objective card from

$$
\{\mathrm{Win},\mathrm{Lose},\mathrm{Draw}\}.
$$

After the ordinary board result, let

$$
m_i=\mathbf 1[\text{player }i\text{'s result matches player }i\text{'s card}]
$$

and

$$
u_i=m_i-m_{-i}.
$$

Exactly one matching player wins; both or neither matching gives zero utility.
This is a two-player zero-sum imperfect-information game. The packed exhaustive
solver supplies exact targets and exploitability checks. Both neural
architectures run here at matched approximate parameter counts.

The Incomplete Tic-Tac-Toe game library owns its lossless canonicalizer. For a
square board, $G=D_4$ consists of four rotations and four reflections. The
canonical information-state key is based on

$$
(\text{acting player},\ \text{own objective},\
  \text{ordered transformed public moves}).
$$

It must not use only current occupancy: different move orders can reach the
same board while conveying different evidence about a hidden objective. It
must not include the opponent's private objective. The adapter precomputes the
eight cell permutations, uses them for the ordered move sequence and legal
actions, chooses a deterministic canonical history, and retains both directions
of the selected action permutation. A pair of 9-bit occupancy masks may cache
board tests, but occupancy does not replace ordered history in the information
key. The initial design retains a full canonical action row instead of also
compressing action orbits.

Runs compare identity representation with $D_4$ canonical representation and
must reproduce the same game value and exploitability within numerical
tolerance. Tests rotate and reflect entire observable histories, verify action
round trips and ties on self-symmetric positions, and confirm that distinct
move orders reaching the same board remain distinct.

### 10.2 Hidden-information $4\times4$ tic-tac-toe

Use four in a row and the same private-card payoff. This keeps rules and
information structure fixed while increasing the state and history space. It is
the first meaningful function-approximation and throughput test. Its game
adapter reuses the same $D_4$ history-and-action algorithm with precomputed
16-cell permutations and 16-bit occupancy masks; no CFR component changes.

### 10.3 Hidden-information $3\times3\times3$ tic-tac-toe

Use three in a row across all valid axial, face-diagonal, and space-diagonal
lines. This validates 3D spatial encoding and distinguishes the pure history
transformer from the 3D convolution-transformer hybrid. Its game adapter uses
precomputed permutations for the cube's 24 rotations. Reflections may be added
only after verifying that they preserve the complete rules and utilities; if
so, the exact transform set has 48 elements. Ordered public history and the
27-cell action permutation use the same selected transform, while packed
27-bit occupancy masks support allocation-free board tests.

### 10.4 Mini Hold'em

Use a fixed short-deck, heads-up community-card game with:

- ranks Six through Ace and two copies of each rank;
- two private cards per player;
- a two-card flop followed by a one-card turn;
- fixed-size bets and a fixed raise cap in each of three betting rounds;
- locally dense legal choices drawn from check, call, raise, and fold;
- five-card ranks limited to high card, pair, two pair, and straight; and
- terminal utility equal to net chips won or lost.

The reduced deck deliberately makes three of a kind, a full house, and a flush
impossible. This keeps hand comparison auditable while retaining private
cards, public chance reveals, multiple betting rounds, folds, and changing
legal-action sets. A one-card, three-rank, one-round configuration must reduce
to ordinary Kuhn poker and reproduce its known value before the larger game is
accepted.

Mini Hold'em is primarily a history-transformer and sampled-traversal fixture;
it does not need a spatial encoder. It also provides a compact interactive
policy test before the word games introduce very large candidate sets.

### 10.5 Mini-Jotto

Use ordered three-letter strings without repeated letters over a six-letter
alphabet, giving

$$
6\cdot5\cdot4=120
$$

possible secrets. Both players choose a secret without observing the other's
choice, alternate guesses, and truthfully report the number of shared letters.
The first exact guess wins; a fixed guess limit produces a draw if neither
succeeds.

Mini-Jotto validates private word state, variable candidate actions,
action-conditioned scoring, and exact comparison without requiring an external
dictionary.

### 10.6 Five-letter Jotto

Replace synthetic strings with a fixed, versioned five-letter lexicon and use a
documented repeated-letter rule. The initial rules should forbid repeated
letters because this makes shared-letter feedback unambiguous. This stage tests
dictionary-scale candidate encoding and determines whether external sampling
still has acceptable branching cost. Jotto's traditional two-player structure
is summarized [here](https://people.sc.fsu.edu/~jburkardt/fun/wordplay/jotto.html).

### 10.7 Mini-Scrabble

Use a controlled Scrabble-like game with:

- a $7\times7$ board;
- five-tile hidden racks;
- a reduced, versioned tile bag and lexicon;
- words of length two through five;
- deterministic word validity;
- no premium squares, challenge procedure, or bingo bonus;
- explicit pass and exchange actions; and
- terminal utility equal to score difference.

This is the first game that combines a spatial public state, private rack,
chance draws, variable word-placement actions, and direct board interaction.
It validates the hybrid encoder and candidate-action scorer before standard
board and lexicon sizes.

### 10.8 Full two-player Scrabble

Add the standard board, rack size, premium squares, tile values, bag, scoring,
and end conditions under one fixed ruleset and versioned lexicon. Begin with two
players and utility

$$
u_0=s_0-s_1,
\qquad
u_1=-u_0.
$$

A deterministic legal-move generator must enumerate or sample structured
candidates containing word, coordinate, direction, tiles used, and score. The
neural model scores candidates; it does not replace legality checking. Hasbro's
[official rules page](https://instructions.hasbro.com/en-us/instruction/scrabble-board-game)
is the starting rules reference, but the implemented lexicon and challenge
policy must be fixed explicitly for reproducible experiments.

## 11. Proposed extension milestones

These letters deliberately avoid collision with the numbered production-core
stages in `CFR.md`.

### Extension A: abstraction and lazy discovery

Implement the pure coarse-graining adapter, resolver boundary, lazy dense
interner, action-schema validation, deterministic growth, and memory-cap
behavior. The adapter accepts already canonical, action-aligned game inputs; it
does not implement finite groups or geometric action permutations. Validate
identity and deliberate coarse mappings on small games.

**Exit:** coarse mappings are deterministic; incompatible actions fail
immediately; already canonical action order is preserved; and memory behavior
is measured and documented.

### Extension B: neural substrate and history transformer

Create the separate library, packed reservoirs, external-sampling training
loop, advantage transformer, average-policy transformer, and exact-oracle
tests. Implement hidden-information $3\times3$ tic-tac-toe, including its
game-owned $D_4$ history canonicalizer and bidirectional action permutations.

**Exit:** the learned policy improves from uniform play, average-policy output
is distinct and queryable, identity and symmetry-reduced games produce the same
value and exploitability, every action permutation round-trips, observable move
order remains distinct, exact small-game diagnostics pass, and training memory
is bounded.

### Extension C: spatial hybrid and board scaling

Add the 2D and 3D convolution stems, contextual cell scoring,
hidden-information $4\times4$ tic-tac-toe, and hidden-information
$3\times3\times3$ tic-tac-toe. Extend the game-owned canonicalizers to the
16-cell $D_4$ maps and verified cube transforms.

**Exit:** transformer and hybrid are compared at matched workloads; the 3D
model preserves action alignment; identity and canonical representations remain
strategically equivalent; throughput and memory are benchmarked.

### Extension D: Mini Hold'em

Implement the reduced deck, exact hand evaluator, public chance reveals,
three capped betting rounds, legal-action schemas, and an interactive policy
driver. Start with packed exhaustive CFR as the oracle and compare external
sampling and the history transformer only after the configured game outgrows
comfortable exhaustive traversal.

**Exit:** the degenerate Kuhn configuration reproduces its known value; card
dealing and chip accounting are exhaustive-test correct; private cards do not
leak into opponent information states; every emitted action is legal; and a
fixed-seed trained policy can complete an interactive match.

### Extension E: variable word actions

Add action-conditioned scoring, Mini-Jotto, and five-letter Jotto with a fixed
lexicon.

**Exit:** Mini-Jotto matches exact small-game checks; candidate identity is
stable through reservoirs and policy output; five-letter branching cost is
measured.

### Extension F: nearest-neighbor abstraction

Add the feature/distance/radius resolver using a deterministic exact scan first.
Compare it with explicit coarse graining on games with a known controlled
abstraction.

**Exit:** assignments are stable, player/action compatibility is enforced,
distance behavior is deterministic, and approximation quality is reported
separately from neural error.

### Extension G: Mini-Scrabble

Implement the controlled board, rack, bag, lexicon, legal-move generator, and
structured candidate encoder.

**Exit:** all generated actions are legal, hidden racks do not leak, chance
probabilities are correct, and external-sampling cost is characterized.

### Extension H: outcome sampling and full Scrabble

Implement outcome sampling only if the Jotto or Mini-Scrabble measurements show
that external sampling is the limiting factor. After its correctness and
variance gates pass, scale the game and move generator to the fixed full
two-player Scrabble ruleset.

**Exit:** outcome sampling wins a representative high-branching benchmark,
full-game runs remain memory-bounded, and every active action or information
abstraction is reported.

At each completed extension milestone, update this proposal and `CFR.md` in
place, append the dated implementation history required by `AGENTS.md`, and
append rather than overwrite benchmark results in the consolidated root
benchmark document.

## 12. Product targets and prototype references

The validation games also support two deliberately small playable products.
Training remains an offline concern: released clients consume a versioned,
frozen average policy through a narrow inference boundary and do not embed the
CFR trainer. The game rules, information-state encoding, legal-action logic,
and policy protocol live in shared libraries; UI projects do not reimplement
them.

### 12.1 Incomplete Tic-Tac-Toe

Incomplete Tic-Tac-Toe is one game with three selectable variants:

1. the $3\times3$ board;
2. the $4\times4$ board; and
3. the $3\times3\times3$ board.

It targets Android and the web. Both clients use the same private-objective
rules, payoff calculation, lossless history canonicalization, concrete-to-
canonical action mappings, saved-policy format, and variant identifier. These
game-specific transforms live in the shared game library, not the CFR core or
generic abstraction extension. The prior `incomplete-tictactoe` Fable
application is a behavioral reference for rules presentation,
human-versus-AI flow, and browser policy loading. Its learned strategy and
payoff implementation are not correctness
oracles and must not be imported without validation against the new core.

The first playable release follows Extension C. It may initially use an exact
or tabular policy for $3\times3$ while the larger variants use the validated
model boundary. Android-framework selection and web-framework selection remain
application decisions and must not introduce dependencies into the CFR core.

### 12.2 Mini Hold'em

Mini Hold'em targets the web and a terminal console application. Here
"console" means a command-line text UI, not a game-console platform. Both
frontends use the same deck, deal, betting, hand-ranking, chip-accounting, and
policy code.

The prior Jupyter `cfr.fsx` experiment is a behavioral reference for the
interactive betting loop, public-card reveals, hand display, and playing
against an average strategy. Its traversal, information-set construction, and
trained data are not treated as correct. The new implementation migrates the
rules deliberately and proves the reduced Kuhn configuration before recreating
the interactive experience.

The first playable release follows Extension D and uses a fixed-seed,
versioned average policy. Optional on-demand retraining from the old prototype
is excluded from the initial clients because it would mix training behavior
with deterministic inference and greatly complicate reproducibility.

## 13. Principal risks

1. **Representation leakage.** A convenient state serialization may expose
   private opponent data and make training appear better while solving the
   wrong game.
2. **Action mismatch.** Grouping or canonicalizing information states without
   identical action semantics corrupts regret and policy rows.
3. **Approximation feedback.** Advantage error changes the policy that creates
   later data, so held-out MSE alone can conceal strategic failure.
4. **Average-policy error.** A good current advantage model does not guarantee
   an accurate deployable average policy.
5. **Branching explosion.** External sampling may become infeasible for word
   actions long before model memory is exhausted.
6. **Moving nearest-neighbor clusters.** Reassigning old states changes the
   meaning of stored regrets; assignments must remain stable unless regret
   transfer is designed explicitly.
7. **Framework overhead.** Small models can spend more time constructing
   tensors than multiplying them; batching and packed encoded buffers are part
   of the design, not optional cleanup.
8. **False general-sum claims.** Mechanical support for multiple utility
   vectors does not extend two-player zero-sum equilibrium guarantees.

The proposal succeeds if it scales to larger information spaces while the
packed solver remains small, fast, understandable, and usable without any of
these extensions.
