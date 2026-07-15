I have a strong interest in probabilistic programming and believe it plays a key role in most paths leading to more intelligent and helpful machines. I've worked on a Hansei Port, a couple probability monads, worked with [Infer.Net](http://infer.Net) and Stan plus learned a great deal from WebPPL materials. Each of these has its advantages and downsides:

**Hansei** 

**Pro**: Fast when space is constrained, shares many advantages of logic based programming, can be viewed as Enhanced Sequence Comprehensions Augmented with a backtracking, weight tracking search operator.  

That last is important and I believe this form of control flow is the most under-used. Unlike sampling based approaches, sampling in this regime is simply lazily unfolding a search tree—recursion is natural and constraints are most easily applied here. Because of its iterative nature, generated samples (depending on search procedure) need not not be revisited.

**Con**: Discrete and so can be very memory intensive. Computation time can blow up in large spaces. One must be very careful to avoid such blow ups.

**Infer.Net**

**Pro**: Fast, scalable (arguably the most scalable bayesian approach that is not a variational approach—even its variational message passing seems more principled than run of the mill variational bayes).  Although you’re restricted in the class of computations you can represent, it is nonetheless a wide class. Indeed, this approach to probabilistic programming is probably not sufficiently well explored.

**Con**: Infer.net operates on and extends graphical models and works by hand defined primitive distributions. If your problem does not fit within this framework then you're stuck.

**Stan and HMC**

**Pro:** HMC is a highly principled approach to probabilistic programming, effective, built in diagnostic tools, gradients allow scaling to larger spaces. State of the art research augmenting basic HMC (not just NUTs).

**Con:** Persnickety (: to keep good properties, operations violating volume preservation or reversibility etc. are out, attempts to account for these violations tend to be fishy). Need to be differentiable, cannot easily do discrete mixture models or non-parametric unbounded growth models. Stan specifically, is very heavy, with a *complex* install workflow requiring an entire C++ compiler. Weak to multimodality ([https://arxiv.org/abs/1808.03230](https://arxiv.org/abs/1808.03230)) but note that *everything is weak multimodality, including brains (possible exceptions are planet-scale inference processes ala Evolution).*

*Minor*: Parameters must be tuned (work-arounds include NUTs but also, using hyper gradients on log probability of random subsample of data penalized by a term to encourage exploration).

**WebPPL or Probability Monads ala Gordon, Scibior and Ghahramani**

Pro: fully flexible, can easily write arbitrary models.

Con: For both, flexibility comes with a steep price: inference can be quite costly. For the monadic approach, extra costs as running interpreters via the Free monad is *slow.*

# EvoBayes

EvoBayes starting point is the observation that most Bayesian approaches tend to blow up on needed computational resources, are (one or more of) slow, heavy or with a cumbersome interface. Meanwhile, simulated annealing, which is not so different from Metropolis Hastings, is surprisingly fast and scales to quite hostile combinatorial spaces. A similar observation can be made about the ease of use of evolutionary or genetic algorithm packages, which are commonly deployed while also maintaining a population (their flexibility being of interest more so than their optimality). This caused me to consider a method combining all these approaches. 

EvoBayes is a lightweight F# toolkit for sampling and scoring distributions,
then using them with transition-based inference. Its smallest common
abstraction is `Distribution<'T>`: a value that can both produce a sample and
evaluate the log likelihood of a value. The library supplies common scalar and
discrete distributions such as normal, beta, Bernoulli, categorical, and
Poisson distributions.

For example, the same beta prior can generate a simulation and score a
candidate value:

```fsharp
open EvolutionaryBayes.Distributions
open EvolutionaryBayes.Extras

let winRatePrior = beta 6. 10.
let simulatedRates = winRatePrior.SampleN(100_000)
let logDensityAtHalf = winRatePrior.LogLikelihood 0.5
```

The compiled library intentionally has no general-purpose model computation
expression. Models are ordinary typed F# functions, while inference algorithms
remain explicit about the additional pieces they need—a complete log target
for MCMC, or a prior, non-negative scorer, and valid move kernel for particle
methods. This separation keeps trees, graphs, variable-size values, and other
domain-specific state available without requiring a model compiler.

The library is deliberately permissive about state representations, but not
every use of that flexibility is equally Bayesian. Correctness depends on the
contract of the selected algorithm: Hastings proposals must include the right
forward/reverse correction, and resample–move particle mutations must preserve
the current annealed target. Supplying an arbitrary search mutation is still
useful, but then the result should be treated as heuristic optimization rather
than posterior inference.

## Metropolis Hastings

`MCMC<'Data,'State>` runs a stateful Metropolis–Hastings chain against an
explicit log target. It supports symmetric and asymmetric proposals, optional
annealing, burn-in, thinning, continuation across calls, and acceptance
diagnostics. When annealing starts above temperature 1, warm-up states are
recorded in the trajectory but only temperature-1 states enter the posterior
sample collection.

Here is a complete one-parameter posterior. The proposal helper returns a
proper `Proposal<float>` and is temperature-aware when annealing is enabled:

```fsharp
open EvolutionaryBayes
open EvolutionaryBayes.Distributions
open EvolutionaryBayes.MutationKernels

let observations = [| 5.; 10.; 4. |]

let logTarget mean =
    (normal 0. 10.).LogLikelihood mean
    + (observations
       |> Array.sumBy (fun y -> (normal mean 1.).LogLikelihood y))

let chain =
    MCMC<unit, float>(
        ProposalKernels.gaussianRandomWalk 0.5,
        (fun _ mean -> logTarget mean),
        0.,
        nsamples = 5_000,
        burnin = 1_000,
        thin = 5)

let posteriorSamples = chain.RunChain()
let diagnostics = chain.Diagnostics
```

The log target includes both the prior and observation terms. Keeping that
function explicit makes it clear what density the chain targets and avoids
confusing a convenient prior generator with the full posterior.

## Particle Filter

`ParticleFilters.PopulationSampler` implements annealed resample–move
sequential Monte Carlo for a static target. It draws particles from a prior,
increments the influence of a non-negative likelihood or score as temperature
falls toward 1, resamples when effective sample size is low, and rejuvenates
particles with a target-preserving mutation kernel.

```fsharp
open System
open EvolutionaryBayes.Distributions
open EvolutionaryBayes.Extras
open EvolutionaryBayes.MutationKernels
open EvolutionaryBayes.ParticleFilters

let prior = normal 0. 10.
let observations = [| 5.; 10.; 4. |]

let likelihood mean =
    observations
    |> Array.sumBy (fun y -> (normal mean 1.).LogLikelihood y)
    |> exp

let sampler =
    PopulationSampler(
        prior,
        likelihood,
        Standard.gaussianRandomWalk 0.35,
        temperature = 8.,
        attenuate = 0.85)

let posterior =
    sampler.RecursiveMonteCarloSamples(
        samplespergen = 1_000,
        generations = 30)

let summary =
    posterior.SampleAndGroup(2_000, fun x -> Math.Round(x, 1))
```

`RecursiveMonteCarloSamples` is the direct resample–move variant.
`EvolveSequence` uses a bounded archive of earlier particle approximations as
additional proposal support, with a Metropolis–Hastings correction; the archive
helps proposals cross difficult spaces but does not give old generations
posterior mass. See [Mutation kernels](EvolutionaryBayes/MutationKernels.md)
for the exact move-kernel contract and archive details.

## Transitions/Perturbations/Mutations:

Transitions are explicit because the library is intended to work with more
than Euclidean parameter vectors. A proposal may alter one coordinate, replace
a subtree, rewire a graph, or regenerate part of a symbolic object. The move
must nevertheless match the inference algorithm's contract.

For ordinary numeric spaces, start with the supplied kernels:

```fsharp
let scalarMove = Standard.gaussianRandomWalk 0.25
let vectorMove = Standard.gaussianCoordinateWalk 0.15
```

Both particle kernels apply Metropolis correction against the annealed target.
For an asymmetric move, provide its forward and reverse log proposal densities
instead of pretending it is symmetric. For example, this log-normal random
walk stays in the positive reals:

```fsharp
open MathNet.Numerics.Distributions
open EvolutionaryBayes.MutationKernels

let positiveMove =
    Metropolis.fromProposal (fun _ x ->
        let forward = LogNormal(log x, 0.2)
        let proposed = forward.Sample()
        let reverse = LogNormal(log proposed, 0.2)

        { Value = proposed
          LogForward = forward.DensityLn proposed
          LogReverse = reverse.DensityLn x })
```

The same accounting applies to structural edits: include the probability of
choosing the edit kind, location, and replacement in both directions. If a
move truly is symmetric, `Metropolis.symmetric` is the concise particle form;
`Proposal.symmetric` and `ProposalKernels` provide the corresponding MCMC
proposal forms.

# Limitations

EvoBayes is a small inference toolkit, not a complete probabilistic-programming
compiler. In particular:

- it does not derive a posterior, proposal ratio, or mutation correction from
  an arbitrary generative function;
- it does not provide automatic differentiation, HMC/NUTS, variational
  inference, or the diagnostics of a mature dedicated inference system;
- MCMC and particle methods can still mix poorly in high-dimensional or
  strongly multimodal targets, and a population does not by itself solve that;
- recursive model and sampler functions execute as ordinary calls and can
  exhaust the stack—they are not lazily explored probability trees; and
- arbitrary mutations are permitted for search experiments, but they cease to
  define exact posterior inference unless they preserve the stated target.

For example, use a mature HMC system for a large smooth differentiable model,
Hansei-style search when exact discrete control flow is central, and Infer.NET
when the problem fits its scalable graphical-model family. EvoBayes is most at
home when the state or move is domain-specific and you want a compact F# API
for simulation, MCMC, annealing, or particle inference without hiding those
algorithmic choices.

# Regret Learning

EvoBayes is organized around one recurring operation: differential growth of
mass over a set of alternatives. Given a current distribution $x_t$ and a
non-negative growth factor $g_t$ with positive total mass, the common update is

$$
x_{t+1}(i)
=\frac{x_t(i)g_t(i)}{\sum_j x_t(j)g_t(j)}.
$$

The surrounding domain determines how that growth factor is obtained. For a
Bayesian hypothesis it is a likelihood $L_t(i)$. For an MWUA action it is
$\exp(\eta r_t(i))$. For an evolutionary strategy over a finite time step it is
$\exp(\Delta t\,f_i(x_t))$. In every case, alternatives with greater relative
evidence, payoff, or reproductive success acquire a greater share of the next
distribution. The interpretations add domain meaning and constraints, but the
selection geometry underneath them is the same.

The MWUA–Bayes connection is therefore exact under a change of coordinates,
not merely an implementation analogy. Setting

$$
r_t(i)=\frac{1}{\eta}\log L_t(i)
$$

makes the MWUA update identical to sequential Bayesian updating. Conversely,
exponentiated reward acts mathematically as a likelihood factor. Whether that
factor represents observed evidence, decision utility, or another measure of
success is a modeling interpretation layered over the same normalized
multiplicative process.

The same identity gives the evolutionary connection. The finite replicator
step implemented in this library is

$$
x_i(t+\Delta t) \propto x_i(t)\exp(\Delta t\,f_i(x(t))),
$$

which is MWUA with population-dependent fitness as its growth rate.
`Replicator.step` and MWUA consequently use the same stable exponential
reweighting kernel. Their APIs reflect different views of the process: online
learning exposes a `Choose`/`Learn` interaction and an average played strategy;
evolution exposes the current population and its continuous-time derivative.

The current replicator module has continuous population shares over a finite
strategy set. The planned continuous-trait extension will represent numeric or
structured traits with particles. It will reuse a shared particle core for
normalization, effective sample size, resampling, mutation scheduling, and
diagnostics, while remaining a separate consumer from Bayesian SMC: selection
will reweight by fitness, and biological mutation will intentionally change the
population instead of applying Metropolis–Hastings correction to preserve a
posterior. See [Evolutionary dynamics](EvolutionaryDynamics.md) for that staged
`ReplicatorMutator` plan.

Bandit and regret algorithms cover the decision-learning branch. EXP3 adds
importance-weighted feedback when only the chosen action is observed, while
regret matching learns from how every action would have performed relative to
the current mixture.

CFR is the tree-shaped extension of that last idea. A flat regret matcher learns
one distribution over actions. CFR places a local regret matcher at every
information set, then uses counterfactual reach probabilities to assign credit
through a sequential, partially observed game. It therefore turns a global
strategy problem into a collection of coupled local online-learning problems.

## Counterfactual Regret Minimization

CFR minimizes regret separately at every information set in a finite
extensive-form game. Unlike ordinary regret matching, it traverses a game tree
and weights each local update by the probability that the other players and
chance reach that decision. The average of the learned local strategies is the
policy used for play. In two-player zero-sum games with perfect recall, those
average strategies approach a Nash equilibrium; in multiplayer or general-sum
games, the solver still minimizes counterfactual regret, but that statement no
longer implies convergence to a Nash profile.

The most important part of a game adapter is `InformationSetId`. States receive
the same ID exactly when the acting player cannot distinguish them. Hidden
matching pennies is the smallest useful example: player 1 acts after player 0,
but cannot observe player 0's choice, so both underlying player-1 states share
information-set ID 1.

```fsharp
open EvolutionaryBayes.CFRCore

type State =
    | Player0Turn
    | Player1Turn of hiddenPlayer0Action: int
    | Terminal of player0Action: int * player1Action: int

let game =
    { new IGame<State> with
        member _.TerminalUtility(state, targetPlayer) =
            match state with
            | Terminal(player0Action, player1Action) ->
                let utility0 =
                    if player0Action = player1Action then 1. else -1.

                ValueSome(if targetPlayer = 0 then utility0 else -utility0)
            | _ -> ValueNone

        member _.Actor state =
            match state with
            | Player0Turn -> 0
            | Player1Turn _ -> 1
            | Terminal _ -> invalidOp "Terminal state"

        member _.InformationSetId state =
            match state with
            | Player0Turn -> 0
            | Player1Turn _ -> 1 // Player 0's action is hidden here.
            | Terminal _ -> invalidOp "Terminal state"

        member _.ActionCount state =
            match state with
            | Player0Turn | Player1Turn _ -> 2 // 0 = Heads, 1 = Tails
            | Terminal _ -> invalidOp "Terminal state"

        member _.NextState(state, action) =
            match state with
            | Player0Turn -> Player1Turn action
            | Player1Turn player0Action -> Terminal(player0Action, action)
            | Terminal _ -> invalidOp "Terminal state"

        member _.ChanceProbability(_state, _action) =
            invalidOp "This game has no chance nodes" }

let informationSets =
    [| { Id = 0; Owner = 0; ActionCount = 2 }
       { Id = 1; Owner = 1; ActionCount = 2 } |]

let solver =
    Solver.create
        SolverMode.CFRPlus
        2
        game
        informationSets
        2       // maximum nonterminal depth
        2       // maximum local action count
        1729    // seed, used by sampled modes

let training = Solver.run solver 10_000 100 Player0Turn
let player0Policy = Solver.averageStrategy solver 0
let player1Policy = Solver.averageStrategy solver 1
```

Both policies approach `[| 0.5; 0.5 |]`. The `100` passed to `Solver.run` is
average-strategy burn-in: regret learning still occurs, but the first 100
iterations do not contribute to the returned playing policy. Repeated calls
continue the same solver; it owns iteration numbering so CFR+ linear averaging
cannot accidentally repeat or skip an iteration.

The four modes make traversal and update semantics explicit:

| Mode | Tree traversal | Regret update | Average weighting | Typical use |
| --- | --- | --- | --- | --- |
| `CFR` | exhaustive | signed | uniform | deterministic baseline and small trees |
| `CFRPlus` | exhaustive | clipped at zero | linear | faster practical convergence when exhaustive traversal fits |
| `MCCFR` | external sampling | signed | uniform | larger trees where full traversal is too costly |
| `MCCFRPlus` | external sampling | clipped at zero | linear | sampled CFR+ experiments; validate under sampling variance |

Actions are dense local indices `0 .. ActionCount(state) - 1`; illegal global
actions and Boolean masks never enter the traversal. `Actor` may return any
player, allowing consecutive or skipped turns. For an explicit chance node it
returns `ChanceActor`, while `ChanceProbability` supplies a normalized
distribution over that node's local actions. General-sum games return the
requested player's own payoff from `TerminalUtility` rather than relying on
two-player negation.

`Solver.create` returns an opaque solver that owns packed regret and mandatory
average-strategy tables plus reusable traversal workspace. `Solver.run` trains
for a fixed additional budget; `Solver.runUntil` uses a caller-defined error
measure; `Solver.evaluateAverage` exactly evaluates the current average profile;
and `Solver.averageStrategy` returns one normalized information-set row.

See [Counterfactual Regret Minimization](CFR.md) for the mathematical audit,
memory model, implementation history, and complete production API. The
runnable
[`hidden-matching-pennies.fsx`](EvolutionaryBayes.CFR.Tests/games/hidden-matching-pennies.fsx)
expands the example above. Kuhn poker and Mini Dudo in the same directory
demonstrate explicit chance, exact average-profile evaluation, and
tolerance-driven training.

```powershell
dotnet build EvolutionaryBayes.sln -c Release -v:minimal
dotnet run --project EvolutionaryBayes.CFR.Tests -c Release
dotnet run --project EvolutionaryBayes.CFR.Benchmarks -c Release -- --revision <revision>
```

Auditable runtime, memory, and allocation results are consolidated in
[`CFR_BENCHMARK_RESULTS.md`](CFR_BENCHMARK_RESULTS.md).

## Flat and contextual regret minimization

For repeated decisions without a game tree, the library provides
full-information online learners for minimizing external regret. Exponential
multiplicative weights and standalone regret matching share the same small API.

```fsharp
open EvolutionaryBayes

type Action =
    | Go
    | Stop

let reward =
    function
    | (Go, Go) -> -100.
    | (Go, Stop) -> 1.
    | (Stop, Go) -> 0.
    | (Stop, Stop) -> 0.

let learner = Mwua.create 0.05 [| Go; Stop |]
let action = learner.Choose()

// Full-information feedback: score every action against the observation.
let observedOpponentAction = Stop
learner.Learn(fun candidate -> reward (candidate, observedOpponentAction))

// A sequence is processed one round at a time in enumeration order.
learner.LearnBatch
    [ fun candidate -> reward (candidate, Go)
      fun candidate -> reward (candidate, Stop) ]

let currentStrategy = learner.Strategy
let playedAverage = learner.AverageStrategy

// Standard regret matching and the clipped plus variant have the same surface.
let regretMatcher = RegretMatching.create [| Go; Stop |]
let regretMatcherPlus = RegretMatching.createPlus [| Go; Stop |]

```

When only the selected action's reward is observable, use EXP3 instead of the
full-information learners. The environment step remains explicit and pure: it
returns both its next state and the reward that is passed to `Learn`.

```fsharp
let banditStep target action =
    let observedReward = if action = target then 1.0 else 0.0
    let nextTarget = if target = Go then Stop else Go
    nextTarget, observedReward

let bandit = Exp3.create 0.05 0.1 [| Go; Stop |]
let choice = bandit.Choose()
let nextTarget, observedReward = banditStep Go choice.Action
bandit.Learn(choice, observedReward)

// Batch feedback is the same loop. Each function describes one reward round
// and is evaluated only for the action sampled in that round.
bandit.LearnBatch
    [ fun action -> if action = Go then 1.0 else 0.0
      fun action -> if action = Stop then 1.0 else 0.0 ]

let currentBanditStrategy = bandit.Strategy
let playedBanditAverage = bandit.AverageStrategy
```

EXP3 rewards must be finite values in `[0, 1]`. Its learning rate is the first
numeric argument and its explicit exploration fraction is the second. A
`BanditChoice` records the sampled action and probability; return it to the
same learner exactly once before requesting another choice.

For a small repeating set of contexts, exact-context variants keep one
independent learner per context. Their constructors hide the per-context
learner factory, and the first observation for a new context is learned rather
than discarded.

```fsharp
type Situation =
    | Quiet
    | Busy

let bySituation: ExactContextBanditRegretMinimizer<Situation, Action> =
    ContextualExp3.create 0.05 0.1 [| Go; Stop |]

let contextualChoice = bySituation.Choose(Quiet)
let observedReward = if contextualChoice.Action = Go then 1.0 else 0.0
bySituation.Learn(contextualChoice, observedReward)

bySituation.LearnBatch
    [ Quiet, (fun action -> if action = Go then 1.0 else 0.0)
      Busy, (fun action -> if action = Stop then 1.0 else 0.0) ]
```

`ContextualMwua` and `ContextualRegretMatching` provide the same exact-context
idea for full-information feedback. Exact-context tables memorize; they do not
generalize between different contexts.

For generalization, EXP4 learns a mixture over contextual policies. Evidence
changes a policy's global weight, so its recommendations can transfer to
contexts that have not yet appeared.

```fsharp
let policies: ContextualPolicy<Situation, Action>[] =
    [| (fun situation -> if situation = Quiet then Go else Stop)
       (fun situation -> if situation = Quiet then Stop else Go)
       (fun _ -> Stop) |]

let contextualBandit =
    Exp4.create 0.05 0.1 [| Go; Stop |] policies

let policyChoice = contextualBandit.Choose(Quiet)
let observedReward = if policyChoice.Action = Go then 1.0 else 0.0
contextualBandit.Learn(policyChoice, observedReward)

let actionDistribution = contextualBandit.Strategy(Busy)
let currentPolicyMixture = contextualBandit.PolicyStrategy
let playedPolicyAverage = contextualBandit.AveragePolicyStrategy
```

EXP4 policies must return one of the configured distinct actions. As with
EXP3, rewards are finite values in `[0, 1]`, and each opaque contextual choice
must be learned exactly once before another choice is requested.

# Evolutionary Dynamics

`Replicator` evolves a normalized finite population under population-dependent
fitness. It is a pure evolutionary API: there is no sampled action, learning
token, or average strategy.

```fsharp
let fitness (population: Replicator.Population<Action>) action =
    let goShare = population.Share Go

    match action with
    | Go -> -100.0 * goShare + (1.0 - goShare)
    | Stop -> 0.0

let initialPopulation = Replicator.create [| Go, 0.5; Stop, 0.5 |]

// A continuous-time diagnostic; step and run do not consume this value.
let instantaneousChange = Replicator.derivative fitness initialPopulation
let nextPopulation = Replicator.step 0.01 fitness initialPopulation

// Equivalent to applying the same step sequentially 5,000 times.
let evolvedPopulation =
    Replicator.run 5_000 0.01 fitness initialPopulation

let finalDistribution = evolvedPopulation.Distribution
```

Input masses are normalized. Strategy names must be distinct, shares must be
finite and nonnegative with positive total mass, time steps must be finite and
positive, and fitness must be finite. The exponential step preserves the
simplex and keeps zero-share strategies absent. It uses the same internal
max-shifted exponential reweighting primitive as MWUA, while retaining a pure
population API rather than MWUA's mutable learning and averaging state.

