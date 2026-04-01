# Mutation Kernels for `PopulationSampler`

`recursiveMonteCarloSamples` is now designed as a resample-move SMC method.
That means the `mutate` function is not just a search heuristic: it is part of the transition kernel.

## Rule of thumb

- If `mutate` is the identity function, the algorithm is still a valid annealed SMC sampler, but particle diversity can collapse after repeated resampling.
- If `mutate` preserves the current annealed target distribution, the method is a proper resample-move SMC sampler.
- If `mutate` does not preserve the current target, the method becomes heuristic rather than exact.

The current target is proportional to:

`prior(x) * likelihood(x)^(1 / T)`

where `T` is the current temperature.

## The helper file

See `MutationKernels.fs` for:

- `MutationContext<'a>`
- `MutationContext.create`
- `Proposal<'a>`
- `Proposal.symmetric`
- `Metropolis.fromProposal`
- `Metropolis.symmetric`
- `Metropolis.withLogProposalRatio`
- `Metropolis.repeat`
- `Standard.gaussianRandomWalk`
- `Standard.gaussianVectorRandomWalk`
- `Standard.gaussianCoordinateWalk`

## What `MutationContext` gives you

- `Population`
  A sampler for the current particle approximation. Useful as a proposal source.
- `WeightedPopulation`
  The current weighted particles when available. In `recursiveMonteCarloSamples` this is usually `Some`.
- `Temperature`
  The current annealing temperature.
- `Likelihood`
  The raw score function passed into the particle filter.
- `PriorLogLikelihood`
  The prior log density or log mass.
- `LogAnnealedScore`
  A log score proportional to the current target.
- `AnnealedScore`
  The same target score in probability space.

`LogAnnealedScore` is usually the safest thing to use inside accept/reject logic.

## Standard numeric spaces

For ordinary parameter spaces like `float` or `float[]`, start with a symmetric random walk:

```fsharp
open EvolutionaryBayes.MutationKernels

let mutate = Standard.gaussianRandomWalk 0.25

let sampler =
    PopulationSampler(prior, scorer, mutate, temperature = 8., attenuate = 0.92)
```

For vectors:

```fsharp
let mutate = Standard.gaussianCoordinateWalk 0.15
```

This uses Metropolis-Hastings internally, so the move preserves the current annealed target.

## Custom symmetric proposals

If your proposal is symmetric but domain-specific, wrap it with `Metropolis.symmetric`:

```fsharp
let mutate =
    Metropolis.symmetric (fun ctx x ->
        let step = if ctx.Temperature > 2. then 0.5 else 0.1
        x + step)
```

The proposal above is only an illustration. In practice, the proposal should use randomness.

## Trees, graphs, and ASTs

For structured spaces, use `Metropolis.fromProposal`.
You provide a proposal plus the forward and reverse proposal probabilities in log space.

```fsharp
open EvolutionaryBayes.MutationKernels

let mutateAst =
    Metropolis.fromProposal (fun ctx ast ->
        let editedAst = ast // replace with a real edit
        let logForward = 0.0
        let logReverse = 0.0
        { Value = editedAst
          LogForward = logForward
          LogReverse = logReverse })
```

This is the right pattern when your move is something like:

- subtree replacement
- insert/delete node
- relabel node
- edge add/remove/rewire
- grammar-guided regeneration

## Designing good structural proposals

- Make edits reversible when possible.
- Keep track of the probability of selecting the move type.
- Keep track of the probability of selecting the edit location.
- Keep track of the probability of the replacement object or subtree.
- Compute the reverse move probability for the edited object.
- Use `ctx.LogAnnealedScore` in the MH acceptance ratio.

If the forward and reverse proposal probabilities are equal, you can treat the move as symmetric and use `Metropolis.symmetric` instead.

## Population-guided proposals

For large or irregular spaces, you may want to use the particle population as a proposal source.
Good uses:

- choose a subtree library from `WeightedPopulation`
- borrow a fragment from another particle
- bias a repair step toward high-weight particles

Use the population as part of the proposal, not as a replacement for acceptance correction.
If the proposal is not symmetric, include the forward and reverse proposal probabilities.

## Repeated local moves

If one mutation step is too weak, compose several:

```fsharp
let mutate =
    Metropolis.repeat 3 (Standard.gaussianCoordinateWalk 0.1)
```

This is still a valid kernel as long as each repeated step is valid.

## Practical tuning

- High temperature: broader moves are usually okay.
- Low temperature: use smaller, more conservative moves.
- If acceptance is near zero, your proposal is too aggressive.
- If particles become near-duplicates, increase move strength or do more MH steps.
- If the proposal is hard to reverse, prefer a different edit family.

## Recommended starting point

- Standard numeric spaces: `Standard.gaussianRandomWalk` or `Standard.gaussianCoordinateWalk`
- Structured spaces: `Metropolis.fromProposal`
- Unknown space but easy local edit: `Metropolis.symmetric`

## Future improvements

- Multiple MH steps per annealing stage
  This is likely the biggest practical improvement for the current sampler. Repeating a valid mutation kernel several times after each reweight or resample stage usually reduces duplicate particles and improves local exploration more than small formula tweaks elsewhere.
- Temperature-aware proposal scaling
  Use larger moves at high temperature and smaller moves near `T = 1` to balance exploration and acceptance.
- Mixture proposals
  Combine several proposal families in one kernel, for example mostly local moves plus occasional larger jumps or population-guided proposals.
- Adaptive proposals for vector spaces
  Learn proposal scale or covariance from the current weighted particle cloud instead of using a fixed scalar step size.
- Population-guided structural proposals
  For trees, graphs, and ASTs, mine reusable fragments or substructures from `WeightedPopulation` and use them inside a properly corrected MH proposal.
- Domain-specific reversible edit libraries
  Build reusable reversible move sets for common spaces such as subtree replace/insert/delete, graph edge toggle/rewire, or grammar-guided regeneration.
- Better diagnostics
  Track acceptance rate, ESS trajectory, unique particle count, and rough ancestry collapse so tuning is based on actual sampler behavior.
- Configurable ESS thresholds
  The current threshold is a reasonable default, but some targets benefit from resampling earlier or later.
- More robust proposal composition helpers
  Add reusable helpers for proposal mixtures, adaptive random walks, and structured edit kernels so users do not have to hand-roll them.

## Next biggest impact step

Add a first-class helper for repeated, temperature-aware MH rejuvenation, plus a few default mixture kernels.

Why this is the best next step:

- it directly attacks residual particle degeneracy,
- it improves mixing without changing the outer SMC logic,
- it helps both ordinary numeric spaces and structured spaces,
- and it gives users a much better default than a single weak mutation step.
