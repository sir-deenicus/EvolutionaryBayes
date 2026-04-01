module EvolutionaryBayes.MutationKernels

open System
open MathNet.Numerics.Distributions
open EvolutionaryBayes.ProbMonad

let private rng = Random()

type MutationContext<'a when 'a : equality> =
    { Population : Distribution<'a>
      WeightedPopulation : ('a * float) [] option
      Temperature : float
      Likelihood : 'a -> float
      PriorLogLikelihood : 'a -> float
      LogAnnealedScore : 'a -> float
      AnnealedScore : 'a -> float }

module MutationContext =
    let create
        (temperature : float)
        (likelihood : 'a -> float)
        (prior : Distribution<'a>)
        (population : Distribution<'a>)
        (weightedPopulation : ('a * float) [] option) =

        let safeTemperature = max 1. temperature
        let beta = 1. / safeTemperature

        let logAnnealedScore x =
            let lik = max 0. (likelihood x)
            if lik <= 0. then -infinity
            else prior.LogLikelihood x + beta * log lik

        let annealedScore x =
            let score = logAnnealedScore x
            if Double.IsNegativeInfinity score then 0.
            else exp score

        { Population = population
          WeightedPopulation = weightedPopulation
          Temperature = safeTemperature
          Likelihood = likelihood
          PriorLogLikelihood = prior.LogLikelihood
          LogAnnealedScore = logAnnealedScore
          AnnealedScore = annealedScore }

    let annealingPower (context : MutationContext<'a>) =
        1. / context.Temperature

type Proposal<'a> =
    { Value : 'a
      LogForward : float
      LogReverse : float }

module Proposal =
    let symmetric value =
        { Value = value
          LogForward = 0.
          LogReverse = 0. }

module ProposalKernels =
    let private validateStepSize stepSize =
        if stepSize <= 0. then invalidArg (nameof stepSize) "stepSize must be positive."

    let symmetricProposal perturb =
        fun (_ : float) x -> Proposal.symmetric (perturb x)

    let temperatureScaledSymmetric perturb =
        fun temperature x -> Proposal.symmetric (perturb temperature x)

    let gaussianRandomWalk stepSize =
        validateStepSize stepSize
        temperatureScaledSymmetric (fun temperature x ->
            let scaled = stepSize * sqrt temperature
            x + Normal.Sample(rng, 0., scaled))

    let gaussianVectorRandomWalk stepSize =
        validateStepSize stepSize
        temperatureScaledSymmetric (fun temperature xs ->
            let scaled = stepSize * sqrt temperature
            xs |> Array.map (fun x -> x + Normal.Sample(rng, 0., scaled)))

    let gaussianCoordinateWalk stepSize =
        validateStepSize stepSize
        temperatureScaledSymmetric (fun temperature xs ->
            if Array.isEmpty xs then xs
            else
                let scaled = stepSize * sqrt temperature
                let ys = Array.copy xs
                let i = rng.Next(xs.Length)
                ys.[i] <- ys.[i] + Normal.Sample(rng, 0., scaled)
                ys)

module Metropolis =
    let acceptLogAcceptanceRatio logAlpha =
        if Double.IsNaN logAlpha then false
        elif logAlpha >= 0. then true
        else rng.NextDouble() < exp logAlpha

    let acceptanceLogRatio
        (context : MutationContext<'a>)
        current
        proposed
        logProposalRatio =
        context.LogAnnealedScore proposed
        - context.LogAnnealedScore current
        + logProposalRatio

    let fromProposal
        (proposal : MutationContext<'a> -> 'a -> Proposal<'a>)
        (context : MutationContext<'a>)
        current =

        let proposed = proposal context current
        let logAlpha =
            acceptanceLogRatio context current proposed.Value
                (proposed.LogReverse - proposed.LogForward)

        if acceptLogAcceptanceRatio logAlpha then proposed.Value
        else current

    let symmetric
        (propose : MutationContext<'a> -> 'a -> 'a)
        (context : MutationContext<'a>)
        current =
        fromProposal (fun ctx x -> Proposal.symmetric (propose ctx x)) context current

    let withLogProposalRatio
        (propose : MutationContext<'a> -> 'a -> 'a)
        (logProposalRatio : MutationContext<'a> -> 'a -> 'a -> float)
        (context : MutationContext<'a>)
        current =

        let proposed = propose context current
        let logAlpha = acceptanceLogRatio context current proposed (logProposalRatio context current proposed)
        if acceptLogAcceptanceRatio logAlpha then proposed
        else current

    let repeat steps
        (kernel : MutationContext<'a> -> 'a -> 'a)
        (context : MutationContext<'a>)
        initial =

        let rec loop n state =
            if n <= 0 then state
            else loop (n - 1) (kernel context state)

        loop steps initial

module Standard =
    let gaussianRandomWalk stepSize =
        if stepSize <= 0. then invalidArg (nameof stepSize) "stepSize must be positive."
        Metropolis.symmetric (fun _ x -> x + Normal.Sample(rng, 0., stepSize))

    let gaussianVectorRandomWalk stepSize =
        if stepSize <= 0. then invalidArg (nameof stepSize) "stepSize must be positive."
        Metropolis.symmetric (fun _ xs ->
            xs |> Array.map (fun x -> x + Normal.Sample(rng, 0., stepSize)))

    let gaussianCoordinateWalk stepSize =
        if stepSize <= 0. then invalidArg (nameof stepSize) "stepSize must be positive."
        Metropolis.symmetric (fun _ xs ->
            if Array.isEmpty xs then xs
            else
                let ys = Array.copy xs
                let i = rng.Next(xs.Length)
                ys.[i] <- ys.[i] + Normal.Sample(rng, 0., stepSize)
                ys)
