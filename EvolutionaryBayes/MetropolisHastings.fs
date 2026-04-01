namespace EvolutionaryBayes

open System
open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions
open EvolutionaryBayes.MutationKernels
open Prelude 
 
type ChainStep<'sample> =
    { State : 'sample
      LogTarget : float
      Temperature : float
      Accepted : bool }

type ChainRun<'sample> =
    { Steps : ChainStep<'sample> []
      RetainedSamples : 'sample []
      FinalState : 'sample * float
      FinalTemperature : float
      AcceptedCount : int }

type ChainDiagnostics =
    { TotalTransitions : int
      AcceptedTransitions : int
      AcceptanceRate : float
      RetainedPosteriorSamples : int
      BurnIn : int
      Thin : int
      CurrentTemperature : float }

module MetropolisHastings =
    let internal validateAnnealingSchedule T attenuate =
        if T < 1. then invalidArg (nameof T) "Temperature T must be >= 1."
        if attenuate <= 0. || attenuate > 1. then
            invalidArg (nameof attenuate) "attenuate must be in the interval (0, 1]."

    let internal warmupStepsUntilPosterior temperature attenuate =
        if temperature <= 1. then 0
        elif attenuate = 1. then Int32.MaxValue
        else
            let raw = log (1. / temperature) / log attenuate
            max 0 (int (ceil raw))

    let internal iterateSimple n (logLikelihood : 'a -> float)
        (proposal : Distribution<_>) (initial : 'a * float) =
        let rec loop newChain (currentState, logp) remaining =
            if remaining = 0 then newChain
            else
                let candidate = proposal.Sample()
                let logp' = logLikelihood candidate
                let accepted = Metropolis.acceptLogAcceptanceRatio (logp' - logp)

                let next =
                    if accepted then (candidate, logp')
                    else (currentState, logp)

                loop (fst next :: newChain) next (remaining - 1)

        if n <= 0 then [ fst initial ]
        else loop [ fst initial ] initial n

    let iterateProposal attenuate T0 n (proposal : float -> 'a -> Proposal<'a>) (logTarget : 'a -> float)
        (initial : 'a * float) =
        validateAnnealingSchedule T0 attenuate

        let rec loop T retained steps (currentState, currentLogTarget) remaining acceptedCount =
            if remaining = 0 then
                { Steps = List.rev steps |> List.toArray
                  RetainedSamples = List.rev retained |> List.toArray
                  FinalState = (currentState, currentLogTarget)
                  FinalTemperature = T
                  AcceptedCount = acceptedCount }
            else
                let proposed = proposal T currentState
                let proposedLogTarget = logTarget proposed.Value
                let logAlpha =
                    ((proposedLogTarget - currentLogTarget) / T)
                    + proposed.LogReverse
                    - proposed.LogForward

                let accepted = Metropolis.acceptLogAcceptanceRatio logAlpha

                let nextState, nextLogTarget =
                    if accepted then proposed.Value, proposedLogTarget
                    else currentState, currentLogTarget

                let retained' =
                    if T = 1. then nextState :: retained
                    else retained

                let step =
                    { State = nextState
                      LogTarget = nextLogTarget
                      Temperature = T
                      Accepted = accepted }

                let nextT = max 1. (T * attenuate)
                loop nextT retained' (step :: steps) (nextState, nextLogTarget) (remaining - 1)
                    (if accepted then acceptedCount + 1 else acceptedCount)

        loop T0 [] [] initial n 0

    let iterate attenuate T0 n perturb (logTarget : 'a -> float)
        (initial : 'a * float) =
        iterateProposal attenuate T0 n (ProposalKernels.symmetricProposal perturb) logTarget initial
  
    let sampleHastings attenuate T logTarget proposal (n : int) init =
        let initial = init, logTarget init
        let run = iterateProposal attenuate T n proposal logTarget initial
        Array.append [| init |] (run.Steps |> Array.map (fun step -> step.State))
        |> Array.toList

    let sample attenuate T logTarget perturb (n : int) init =
        sampleHastings attenuate T logTarget (ProposalKernels.symmetricProposal perturb) n init

    let sampleBasic logLikelihood (prior : Distribution<_>) (n : int) =
        let x = prior.Sample()
        iterateSimple n logLikelihood prior (x, logLikelihood x)
         

///The perturbation/mutation step can be a sampler whose parameters are the current state. 
type MCMC<'data, 'sample when 'sample: equality>
        (proposal : float -> 'sample -> Proposal<'sample>, logTarget, init, ?nsamples, ?Temperature, ?attenuate, ?burnin, ?thin) =

    let T = defaultArg Temperature 1.
    let atten = defaultArg attenuate 1.
    let burnIn = defaultArg burnin 0
    let thin = defaultArg thin 1
    do
        if T > 1. && atten = 1. then
            invalidArg "attenuate" "MCMC retains posterior samples only. Use attenuate < 1 when Temperature > 1, or use MetropolisHastings.sampleHastings for fixed-temperature chains."
        MetropolisHastings.validateAnnealingSchedule T atten
        if burnIn < 0 then invalidArg "burnin" "burnin must be >= 0."
        if thin <= 0 then invalidArg "thin" "thin must be >= 1."

    let numsamples = defaultArg nsamples 100_000 
    let samples = ResizeArray<'sample>()
    let history = ResizeArray<ChainStep<'sample>>()
    let mutable observations = ResizeArray<'data>()
    let mutable currentTemperature = T
    let initialState = init
    let mutable currentState = init, 0.
    let mutable posteriorSeen = 0
    let mutable acceptedTransitions = 0
    let mutable totalTransitions = 0
    let mutable lastRunDiagnostics =
        { TotalTransitions = 0
          AcceptedTransitions = 0
          AcceptanceRate = 0.
          RetainedPosteriorSamples = 0
          BurnIn = burnIn
          Thin = thin
          CurrentTemperature = currentTemperature }

    let currentLogTarget() = logTarget observations

    let refreshCurrentState() =
        let state = fst currentState
        currentState <- state, currentLogTarget() state

    let retainPosteriorSamples (posteriorSamples : 'sample []) =
        posteriorSamples
        |> Array.choose (fun sample ->
            let index = posteriorSeen
            posteriorSeen <- posteriorSeen + 1

            let keep =
                index >= burnIn
                && ((index - burnIn) % thin = 0)

            if keep then Some sample else None)

    let transitionsToCollect desiredPosteriorSamples =
        let warmup = MetropolisHastings.warmupStepsUntilPosterior currentTemperature atten
        if warmup = Int32.MaxValue then
            invalidArg "attenuate" "This chain never reaches T = 1, so it cannot produce posterior samples."

        let posteriorTransitionsNeeded =
            if desiredPosteriorSamples <= 0 then 0
            else
                let mutable seen = posteriorSeen
                let mutable kept = 0
                let mutable transitions = 0

                while kept < desiredPosteriorSamples do
                    let keep =
                        seen >= burnIn
                        && ((seen - burnIn) % thin = 0)

                    seen <- seen + 1
                    transitions <- transitions + 1
                    if keep then kept <- kept + 1

                transitions

        warmup + posteriorTransitionsNeeded

    do refreshCurrentState()

    new(proposal : 'sample -> Proposal<'sample>, logTarget, init, ?nsamples, ?Temperature, ?attenuate, ?burnin, ?thin) =
        MCMC<'data, 'sample>(
            ((fun (_ : float) (x : 'sample) -> proposal x) : float -> 'sample -> Proposal<'sample>),
            logTarget,
            init,
            ?nsamples = nsamples,
            ?Temperature = Temperature,
            ?attenuate = attenuate,
            ?burnin = burnin,
            ?thin = thin)

    new(mutator : 'sample -> 'sample, logTarget, init, ?nsamples, ?Temperature, ?attenuate, ?burnin, ?thin) =
        MCMC<'data, 'sample>(
            ProposalKernels.symmetricProposal mutator,
            logTarget,
            init,
            ?nsamples = nsamples,
            ?Temperature = Temperature,
            ?attenuate = attenuate,
            ?burnin = burnin,
            ?thin = thin)

    member __.ClearSamples() =
        samples.Clear()
        history.Clear()

    member __.ResetChain() =
        samples.Clear()
        history.Clear()
        currentTemperature <- T
        currentState <- initialState, snd currentState
        posteriorSeen <- 0
        acceptedTransitions <- 0
        totalTransitions <- 0
        lastRunDiagnostics <-
            { TotalTransitions = 0
              AcceptedTransitions = 0
              AcceptanceRate = 0.
              RetainedPosteriorSamples = 0
              BurnIn = burnIn
              Thin = thin
              CurrentTemperature = currentTemperature }
        refreshCurrentState()

    member __.Samples = Seq.toArray samples

    member __.Trajectory = Seq.toArray history

    member __.AcceptanceRate =
        if totalTransitions = 0 then 0.
        else float acceptedTransitions / float totalTransitions

    member __.Diagnostics =
        { TotalTransitions = totalTransitions
          AcceptedTransitions = acceptedTransitions
          AcceptanceRate = if totalTransitions = 0 then 0. else float acceptedTransitions / float totalTransitions
          RetainedPosteriorSamples = samples.Count
          BurnIn = burnIn
          Thin = thin
          CurrentTemperature = currentTemperature }

    member __.LastRunDiagnostics = lastRunDiagnostics

    /// <summary>
    /// Gets the distribution of the posterior samples, optionally grouped by a provided function.
    /// For annealed chains, only samples generated after the temperature reaches 1 are included.
    /// </summary>
    /// <param name="groupWith">An optional function to group the samples with. Defaults to the identity function if not provided.</param>
    /// <returns>The distribution of the posterior samples as a categorical distribution.</returns>
    member __.GetDistribution(?groupWith) =
        SampleSummarize.roundAndGroupSamplesWith (defaultArg groupWith id) samples
        |> categorical2
        
    /// <summary>
    /// Samples a specified number of elements from the retained posterior samples.
    /// For annealed chains, this excludes warmup samples collected before the temperature reaches 1.
    /// </summary>
    /// <param name="n">The number of samples to draw.</param>
    /// <param name="groupWith">An optional function to group the samples with. Defaults to the identity function if not provided.</param>
    /// <returns>A sequence of sampled elements.</returns>
    member __.SampleN(n, ?groupWith) =   
        SampleSummarize.roundAndGroupSamplesWith (defaultArg groupWith id) samples
        |> Sampling.discreteSampleN n        

    member __.Observations = observations
    
    member d.RunChain(?samplecount) =
        let posteriorSamplesRequested = defaultArg samplecount numsamples
        refreshCurrentState()

        let run =
            MetropolisHastings.iterateProposal atten currentTemperature
                (transitionsToCollect posteriorSamplesRequested)
                proposal (currentLogTarget()) currentState

        history.AddRange run.Steps
        let retained = retainPosteriorSamples run.RetainedSamples
        samples.AddRange retained
        currentState <- run.FinalState
        currentTemperature <- run.FinalTemperature
        acceptedTransitions <- acceptedTransitions + run.AcceptedCount
        totalTransitions <- totalTransitions + run.Steps.Length
        lastRunDiagnostics <-
            { TotalTransitions = run.Steps.Length
              AcceptedTransitions = run.AcceptedCount
              AcceptanceRate = if run.Steps.Length = 0 then 0. else float run.AcceptedCount / float run.Steps.Length
              RetainedPosteriorSamples = retained.Length
              BurnIn = burnIn
              Thin = thin
              CurrentTemperature = currentTemperature }
        d.Samples
