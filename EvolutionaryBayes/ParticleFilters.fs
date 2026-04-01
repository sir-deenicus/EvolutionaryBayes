module EvolutionaryBayes.ParticleFilters

open EvolutionaryBayes.ProbMonad
open System
open Prelude.Common
open Prelude.Math 
open EvolutionaryBayes
open System.Collections.Generic 
open Prelude.Sampling
open EvolutionaryBayes.Distributions
open Prelude.ProbabilityTools
open Extras
open Prelude
open EvolutionaryBayes.MutationKernels

/// <summary>
/// Generates a set of samples from a prior distribution and weights them according to a likelihood function.
/// </summary>
/// <param name="likelihood">A function that computes a non-negative likelihood or score for a sample.</param>
/// <param name="n">The number of samples to generate.</param>
/// <param name="prior">The prior distribution to sample the initial particles from.</param>
/// <returns>An array of samples paired with their normalized weights.</returns>
/// <remarks>
/// This function samples from the prior distribution, computes the likelihood for each sample,
/// and then normalizes the weights so that they sum to 1. This function works best for larger n's.
/// </remarks>
let private uniformLogWeight count =
    if count <= 0 then -infinity
    else -log (float count)

let private safeLogScore score =
    if Double.IsNaN score || score < 0. then -infinity
    elif score = 0. then -infinity
    elif Double.IsPositiveInfinity score then infinity
    else log score

let private logSumExp (values : float []) =
    if values |> Array.exists Double.IsPositiveInfinity then infinity
    else
        let finite =
            values
            |> Array.filter (fun x -> not (Double.IsNaN x) && not (Double.IsNegativeInfinity x))

        if finite.Length = 0 then -infinity
        else
            let m = Array.max finite
            m + log (finite |> Array.sumBy (fun x -> exp (x - m)))

let private normalizeLogSampleWeights (samples : ('a * float) []) =
    if samples.Length = 0 then samples
    else
        let sanitized =
            samples
            |> Array.map (fun (x, logw) ->
                let logw' = if Double.IsNaN logw then -infinity else logw
                x, logw')

        let positiveInfinityCount =
            sanitized
            |> Array.sumBy (fun (_, logw) -> if Double.IsPositiveInfinity logw then 1 else 0)

        if positiveInfinityCount > 0 then
            let logUniform = uniformLogWeight positiveInfinityCount
            sanitized
            |> Array.map (fun (x, logw) ->
                x, if Double.IsPositiveInfinity logw then logUniform else -infinity)
        else
            let logTotal = sanitized |> Array.map snd |> logSumExp
            if Double.IsNegativeInfinity logTotal then
                let logUniform = uniformLogWeight sanitized.Length
                sanitized |> Array.map (fun (x, _) -> x, logUniform)
            else
                sanitized |> Array.map (fun (x, logw) -> x, logw - logTotal)

let private logWeightsToWeights (samples : ('a * float) []) =
    samples
    |> Array.map (fun (x, logw) ->
        let w =
            if Double.IsNegativeInfinity logw then 0.
            else exp logw
        x, w)

let private aggregateLogSampleWeights (samples : ('a * float) []) =
    samples
    |> Array.groupBy fst
    |> Array.map (fun (x, xs) -> x, xs |> Array.map snd |> logSumExp)
    |> normalizeLogSampleWeights

let private effectiveSampleSize (samples : ('a * float) []) =
    let denom =
        samples
        |> Array.sumBy (fun (_, logw) ->
            if Double.IsNegativeInfinity logw then 0.
            else
                let w = exp logw
                w * w)

    if denom <= 0. then 0.
    else 1. / denom

let private annealedLogWeights temperature (likelihood : 'a -> float) (samples : 'a []) =
    let beta = 1. / max 1. temperature
    samples
    |> Array.map (fun x -> x, beta * safeLogScore (likelihood x))
    |> normalizeLogSampleWeights

let private validateAnnealingSchedule T attenuate =
    if T < 1. then invalidArg (nameof T) "Temperature T must be >= 1 for the annealing schedule."
    if attenuate <= 0. || attenuate > 1. then
        invalidArg (nameof attenuate) "attenuate must be in the interval (0, 1]."

let weightedSamples (likelihood : 'a -> float) (n : int)
    (prior : Distribution<_>) = 
    prior.SampleN n
    |> annealedLogWeights 1. likelihood
    |> logWeightsToWeights


type MapFn<'a, 'b> =
    | Parallel 
    | Sequential
    | Custom of (('a -> 'b) -> 'a [] -> 'b [])

/// <summary>
/// Performs annealed sequential Monte Carlo with multinomial resampling and an optional rejuvenation move.
/// </summary>
/// <param name="T">The initial temperature parameter for likelihood weighting.</param>
/// <param name="attenuate">The attenuation factor to reduce the temperature parameter at each step.</param>
/// <param name="mutateprob">The probability of mutating a sample at each step.</param>
/// <param name="mutate">A rejuvenation kernel that receives the current annealed target context and the current sample.</param>
/// <param name="likelihood">A function that computes a non-negative likelihood or score for a sample.</param>
/// <param name="numparticles">The number of particles (samples) to use.</param>
/// <param name="numsteps">The number of steps to perform the sampling.</param>
/// <param name="mapf">An optional mapping function to apply to the samples. Defaults to sequential Array.map if not provided.</param>
/// <param name="prior">The prior distribution to sample the initial particles from.</param>
/// <returns>A distribution of particles after performing the recursive importance sampling.</returns>
/// <remarks>
/// This method targets an annealed sequence of distributions proportional to
/// prior(x) * likelihood(x)^(1 / T_t). At each step it:
/// reweights particles by the incremental annealing update,
/// resamples only when the effective sample size (ESS) drops too low,
/// and optionally applies a mutation or rejuvenation move.
///
/// Correctness depends on the mutation step. If mutate leaves the current annealed target invariant,
/// such as a Metropolis-Hastings, Gibbs, or other transition kernel with the current target as its invariant distribution,
/// then the overall procedure is a proper resample-move SMC method. If mutate is the identity function, the algorithm
/// remains a valid annealed SMC sampler without rejuvenation, but may suffer from path degeneracy.
/// If mutate does not preserve the current target, the method becomes a heuristic search procedure rather than an exact SMC update.
///
/// Guidelines for writing mutate:
/// 1. Treat MutationContext.Population as a proposal source and MutationContext.WeightedPopulation as the current empirical particle approximation when available.
/// 2. Preserve the current target distribution. A safe default is an MH step:
///    propose x' from a proposal q(x' | x), then accept with probability
///    min(1, [pi_t(x') q(x | x')] / [pi_t(x) q(x' | x)]),
///    where pi_t(x) is proportional to prior(x) * likelihood(x)^(1 / T_t).
///    MutationContext.LogAnnealedScore and MutationContext.AnnealedScore provide this target up to normalization.
/// 3. If the proposal is symmetric, the q terms cancel and the acceptance ratio depends only on pi_t.
/// 4. Keep support broad enough that any region with non-zero target probability can still be reached.
/// 5. Prefer small local moves or a short MH chain per particle over aggressive jumps that are almost always rejected.
/// 6. MutationContext.Temperature lets the move adapt to the current annealing stage.
/// 7. If mutate samples from the population argument directly, use it as a proposal source or mixture component,
///    not as a replacement for the acceptance correction.
///    See EvolutionaryBayes.MutationKernels for reusable helpers.
/// </remarks>
let recursiveMonteCarloSamples T attenuate mutateprob mutate (likelihood : 'a -> float)
    (numparticles : int) (numsteps : int) mapf (prior : Distribution<_>) =
    validateAnnealingSchedule T attenuate

    let mapfn = 
        match mapf with 
        | None -> Array.map
        | Some Parallel -> Array.Parallel.map
        | Some Sequential -> Array.map
        | Some (Custom f) -> f

    let essResampleThreshold = 0.5 * float numparticles
    let choices = prior.SampleN numparticles |> annealedLogWeights T likelihood

    let rec loop currentT samples dist =
        if samples = 0 then 
            aggregateLogSampleWeights dist
            |> logWeightsToWeights
        else
            let nextT = max 1. (currentT * attenuate)
            let deltaBeta = (1. / nextT) - (1. / currentT)

            let reweightedDist =
                dist
                |> Array.map (fun (x, logw) ->
                    x, logw + deltaBeta * safeLogScore (likelihood x))
                |> normalizeLogSampleWeights

            let weightedDist = reweightedDist |> logWeightsToWeights
            let ess = effectiveSampleSize reweightedDist
            let shouldResample = ess <= essResampleThreshold
            let mutationContext =
                MutationContext.create nextT likelihood prior (categorical2 weightedDist) (Some weightedDist)

            let particlesForMove =
                if shouldResample then
                    let logUniform = uniformLogWeight numparticles
                    discreteSampleN numparticles weightedDist
                    |> Array.map (fun x -> x, logUniform)
                else
                    reweightedDist

            let movedDist =
                particlesForMove
                |> mapfn (fun (x, logw) ->
                    let x' =
                        if random.NextDouble() < mutateprob then mutate mutationContext x
                        else x
                    x', logw)

            loop nextT (samples - 1) movedDist
    loop T numsteps choices

/// <summary>
/// Inspired by evolution and based on sequential monte carlo, this method maintains a memory where each element is a set of samples weighted by their average performance.
/// The head of this list is the most recent generation. Worst performing generations are evicted when the memory size limit is reached.
/// Hopefully, by then their descendants can be found in later generations.
/// At each step, a generation is sampled and then a member of that generation.
/// Because the memory size is bounded, there is a fixed upper bound on total samples with the population mixing across generations and hopefully drifting towards the most fit.
/// This in effect allows the model to backtrack and is best for problems where some complex state space is being explored.
/// Use <see cref="recursiveImportanceSample"/> for simpler problems.
/// If the likelihood is too uninformative (such as for a too complicated space), this will usually end up as little progress for evolution,
/// causing exploration to stick close to the initial state and wander randomly, since it will be hard to beat any given random initial configuration.
/// </summary>
/// <param name="T">The initial temperature parameter for likelihood weighting.</param>
/// <param name="atten">The attenuation factor to reduce the temperature parameter at each step.</param>
/// <param name="mutateprob">The probability of mutating a sample at each step.</param>
/// <param name="maxsize">The maximum size of the memory to retain past generations.</param>
/// <param name="mem">The initial memory of past generations.</param>
/// <param name="mutate">A function to mutate a sample given the current mutation context and the current sample.</param>
/// <param name="likelihood">A function that computes a non-negative likelihood or score for a sample.</param>
/// <param name="samplesPerGen">The number of samples to generate per generation.</param>
/// <param name="numSteps">The number of steps to perform the sampling.</param>
/// <param name="mapf">An optional mapping function to apply to the samples. Defaults to sequential Array.map if not provided.</param>
/// <param name="prior">The prior distribution to sample the initial particles from.</param>
/// <returns>A memory of generations after performing the evolutionary sequence.</returns>
/// <remarks>
/// This method maintains sets of particles in a memory, where each element is a set of samples weighted by their average performance.
/// The particles are sampled from a generation and then reweighted and resampled at each step. The particles are mutated with a given probability, and the temperature parameter is attenuated
/// to control the influence of the likelihood function over time. The memory retains past generations, with the worst performing generations being evicted
/// when the memory size limit is reached. This allows the model to backtrack and explore different parts of the state space, improving the chances of finding
/// a more optimal solution.
/// </remarks>
let evolveSequence T atten mutateprob maxsize mem mutate
    (likelihood : 'a -> float) (samplesPerGen : int) (numSteps : int)
    mapf
    (prior : Distribution<_>) =

    validateAnnealingSchedule T atten

    let map = 
        match mapf with
        | Some Sequential
        | None -> Array.map
        | Some Parallel -> Array.Parallel.map
        | Some (Custom f) -> f 

    let rec loop T steps mem =
        if steps = 0 then mem
        else
            let population =  
                match mem with
                | [] -> prior
                | _ ->
                    distBuilder (fun () ->
                        let history =
                            discreteSample //sample a generation
                                ([| for (m, p) in mem -> m, (1. / T) * safeLogScore p |]
                                 |> normalizeLogSampleWeights
                                 |> logWeightsToWeights)

                        let hypothesis =
                            discreteSample //sample a hypothesis from selected generation
                                (history
                                 |> Array.map (fun (x, p) -> x, (1. / T) * safeLogScore p)
                                 |> normalizeLogSampleWeights
                                 |> logWeightsToWeights)
                        hypothesis)

            let samples =
                population.SampleN samplesPerGen
                |> map (fun sample ->
                    let mutationContext =
                        MutationContext.create T likelihood prior population None
                    let sample' =
                        if random.NextDouble() < mutateprob then
                            mutate mutationContext sample
                        else sample
                    sample', (1. / T) * safeLogScore (likelihood sample'))
                |> normalizeLogSampleWeights
                |> logWeightsToWeights
                |> discreteSampleN samplesPerGen

            let len = float samples.Length
            let samplesDist = samples |> Array.map (fun x -> x, 1. / len)
            let w = Array.averageBy (fst >> likelihood) samplesDist

            let memory =
                let mem' = (samplesDist, w) :: mem
                if mem'.Length > maxsize then
                    List.sortByDescending snd mem' |> List.take maxsize
                else mem'
            loop (max 1. (T * atten)) (steps - 1) memory
    
    let guesses = weightedSamples likelihood samplesPerGen prior
    let mem' = (guesses, Array.averageBy (fst >> likelihood) guesses) :: mem
    loop T numSteps mem'

/// <summary>
/// Represents a population sampler that uses evolutionary and sequential monte carlo sampling inspired techniques.
/// </summary>
/// <typeparam name="'a">The type of the samples.</typeparam>
/// <param name="prior">The prior distribution to sample the initial particles from.</param>
/// <param name="scorer">A function that computes a non-negative likelihood or score for a sample.</param>
/// <param name="mutate">A function to mutate a sample given the current mutation context and the current sample.</param>
/// <param name="temperature">An optional initial temperature parameter for likelihood weighting. Defaults to 1 if not provided.</param>
/// <param name="attenuate">An optional attenuation factor to reduce the temperature parameter at each step. Defaults to 1 if not provided.</param>
type PopulationSampler<'a when 'a : equality>
    (prior : Distribution<'a>, scorer, mutate : MutationContext<'a> -> 'a -> 'a, ?temperature, ?attenuate) = 

    let popMutateSeq context x = mutate context x

    let T, atten = defaultArg temperature 1., defaultArg attenuate 1.
    do validateAnnealingSchedule T atten

    /// <summary>
    /// Initializes a new instance of the <see cref="PopulationSampler{T}"/> class with a simpler mutation function.
    /// </summary>
    /// <param name="prior">The prior distribution to sample the initial particles from.</param>
    /// <param name="scorer">A function that computes a non-negative likelihood or score for a sample.</param>
    /// <param name="mutate">A function to mutate a sample.</param>
    /// <param name="temperature">An optional initial temperature parameter for likelihood weighting. Defaults to 1 if not provided.</param>
    /// <param name="attenuate">An optional attenuation factor to reduce the temperature parameter at each step. Defaults to 1 if not provided.</param>
    new(prior, scorer, (mutate:'a -> 'a), ?temperature, ?attenuate) =
        PopulationSampler<'a>(
            prior,
            scorer,
            ((fun (_ : MutationContext<'a>) (x : 'a) -> mutate x) : MutationContext<'a> -> 'a -> 'a),
            ?temperature = temperature,
            ?attenuate = attenuate)

    /// <summary>
    /// Initializes a new instance of the <see cref="PopulationSampler{T}"/> class with a mutation function that uses the current population distribution.
    /// </summary>
    /// <param name="prior">The prior distribution to sample the initial particles from.</param>
    /// <param name="scorer">A function that computes the likelihood of a sample. Should return a non-negative value.</param>
    /// <param name="mutate">A function to mutate a sample given the current population distribution and the current sample.</param>
    /// <param name="temperature">An optional initial temperature parameter for likelihood weighting. Defaults to 1 if not provided.</param>
    /// <param name="attenuate">An optional attenuation factor to reduce the temperature parameter at each step. Defaults to 1 if not provided.</param>
    new(prior, scorer, (mutate:Distribution<'a> -> 'a -> 'a), ?temperature, ?attenuate) =
        PopulationSampler<'a>(
            prior,
            scorer,
            ((fun (context : MutationContext<'a>) (x : 'a) -> mutate context.Population x) : MutationContext<'a> -> 'a -> 'a),
            ?temperature = temperature,
            ?attenuate = attenuate)

    member __.Scorer = scorer

    /// <summary>
    /// Recursively generates importance weighted samples with optional parameters for mutation probability, samples per generation, and number of generations.
    /// </summary>
    /// <param name="mutateprob">An optional mutation probability. Defaults to 0.8 if not provided.</param>
    /// <param name="samplespergen">An optional number of samples per generation (iteration). Defaults to 500 if not provided.</param>
    /// <param name="generations">An optional number of generations. Defaults to 50 if not provided. This is the number of steps to iterate.</param>
    /// <param name="mapf">An optional mapping function to apply to the samples. Defaults to sequential Array.map if not provided.</param>
    /// <returns>A categorical distribution of the recursive importance samples.</returns>
    /// <remarks>
    /// This method maintains a set of particles that are reweighted and resampled at each step.
    /// The particles are mutated with a given probability, and the temperature parameter is attenuated
    /// to control the influence of the likelihood function over time.
    /// </remarks>
    member __.RecursiveMonteCarloSamples(?mutateprob, ?samplespergen, ?generations, ?mapf) =
        let mp = defaultArg mutateprob 0.8
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        recursiveMonteCarloSamples T atten mp popMutateSeq scorer samplespergen gens mapf prior
        |> categorical2

    /// <summary>
    /// Inspired by evolution and based on sequential monte carlo, this method maintains a memory where each element is a set of samples weighted by their average performance.
    /// The head of this list is the most recent generation. Worst performing generations are evicted when the memory size limit is reached.
    /// Hopefully, by then their descendants can be found in later generations.
    /// At each step, a generation is sampled and then a member of that generation.
    /// Because the memory size is bounded, there is a fixed upper bound on total samples with the population mixing across generations and hopefully drifting towards the most fit.
    /// This in effect allows the model to backtrack and is best for problems where some complex state space is being explored.
    /// Use <see cref="recursiveImportanceSample"/> for simpler problems.
    /// If the likelihood is too uninformative (such as for a too complicated space), this will usually end up as little progress for evolution,
    /// causing exploration to stick close to the initial state and wander randomly, since it will be hard to beat any given random initial configuration.
    /// </summary>
    /// <param name="mutateprob">An optional mutation probability. Defaults to 0.8 if not provided.</param>
    /// <param name="maxpoplen">An optional maximum population length. Defaults to 100 if not provided.</param>
    /// <param name="samplespergen">An optional number for samples per generation. Defaults to 500 if not provided.</param>
    /// <param name="generations">Optional number of generations (iterations). Defaults to 50 if not provided.</param>
    /// <param name="mapf">An optional mapping function to apply to the samples. Defaults to sequential Array.map if not provided.</param>
    /// <returns>A categorical distribution of the evolved sequence of populations.</returns>
    /// <remarks>
    /// This method maintains sets of particles in a memory, where each element is a set of samples weighted by their average performance.
    /// The particles are sampled from a generation and then reweighted and resampled at each step. The particles are mutated with a given probability, and the temperature parameter is attenuated
    /// to control the influence of the likelihood function over time. The memory retains past generations, with the worst performing generations being evicted
    /// when the memory size limit is reached. This allows the model to backtrack and explore different parts of the state space, improving the chances of finding
    /// a more optimal solution.
    /// </remarks>
    member __.EvolveSequence(?mutateprob, ?maxpoplen, ?samplespergen,
                             ?generations, ?mapf) =
        let mp = defaultArg mutateprob 0.8
        let maxpopsize = defaultArg maxpoplen 100
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        let pops =
            evolveSequence T atten mp maxpopsize [] mutate scorer
                samplespergen gens mapf prior
            |> List.toArray
            |> Array.normalizeWeights
            |> categorical2
        dist { let! pop = pops 
               return (discreteSample pop) }
