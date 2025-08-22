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

/// <summary>
/// Generates a set of samples from a prior distribution and weights them according to a likelihood function.
/// </summary>
/// <param name="likelihood">A function that computes the likelihood of a sample. Should return a value between 0 and 1.</param>
/// <param name="n">The number of samples to generate.</param>
/// <param name="prior">The prior distribution to sample the initial particles from.</param>
/// <returns>An array of samples paired with their normalized weights.</returns>
/// <remarks>
/// This function samples from the prior distribution, computes the likelihood for each sample,
/// and then normalizes the weights so that they sum to 1. This function works best for larger n's.
/// </remarks>
let weightedSamples (likelihood : 'a -> float) (n : int)
    (prior : Distribution<_>) = 
    prior.SampleN n 
    |> Array.map (fun x -> x, likelihood x) 
    |> Array.normalizeWeights


type MapFn<'a, 'b> =
    | Parallel 
    | Sequential
    | Custom of (('a -> 'b) -> 'a [] -> 'b [])

/// <summary>
/// Performs recursive importance sampling with mutation and attenuation over multiple steps.
/// </summary>
/// <param name="T">The initial temperature parameter for likelihood weighting.</param>
/// <param name="attenuate">The attenuation factor to reduce the temperature parameter at each step.</param>
/// <param name="mutateprob">The probability of mutating a sample at each step.</param>
/// <param name="mutate">A function to mutate a sample given the choices and the current sample.</param>
/// <param name="likelihood">A function that computes the likelihood of a sample. Should return a value between 0 and 1.</param>
/// <param name="numparticles">The number of particles (samples) to use.</param>
/// <param name="numsteps">The number of steps to perform the sampling.</param>
/// <param name="mapf">An optional mapping function to apply to the samples. Defaults to sequential Array.map if not provided.</param>
/// <param name="prior">The prior distribution to sample the initial particles from.</param>
/// <returns>A distribution of particles after performing the recursive importance sampling.</returns>
/// <remarks>
/// This method maintains a set of particles that are reweighted and resampled at each step.
/// The particles are mutated with a given probability, and the temperature parameter is attenuated
/// to control the influence of the likelihood function over time.
/// </remarks>
let recursiveMonteCarloSamples T attenuate mutateprob mutate (likelihood : 'a -> float)
    (numparticles : int) (numsteps : int) mapf (prior : Distribution<_>) =
    let mapfn = 
        match mapf with 
        | None -> Array.map
        | Some Parallel -> Array.Parallel.map
        | Some Sequential -> Array.map
        | Some (Custom f) -> f

    let choices = weightedSamples likelihood numparticles prior 

    let rec loop T samples dist =
        if samples = 0 then 
            dist
            |> Array.groupBy id 
            |> Array.map (fun (x, xs) -> x, float xs.Length)
            |> Array.normalizeWeights
        else
            let new_weighted_dist =
                discreteSampleN numparticles dist // Resample
                |> mapfn (fun x -> // Mutate & Reweight
                    let x' =
                        if random.NextDouble() < mutateprob then mutate choices x
                        else x
                    x', (likelihood x') ** (1. / T)
                )
                |> Array.normalizeWeights

            loop (max 1. (T * attenuate)) (samples - 1) new_weighted_dist
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
/// <param name="mutate">A function to mutate a sample given the population and the current sample.</param>
/// <param name="likelihood">A function that computes the likelihood of a sample. Should return a value between 0 and 1.</param>
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
                                (Array.normalizeWeights [| for (m, p) in mem -> m, p ** (1. / T) |]) 

                        let hypothesis =
                            discreteSample //sample a hypothesis from selected generation
                                (Array.map (fun (x, p) -> x, p ** (1. / T)) history
                                |> Array.normalizeWeights) 
                        hypothesis)

            let samples =
                population.SampleN samplesPerGen
                |> map (fun sample ->
                    let sample' =
                        if random.NextDouble() < mutateprob then
                            mutate population sample
                        else sample
                    sample', (likelihood sample') ** (1. / T))
                |> Array.normalizeWeights
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
/// <param name="scorer">A function that computes the likelihood of a sample. Should return a value between 0 and 1.</param>
/// <param name="mutate">A function to mutate a sample given the population and the current sample.</param>
/// <param name="temperature">An optional initial temperature parameter for likelihood weighting. Defaults to 1 if not provided.</param>
/// <param name="attenuate">An optional attenuation factor to reduce the temperature parameter at each step. Defaults to 1 if not provided.</param>
type PopulationSampler<'a when 'a : equality>(prior : Distribution<'a>, scorer, mutate, ?temperature, ?attenuate) = 

    let popMutateSeq ps x = 
        mutate (distBuilder (fun () -> discreteSample ps)) x 

    let T, atten = defaultArg temperature 1., defaultArg attenuate 1.

    /// <summary>
    /// Initializes a new instance of the <see cref="PopulationSampler{T}"/> class with a simpler mutation function.
    /// </summary>
    /// <param name="prior">The prior distribution to sample the initial particles from.</param>
    /// <param name="scorer">A function that computes the likelihood of a sample. Should return a value between 0 and 1.</param>
    /// <param name="mutate">A function to mutate a sample.</param>
    /// <param name="temperature">An optional initial temperature parameter for likelihood weighting. Defaults to 1 if not provided.</param>
    /// <param name="attenuate">An optional attenuation factor to reduce the temperature parameter at each step. Defaults to 1 if not provided.</param>
    new(prior, scorer, (mutate:'a -> 'a), ?temperature, ?attenuate) =
        PopulationSampler(prior, scorer, (fun _ x -> mutate x), ?temperature = temperature, ?attenuate = attenuate)

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
