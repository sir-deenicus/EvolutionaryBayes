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

let inline reweightWith likelihood samples =
    Array.countBy id samples
    |> Array.normalizeWeightsWith float
    |> Array.map (fun (x, p) -> x, p * likelihood x)
    |> Array.normalizeWeights

let weightWithT T likelihood samples =
    samples
    |> Array.map (fun x -> x, (likelihood x) ** (1./T))
    |> Array.normalizeWeights

let weightWith likelihood samples = weightWithT 1. likelihood samples

let importanceSamples (likelihood : 'a -> float) (n : int)
    (prior : Distribution<_>) = prior.SampleN n |> reweightWith likelihood

let importanceSamplesArray (likelihood : 'a -> float) (n : int)
    (samples) = discreteSampleN n samples |> reweightWith likelihood

///NOTE: likelihood should return a number between 0 and 1.
let recursiveImportanceSample T attenuate mutateprob mutate (likelihood : 'a -> float)
    (numparticles : int) (numsteps : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood numparticles prior 

    let rec loop T samples dist =
        if samples = 0 then dist
        else
            discreteSampleN numparticles dist
            |> Array.map (fun x ->
                   if random.NextDouble() < mutateprob then mutate choices x
                   else x)
            |> weightWithT T likelihood 
            |> importanceSamplesArray likelihood numparticles 
            |> loop (max 1. (T * attenuate)) (samples - 1)
    loop T numsteps choices

///Inspired by evolution (see papers on regret minimization's connection to evolution), this
///method maintains a memory where each element is a set of samples weighted by their average
///performance. The head of this list is the most recent generation. Worst performing generations
///are evicted when memory size limit is reached. Hopefully, by then their descendants can be found
///in later generations. At each step, a generation is sampled and then a member of that generation.
///Because the memory size is bounded there is a fixed upper bound on total samples with the population
///mixing and hopefully drifting towards the most fit.
///If the likelihood is too uninformative (such as for a complicated space), this will 
///usually end up as little progress for evolution, 
///causing exploration to stick close to the initial state and wander randomly. 
///NOTE: Likelihood should return a number between 0 and 1.
let evolveSequence T atten mutateprob maxsize mem mutate
    (likelihood : 'a -> float) (samplesPerGen : int) (numSteps : int)
    (prior : Distribution<_>) =

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
                |> Array.map (fun sample ->
                       if random.NextDouble() < mutateprob then
                           mutate population sample
                       else sample)
                |> weightWithT T likelihood
                |> importanceSamplesArray likelihood samplesPerGen

            let w = Array.averageBy (fst >> likelihood) samples

            let memory =
                let mem' = (samples, w) :: mem
                if mem'.Length > maxsize then
                    List.sortByDescending snd mem' |> List.take maxsize
                else mem'
            loop (max 1. (T * atten)) (steps - 1) memory
    
    let guesses = importanceSamples likelihood samplesPerGen prior
    let mem' = (guesses, Array.averageBy (fst >> likelihood) guesses) :: mem
    loop T numSteps mem'

///NOTE: scorer should return a number between 0 and 1.
type PopulationSampler<'a when 'a : equality>(prior : Distribution<'a>, scorer, mutate, ?temperature, ?attenuate) = 

    let popMutateSeq ps x = 
        mutate (distBuilder (fun () -> discreteSample ps)) x 

    let T, atten = defaultArg temperature 1., defaultArg attenuate 1.

    new(prior, scorer, (mutate:'a -> 'a), ?temperature, ?attenuate) =
        PopulationSampler(prior, scorer, (fun _ x -> mutate x), ?temperature = temperature, ?attenuate = attenuate)

    member __.Scorer = scorer

    member __.RecursiveImportanceSample(?mutateprob, ?samplespergen, ?generations) =
        let mp = defaultArg mutateprob 0.4
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        recursiveImportanceSample T atten mp popMutateSeq scorer samplespergen gens prior
        |> categorical2

    ///WARNING: If the mutation function makes use of a population distribution, then this will simply pass the prior for that purpose. Prefer the recursive samplers instead.
    member __.SampleChain n =
        MetropolisHastings.sample atten T (scorer>>log) (mutate prior) n (prior.Sample())
        |> SampleSummarize.roundAndGroupSamplesWith id
        |> categorical2 

    member __.EvolveSequence(?mutateprob, ?maxpoplen, ?samplespergen,
                             ?generations) =
        let mp = defaultArg mutateprob 0.65
        let maxpopsize = defaultArg maxpoplen 100
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        let pops =
            evolveSequence T atten mp maxpopsize [] mutate scorer
                samplespergen gens prior
            |> List.toArray
            |> Array.normalizeWeights
            |> categorical2
        dist { let! pop = pops 
               return (discreteSample pop) }
