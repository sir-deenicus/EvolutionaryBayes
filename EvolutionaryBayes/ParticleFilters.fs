module ParticleFilters

open EvolutionaryBayes.ProbMonad
open System
open Prelude.Common
open Prelude.Math
open Helpers
open EvolutionaryBayes

let reweightWith likelihood samples =
    Array.countElements samples
    |> Map.toArray
    |> Array.normalizeWeights
    |> Array.map (fun (x, p) -> x, p * likelihood x)


let weightWith likelihood samples =
    samples |> Array.map (fun x -> x, likelihood x)

let importanceSamples (likelihood : 'a -> float) (n : int)
    (prior : Distribution<_>) = prior.SampleN n |> reweightWith likelihood

let sequenceSamples mutateprob mutate (likelihood : 'a -> float) (numparticles : int)
    (numsamples : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood numparticles prior
    let dist' = Distributions.categorical2 choices

    let rec loop samples (dist : Distribution<'a>) =
        if samples = 0 then dist
        else
            dist.SampleN numparticles
            |> Array.map (fun x ->
                   if random.NextDouble() < mutateprob then mutate x
                   else x)
            |> weightWith likelihood                     |> Distributions.categorical2
            |> importanceSamples likelihood numparticles |> Distributions.categorical2
            |> loop (samples - 1)
    loop numsamples dist'

///Inspired by evolution (see papers on regret minimization's connection to evolution), this
///method maintains a memory where each element is a set of samples weighted by their average
///performance. The head of this list is the most recent generation. Worst performing generations
///are evicted when memory size limit is reached. Hopefully, by then their descendants can be found
///in later generations. At each step, a generation is sampled and then a member of that generation.
///Because the memory size is bounded there is a fixed upper bound on total samples with the population
///mixing and hopefully drifting towards the most fit.
let evolveSequence mutateprob maxsize mem mutate (likelihood : 'a -> float)
    (samplesPerGen : int) (numSteps : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood samplesPerGen prior
    let dist' = Distributions.categorical2 choices, Array.averageBy snd choices

    let rec loop steps mem =
        if steps = 0 then mem
        else
            let population =
                dist { let! history = Distributions.categorical2
                                          (List.toArray mem) //sample a generation
                       let! hypothesis = history //sample a hypothesis from selected generation
                       return hypothesis }

            let samples =
                population.SampleN samplesPerGen
                |> Array.map (fun sample ->
                       if random.NextDouble() < mutateprob then
                           mutate population sample
                       else sample)
                |> weightWith likelihood
                |> Distributions.categorical2
                |> importanceSamples likelihood samplesPerGen

            let w = Array.averageBy snd samples 

            let memory =
                let history = (Distributions.categorical2 samples, w) :: mem
                if history.Length > maxsize then
                    List.sortByDescending snd history |> List.take maxsize
                else history
            loop (steps - 1) memory 
    loop numSteps (dist'::mem)

///The purpose of this function is to take a set of weighted samples and inject or remember it into
///existing memory by drawing the most probable samples and comparing against an existing population.
///This is done for just 3 steps, which should be enough to get good mixing into existing pop of 
///fittest members of this sample.
let remember likelihood mutate maxsize mem (samples : _ []) =
    Distributions.categorical2 samples
    |> evolveSequence 0.1 maxsize mem mutate likelihood samples.Length 3