﻿module EvolutionaryBayes.ParticleFilters

open EvolutionaryBayes.ProbMonad
open System
open Prelude.Common
open Prelude.Math
open Helpers
open EvolutionaryBayes
open System.Collections.Generic
open EvolutionaryBayes.Distributions

let dictToArray d =
    [| for (x : KeyValuePair<_, _>) in d -> x.Key, float x.Value |]

let countElements (xs : _ []) =
    let countdict = Dict()
    for x in xs do
        let b, v = countdict.TryGetValue x
        if b then countdict.[x] <- v + 1
        else countdict.[x] <- 1
    dictToArray countdict

let reweightWith likelihood samples =
    countElements samples
    |> Array.normalizeWeights
    |> Array.map (fun (x, p) -> x, p * likelihood x)
    |> Array.normalizeWeights

let weightWith likelihood samples =
    samples
    |> Array.map (fun x -> x, likelihood x)
    |> Array.normalizeWeights

let importanceSamples (likelihood : 'a -> float) (n : int)
    (prior : Distribution<_>) = prior.SampleN n |> reweightWith likelihood

let sequenceSamples mutateprob mutate (likelihood : 'a -> float)
    (numparticles : int) (numsteps : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood numparticles prior
    let dist' = Distributions.categorical2 choices

    let rec loop samples (dist : Distribution<'a>) =
        if samples = 0 then dist
        else
            dist.SampleN numparticles
            |> Array.map (fun x ->
                   if random.NextDouble() < mutateprob then mutate x
                   else x)
            |> weightWith likelihood
            |> Distributions.categorical2
            |> importanceSamples likelihood numparticles
            |> Distributions.categorical2
            |> loop (samples - 1)
    loop numsteps dist'

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
    let dist' =
        Distributions.categorical2 choices,
        Array.averageBy (fst >> likelihood) choices

    let rec loop steps mem =
        if steps = 0 then mem
        else
            let population =
                dist { let! history = Distributions.categorical2
                                          (Array.normalizeWeights
                                               (List.toArray mem)) //sample a generation
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
    loop numSteps (dist' :: mem)

///The purpose of this function is to take a set of weighted samples and inject or remember it into
///existing memory by drawing the most probable samples and comparing against an existing population.
///This is done for just 3 steps, which should be enough to get good mixing into existing pop of
///fittest members of this sample.
let remember likelihood mutate maxsize mem (samples : _ []) =
    Distributions.categorical2 samples
    |> evolveSequence 0.1 maxsize mem mutate likelihood samples.Length 3

type SolverEvolver<'a when 'a : equality>(generator : Distribution<'a>, mutate, mutateOnPopulation, scorer) =

    member __.SequenceSamples(?mutateprob, ?samplespergen, ?generations) =
        let mp = defaultArg mutateprob 0.4
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        sequenceSamples mp mutate scorer samplespergen gens generator

    member __.SampleChain n =
        MetropolisHastings.sample 1. scorer mutate generator n
        |> Sampling.roundAndGroupSamplesWith id
        |> categorical2

    member __.EvolveSequence(?mutateprob, ?maxpopsize, ?samplespergen,
                             ?generations) =
        let mp = defaultArg mutateprob 0.4
        let maxpopsize = defaultArg maxpopsize 250
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50

        let pops =
            evolveSequence mp maxpopsize [] mutateOnPopulation scorer
                samplespergen gens generator
            |> List.toArray
            |> Array.normalizeWeights
            |> categorical2
        dist { let! pop = pops
               let! memberx = pop
               return memberx }

    member __.SampleFrom n (dist : Distribution<_>) =
        dist.SampleN(n)
        |> Array.map (fun x -> x, scorer x)
        |> Array.normalizeWeights
        |> Helpers.Sampling.compactMapSamples id
        |> Array.sortByDescending snd

    member __.SampleFromRaw n (dist : Distribution<_>) =
        dist.SampleN(n)
        |> Array.map (fun x -> x, scorer x)
        |> Array.normalizeWeights
        |> Helpers.Sampling.compactMapSamples id
        |> Array.map (fun (x, _) -> x, scorer x)
        |> Array.sortByDescending snd
