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

let sequenceSamples mutateprob mutate (likelihood : 'a -> float) (n : int)
    (k : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood n prior
    let dist' = Distributions.categorical2 choices

    let rec loop c (dist : Distribution<'a>) =
        if c = 0 then dist
        else
            dist.SampleN n
            |> Array.map (fun x ->
                   if random.NextDouble() < mutateprob then mutate x
                   else x)
            |> weightWith likelihood          |> Distributions.categorical2
            |> importanceSamples likelihood n |> Distributions.categorical2
            |> loop (c - 1)
    loop k dist'

let sequenceSamplesMem mutateprob mutate maxsize (likelihood : 'a -> float)
    (n : int) (k : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood n prior
    let dist' = Distributions.categorical2 choices

    let rec loop c mem (distr : Distribution<'a>) =
        if c = 0 then (mem, distr)
        else
            let population =
                dist { let! history = Distributions.categorical2
                                          (List.toArray mem)
                       let! hypothesis = history
                       return hypothesis }

            let samples =
                population.SampleN n
                |> Array.map (fun sample ->
                       if random.NextDouble() < mutateprob then
                           mutate population sample
                       else sample)
                |> weightWith likelihood
                |> Distributions.categorical2
                |> importanceSamples likelihood n

            let w = Array.averageBy snd samples 

            let memory =
                let history = (Distributions.categorical2 samples, w) :: mem
                if history.Length > maxsize then
                    List.sortByDescending snd history |> List.take maxsize
                else history
            loop (c - 1) memory (Distributions.categorical2 samples)
    loop k [] dist'