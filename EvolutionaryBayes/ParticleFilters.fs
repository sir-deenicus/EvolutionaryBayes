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
            |> reweightWith likelihood
            |> Distributions.categorical2
            |> importanceSamples likelihood n
            |> Distributions.categorical2
            |> loop (c - 1)
    loop k dist'
