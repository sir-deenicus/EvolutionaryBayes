module ParticleFilters
open EvolutionaryBayes.ProbMonad
open System
open Prelude.Common
open Prelude.Math
open Helpers

let importanceSamples (prior:Distribution<_>) likelihood n =
    let samples = prior.SampleN n
    Array.countElements samples 
    |> Map.toArray
    |> Array.normalizeWeights
    |> Array.map (fun (x,p) -> x, p * likelihood x)