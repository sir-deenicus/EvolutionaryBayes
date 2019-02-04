module Helpers
open Prelude.Math
open Prelude.Common
open System
open EvolutionaryBayes
open EvolutionaryBayes.ProbMonad
    
type Distribution<'T>  with    
    member d.SampleN n =  [|for _ in 1..n -> d.Sample()|]  

module Sampling =
    let roundAndGroupSamplesWith f samples =
        samples
        |> Seq.toArray
        |> Array.map f
        |> Array.groupBy id
        |> Array.map (keepLeft (Array.length >> float))
        |> Array.normalizeWeights
    
    let compactMapSamples f samples =
        Array.map (fun (x, p : float) -> f x, p) samples
        |> Array.groupBy fst
        |> Array.map (fun (x, xs) -> x, Array.sumBy snd xs)
        |> Array.normalizeWeights
    
    let aggregateSamples nIters sampler =
        let updaterate = max 1 (nIters / 10)
        let mutable nelements = 0
        [| for i in 0..nIters do
               let samples = sampler() 
               let els = Array.countElements samples
               nelements <- nelements + els.Count
               if i % updaterate = 0 then 
                   printfn "%d of %d" i nIters
                   printfn "%d elements" nelements
               yield (els) |]
        |> Array.fold (fun fm m -> Map.merge (+) id m fm) Map.empty
        |> Map.toArray