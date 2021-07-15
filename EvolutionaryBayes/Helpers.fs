module EvolutionaryBayes.Extras
open Prelude.Math
open Prelude.Common
open System
open EvolutionaryBayes
open EvolutionaryBayes.ProbMonad
open Prelude

type Distribution<'T when 'T: equality> with
    member d.SampleN n = [| for _ in 1 .. n -> d.Sample() |]
    
    member d.Likelihood x = exp (d.LogLikelihood x)
    
    member dist.SampleAndGroup(n, ?groupWith) =
        dist.SampleN(n)
        |> SampleSummarize.roundAndGroupSamplesWith (defaultArg groupWith id)
        |> Array.sortByDescending snd
    
    member dist.DrawSamples(n, ?scorer, ?groupWith, ?sum) =
        let score = defaultArg scorer dist.Likelihood

        let aggr =
            match sum with
            | None
            | Some false -> SampleSummarize.compactMapSamplesAvg
            | Some true -> SampleSummarize.compactMapSamplesSum

        dist.SampleN(n)
        |> Array.map (fun x -> x, score x)
        |> aggr (defaultArg groupWith id) 
        |> Array.normalizeWeights
        |> Array.sortByDescending snd

    ///Sample and rank with a function, does not normalize
    member dist.SampleAndScore(n, ?scorer) =
        let score = defaultArg scorer dist.Likelihood

        dist.SampleN(n)
        |> Array.map (fun x -> x, score x)
        |> Array.sortByDescending snd

    member dist.SampleAndNormalizeScore(n, ?scorer) =
        dist.SampleAndScore(n, ?scorer = scorer)
        |> Array.normalizeWeights
         