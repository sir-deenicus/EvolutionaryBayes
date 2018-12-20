#r @"bin\Debug\net45\Prelude.dll"
#r @"..\Hansei.Rational\bin\Release\net45\MathNet.Numerics.dll"

open Prelude.Common
open System
open MathNet.Numerics.Distributions

let n = dist { return! normal 0. 1. } 

let lik = observe (fun x y -> Normal(x, 1.).Density y) [ 5.; 10.; 4. ]

MHMC.MH lik (fun x -> 
    if (bernoulli 0.5).Sample() then x + 0.1
    else x - 0.1) n 1000
|> Sampling.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd

let z =
    SimpleMCMC.MH lik n 10000
    |> Sampling.roundAndGroupSamplesWith (round 1)
    |> Array.sortByDescending snd

Sampling.computeSamplesMCMC 10000 1000 lik n
|> Array.sortByDescending snd
|> Sampling.compactMapSamples (round 1)
[ for i in 0..999 -> n.Sample() |> round 1 ]
|> Seq.countBy id
|> Seq.toArray
