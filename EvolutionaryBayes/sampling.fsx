#r @"C:\Users\cybernetic\Code\Libs\net4+\Prelude\Prelude.dll"
#r @"bin\Debug\netcoreapp2.1\EvolutionaryBayes.dll"
#r @"C:\Users\cybernetic\Code\Libs\MathNet\lib\net40\MathNet.Numerics.dll"
#time

open Prelude.Common
open System
open MathNet.Numerics.Distributions
open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions
open Helpers

let n = dist { return! normal 0. 1. } 
let data = [for _ in 0..999 -> Normal(10., 1.).Sample()]
 
let lik = observe (fun x y -> Normal(x, 10.).Density y) data //[ 5.; 10.; 4. ] 

EvolutionaryBayes.MetropolisHastings.sample lik (fun x -> 
    if (bernoulli 0.5).Sample() then x + 0.1
    else x - 0.1) n 100000
|> Sampling.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd

