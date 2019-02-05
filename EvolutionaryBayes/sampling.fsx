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
let data = [for _ in 0..999 -> Normal(10., 10.).Sample()]
 
let lik = observe (fun x y -> Normal(x, 1.).Density y) data //[ 5.; 10.; 4. ]// 

EvolutionaryBayes.MetropolisHastings.sample lik (fun x -> 
    if (bernoulli 0.5).Sample() then x + 0.1
    else x - 0.1) (normal 0. 1.) 100000
|> Sampling.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd 

let game2 a b c d =
    dist {
        let! p = beta a b
        let! q = beta c d
        let! p1res = bernoulliChoice "P1 win" "P2 win" p
        let! p2res = bernoulliChoice "P2 win" "P1 win" q
        if p1res = p2res then return p1res
        else 
            let tiebreak =
                if p > q then "P1 win"
                elif p = q then "can't say"
                else "P2 win"
            return tiebreak
    }

(game2 6. 10. 2. 10.).SampleN(1_000_000)
|> Sampling.roundAndGroupSamplesWith id
|> Array.sortByDescending snd 
