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
open Prelude.Math
open EvolutionaryBayes

let data = [for _ in 0..999 -> Normal(10., 10.).Sample()]
 
let lik = observe (fun parameters x -> Normal(parameters, 1.).Density x) [ 10.; ]//data // [ 5.; 10.; 4. ]// 

EvolutionaryBayes.MetropolisHastings.sample 0.5 lik (fun x -> 
    if (bernoulli 0.5).Sample() then x + 0.1
    else x - 0.1) (normal 0. 1.) 100_000
|> Sampling.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd  

EvolutionaryBayes.MetropolisHastings.sample 0. lik id (normal 0. 1.) 100_000
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


let points =
    [ (218.4, 11.95833333)
      (222.8, 15.125)
      (222.8, 14.78194444)
      (218.4, 11.34305556)
      (222.8, 16.40277778)
      (222.8, 11.70138889)
      (222.8, 15.3125)
      (436.8, 23.52083333)
      (436.8, 21.72083333)
      (218.4, 12.3625) ]

let prior =
    dist { let! m = normal 10. 10.
           let! b = normal 0. 10.
           return (m, b) }

let lik2 = observe (fun (m, b) (x, y) -> Normal(m * x + b, 1.).Density y) points // [ 5.; 10.; 4. ]//

let rs =
    MetropolisHastings.sample 0.5 lik2 (fun (m, b) ->
        let ps = [| m; b |]
        let i = random.Next(0, ps.Length)
        ps.[i] <- ps.[i] + random.NextDouble(-0.1, 0.1)
        ps.[0], ps.[1]) prior 100_000

        
rs
|> Sampling.roundAndGroupSamplesWith (fun (m, b) -> round 1 (m * 218.4 + b))
|> Array.sortByDescending snd

