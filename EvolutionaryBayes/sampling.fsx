#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\netstandard2.1\Prelude.dll"
#r @"bin\Debug\netstandard2.1\EvolutionaryBayes.dll"
#r @"C:\Users\cybernetic\Code\Libs\MathNet\lib\net40\MathNet.Numerics.dll"
#time
 
open Prelude.Common
open System
open MathNet.Numerics.Distributions
open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions
open Prelude.Math
open EvolutionaryBayes
open Prelude.ProbabilityTools
open Prelude
open EvolutionaryBayes.Extras
 
   
let rec randomWalk n s =
    dist {
        if n = 0 then return s 
        else 
            let! dir = bernoulliChoice 1. -1. 0.5
            return! randomWalk (n-1) (s+dir)
    }

let rec randomwalk2 n s =
    dist {
        if n = 0 then return (s**2.)
        else 
            let! dir = normal 0. 5.
            return! randomwalk2 (n-1) (s+dir)
    }

(randomwalk2 1000 0.).SampleN(500) 
|> Array.average
|> sqrt

//The perturbation step can be a sample whose parameters are the current state. 

let data = [for _ in 0..999 -> Normal(10., 10.).Sample()]
 
let lik = observe ((normal 0. 1.).LogLikelihood) (fun parameters x -> Normal(parameters, 1.).DensityLn x) [ 5.; 10.; 4. ]// [ 10.; ]//data // [ 5.; 10.; 4. ]// 
  
EvolutionaryBayes.MetropolisHastings.sample 1. 1. lik (fun x -> 
    if (bernoulli 0.5).Sample() then x + 0.1
    else x - 0.1) 100_000 (0.)
|> SampleSummarize.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd  

let flik = observePriorLess (fun parameters x -> Normal(parameters, 1.).DensityLn x) [ 5.; 10.; 4. ]

EvolutionaryBayes.MetropolisHastings.sampleBasic lik (normal 0. 1.) 100_000
|> SampleSummarize.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd 

EvolutionaryBayes.MetropolisHastings.sample 1. 1. flik (fun x -> x + Array.sampleOne [|-0.1;0.1|]) 100_000 0.
|> SampleSummarize.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd 

EvolutionaryBayes.MetropolisHastings.sample 1. 1. lik (fun x -> x + Array.sampleOne [|-0.1;0.1|]) 100_000 0.
|> SampleSummarize.roundAndGroupSamplesWith (round 1)
|> Array.sortByDescending snd 


EvolutionaryBayes.MetropolisHastings.sample 1. 1. lik (fun x -> Normal(x,1.).Sample()) 100_000 0.
|> SampleSummarize.roundAndGroupSamplesWith (round 1)
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
|> SampleSummarize.roundAndGroupSamplesWith id
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

let prior (m,b) = toLogLikelihood [m,normal 0. 10.; b, normal 0. 10.] 


prior (3.,5.)



let prior2 = Distribution.flatten [normal 0. 10.; normal 0. 10.]

let prior3 = Distribution.zip (normal 0. 10.) (normal 0. 10.)
 
let prior4 =
    distzip {
        let! _ = normal 0. 10.
        and! _ = normal 0. 10.
        return ()
    } 

prior2.Sample()
prior4.Sample()

prior (3.,5.) ,  prior2.LogLikelihood [|3.;5.|], prior3.LogLikelihood (3., 5.), prior4.LogLikelihood (3., 5.)

let likb = observe prior (fun (m, b) (x, y) -> Normal(m * x + b, 1.).DensityLn y) points // [ 5.; 10.; 4. ]//

let rs =
    MetropolisHastings.sample 0.9 100. likb (fun (m, b) ->
        let ps = [| m; b |]
        let i = random.Next(0, ps.Length)
        ps.[i] <- ps.[i] + random.NextDouble(-0.1, 0.1)
        ps.[0], ps.[1]) 100_000 (10.,0.)

        
rs
|> SampleSummarize.roundAndGroupSamplesWith (fun (m, b) -> round 1 (m * 218.4 + b))
|> Array.sortByDescending snd
 

MetropolisHastings.sample 0.9 100. likb (fun (m, b) ->
    if random.NextDouble() < 0.5 then
        Normal(m, 1.).Sample(), b
    else m, Normal(b, 1.).Sample()) 100_000 (10.,0.)


MetropolisHastings.sample 0.9 100. likb (fun (m, b) ->
    let m' = Normal(m, 1.).Sample()
    let b' = Normal(b, 1.).Sample()
    m', b') 100_000 (10.,0.)

let m = ParticleFilters.PopulationSampler(normal 80. 15., (logisticRange 115. 130.), (fun p -> (normal p 15.).Sample()))

let r = m.EvolveSequence(generations = 10)


r.SampleAndGroup(1000,round 2)


