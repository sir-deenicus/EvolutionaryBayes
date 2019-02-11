#r @"C:\Users\cybernetic\Code\Libs\net4+\Prelude\Prelude.dll"
#r @"bin\Debug\netcoreapp2.1\EvolutionaryBayes.dll"
#r @"C:\Users\cybernetic\Code\Libs\MathNet\lib\net40\MathNet.Numerics.dll"
#time

open MathNet.Numerics.Distributions

(* Feb 10, 2019
Currently, the Metropolis Hastings sampler is not bayesian in the sense of pulling towards some prior. 
Only the likelihood is looked at when perturbations are in use. This document contains a simple idea to
allow priors: provide the sample and its joint probability and write the prior a second time for the 
perturbations to use. This is then added to the log probability of the likelihood. If it is not easy
to get either a prior or likelihood (which is the main expected use case of this library), and especially 
when there is lots of data then simply returning a log 1. for the prior joint is fine. 
Indeed this is only here for correcterness. Note that this is essentially a non-issue 
for the sequential samplers.
*) 
type Distribution<'T> =
    abstract Sample : unit -> 'T * float

let bind (dist : Distribution<'T>) (k : 'T -> Distribution<'U>) =
    { new Distribution<'U> with
          member d.Sample() =
              let s, p = dist.Sample()
              let s', p' = (k s).Sample()
              s', p * p' }

/////////////////////////////////////
let always x =
    { new Distribution<'T> with
          member d.Sample() = x, 1. }

let bernoulli p =
    let b = Bernoulli(p)
    { new Distribution<_> with
          member d.Sample() =
              let t = b.Sample()
              t = 1, b.Probability t }

let normal m s =
    let n = Normal(m, s)
    { new Distribution<float> with
          member d.Sample() =
              let s = n.Sample()
              s, n.Density s }

type DistributionBuilder() =
    member d.Delay f = bind (always()) f
    member d.Bind(dist, f) = bind dist f
    member d.Return v = always v
    member d.ReturnFrom vs = vs

let dist = DistributionBuilder()

let dn =
    dist { let! m = normal 0. 1.
           let! x = normal m 1.
           return m, x }
let bn =
    dist { let! b = bernoulli 0.35
           let! b2 = bernoulli 0.5
           let! c = bernoulli 0.5
           return (b, b2, c) }

let priorb (a,b,c) =
    [ Bernoulli(0.35).Probability a
      Bernoulli(0.5).Probability b
      Bernoulli(0.5).Probability c]
    |> List.fold (*) 1.

let prior (m, x) =
    [ Normal(0., 1.).Density m
      Normal(m, 1.).Density x ]
    |> List.fold (*) 1.

0.5 ** 3.
bn.Sample()

prior (-0.9172276288, -1.290191905)
dn.Sample()