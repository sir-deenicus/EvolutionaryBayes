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

Nov 27, 2019

Revisiting this, my thoughts on how to implement support for priors were not fully correct. 
Especially, the before mentioned concept of tracking joints does not make sense.
Ultimately, constructing a prior that can also compute likelihoods with computation expressions is
probably not the best way to go, likely requiring the use of symbols and adjusting the distributions
to accept them. But that's not all since getting something that can be both sampled from and computes likelihoods
seems very tricky--there seems no obvious way to apply the needed arbitrary projection to the intermediate tensor products.

It is much easier instead to implement a way for base distributions to compute likelihoods and then
manually compute the prior and the sum for the likelihood. With included helpers it should not take much
more work than doing so with the aid of a computation expression.

Computation expressions are a convenient way to sequence expressions & for probablistic
programming, seem to work best when also incrementally building trees or lists. 

I also have hacks for interpreting lists that should work for restricted cases (when all types
are the same and either no input or a simple threading of inputs).

The prior is not worth too much focus as perubation and likelihood computations 
are more important anyways wrt intended usage.

Priors are more relevant for metropolis hastings--not importance. And 
they are now naturally incorporated now by adding the prior's density for the parameters to the
likelihoods.
*) 
 
type Distribution<'T> =
    abstract Sample : unit -> 'T * float  
    abstract LogLikelihood : 'T -> float
     
////// 
module Ext =
    type Distribution<'T> with 
        member d.SampleN n = [for i in 1..n -> d.Sample()]
        member d.Likelihood x = exp(d.LogLikelihood x)

open Ext

let bind (dist : Distribution<'T>) (k : 'T -> Distribution<'U>) =
    { new Distribution<'U> with
          member d.Sample() =
              let s, p = dist.Sample()
              let s', p' = (k s).Sample()
              s', p * p'
          member __.LogLikelihood x = 
            failwith "cannot compute likelihood with compound distrs in a meaningful manner" }
           
 
/////////////////////////////////////
let boolToInt = function true -> 1 | false -> 0

let always x =
    { new Distribution<'T> with
          member __.Sample() = x, 1.
          member __.LogLikelihood y = if x = y then 0. else -infinity}
                     
let bernoulli p =
    let b = Bernoulli(p)
    { new Distribution<_> with
          member d.Sample() =
              let t = b.Sample()
              t = 1, b.Probability t
          member d.LogLikelihood x = b.ProbabilityLn (boolToInt x)}

let normal m s =
    let n = Normal(m, s)
    { new Distribution<float> with
          member d.Sample() =
              let s = n.Sample()
              s, n.Density s
          member __.LogLikelihood x = n.DensityLn x}
            
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

let toLikelihood2 distrs parameters = 
    List.map2 (fun (d:Distribution<_>) p -> d.LogLikelihood p) distrs parameters
    |> List.sum 

let toLikelihood distrs = 
    List.sumBy (fun (x, d:Distribution<_>) -> d.LogLikelihood x) distrs 
     
let flattenDistributions distrs =
    { new Distribution<_> with
          member __.Sample() =
              distrs
              |> List.map (fun (p : Distribution<_>) -> p.Sample() |> fst), 1.
          member __.LogLikelihood ps = toLikelihood2 distrs ps }
           
type ProbState<'a> = Val of Distribution<'a> | Thunk of ('a -> Distribution<'a>)

let flattenDistributions2 distrs =
    { new Distribution<_> with 
          member __.Sample() =
              let sample, _ =
                  distrs
                  |> List.fold (fun (l, previous : _ option) (probstate : ProbState<'a>) ->
                        let next =
                            match probstate with
                            | Val p -> p.Sample() |> fst
                            | Thunk f -> (f previous.Value).Sample() |> fst
                        next :: l, Some next) ([], None)
              sample, 1.

          member __.LogLikelihood parameters =
              let rec loop likelihood previous probstates points =
                  match probstates, points with
                  | [], [] -> likelihood
                  | probstate :: probstates', point :: points' ->
                      match probstate with
                      | Val p ->
                          loop (likelihood + p.LogLikelihood point) point probstates' points'
                      | Thunk f ->
                          loop
                              (likelihood + (f previous).LogLikelihood point)
                              point probstates' points'
                  | _ -> failwith "sampling error"
              loop 0. parameters.[0] distrs parameters }

let prior0(m,x) = 
    toLikelihood
        [ m, normal 0. 1.
          x, normal m 1.]  

let prior0b =
    flattenDistributions2
        [ Val (normal 0. 1.)
          Thunk(fun m -> normal m 1.)] 

let (!) (d:Distribution<_>) = d.Sample() |> fst

let distBuilder sampler loglikelihood =
    { new Distribution<_> with 
        member __.Sample() = sampler()
        member __.LogLikelihood x = loglikelihood x}

let prior0c = 
    distBuilder
        (fun () ->
            let m = !(normal 0. 1.) 
            (m, !(normal m 1.)),1.)
        (fun ((m,x)) ->
            (normal 0. 1.).LogLikelihood m + (normal m 1.).LogLikelihood x )
     
let priorb2 ps = 
    toLikelihood2 [ bernoulli 0.35; bernoulli 0.5; bernoulli 0.5] ps

let b3 = flattenDistributions [ bernoulli 0.35; bernoulli 0.5; bernoulli 0.5]

b3.SampleN(30) |> List.map fst 

b3.Likelihood [false; false;true]

priorb2 [false; false;true]
priorb (0,0,1)
 
bn.SampleN 99

prior (-0.9172276288, -1.290191905)

prior0 (-0.9172276288, -1.290191905)
 

prior0b.Likelihood [-0.9172276288; -1.290191905]  

prior0b.Sample()
prior0c.Likelihood (-0.9172276288, -1.290191905) 
prior0c.Sample()

dn.Sample() 