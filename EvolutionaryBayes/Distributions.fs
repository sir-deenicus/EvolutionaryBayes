module EvolutionaryBayes.Distributions

open ProbMonad
open MathNet.Numerics.Distributions
open Prelude.Math
open System

let normal m s =
    let n = Normal(m,s)
    { new Distribution<float> with
          member d.Sample() = n.Sample() }

let bernoulli p =
    let b = Bernoulli(p)
    { new Distribution<_> with
          member d.Sample() = b.Sample() = 1 }

let categorical (items:_[]) (pmf:float[]) = 
    let c = Categorical(pmf)
    { new Distribution<_> with
          member d.Sample() = items.[c.Sample()]  }

let discreteUniform (lower,upper)  = 
    let du = DiscreteUniform(lower,upper)
    { new Distribution<_> with
          member d.Sample() = du.Sample() }

let continuousUniform (lower,upper)  = 
    let cu = ContinuousUniform(lower,upper) 
    { new Distribution<_> with
          member d.Sample() = cu.Sample()  }

let uniform (items:_[]) =  
    { new Distribution<_> with
          member d.Sample() =  Array.sampleOne items }
        