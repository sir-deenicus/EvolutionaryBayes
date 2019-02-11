module EvolutionaryBayes.Distributions

open ProbMonad
open MathNet.Numerics.Distributions
open Prelude.Math
open System
open MathNet.Numerics.LinearAlgebra.Double

let normal m s =
    let n = Normal(m, s)
    { new Distribution<float> with
          member d.Sample() = n.Sample() }

let lognormal m s =
    let n = LogNormal.WithMeanVariance(m, s ** 2.)
    { new Distribution<float> with
          member d.Sample() = n.Sample() }

let beta a b =
    let b = Beta(a, b)
    { new Distribution<float> with
          member d.Sample() = b.Sample() }

let gamma shape rate =
    let g = Gamma(shape, rate)
    { new Distribution<float> with
          member d.Sample() = g.Sample() }

let bernoulli p =
    let b = Bernoulli(p)
    { new Distribution<_> with
          member d.Sample() = b.Sample() = 1 }

let bernoulliChoice choice1 choice2 p =
    dist {
        let! b = bernoulli p
        return (if b then choice1
                else choice2)
    }

let categorical (items : _ []) (pmf : float []) =
    let c = Categorical(pmf)
    { new Distribution<_> with
          member d.Sample() = items.[c.Sample()] }

let categorical2 (items : _ []) =
    let items, pmf = Array.unzip items
    let c = Categorical(pmf)
    { new Distribution<_> with
          member d.Sample() = items.[c.Sample()] }

let dirichlet alpha =
    let dir = Dirichlet(alpha)
    { new Distribution<_> with
          member d.Sample() = dir.Sample() }

let discreteUniform (lower, upper) =
    let du = DiscreteUniform(lower, upper)
    { new Distribution<_> with
          member d.Sample() = du.Sample() }

let continuousUniform (lower, upper) =
    let cu = ContinuousUniform(lower, upper)
    { new Distribution<_> with
          member d.Sample() = cu.Sample() }

let uniform (items : _ []) =
    { new Distribution<_> with
          member d.Sample() = Array.sampleOne items }

let wishart degreesOfFreedom scale = 
    let w = Wishart(degreesOfFreedom, scale)
    { new Distribution<_> with
          member d.Sample() = w.Sample() }

let multiVariateNormal cv (meanVector : float []) =
    let m =
        MatrixNormal
            (DenseMatrix.OfRowArrays(meanVector),
             DenseMatrix.OfRowArrays([| [| 1.0 |] |]), cv)
    { new Distribution<float []> with
          member d.Sample() = m.Sample().Row(0).ToArray() }

let studentT dof loc scale =
    let s = StudentT(loc, scale, dof)
    { new Distribution<float> with
          member d.Sample() = s.Sample() }

let exponential rate =
    let e = Exponential(rate)
    { new Distribution<float> with
          member d.Sample() = e.Sample() }

let poisson lambda =
    let p = Poisson(lambda)
    { new Distribution<int> with
          member d.Sample() = p.Sample() }

let cauchy loc scale =
    let c = Cauchy(loc , scale)
    { new Distribution<float> with
          member d.Sample() = c.Sample() }
