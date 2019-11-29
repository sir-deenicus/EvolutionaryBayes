module EvolutionaryBayes.Distributions

open ProbMonad
open MathNet.Numerics.Distributions
open Prelude.Math
open System
open MathNet.Numerics.LinearAlgebra.Double
open Prelude.Common

let boolToInt = function true -> 1 | false -> 0 
 
let normal m s =
    let n = Normal(m, s)
    { new Distribution<float> with
        member d.Sample() = n.Sample()
        member __.LogLikelihood x = n.Density x}

let lognormal m s =
    let n = LogNormal.WithMeanVariance(m, s ** 2.)
    { new Distribution<float> with
        member d.Sample() = n.Sample()
        member __.LogLikelihood x = n.Density x}

let beta a b =
    let b = Beta(a, b)
    { new Distribution<float> with
        member d.Sample() = b.Sample()
        member __.LogLikelihood x = b.Density x}

let gamma shape rate =
    let g = Gamma(shape, rate)
    { new Distribution<float> with
        member d.Sample() = g.Sample()
        member __.LogLikelihood x = g.Density x}

let bernoulli p =
    let b = Bernoulli(p)
    { new Distribution<_> with
        member d.Sample() = b.Sample() = 1
        member d.LogLikelihood x = b.Probability (boolToInt x)}

let bernoulliChoice choice1 choice2 p =
    let b = Bernoulli(p)
    { new Distribution<_> with
        member d.Sample() = if b.Sample() = 1 then choice1 else choice2
        member d.LogLikelihood x = 
            let c = match x with
                    | v when v = choice1 -> 1 
                    | v when v = choice2 -> 0 
                    | _ -> failwith "unrecognized choice"
            b.Probability c} 

let categorical (items : _ []) (pmf : float []) =
    let c = Categorical(pmf)
    let dictindex = Dict.ofSeq [|for i in 0..items.Length - 1 -> items.[i], i|]
    { new Distribution<_> with
        member d.Sample() = items.[c.Sample()] 
        member __.LogLikelihood x = c.ProbabilityLn (dictindex.[x]) }

let categorical2 (ps : _ []) =
    let items, pmf = Array.unzip ps
    categorical items pmf

let dirichlet alpha =
    let dir = Dirichlet(alpha)
    { new Distribution<_> with
        member d.Sample() = dir.Sample()
        member d.LogLikelihood x = dir.DensityLn x }

let discreteUniform (lower, upper) =
    let du = DiscreteUniform(lower, upper)
    { new Distribution<_> with
        member d.Sample() = du.Sample()
        member d.LogLikelihood x = du.ProbabilityLn x}

let continuousUniform (lower, upper) =
    let cu = ContinuousUniform(lower, upper)
    { new Distribution<_> with
        member d.Sample() = cu.Sample()
        member d.LogLikelihood x = cu.DensityLn x}

let uniform (items : _ []) =
    let dictindex = Hashset items
    let p = log (1./(float (items.Length - 1)))
    { new Distribution<_> with
        member d.Sample() = Array.sampleOne items
        member d.LogLikelihood x = if dictindex.Contains x then p else -infinity}

let wishart degreesOfFreedom scale = 
    let w = Wishart(degreesOfFreedom, scale)
    { new Distribution<_> with
        member d.Sample() = w.Sample()
        member d.LogLikelihood x = log(w.Density x)}

let multiVariateNormal cv (meanVector : float []) =
    let m =
        MatrixNormal
            (DenseMatrix.OfRowArrays(meanVector),
             DenseMatrix.OfRowArrays([| [| 1.0 |] |]), cv)
    { new Distribution<float []> with
        member d.Sample() = m.Sample().Row(0).ToArray()
        member d.LogLikelihood x = log(m.Density (DenseMatrix.OfRowArrays x))}

let studentT dof loc scale =
    let s = StudentT(loc, scale, dof)
    { new Distribution<float> with
        member d.Sample() = s.Sample() 
        member d.LogLikelihood x = s.DensityLn x}

let exponential rate =
    let e = Exponential(rate)
    { new Distribution<float> with
        member d.Sample() = e.Sample()
        member d.LogLikelihood x = e.DensityLn x}

let poisson lambda =
    let p = Poisson(lambda)
    { new Distribution<int> with
        member d.Sample() = p.Sample()
        member d.LogLikelihood x = p.ProbabilityLn x}

let cauchy loc scale =
    let c = Cauchy(loc , scale)
    { new Distribution<float> with
        member d.Sample() = c.Sample()
        member d.LogLikelihood x = c.DensityLn x}
