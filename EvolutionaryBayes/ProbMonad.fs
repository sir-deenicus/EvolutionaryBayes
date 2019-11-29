module EvolutionaryBayes.ProbMonad

type Distribution<'T> =
    abstract Sample : unit -> 'T
    abstract LogLikelihood : 'T -> float

let bind (dist : Distribution<'T>) (k : 'T -> Distribution<'U>) =
    { new Distribution<'U> with
        member __.Sample() =
            let dist' = k (dist.Sample())
            dist'.Sample()
        member __.LogLikelihood x = 
            failwith "cannot compute likelihood with compound distributions in a meaningful manner" }

/////////////////////////////////////
let always x =
    { new Distribution<'T> with
        member __.Sample() = x 
        member __.LogLikelihood y = if x = y then 0. else -infinity}

type DistributionBuilder() =
    member d.Delay f = bind (always()) f
    member d.Bind(dist, f) = bind dist f
    member d.Return v = always v
    member d.ReturnFrom vs = vs

let dist = DistributionBuilder()

let distBuilder sampler loglikelihood =
    { new Distribution<_> with 
        member __.Sample() = sampler()
        member __.LogLikelihood x = loglikelihood x}

let distBuilder2 sampler =
    { new Distribution<_> with 
        member __.Sample() = sampler()
        member __.LogLikelihood _ = failwith "No likelihood function"}

let observep prior pdf observations parameters =
    prior parameters
    + List.sumBy (fun x -> log (max System.Double.Epsilon (pdf parameters x)))
          observations

let inline observe prior pdf observations parameters =
    prior parameters
    + List.sumBy (pdf parameters) observations

let inline observePriorLess pdf observations parameters =
    List.sumBy (pdf parameters) observations

let inline logdiv a b = exp(a - b)

let inline logdivT T a b = exp((a - b) / T)
