module EvolutionaryBayes.ProbMonad

type Distribution<'T when 'T: equality> =
    abstract Sample : unit -> 'T
    abstract LogLikelihood : 'T -> float

module Distribution =
    let bind (dist : Distribution<'T>) (k : 'T -> Distribution<'U>) =
        { new Distribution<'U> with
            member __.Sample() =
                let dist' = k (dist.Sample())
                dist'.Sample()
            member __.LogLikelihood _ = 
                failwith "cannot compute likelihood with compound distributions in a meaningful manner" }
             
    let zip (distA: Distribution<'T>) (distB: Distribution<'U>) =
        { new Distribution<'T * 'U> with
            member __.Sample() = distA.Sample(), distB.Sample()

            member __.LogLikelihood((x, y)) =
                distA.LogLikelihood x + distB.LogLikelihood y }

    let zip3 (distA: Distribution<'T>) (distB: Distribution<'U>) 
        (distC: Distribution<'V>) =
        { new Distribution<'T * 'U * 'V> with
            member __.Sample() =
                distA.Sample(), distB.Sample(), distC.Sample()

            member __.LogLikelihood((x, y, z)) =
                distA.LogLikelihood x
                + distB.LogLikelihood y
                + distC.LogLikelihood z }

    let zip4 (distA: Distribution<'T>) (distB: Distribution<'U>) 
        (distC: Distribution<'V>) (distD: Distribution<'W>) =
        { new Distribution<'T * 'U * 'V * 'W> with
            member __.Sample() =
                distA.Sample(), distB.Sample(), distC.Sample(), distD.Sample()

            member __.LogLikelihood((x, y, z, w)) =
                distA.LogLikelihood x
                + distB.LogLikelihood y
                + distC.LogLikelihood z
                + distD.LogLikelihood w }

    let flatten (distrs: Distribution<'T> seq) =
        { new Distribution<'T []> with
            member __.Sample() = [| for d in distrs -> d.Sample() |]

            member __.LogLikelihood xs =
                (distrs, xs)
                ||> Seq.fold2 (fun ll d x -> ll + d.LogLikelihood x) 0. }
             
  
/////////////////////////////////////
open Distribution

let always x =
    { new Distribution<'T> with
        member __.Sample() = x 
        member __.LogLikelihood y = if x = y then 0. else -infinity}

type DistributionBuilder() =
    member d.Delay f = bind (always()) f
    member d.Bind(dist, f) = bind dist f
    member d.MergeSource (p,q) = zip p q
    member d.MergeSource3 (p,q,r) = zip3 p q r
    member d.MergeSource4 (p,q,r,s) = zip4 p q r s
    member d.Return v = always v
    member d.ReturnFrom vs = vs

let dist = DistributionBuilder()

let distBuilderLL sampler loglikelihood =
    { new Distribution<_> with 
        member __.Sample() = sampler()
        member __.LogLikelihood x = loglikelihood x} 

let distBuilder sampler =
    { new Distribution<_> with 
        member __.Sample() = sampler()
        member __.LogLikelihood _ = failwith "No likelihood function"}
           
     
let inline observe prior logpdf observations parameters =
    prior parameters
    + Seq.sumBy (logpdf parameters) observations

let inline logdiv a b = exp(a - b)

let inline logdivT T a b = exp((a - b) / T)

/////

///A hack to allow joining independent distributions of different types while maintaining a likelihood
type DistributionZipBuilder() = 
    member d.MergeSources (p,q) = zip p q 
    member d.BindReturn(m:Distribution<_>, _) = m

///A hack to allow joining independent distributions of different types while maintaining a likelihood
let distzip = DistributionZipBuilder()
