module EvolutionaryBayes.ProbMonad

type Distribution<'T> =
    abstract Sample : unit -> 'T
    

let bind (dist : Distribution<'T>) (k : 'T -> Distribution<'U>) =
    { new Distribution<'U> with
          member d.Sample() =
              let dist' = k (dist.Sample())
              dist'.Sample() }

/////////////////////////////////////
let always x =
    { new Distribution<'T> with
          member d.Sample() = x }

type DistributionBuilder() =
    member d.Delay f = bind (always()) f
    member d.Bind(dist, f) = bind dist f
    member d.Return v = always v
    member d.ReturnFrom vs = vs

let dist = DistributionBuilder()

let observe pdf l y = 
    List.fold (fun s x -> 
        let d = max System.Double.Epsilon (pdf x y)
        s + log d) 0. l   
 
let inline logdiv a b = exp(a - b)