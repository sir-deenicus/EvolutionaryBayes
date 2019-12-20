#r "netstandard"
#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\net47\prelude.dll"
#r @"bin\Release\net47\EvolutionaryBayes.dll"
#r @"C:\Users\cybernetic\Code\Libs\MathNet\lib\net40\MathNet.Numerics.dll"

#I @"C:\Users\cybernetic\Documents\Papers"  
#load @"disputil.fsx"
#load @"literate.fsx"  
#r @"LiterateUtils\LiterateUtils\bin\Release\net47\LiterateUtils.dll"  

#time "on"

open Parseutils
open XPlot.Plotly

open MathNet.Numerics.Distributions
open Prelude.Common
open System 
open Prelude.Math
open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions
open EvolutionaryBayes.Helpers 
open EvolutionaryBayes 

let rec flips n bs p =
    dist {
        if n = 0 then return bs
        else 
            let! b = bernoulliChoice 5. -5. p
            return! flips (n-1) (b::bs) p
    }

let bin = bernoulli 0.5 

let mix (l1: _ list) (l2: _ list) =
    let order = [|0..l1.Length-1|] 
    order.permuteYates() 
    [for i in order -> 
        if bin.Sample() then l1.[i] else l2.[i]] 

let populationFit p =
    let d = flips 16 [] p
    let avg =
        d.SampleN 1000
        |> Array.averageBy (List.sum >> ((+) 100.))
    1./(0.0001 + abs(avg - 80.))

let p = flips 16 [] 0.5

let p1 = p.Sample()
let p2 = p.Sample()

100. + List.sum p2

let a () = mix p1 p2 |> List.sum

[|for i in 1..1000 -> 100. + a()|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

[|for i in 1..1000 -> 100. + a()|]
|> Sampling.roundAndGroupSamplesWith id
|> SampleSummarize.computeDistAverage id

p.SampleN 10000
|> Array.averageBy (List.sum >> ((+) 100.)) 


let m1 =
    ParticleFilters.PopulationSampler
        (generator = always 0.5,
         mutate = (fun p -> p + random.NextDouble(-0.02, 0.02)),
         scorer = populationFit)

let res = m1.EvolveSequence(generations = 100, samplespergen = 50, forgetPrior = true )

let res2 = m1.RecursiveImportanceSample (generations = 100, samplespergen = 50)
let res3 = m1.SampleChain(1000)

res3 |> m1.SampleFrom 1000
|> Sampling.compactMapSamplesAvg (round 2)
//|> Chart.Column
|> Array.sumBy (fun (x,p) -> x * p)

res.SampleN(100)
|> Array.map (fun x -> x, populationFit x)
|> Array.normalizeWeights
|> Sampling.compactMapSamplesAvg id
 
p.SampleN 10000
|> Array.map (List.sum >> ((+) 100.))
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

let newgen l =
    let p =
        l |> List.sum
          |> scaleTo 0. 1. -80. 80.
    (flips 16 [] p).Sample()

mix p1 p2 |> newgen |> List.sum |> (+) 100.

[|for i in 1..1000 -> mix p1 p2  |> newgen |> List.sum |> (+) 100.|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

mix p1 p2  |> List.sum |> scaleTo 0. 1. -80. 80. 

let m =
    ParticleFilters.PopulationSampler
        (generator = flips 16 [] 0.5, 
         popMutate = (fun p m -> newgen (mix (p.Sample()) m)),
         scorer =
             (List.sum
              >> ((+) 100.)
              >> logisticRange 50. 100.))

let resx = m.EvolveSequence(generations = 200, samplespergen = 500, maxpoplen = 50)

resx.SampleN(10000) |> Array.map (List.sum >> ((+) 100.))
|> Sampling.roundAndGroupSamplesWith id
|> SampleSummarize.computeDistAverage id

let zzx pr g =
    let m =
        ParticleFilters.PopulationSampler
            (generator = flips 16 [] pr, 
             popMutate = (fun p m -> newgen (mix (p.Sample()) m)),
             scorer =
                 (List.sum
                  >> ((+) 100.)
                  >> logisticRange 70. 130.))

    let resx = 
        m.EvolveSequence(generations = g, samplespergen = 500, maxpoplen = 50)

    resx.SampleN(10000)
    |> Array.map (List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id
    |> SampleSummarize.computeDistAverage id


let dd = [for i in 0..10..300 -> zzx 0.37 i]
let dd2 = [for i in 0..10..300 -> zzx 0.5 i]

[dd |> List.indexed;dd2|> List.indexed] |> Chart.Line |> drawPlot

resx.SampleN(10000) |> Array.map (List.sum >> ((+) 100.))
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot
 