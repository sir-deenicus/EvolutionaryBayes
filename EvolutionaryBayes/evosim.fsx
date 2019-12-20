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
open EvolutionaryBayes

let rec flips n bs p =
    dist {
        if n = 0 then return bs
        else  
            let! b = bernoulliChoice 3. -3. p
            return! flips (n-1) (b::bs) p
    }

let bin = bernoulli 0.5 

let mix (l1: _ list) (l2: _ list) =
    let order = [|0..l1.Length-1|] 
    order.permuteYates() 
    [for i in order -> 
        if bin.Sample() then l1.[i] else l2.[i]] 

let populationFit p =
    let d = flips 27 [] p
    let avg =
        d.SampleN 1000
        |> Array.averageBy (List.sum >> ((+) 100.))
    1./(0.0001 + abs(avg - 80.))

let p = flips 27 [] 0.5

let p1 = p.Sample()
let p2 = p.Sample()

100. + List.sum p2
100. + List.sum p1

let a () = mix p1 p2 |> List.sum

[|for i in 1..1000 -> 100. + a()|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

[|for i in 1..1000 -> 100. + a()|]
|> Sampling.roundAndGroupSamplesWith id
|> SampleSummarize.computeDistAverage id

(116. + 92.)/2.

p.SampleN 10000
|> Array.averageBy (List.sum >> ((+) 100.)) 

[|for i in 1..1000 -> (p.Sample() |> List.sum) + 100.|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot
 

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
          |> scaleTo 0. 1. -81. 81.
    (flips 27 [] p).Sample()

mix p1 p2 |> newgen |> List.sum |> (+) 100.

[|for i in 1..1000 -> mix p1 p2  |> newgen |> List.sum |> (+) 100.|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

[|for i in 1..1000 -> mix p1 p2  |> newgen |> List.sum |> (+) 100.|]
|> Sampling.roundAndGroupSamplesWith id
|> SampleSummarize.computeDistAverage id


  
let m =
    ParticleFilters.PopulationSampler
        (generator = flips 27 [] 0.5, 
         popMutate = (fun p m -> newgen (mix (p.Sample()) m)),
         scorer =
             (List.sum
              >> ((+) 100.)
              >> logisticRange 65. 95.))

let resx = m.EvolveSequence(generations = 100, samplespergen = 2000, maxpoplen = 50)

let rate p =
    let q = List.sum p + 100.
    let acc = bernoulli (logisticRange 95. 120. q)
    let g = bernoulli 0.05 
    acc.Sample() && g.Sample()
 
let smarts, regs = resx.SampleN(1000) |> Array.partition rate

let todist = ParticleFilters.weightWith (List.sum >> ((+) 100.) >> logisticRange 65. 95.) >> categorical2 

let g1 =
    smarts |> Array.map (List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id

let g2 =
    regs |> Array.map (List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id

[g1;g2] |> Chart.Column |> drawPlot

g1 |> SampleSummarize.computeDistAverage id
g2 |> SampleSummarize.computeDistAverage id
 

let zzx pr g =
    let m =
        ParticleFilters.PopulationSampler
            (generator = pr, 
             popMutate = (fun p m -> newgen (mix (p.Sample()) m)),
             scorer =
                 (List.sum
                  >> ((+) 100.)
                  >> logisticRange 65. 95.))

    let resx = 
        m.EvolveSequence(generations = g, samplespergen = 500, maxpoplen = 50)

    resx.SampleN(10000)
    |> Array.map (List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id
    |> SampleSummarize.computeDistAverage id, resx

let smd = (todist smarts) 
let rd = (todist regs) 
let dd = [for i in 0..10..250 -> zzx smd i]
let dd2 = [for i in 0..10..250 -> zzx rd i]

List.map (List.map fst >> List.indexed) [dd;dd2] |> Chart.Line |> drawPlot

let d11 = (dd |> List.last |> snd).SampleN(10000) |> Array.map (List.sum >> ((+) 100.)) |> Sampling.roundAndGroupSamplesWith id

let d21 = (dd2 |> List.last |> snd).SampleN(10000) |> Array.map (List.sum >> ((+) 100.)) |> Sampling.roundAndGroupSamplesWith id
[d11;d21] |> Chart.Column |> drawPlot