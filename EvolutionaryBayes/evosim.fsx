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
            let! b = bernoulliChoice 3. -3. p
            return! flips (n-1) (b::bs) p
    }

let rec flips2 n bs p =
    dist {
        if n = 0 then return bs, p
        else  
            let! b = bernoulliChoice 3. -3. p
            return! flips2 (n-1) (b::bs) p
    }

let bin = bernoulli 0.5 
let mutate = bernoulli 0.01 
let gene p = (bernoulliChoice 3. -3. p).Sample()

let mix (l1: _ list) (l2: _ list) =
    let order = [|0..l1.Length-1|] 
    order.permuteYates() 
    let a1, a2 = List.toArray l1, List.toArray l2
    [for i in order -> 
        if bin.Sample() then a1.[i] else a2.[i]] 
   
let newgen2 (l1: float list, p1) (l2: _ list, p2) =
    let order = [|0..l1.Length-1|] 
    order.permuteYates() 
    let a1, a2 = List.toArray l1, List.toArray l2 
    let m = if mutate.Sample() then -1. else 1.
    let genome =
        [for i in order -> 
            if bin.Sample() then 
                m * (if bin.Sample() then a1.[i] else gene p1)
            else 
                m * (if bin.Sample() then a2.[i] else gene p2)]
    let ps = List.sum genome
    genome,  0.5//scaleTo 0. 1. -81. 81. ps

let rec drift n avgs (pop:Distribution<_>) = 
        if n = 0 then pop, List.rev avgs
        else
            let p1 = pop.Sample()
            let p2 = pop.Sample()

            let pop' = [|for _ in 1..500 -> if random.NextDouble() < 0.8 then p1 else newgen2 p1 p2|]
            let t = Array.averageBy (fst >> List.sum >> (+) 100.) pop'
            let next =
                pop'    
                |> Sampling.roundAndGroupSamplesWith id
                |> categorical2
            drift (n-1) (t::avgs) next

        
let newgen l =
    let p =
        l |> List.sum
          |> scaleTo 0. 1. -81. 81.
    (flips 27 [] p).Sample()    
    
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
  
p.SampleN 10000
|> Array.averageBy (List.sum >> ((+) 100.)) 

[|for i in 1..1000 -> (p.Sample() |> List.sum) + 100.|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot 
  
[|for i in 1..1000 -> mix p1 p2  |> newgen |> List.sum |> (+) 100.|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

[|for i in 1..1000 -> mix p1 p2  |> newgen |> List.sum |> (+) 100.|]
|> Sampling.roundAndGroupSamplesWith id
|> SampleSummarize.computeDistAverage id 


let pb = flips2 27 [] 0.5

let p3 = pb.Sample()
let p4 = pb.Sample()

100. + List.sum (fst p3)
100. + List.sum (fst p4)
 
[|for i in 1..1000 -> newgen2 p3 p4 |> fst |> List.sum |> (+) 100.|]
|> Sampling.roundAndGroupSamplesWith id
//|> SampleSummarize.computeDistAverage id 
|> Chart.Column
|> drawPlot

[|for i in 1..1000 -> newgen2 p3 p4 |> snd|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

let df,al = drift 50 [] pb

al |> Chart.Line |> drawPlot

[|for i in 1..1000 -> df.Sample() |> snd|]
|> Sampling.roundAndGroupSamplesWith id
|> Chart.Column
|> drawPlot

[|for i in 1..1000 -> df.Sample() |> fst |> List.sum |> (+) 100.|]
|> Sampling.roundAndGroupSamplesWith id
//|> SampleSummarize.computeDistAverage id 
|> Chart.Column
|> drawPlot

let m =
    ParticleFilters.PopulationSampler
        (generator = flips 27 [] 0.5, 
         popMutate = (fun p m -> newgen (mix (p.Sample()) m)),
         scorer =
             (List.sum
              >> ((+) 100.)
              >> logisticRange 65. 95.))

let m2 =
    ParticleFilters.PopulationSampler
        (generator = flips2 27 [] 0.5, 
         popMutate = (fun p m -> newgen2 (p.Sample()) m),
         scorer =
             (fst
              >> List.sum
              >> ((+) 100.)
              >> logisticRange 65. 95.))

let resx = m2.EvolveSequence(generations = 100, samplespergen = 2000, maxpoplen = 50)

let rate p =
    let q = List.sum p + 100.
    let acc = bernoulli (logisticRange 105. 130. q)// (logisticRange 95. 120. q)
    let g = bernoulli 0.05 
    acc.Sample() && g.Sample()
 
let smarts, regs = resx.SampleN(10000) |> Array.partition (fst >> rate)

//let todist = ParticleFilters.weightWith (List.sum >> ((+) 100.) >> logisticRange 65. 95.) >> categorical2 
let todist = ParticleFilters.weightWith (List.sum >> ((+) 100.) >> logisticRange 25. 45.) >> categorical2 
let todist2 l = l |> ParticleFilters.weightWith (fst >> List.sum >> ((+) 100.) >> logisticRange 25. 45.) |> categorical2 

let g1 =
    smarts |> Array.map (fst >> List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id

let g2 =
    regs |> Array.map (fst >> List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id

[g1;g2] |> Chart.Column |> drawPlot

regs |> Array.map snd
|> Sampling.roundAndGroupSamplesWith id |> Chart.Column |> drawPlot

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
                  >> logisticRange 25. 45.))//>> logisticRange 65. 95.))

    let resx = 
        m.EvolveSequence(generations = g, samplespergen = 500, maxpoplen = 50)

    resx.SampleN(10000)
    |> Array.map (List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id
    |> SampleSummarize.computeDistAverage id, resx

let zzx2 pr g =
    let m =
        ParticleFilters.PopulationSampler
            (generator = pr, 
             popMutate = (fun p m -> newgen2 (p.Sample()) m),
             scorer =
                 (fst >> List.sum
                  >> ((+) 100.)
                  >> logisticRange 25. 45.))//>> logisticRange 65. 95.))

    let resx = 
        m.EvolveSequence(generations = g, samplespergen = 500, maxpoplen = 50)

    resx.SampleN(10000)
    |> Array.map (fst >> List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id
    |> SampleSummarize.computeDistAverage id, resx

        
let smd = (todist2 smarts) 
let rd = (todist2 regs) 
let dd = [for i in 0..10..150 -> zzx2 smd i]
let dd2 = [for i in 0..10..150 -> zzx2 rd i]

List.map (List.map fst >> List.indexed) [ dd; dd2 ]
|> Chart.Line
|> drawPlot

let d11 =
    (dd |> List.last |> snd).SampleN(10000)
    |> Array.map (fst >> List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id

let d21 =
    (dd2 |> List.last |> snd).SampleN(10000)
    |> Array.map (fst >> List.sum >> ((+) 100.))
    |> Sampling.roundAndGroupSamplesWith id

[ d11; d21 ]
|> Chart.Column
|> drawPlot
