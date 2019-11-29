module Helpers
open Prelude.Math
open Prelude.Common
open System
open EvolutionaryBayes
open EvolutionaryBayes.ProbMonad
    
type Distribution<'T> with    
    member d.SampleN n =  [|for _ in 1..n -> d.Sample()|]  
    member d.Likelihood x = exp(d.LogLikelihood x)

module Ops =    
    let (!) (d:Distribution<_>) = d.Sample() |> fst

module Sampling =
    let roundAndGroupSamplesWith f samples =
        samples
        |> Seq.toArray
        |> Array.map f
        |> Array.groupBy id
        |> Array.map (keepLeft (Array.length >> float))
        |> Array.normalizeWeights
    
    let compactMapSamples f samples =
        Array.map (fun (x, p : float) -> f x, p) samples
        |> Array.groupBy fst
        |> Array.map (fun (x, xs) -> x, Array.sumBy snd xs)
        |> Array.normalizeWeights
    
    let aggregateSamples nIters sampler =
        let updaterate = max 1 (nIters / 10)
        let mutable nelements = 0
        [| for i in 0..nIters do
               let samples = sampler() 
               let els = Array.countElements samples
               nelements <- nelements + els.Count
               if i % updaterate = 0 then 
                   printfn "%d of %d" i nIters
                   printfn "%d elements" nelements
               yield (els) |]
        |> Array.fold (fun fm m -> Map.merge (+) id m fm) Map.empty
        |> Map.toArray


module SampleSummarize =
    let smoothDistr alpha data =
        data
        |> Array.fold (fun (p_prior, ps) (x, p) ->
               let p' = Stats.exponentialSmoothing id alpha p_prior p
               p', (x, p') :: ps) (snd data.[0], [])
        |> snd
        |> List.toArray
        |> Array.normalizeWeights

    let inline computeDistrAverage transform d = Array.sumBy (fun (x, p) -> transform x * p) d

    let computeCredible sortBy roundpsTo p1 p2 dat =
        let rec loop cumulativeProb ps =
            function
            | [] -> ps, cumulativeProb
            | _ when cumulativeProb > p2 -> ps, cumulativeProb
            | (x, p) :: dat ->
                let ps' =
                    if cumulativeProb + p < p1 then ps
                    else ((x, p, round roundpsTo (p + cumulativeProb)) :: ps)
                loop (cumulativeProb + p) ps' dat

        let cump, _ = loop 0. [] (List.ofSeq dat |> List.sortBy sortBy)
        cump, List.minBy third cump |> fst3, List.maxBy third cump |> fst3

    let getLargeProbItems maxp data =
        let rec innerloop curritems cumulativeprob =
            function
            | [] -> curritems
            | _ when cumulativeprob > maxp -> curritems
            | ((_, p) as item :: ps) ->
                innerloop (item :: curritems) (p + cumulativeprob) ps
        innerloop [] 0. data

    let findTopItem (vc : _ []) =
        let topindex, _ =
            vc
            |> Array.indexed
            |> Array.maxBy (snd >> snd)

        let (_, p) as topitem = vc.[topindex]
        topindex, p, [ topitem ]

    let getBulk (minp : float) items =
        let rec loopinner cmin cmax bulkMass sum =
            if sum > minp || (cmin < 0 && cmax >= Array.length items) then
                sum, bulkMass
            else
                let bulkMass' =
                    let frontpart =
                        if cmin < 0 then bulkMass
                        else items.[cmin] :: bulkMass
                    if cmax > items.Length - 1 then frontpart
                    else items.[cmax] :: frontpart

                let currentSum = List.sumBy snd bulkMass'
                loopinner (cmin - 1) (cmax + 1) bulkMass' currentSum

        let topindex, p, root = findTopItem items
        loopinner (topindex - 1) (topindex + 1) root p

    let getBulkAlternating (minp : float) toggle items =
        let rec loopinner toggle cmin cmax bulkMass sum =
            if sum > minp || (cmin < 0 && cmax >= Array.length items) then
                sum, bulkMass
            else
                let cmin', cmax', bulkMass' =
                    match toggle with
                    | true ->
                        cmin, cmax + 1, 
                            (if cmax > items.Length - 1 then bulkMass
                             else items.[cmax] :: bulkMass)
                    | false ->
                        cmin - 1, cmax, 
                           (if cmin >= 0 then items.[cmin] :: bulkMass
                            else bulkMass)

                let currentSum = List.sumBy snd bulkMass'
                loopinner (not toggle) cmin' cmax' bulkMass' currentSum

        let topindex, p, root = findTopItem items
        loopinner toggle (topindex - 1) (topindex + 1) root p

        
module ProbTools =
    let filterWith f data =
        let matches = data |> Array.filter f
        (Array.length matches |> float) / (float data.Length)

    let filterWithCondition conditional f data =
        let sub = Array.filter conditional data
        let matches = sub |> Array.filter f
        (Array.length matches |> float) / (float sub.Length)

    let inline probabilityOf filter m =
        Array.sumBy snd (Array.filter (fun (k, _) -> filter k) m)

    let inline conditionalProbability conditional matchwith m =
        let sub = Array.filter (fun (k, _) -> conditional k) m
        let matches = Array.filter (fun (k, _) -> matchwith k) sub
        (Array.sumBy snd matches) / (Array.sumBy snd sub)

    let toBits x = x / log 2.

    let inline log0 x =
        if x = 0. then 0.
        else log x

    let inline entropy dist = -Seq.sumBy (fun (_, p) -> p * log0 p) dist
    let inline betaMean (a, b) = a / (a + b)

    let inline dirichletMean (a : Map<_, _>) =
        let total = Map.sum a in Map.map (fun _ a_i -> a_i / total) a

    let updateBeta (a, b) t =
        if t then (a + 1., b)
        else (a, b + 1.)

    let updateDirichlet (m : Map<_, _>) x = Map.add x (m.[x] + 1.) m
