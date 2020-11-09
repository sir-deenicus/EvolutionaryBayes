﻿module EvolutionaryBayes.Helpers
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
    
    let compactMapSamplesAgg aggregator f samples =
        Array.map (fun (x, p : float) -> f x, p) samples
        |> Array.groupBy fst
        |> Array.map (fun (x, xs) -> x, aggregator snd xs)
        |> Array.normalizeWeights
      
    let compactMapSamplesSum f samples =
        compactMapSamplesAgg Array.sumBy f samples

    let compactMapSamplesAvg f samples =
        compactMapSamplesAgg Array.averageBy f samples

    let inline cdf (prob : _ []) =
        let cd = Array.create prob.Length prob.[0] 
        for i in 1..prob.Length - 1 do
            let j = (prob.Length - i - 1) 
            let (x, p) = prob.[i]
            cd.[j] <- x, snd cd.[j+1] + p
        cd
      
    let getDiscreteSample (pcdf : ('a * float) []) =
        let k, pcdlen = 
            random.NextDouble() * (snd pcdf.[0]), pcdf.Length - 1
        
        let rec cummProb idx =
            if k > snd pcdf.[idx] then cummProb (idx - 1)
            else idx, pcdf.[idx] 
        let i, (item,_) = cummProb pcdlen 
        i, item
    
    let discreteSampleIndex p = cdf p |> getDiscreteSample

    let inline discreteSample p = discreteSampleIndex p |> snd

    let discreteSampleN n items = [|for _ in 1..n -> discreteSample items|]

    let discreteSampleN_b n items = [|for _ in 1..n -> discreteSampleIndex items|]

    let sampleN_No_Replacements n items =
        let rec sampleN_without_replacement i ch =
            function
            | _ when i >= n -> ch
            | [] -> ch
            | choices ->
                let choice = discreteSample (List.toArray choices)
                let rem = List.filter (fst >> (<>) choice) choices
                sampleN_without_replacement (i + 1) (choice::ch) rem
        sampleN_without_replacement 0 [] items
  
    let inline normalizeWeights data =
        let sum = Array.sumBy snd data |> float
        [|for (x,p) in data -> x, float p / sum|]

module SampleSummarize =
    let smoothDistr alpha data =
        data
        |> Array.fold (fun (p_prior, ps) (x, p) ->
               let p' = Stats.exponentialSmoothing id alpha p_prior p
               p', (x, p') :: ps) (snd data.[0], [])
        |> snd
        |> List.toArray
        |> Array.normalizeWeights

    let inline computeDistAverage transform d = Array.sumBy (fun (x, p) -> transform x * p) d

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
