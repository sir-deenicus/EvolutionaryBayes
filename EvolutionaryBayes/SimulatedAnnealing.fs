module EvolutionaryBayes.SimulatedAnnealing
open Prelude.Math

let optimizeVector loss (ranges : _ []) T n (atten : float) (v : _ []) =
    let rec loop n T best prevLoss (v : _ []) =
        if n = 0 then best
        else
            let i = random.Next(0, v.Length) 

            let v' =
                Array.mapi (fun j x ->
                    if i = j then x + random.NextDouble(-ranges.[i], ranges.[i])
                    else x) v

            let currLoss = loss v'
            if currLoss < prevLoss then
                let best' =
                    if currLoss < fst best then (currLoss, v')
                    else best
                loop (n - 1) (T * atten) best' currLoss v'
            else
                let fLoss, v'' =
                    if random.NextDouble() < exp ((currLoss - prevLoss) / T) then currLoss, v'
                    else prevLoss, v
                loop (n - 1) (T * atten) best fLoss v''

    let eLoss = loss v
    loop n T (eLoss, v) eLoss v

type SimulatedAnnealer(ranges: _ [], loss, ?temperature, ?attenuate) =
    let T = defaultArg temperature 1000.
    let atten = defaultArg attenuate 0.95 

    new(range, loss, len, ?temperature, ?attenuate) =
        let ranges = [| for _ in 1 .. len -> range |]

        SimulatedAnnealer(
            ranges,
            loss,
            defaultArg temperature 1000.,
            defaultArg attenuate 0.95
        )

    member __.OptimizeVector n v =
        optimizeVector loss ranges T n atten v

