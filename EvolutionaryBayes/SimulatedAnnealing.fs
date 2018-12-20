module EvolutionaryBayes.SimulatedAnnealing
open Prelude.Math

let optimizeVector (ranges:_[]) transform loss n T (atten: float) (v: _ []) =
    let rec loop n T best ea (v: _ []) =
        if n = 0 then best
        else 
            let i = random.Next(0,v.Length)
            let v' = Array.copy v
            v'.[i] <- transform (i, v'.[i] + random.NextDouble(-ranges.[i],ranges.[i]))
            let eb = loss v'
            if eb < ea then 
                let best' =
                    if eb < fst best then (eb,v')
                    else best
                loop (n - 1) (T * atten) best' eb v'
            else 
                let e',v'' =
                    if random.NextDouble() < exp((eb - ea) / T) then eb,v'
                    else ea,v
                loop (n - 1) (T * atten) best e' v''
    
    let ea = loss v
    loop n T (ea,v) ea v

type SimulatedAnnealer(ranges:_[],loss, ?temperature, ?attenuate, ?transform) = 
     let T = defaultArg temperature 1000.
     let atten = defaultArg attenuate 0.95
     let f = defaultArg transform snd
     new(range,loss,len, ?temperature, ?attenuate, ?transform) =
        let ranges = [|for _ in 1..len -> range|]
        SimulatedAnnealer(ranges, loss,defaultArg temperature 1000.,defaultArg attenuate 0.95,defaultArg transform snd)

     member __.OptimizeVector n v = optimizeVector ranges f loss n T atten v
     

