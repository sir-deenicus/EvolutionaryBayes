module EvolutionaryBayes.CFR

open Prelude.Common
open Prelude.Math
open System

type StrategyNode =
    { regretSum: double []
      strategySum: double [] }

let newStrategyNode n =
    { regretSum = Array.create n 0.
      strategySum = Array.create n 0. }


let getStrategy weight (strategySum: _ []) regretSum =
    let strategy = Array.map (max 0.) regretSum
    let nSum, numactions = Array.sum strategy, strategy.Length

    let strategy' =
        if nSum <= 0. then
            Array.create numactions (1. / float numactions)
        else
            Array.map (fun a -> a / nSum) strategy

    for a in 0 .. numactions - 1 do
        strategySum.[a] <- strategySum.[a] + weight * strategy'.[a]

    strategy'
 
let getAvgStrategyForced (n:StrategyNode) =
    let nSum = Array.sum n.strategySum

    if nSum <= 0. || Hashset(n.strategySum).Count = 1 then
        let r' = Array.map (max 0.) n.regretSum 
        if Array.sum r'  > 0. then true, Array.normalize r'
        else
            let numactions = n.strategySum.Length 
            true, Array.create numactions (1. / float numactions) 
    else false, Array.map (fun a -> a / nSum) n.strategySum 

let getAvgStrategy strategySum =
    let nSum, numactions = Array.sum strategySum, strategySum.Length

    if nSum <= 0. then
        Array.create numactions (1. / float numactions)
    else
        Array.map (fun a -> a / nSum) strategySum

let basicLookUp nummoves (nodeMap:Dict<_,_>) (infoset:string) =
    match nodeMap.tryFind infoset with
    | None ->
        let n = newStrategyNode nummoves
        nodeMap.Add(infoset, n)
        n
    | Some n -> n

let rec cfr d p0 p1 reward lookup adjustState getActionsMask stringifyContext (nodeMap: Dict<_, _>) (contexts: 'a []) (actions:_[])  (history: string) =
    let player = d % 2
    
    let actionsMask : _ [] = getActionsMask d contexts history
    let nummoves = actions.Length

    match reward contexts player history with
    | Some r -> r
    | None ->

        let infoset = stringifyContext contexts[player] + history

        let node = lookup nummoves nodeMap infoset 

        let strategy =
            getStrategy (if player = 0 then p0 else p1) node.strategySum node.regretSum

        let util = Array.create nummoves 0.
        let mutable nodeUtil = 0.
        for a in 0 .. nummoves - 1 do
            if actionsMask[a] then
                let h' = adjustState contexts player actions.[a] history

                util.[a] <-
                    if player = 0 then
                        -cfr (d + 1) (p0 * strategy.[a]) p1 reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'
                    else
                        -cfr (d + 1) p0 (p1 * strategy.[a]) reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'

                nodeUtil <- nodeUtil + strategy.[a] * util.[a]

        for a in 0 .. nummoves - 1 do
            if actionsMask[a] then
                let regret = util.[a] - nodeUtil

                node.regretSum.[a] <-
                    node.regretSum.[a]
                    + (if player = 0 then p1 else p0) * regret
            else node.regretSum[a] <- Double.MinValue

        nodeUtil 
