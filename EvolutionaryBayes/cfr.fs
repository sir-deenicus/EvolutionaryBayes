module EvolutionaryBayes.CFR

open Prelude.Common
open Prelude.Math
open System
open Prelude
open System.Collections.Generic

type StrategyNode =
    { regretSum: double []
      strategySum: double [] }

let newStrategyNode n =
    { regretSum = Array.create n 0.
      strategySum = Array.create n 0. }

let getStrategyMasked (weight: double) (mask: bool[]) (strategySum: double[]) (regretSum: double[]) =
    // Convert regrets to non-negative
    let tmp = Array.init regretSum.Length (fun i -> if mask.[i] then max 0. regretSum.[i] else 0.)
    let s = Array.sum tmp
    let strategy =
        if s <= 0. then
            // Uniform over legal actions only
            let legalCount = mask |> Array.filter id |> Array.length
            if legalCount = 0 then failwith "getStrategyMasked: no legal actions"
            Array.init tmp.Length (fun i -> if mask.[i] then 1. / float legalCount else 0.)
        else
            Array.map (fun v -> if v = 0. then 0. else v / s) tmp
    // Accumulate average strategy weighted by opponent reach
    for i in 0 .. strategy.Length - 1 do
        if mask.[i] then
            strategySum.[i] <- strategySum.[i] + weight * strategy.[i]
    strategy

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

type ActionMask<'contexts> = int -> 'contexts [] -> string -> bool[]
  
let inline clipRegretsPlus (mask: bool[]) (regrets: double[]) =
    // CFR+: keep cumulative regrets non-negative for legal actions
    for i = 0 to regrets.Length - 1 do
        if mask.[i] && regrets.[i] < 0. then
            regrets.[i] <- 0.

let rec cfr d p0 p1 reward lookup adjustState (getActionsMask: ActionMask<'a>) (stringifyContext:_->string) (nodeMap: Dict<_, _>) (contexts: 'a []) (actions:_[])  (history: string) =
    let player = d % 2
    let actionsMask : _ [] = getActionsMask d contexts history
    let nummoves = actions.Length
    match reward contexts player history with
    | Some r -> r
    | None ->
        // Delimiter in infoset to avoid collisions
        let infoset = $"{stringifyContext contexts.[player]}|{history}"
        let node = lookup nummoves nodeMap infoset
        // Use opponent reach for averaging (π_{-i})
        let oppReach = if player = 0 then p1 else p0
        let strategy = getStrategyMasked oppReach actionsMask node.strategySum node.regretSum
        let util = Array.zeroCreate nummoves
        let mutable nodeUtil = 0.
        for a in 0 .. nummoves - 1 do
            if actionsMask.[a] then
                let h' = adjustState contexts player actions.[a] history
                let nextP0, nextP1 =
                    if player = 0 then (p0 * strategy.[a], p1)
                    else (p0, p1 * strategy.[a])
                util.[a] <- -cfr (d + 1) nextP0 nextP1 reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'
                nodeUtil <- nodeUtil + strategy.[a] * util.[a]
        // Regret update scaled by opponent reach
        for a in 0 .. nummoves - 1 do
            if actionsMask.[a] then
                let regret = util.[a] - nodeUtil
                node.regretSum.[a] <- node.regretSum.[a] + oppReach * regret
            // leave regretSum untouched for illegal actions (safer)
        // CFR+ clipping
        clipRegretsPlus actionsMask node.regretSum
        nodeUtil 
 
// helper: sample one legal action according to `strategy`, return (index, prob)
let sampleIndex (rnd: Random) (strategy: double[]) (mask: bool[]) =
    let legal = [| for i in 0 .. strategy.Length - 1 do if mask.[i] then yield i |]
    if legal.Length = 0 then failwith "sampleIndex: no legal actions"
    let raw = legal |> Array.map (fun i -> max 0. strategy.[i])
    let sum = Array.sum raw
    let probs =
        if sum <= 0. then
            Array.create legal.Length (1. / float legal.Length)
        else
            raw |> Array.map (fun w -> w / sum)

    let r = rnd.NextDouble()
    // Single pass with early exit
    let mutable k = 0
    let mutable cum = probs.[0]
    while k < probs.Length - 1 && r > cum do
        k <- k + 1
        cum <- cum + probs.[k]
    let chosenLocal = legal.[k]
    let chosenProb =
        if sum <= 0. then 1. / float legal.Length
        else probs.[k]            // already normalized over legal actions
    chosenLocal, chosenProb

// External-sampling MCCFR: full expand for `targetPlayer`, sample otherwise.
let rec cfrSampled
    (rnd: Random)
    (targetPlayer: int)
    (d: int)
    (p0: double)
    (p1: double)
    (reward: 'a[] -> int -> string -> double option)
    (lookup: int -> Dict<_,_> -> string -> StrategyNode)
    (adjustState: 'a[] -> int -> 'b -> string -> string)
    (getActionsMask: ActionMask<'a>)
    (stringifyContext: _ -> string)
    (nodeMap: Dict<_, _>)
    (contexts: 'a[])
    (actions: 'b[])
    (history: string)
    : double =

    let player = d % 2
    match reward contexts player history with
    | Some r -> r
    | None ->
        let infoset = $"{stringifyContext contexts.[player]}|{history}"
        let actionsMask = getActionsMask d contexts history
        let node = lookup actions.Length nodeMap infoset
        let oppReach = if player = 0 then p1 else p0
        // Mask-aware strategy
        let strategy = getStrategyMasked oppReach actionsMask node.strategySum node.regretSum
        if player = targetPlayer then
            // Full expansion
            let util = Array.zeroCreate actions.Length
            let mutable nodeUtil = 0.
            for a in 0 .. actions.Length - 1 do
                if actionsMask.[a] then
                    let h' = adjustState contexts player actions.[a] history
                    let nextP0, nextP1 =
                        if player = 0 then (p0 * strategy.[a], p1)
                        else (p0, p1 * strategy.[a])
                    util.[a] <- -cfrSampled rnd targetPlayer (d + 1) nextP0 nextP1
                                    reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'
                    nodeUtil <- nodeUtil + strategy.[a] * util.[a]
            // Regret update
            for a in 0 .. actions.Length - 1 do
                if actionsMask.[a] then
                    let regret = util.[a] - nodeUtil
                    node.regretSum.[a] <- node.regretSum.[a] + oppReach * regret
            // CFR+ clipping only where regrets updated
            clipRegretsPlus actionsMask node.regretSum
            nodeUtil
        else
            // Sample one opponent action according to masked strategy
            let (chosen, chosenProb) = sampleIndex rnd strategy actionsMask
            let h' = adjustState contexts player actions.[chosen] history
            let nextP0, nextP1 =
                if player = 0 then (p0 * chosenProb, p1)
                else (p0, p1 * chosenProb)
            -cfrSampled rnd targetPlayer (d + 1) nextP0 nextP1
                reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'
