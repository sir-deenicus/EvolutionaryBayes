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

//TODO: Are both the sampled and exhaustive implementations of CFR+?
//TODO: How can we replace the dictionary with function approximation? My current thoughts are that if we use an object that
//can be keyed by history such as a small transformer or indRNN, we'd need to find an updating scheme or loss that recovered regret convergence.

// Compute linear averaging weight for iteration (1-based) with optional burn-in.
let inline avgWeight (iter:int) (burnIn:int) =
    let t = iter
    if t <= burnIn then 0. else float (t - burnIn)  // w_t = t - burnIn

let inline private ensureLength name expected actual =
    if actual <> expected then
        invalidArg name $"Expected length {expected}, got {actual}."

let getStrategyMasked (mask: bool[]) (regretSum: double[]) =
    ensureLength "mask" regretSum.Length mask.Length
    // Convert regrets to non-negative (CFR+ style)
    let tmp = Array.init regretSum.Length (fun i -> if mask.[i] then max 0. regretSum.[i] else 0.)
    let s = Array.sum tmp
    if s <= 0. then
        // Uniform over legal actions only
        let legalCount = mask |> Array.filter id |> Array.length
        if legalCount = 0 then failwith "getStrategyMasked: no legal actions"
        Array.init tmp.Length (fun i -> if mask.[i] then 1. / float legalCount else 0.)
    else
        Array.map (fun v -> if v = 0. then 0. else v / s) tmp

let accumulateStrategyMasked (weight: double) (mask: bool[]) (strategySum: double[]) (strategy: double[]) =
    ensureLength "mask" strategy.Length mask.Length
    ensureLength "strategySum" strategy.Length strategySum.Length
    if weight > 0. then
        for i in 0 .. strategy.Length - 1 do
            if mask.[i] then
                strategySum.[i] <- strategySum.[i] + weight * strategy.[i]

type ActionMask<'contexts> = int -> 'contexts [] -> string -> bool[]
  
let inline clipRegretsPlus (mask: bool[]) (regrets: double[]) =
    ensureLength "mask" regrets.Length mask.Length
    // CFR+: keep cumulative regrets non-negative for legal actions
    for i = 0 to regrets.Length - 1 do
        if mask.[i] && regrets.[i] < 0. then
            regrets.[i] <- 0.

let rec cfr iter burnIn d p0 p1 reward lookup adjustState (getActionsMask: ActionMask<'a>) (stringifyContext:_->string) (nodeMap: Dict<_, _>) (contexts: 'a []) (actions:_[])  (history: string) =
    let player = d % 2
    let actionsMask : _ [] = getActionsMask d contexts history
    let nummoves = actions.Length
    ensureLength "getActionsMask" nummoves actionsMask.Length
    match reward contexts player history with
    | Some r -> r
    | None ->
        // Delimiter in infoset to avoid collisions
        let infoset = $"{stringifyContext contexts.[player]}|{history}"
        let node = lookup nummoves nodeMap infoset
        let strategy = getStrategyMasked actionsMask node.regretSum
        let ownReach = if player = 0 then p0 else p1
        let oppReach = if player = 0 then p1 else p0
        accumulateStrategyMasked (avgWeight iter burnIn * ownReach) actionsMask node.strategySum strategy
        let util = Array.zeroCreate nummoves
        let mutable nodeUtil = 0.
        for a in 0 .. nummoves - 1 do
            if actionsMask.[a] then
                let h' = adjustState contexts player actions.[a] history
                let nextP0, nextP1 =
                    if player = 0 then (p0 * strategy.[a], p1)
                    else (p0, p1 * strategy.[a])
                util.[a] <- -cfr iter burnIn (d + 1) nextP0 nextP1 reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'
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
    ensureLength "mask" strategy.Length mask.Length
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
    (rnd: Random) (targetPlayer: int) (iter:int) (burnIn:int)
    (d: int) (p0: double) (p1: double)
    reward lookup adjustState (getActionsMask: ActionMask<'a>) stringifyContext
    (nodeMap: Dict<_, _>) (contexts: 'a[]) (actions: 'b[]) (history: string) : double =

    let player = d % 2
    match reward contexts player history with
    | Some r -> r
    | None ->
        let infoset = $"{stringifyContext contexts.[player]}|{history}"
        let actionsMask = getActionsMask d contexts history
        ensureLength "getActionsMask" actions.Length actionsMask.Length
        let node = lookup actions.Length nodeMap infoset
        let strategy = getStrategyMasked actionsMask node.regretSum
        if player <> targetPlayer then
            // In external sampling, non-traverser infosets are visited with probability equal to their own reach.
            accumulateStrategyMasked (avgWeight iter burnIn) actionsMask node.strategySum strategy
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
                    util.[a] <- -cfrSampled rnd targetPlayer iter burnIn (d + 1) nextP0 nextP1
                                    reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'
                    nodeUtil <- nodeUtil + strategy.[a] * util.[a]
            // External-sampling already estimates counterfactual regret; no extra reach factor is applied here.
            for a in 0 .. actions.Length - 1 do
                if actionsMask.[a] then
                    let regret = util.[a] - nodeUtil
                    node.regretSum.[a] <- node.regretSum.[a] + regret
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
            -cfrSampled rnd targetPlayer iter burnIn (d + 1) nextP0 nextP1
                reward lookup adjustState getActionsMask stringifyContext nodeMap contexts actions h'
