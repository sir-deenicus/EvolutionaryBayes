module EvolutionaryBayes.RegretMinimization

open Prelude.Math
open System
open Prelude.Common

let multiplicativeWeightsUpdate rate minAmount maxAmount (oldweights : float [])
    (results : seq<float>) =
    let lossbounded = scaleTo -1. 1. minAmount maxAmount
    results
    |> Seq.mapi
           (fun i r ->
           oldweights.[i] * (1. + rate * lossbounded (min r maxAmount)))
    |> Seq.toArray
    |> Array.normalize

type Regrets =
    { Weights : float []
      SumRegrets : float []
      mutable Total : float }
    member e.Sample() = Stats.discreteSample e.Weights
    member e.NormalizedWeights =
        Array.map (fun r -> r / e.Total) e.SumRegrets
    member e.WeightedActions actions = 
        if e.Total = 0. then Array.zip e.Weights actions 
        else Array.zip e.NormalizedWeights actions

let newExpert n =
    { Weights = 
        Array.create n 1. 
        |> Array.normalize
      SumRegrets = Array.create n 0.
      Total = 0. }

let getExpert (experts : IDict<_, _>) n k =
    match experts.ContainsKey k with
    | true -> experts.[k]
    | false ->
        let e = newExpert n
        experts.Add(k, e); e

let editOrAdd (experts : IDict<_, _>) k e =
    match experts.ContainsKey k with
    | true -> experts.[k] <- e
    | false -> experts.Add(k, e)

let learningStep minr maxr rate actions reward (e, obsv) =
    let w = e.Weights
    let rs = actions |> Array.map (fun a -> reward (a, obsv))
    Array.Copy
        (multiplicativeWeightsUpdate rate minr maxr w rs, w, actions.Length)
    Array.addInPlaceIntoFirst e.SumRegrets w
    e.Total <- e.Total + 1.

type RegretLearner<'a, 'actions, 'obs when 'a : equality>
    (reward, ?agentActions : 'actions [], ?learningrate, 
     ?minreward, ?maxreward, ?prevweights, ?learner) =

    let experts = defaultArg prevweights (Dict<'a, _>())
    let lr = defaultArg learningrate 0.1
    let minr = defaultArg minreward -1.
    let maxr = defaultArg maxreward 1.
    let learn = defaultArg learner (learningStep minr maxr lr)
    let mutable actions = defaultArg agentActions [||]
    let mutable dim = actions.Length
    
    member __.Get k = experts.tryFind k
    
    member __.GetOrAdd k = getExpert experts dim k

    member __.New(k : 'a) =
        if not (experts.ContainsKey k) then experts.Add(k, newExpert dim)

    member m.New(items : _ seq) =
        for k in items do
            if not (experts.ContainsKey k) then experts.Add(k, newExpert dim)

    member __.SetActions a =
        actions <- a
        dim <- a.Length

    member __.TryFirst() =
        Seq.tryHead experts |> Option.map (fun kv -> kv.Value)
    
    member __.EditOrAddExpert(k, newexpert) = editOrAdd experts k newexpert
    
    member __.Learn(k, observation : 'obs) =
        learn actions reward ((experts.[k]), observation)
    
    member t.Observe(observation) = t.Learn(observation)
    
    member t.Learn(observation : 'obs) =
        t.TryFirst()
        |> Option.iter (fun e -> learn actions reward (e, observation))
    
    member __.Item k = experts.[k]
    
    member __.SampleActionOf k = actions.[experts.[k].Sample()]
    
    member __.Actions = actions

    member __.WeightedActions() =
        [| for (KeyValue(k, e)) in experts -> k, e.WeightedActions actions |]

    member __.WeightedActionsFor k = experts.[k].WeightedActions actions
    
    member t.WeightedActionsZero() =
        t.TryFirst() |> Option.map (fun e -> e.WeightedActions actions)
    
    member __.Experts = experts
    
    member __.Forget() =
        for k in (Seq.toArray experts.Keys) do
            experts.[k] <- newExpert dim