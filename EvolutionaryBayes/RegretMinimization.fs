module EvolutionaryBayes.RegretMinimization

open Prelude.Math
open System
open Prelude.Common
open Prelude

type Regrets =
    { Weights: float []
      SumRegrets: float []
      mutable Total: float } 
       
let newExpert nActions =
    { Weights = Array.create nActions 1. |> Array.normalize
      SumRegrets = Array.create nActions 0.
      Total = 0. } 

let multiplicativeWeightsUpdate rate minAmount maxAmount (oldweights : float [])
    (results : seq<float>) =
    let lossbounded = scaleTo -1. 1. minAmount maxAmount
    results
    |> Seq.mapi
           (fun i r ->
           oldweights.[i] * (1. + rate * lossbounded (min r maxAmount)))
    |> Seq.toArray
    |> Array.normalize

let learningStep minr maxr rate actions reward (e, obsv) =
    let w = e.Weights
    let rs = actions |> Array.map (fun a -> reward (a, obsv))
    Array.Copy
        (multiplicativeWeightsUpdate rate minr maxr w rs, w, actions.Length)
    Array.addInPlaceIntoFirst e.SumRegrets w
    e.Total <- e.Total + 1.

type RegretLearner<'actions, 'obs>
    (reward, possibleActions : 'actions [], ?learningrate, 
     ?minreward, ?maxreward, ?prevweights, ?learner) = 
    let lr = defaultArg learningrate 0.1
    let minr = defaultArg minreward -1.
    let maxr = defaultArg maxreward 1.
    let learn = defaultArg learner (learningStep minr maxr lr) 

    let expert = defaultArg prevweights (newExpert possibleActions.Length) 
     
    member e.ActionWeights = Array.zip possibleActions expert.Weights

    member e.Sample() = possibleActions.[Sampling.discreteSampleIndex expert.Weights]

    member e.NormalizedRegret =
        Array.map (fun r -> r / expert.Total) expert.SumRegrets
    
    member e.LearnedActionWeights() = Array.zip possibleActions e.NormalizedRegret 
    
    member __.Learn(observation : 'obs) =
        learn possibleActions reward (expert, observation)
     
    member __.Expert = expert
    
    member __.Actions = possibleActions

    member __.Reset() =
        expert.Total <- 0.
        for i in 0..possibleActions.Length - 1 do 
            expert.Weights.[i] <- 1./float possibleActions.Length
            expert.SumRegrets.[i] <- 0.

type ContextualExpert(coarseGrainer, distancefn) =
    let seen = Hashset() 

    let predict() = 
        ()
