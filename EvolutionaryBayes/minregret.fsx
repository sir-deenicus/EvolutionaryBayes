#load "MWUA.fs"
#load "ContextualRegret.fs"

open System
open EvolutionaryBayes

type Dir =
    | Stop
    | Go

let actions = [| Go; Stop |]

let reward opponentAction ownAction =
    match ownAction, opponentAction with
    | Go, Go -> -100.0
    | Go, Stop -> 1.0
    | Stop, Go
    | Stop, Stop -> 0.0

let options seed =
    { ExternalRegretOptions.defaults with
        Random = Some(Random seed) }

// Both algorithms implement ExternalRegretMinimizer<Dir>.
let player1 = Mwua.createWith (options 1) 0.05 actions
let player2 = RegretMatching.createWith (options 2) actions
let player1Observations = ResizeArray<Dir>()
let player2Observations = ResizeArray<Dir>()

for _ = 1 to 10_000 do
    let action1 = player1.Choose()
    let action2 = player2.Choose()

    player1Observations.Add action2
    player2Observations.Add action1

    // Full-information feedback scores every possible action against the
    // opponent action observed this round.
    player1.Learn(reward action2)
    player2.Learn(reward action1)

printfn "MWUA average strategy:            %A" player1.AverageStrategy
printfn "Regret-matching average strategy: %A" player2.AverageStrategy

// Bulk learning is the same update loop over a sequence of feedback functions.
// Replay the observations from the game above into fresh learners.
let bulkPlayer1 = Mwua.create 0.05 actions
let bulkPlayer2 = RegretMatching.create actions

let feedback observations =
    observations
    |> Seq.map reward

bulkPlayer1.LearnBatch(feedback player1Observations)
bulkPlayer2.LearnBatch(feedback player2Observations)

printfn "Bulk MWUA average strategy:       %A" bulkPlayer1.AverageStrategy
printfn "Bulk regret-matching average:     %A" bulkPlayer2.AverageStrategy

// Exact contexts memorize one independent learner per repeated context.
type Situation =
    | Quiet
    | Busy

let contextualReward situation action =
    match situation, action with
    | Quiet, Go
    | Busy, Stop -> 1.0
    | _ -> 0.0

let exactBandit: ExactContextBanditRegretMinimizer<Situation, Dir> =
    ContextualExp3.createWith (options 3) 0.05 0.1 actions

let exactChoice = exactBandit.Choose Quiet
let exactObservedReward = contextualReward Quiet exactChoice.Action
exactBandit.Learn(exactChoice, exactObservedReward)

printfn "Exact-context strategy (Quiet):   %A" (exactBandit.Strategy Quiet)

// EXP4 generalizes across contexts by learning weights over policies.
let policies: ContextualPolicy<Situation, Dir>[] =
    [| (fun situation -> if situation = Quiet then Go else Stop)
       (fun situation -> if situation = Quiet then Stop else Go) |]

let policyBandit =
    Exp4.createWith (options 3) 0.05 0.1 actions policies

let policyChoice = policyBandit.Choose Quiet
let policyObservedReward = contextualReward Quiet policyChoice.Action
policyBandit.Learn(policyChoice, policyObservedReward)

printfn "EXP4 policy strategy:             %A" (policyBandit.PolicyStrategy |> Array.map snd)
printfn "EXP4 action strategy (Busy):      %A" (policyBandit.Strategy Busy)

(* TODO: Remove this legacy API example in the final cleanup stage.
   It is retained as a comment while the online-learning API is replaced.

#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\netstandard2.1\Prelude.dll"
#r @"bin\Debug\netstandard2.1\EvolutionaryBayes.dll"

open Prelude
open Prelude.Math
open EvolutionaryBayes.RegretMinimization

let expert1 =
    RegretLearner<_, _>(reward, [| Go; Stop |], minreward = -100., maxreward = 100.)

let expert2 =
    RegretLearner<_, _>(reward, [| Go; Stop |], minreward = -100., maxreward = 100.)

for _ in 0 .. 999999 do
    let action1 = expert1.Sample()
    let action2 = expert2.Sample()

    expert1.Learn(observation = action1)
    expert2.Learn(action2)

expert1.NormalizedRegret
expert1.ActionWeights
expert1.Sample()
expert1.LearnedActionWeights()
expert2.LearnedActionWeights()
*)
