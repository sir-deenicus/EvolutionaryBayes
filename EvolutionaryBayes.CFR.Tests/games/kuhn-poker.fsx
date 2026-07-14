#load "../../EvolutionaryBayes/CFRCore.fs"

open EvolutionaryBayes.CFRCore

// Kuhn poker
// ==========
//
// The deck contains Jack, Queen, and King. Each player receives one private
// card, leaving one card undealt. At each decision, local action 0 means
// check/fold and action 1 means bet/call. The two actions have different names
// only because their meaning depends on whether a bet is outstanding.

type Card =
    | Jack
    | Queen
    | King

    override this.ToString() =
        match this with
        | Jack -> "J"
        | Queen -> "Q"
        | King -> "K"

type State =
    | Deal
    | Play of player0Card: Card * player1Card: Card * history: string

let deals =
    [| Jack, Queen
       Jack, King
       Queen, Jack
       Queen, King
       King, Jack
       King, Queen |]

let cardIndex = function
    | Jack -> 0
    | Queen -> 1
    | King -> 2

let showdown player0Card player1Card stake =
    if cardIndex player0Card > cardIndex player1Card then stake else -stake

let tryPlayer0Utility player0Card player1Card history =
    match history with
    | "PP" -> ValueSome(showdown player0Card player1Card 1.0)
    | "BP" -> ValueSome 1.0
    | "BB" -> ValueSome(showdown player0Card player1Card 2.0)
    | "PBP" -> ValueSome -1.0
    | "PBB" -> ValueSome(showdown player0Card player1Card 2.0)
    | _ -> ValueNone

let actionSymbol = function
    | 0 -> "P"
    | 1 -> "B"
    | action -> invalidArg "action" $"Unknown action {action}."

let game =
    { new IGame<State> with
        member _.TerminalUtility(state, targetPlayer) =
            match state with
            | Play(player0Card, player1Card, history) ->
                match tryPlayer0Utility player0Card player1Card history with
                | ValueSome player0Utility ->
                    ValueSome(if targetPlayer = 0 then player0Utility else -player0Utility)
                | ValueNone -> ValueNone
            | Deal -> ValueNone

        member _.Actor state =
            match state with
            | Deal -> ChanceActor
            | Play(_, _, "")
            | Play(_, _, "PB") -> 0
            | Play(_, _, "P")
            | Play(_, _, "B") -> 1
            | Play(_, _, history) ->
                invalidArg "state" $"Terminal or invalid history '{history}' has no actor."

        member _.InformationSetId state =
            match state with
            | Play(player0Card, _, "") -> cardIndex player0Card
            | Play(_, player1Card, "P") -> 3 + cardIndex player1Card
            | Play(_, player1Card, "B") -> 6 + cardIndex player1Card
            | Play(player0Card, _, "PB") -> 9 + cardIndex player0Card
            | Deal -> invalidArg "state" "A chance state has no information set."
            | Play(_, _, history) ->
                invalidArg "state" $"Terminal or invalid history '{history}' has no information set."

        member _.ActionCount state =
            match state with
            | Deal -> deals.Length
            | Play(player0Card, player1Card, history) ->
                match tryPlayer0Utility player0Card player1Card history with
                | ValueSome _ -> invalidArg "state" "A terminal state has no actions."
                | ValueNone -> 2

        member _.NextState(state, action) =
            match state with
            | Deal ->
                if action < 0 || action >= deals.Length then
                    invalidArg "action" $"Unknown deal {action}."

                let player0Card, player1Card = deals.[action]
                Play(player0Card, player1Card, "")
            | Play(player0Card, player1Card, history) ->
                match tryPlayer0Utility player0Card player1Card history with
                | ValueSome _ -> invalidArg "state" "A terminal state has no successor."
                | ValueNone -> Play(player0Card, player1Card, history + actionSymbol action)

        member _.ChanceProbability(state, action) =
            match state with
            | Deal when action >= 0 && action < deals.Length -> 1.0 / float deals.Length
            | Deal -> invalidArg "action" $"Unknown deal {action}."
            | Play _ -> invalidArg "state" "A player state has no chance probability." }

let informationSets =
    Array.init 12 (fun id ->
        { Id = id
          Owner = if id < 3 || id >= 9 then 0 else 1
          ActionCount = 2 })

let solver =
    Solver.create
        SolverMode.CFR
        2
        game
        informationSets
        4       // chance plus at most three player decisions
        6       // six ordered deals at the chance node
        1729    // random seed; unused by exhaustive CFR

let training = Solver.run solver 50_000 0 Deal

let showRow player card decision infoSetId =
    let strategy = Solver.averageStrategy solver infoSetId
    printfn
        "Player %d, %s, %-14s: check/fold %.4f, bet/call %.4f"
        player
        (string card)
        decision
        strategy.[0]
        strategy.[1]

printfn "Kuhn poker after %d exhaustive CFR iterations" training.IterationsRun

for card in [ Jack; Queen; King ] do
    showRow 0 card "first action" (cardIndex card)
    showRow 1 card "after check" (3 + cardIndex card)
    showRow 1 card "facing bet" (6 + cardIndex card)
    showRow 0 card "check, then bet" (9 + cardIndex card)

let utilities = Array.zeroCreate 2
Solver.evaluateAverage solver Deal utilities
let value = utilities.[0]
let equilibriumValue = -1.0 / 18.0
printfn "Average-profile value for player 0: %.6f (equilibrium %.6f)" value equilibriumValue

if abs (value - equilibriumValue) > 0.01 then
    failwith "The average profile did not approach Kuhn poker's equilibrium value."
