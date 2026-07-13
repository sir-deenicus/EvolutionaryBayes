#load "../../EvolutionaryBayes/CFRCore.fs"

open EvolutionaryBayes.CFRCore

// Hidden matching pennies
// =======================
//
// This is about the smallest possible imperfect-information game:
//
//   1. Player 0 chooses Heads or Tails.
//   2. Player 1 chooses Heads or Tails without seeing player 0's choice.
//   3. Player 0 wins if the choices match; player 1 wins otherwise.
//
// It is written sequentially because CFR walks a game tree. The game is still
// equivalent to simultaneous matching pennies: player 1's two possible
// decision nodes belong to the same information set.

type Choice =
    | Heads
    | Tails

type State =
    | Player0Turn
    | Player1Turn of hiddenPlayer0Choice: Choice
    | Terminal of player0Choice: Choice * player1Choice: Choice

let choiceFromAction = function
    | 0 -> Heads
    | 1 -> Tails
    | action -> invalidArg "action" $"Unknown action {action}."

let game =
    { new IGame<State> with
        member _.TerminalUtility(state, targetPlayer) =
            match state with
            | Terminal(player0Choice, player1Choice) ->
                let player0Utility =
                    if player0Choice = player1Choice then 1.0 else -1.0

                ValueSome(if targetPlayer = 0 then player0Utility else -player0Utility)
            | _ -> ValueNone

        member _.Actor state =
            match state with
            | Player0Turn -> 0
            | Player1Turn _ -> 1
            | Terminal _ -> invalidArg "state" "A terminal state has no actor."

        member _.InformationSetId state =
            match state with
            | Player0Turn -> 0
            | Player1Turn _ ->
                // Crucial imperfect-information rule: this is ID 1 whether
                // player 0 chose Heads or Tails. Player 1 cannot distinguish
                // the two underlying states.
                1
            | Terminal _ ->
                invalidArg "state" "A terminal state has no information set."

        member _.ActionCount state =
            match state with
            | Player0Turn
            | Player1Turn _ -> 2
            | Terminal _ -> invalidArg "state" "A terminal state has no actions."

        member _.NextState(state, action) =
            let choice = choiceFromAction action

            match state with
            | Player0Turn -> Player1Turn choice
            | Player1Turn player0Choice -> Terminal(player0Choice, choice)
            | Terminal _ -> invalidArg "state" "A terminal state has no successor."

        member _.ChanceProbability(_state, _action) =
            invalidOp "This game has no chance nodes." }

// The solver uses dense information-set IDs. Both information sets have the
// same two locally indexed actions: 0 = Heads and 1 = Tails.
let informationSets =
    [| { Id = 0; Owner = 0; ActionCount = 2 }
       { Id = 1; Owner = 1; ActionCount = 2 } |]

let solver =
    Solver<State, IGame<State>>(
        SolverMode.CFR,
        game,
        informationSets,
        2,      // maximum number of nonterminal levels
        2,      // maximum actions at a node
        4,      // sampled-delta capacity; unused by exhaustive CFR
        1729    // random seed; unused by exhaustive CFR
    )

let training = solver.Train(10_000, 0, Player0Turn)

let player0Strategy = solver.AverageStrategy 0
let player1Strategy = solver.AverageStrategy 1

let showStrategy player (strategy: double[]) =
    printfn
        "Player %d: Heads %.4f, Tails %.4f"
        player
        strategy.[0]
        strategy.[1]

printfn "Hidden matching pennies after %d CFR iterations" training.IterationsRun
showStrategy 0 player0Strategy
showStrategy 1 player1Strategy

// The unique Nash equilibrium is uniform play by both players.
let nearHalf probability = abs (probability - 0.5) < 0.01

if not (Array.forall nearHalf player0Strategy
        && Array.forall nearHalf player1Strategy) then
    failwith "The average strategies did not approach the 50/50 equilibrium."
