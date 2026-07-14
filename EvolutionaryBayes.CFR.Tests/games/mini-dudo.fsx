#load "../../EvolutionaryBayes/CFRCore.fs"

open System.Numerics
open EvolutionaryBayes.CFRCore

// Mini Dudo
// =========
//
// Each player rolls one private d6. Player 1 starts; players then alternate
// increasing claims about the two dice or call Dudo. Ones are wild for claims
// of ranks 2..6, but only literal ones count toward a claim on rank 1.

let dieSides = 6
let claimCount = 12
let historyCount = 1 <<< claimCount

// Within each quantity, ranks strengthen as 2, 3, 4, 5, 6, 1. Thus n x 1
// immediately precedes 2n x 2, exactly matching the special wild-rank rule.
let claim strength =
    let quantity = strength / dieSides + 1
    let rankOffset = strength % dieSides
    let rank = if rankOffset = dieSides - 1 then 1 else rankOffset + 2
    struct (quantity, rank)

let claimStrength quantity rank =
    (quantity - 1) * dieSides + (if rank = 1 then dieSides - 1 else rank - 2)

let strongestClaim (history: uint16) = BitOperations.Log2(uint32 history)
let actor (history: uint16) = BitOperations.PopCount(uint32 history) &&& 1

let actionCount (history: uint16) =
    if history = 0us then claimCount else claimCount - strongestClaim history

// Strictly increasing claims make the 12-bit set a lossless encoding of the
// complete public sequence. The packed dice pair is 0..35; -1 means not dealt.
// Caller is -1 during play and 0 or 1 after Dudo.
[<Struct>]
type State =
    { Claims: uint16
      Dice: sbyte
      Caller: sbyte }

let root =
    { Dice = -1y
      Claims = 0us
      Caller = -1y }

let dieAt player diceCode =
    if player = 0 then diceCode / dieSides + 1 else diceCode % dieSides + 1

let informationSetId state =
    let player = actor state.Claims
    int state.Claims * dieSides + dieAt player (int state.Dice) - 1

let game =
    { new IGame<State> with
        member _.TerminalUtility(state, targetPlayer) =
            if state.Caller < 0y then
                ValueNone
            else
                if targetPlayer <> 0 && targetPlayer <> 1 then
                    invalidArg "targetPlayer" "Dudo has exactly two players."

                let caller = int state.Caller
                let claimant = 1 - caller
                let struct (quantity, rank) = strongestClaim state.Claims |> claim
                let diceCode = int state.Dice
                let die0 = dieAt 0 diceCode
                let die1 = dieAt 1 diceCode
                let matches die = die = rank || (rank <> 1 && die = 1)
                let actual = (if matches die0 then 1 else 0) + (if matches die1 then 1 else 0)
                let winner = if actual >= quantity then claimant else caller
                ValueSome(if targetPlayer = winner then 1.0 else -1.0)

        member _.Actor state =
            if state.Caller >= 0y then
                invalidArg "state" "A terminal state has no actor."
            elif state.Dice < 0y then
                ChanceActor
            else
                actor state.Claims

        member _.InformationSetId state =
            if state.Dice < 0y || state.Caller >= 0y then
                invalidArg "state" "Only player states have information sets."

            informationSetId state

        member _.ActionCount state =
            if state.Caller >= 0y then
                invalidArg "state" "A terminal state has no actions."
            elif state.Dice < 0y then
                dieSides * dieSides
            else
                actionCount state.Claims

        member _.NextState(state, action) =
            if state.Caller >= 0y then
                invalidArg "state" "A terminal state has no successor."
            elif state.Dice < 0y then
                if action < 0 || action >= dieSides * dieSides then
                    invalidArg "action" $"Unknown dice roll {action}."

                { Dice = sbyte action
                  Claims = 0us
                  Caller = -1y }
            else
                let count = actionCount state.Claims

                if action < 0 || action >= count then
                    invalidArg "action" $"Unknown local action {action}."

                if state.Claims <> 0us && action = 0 then
                    { state with Caller = sbyte (actor state.Claims) }
                else
                    // Initially local actions 0..11 are claims. Thereafter
                    // action 0 is Dudo and action n raises strength by n.
                    let strength =
                        if state.Claims = 0us then action
                        else strongestClaim state.Claims + action

                    { state with Claims = state.Claims ||| (1us <<< strength) }

        member _.ChanceProbability(state, action) =
            if state.Dice >= 0y then
                invalidArg "state" "Only the root is a chance state."
            elif action < 0 || action >= dieSides * dieSides then
                invalidArg "action" $"Unknown dice roll {action}."

            1.0 / float (dieSides * dieSides) }

// IDs are dense without a lookup table:
//
//   id = 6 * public-history-mask + (private-die - 1)
//
// The owner follows from the number of previous claims. Every subset of claim
// strengths is a reachable increasing history, so all 24,576 rows are useful.
let informationSets =
    Array.init (historyCount * dieSides) (fun id ->
        let history = uint16 (id / dieSides)

        { Id = id
          Owner = actor history
          ActionCount = actionCount history })

let createSolver mode =
    Solver.create
        mode
        2
        game
        informationSets
        14      // chance, twelve possible claims, and Dudo
        36      // all ordered pairs of dice at the chance node
        1729

let expectedValue = -7.0 / 258.0

let train mode iterations burnIn tolerance =
    let solver = createSolver mode
    let convergence = ConvergenceCheck.create tolerance iterations
    let result =
        Solver.runUntil
            solver
            iterations
            burnIn
            root
            convergence
            (Convergence.utilityError 0 expectedValue)

    let utilities = Array.zeroCreate 2
    Solver.evaluateAverage solver root utilities
    struct (result, utilities.[0])

// Small rule checks guard the two details most likely to be implemented
// incorrectly: claim ordering and wild ones. Terminal utility is zero-sum by
// construction: exactly one of the two players is selected as the winner.
let terminalUtility diceCode strength caller targetPlayer =
    let state =
        { Dice = sbyte diceCode
          Claims = 1us <<< strength
          Caller = sbyte caller }

    match game.TerminalUtility(state, targetPlayer) with
    | ValueSome utility -> utility
    | ValueNone -> failwith "A resolved Dudo state was not terminal."

let oneAndThree = 2 // encoded dice (1, 3)

if claim 0 <> struct (1, 2)
   || claim 5 <> struct (1, 1)
   || claim 6 <> struct (2, 2)
   || claim (claimCount - 1) <> struct (2, 1) then
    failwith "Claim ordering endpoints were incorrect."

if claimStrength 1 1 + 1 <> claimStrength 2 2 then
    failwith "The special wild claim ordering was incorrect."

if terminalUtility oneAndThree (claimStrength 1 6) 1 0 <> 1.0 then
    failwith "A wild one did not satisfy a claim of one six."

if terminalUtility oneAndThree (claimStrength 2 6) 1 1 <> 1.0 then
    failwith "The caller did not win against a false claim of two sixes."

if terminalUtility oneAndThree (claimStrength 2 1) 1 1 <> 1.0 then
    failwith "A non-one incorrectly counted toward a claim on ones."

// Optional script arguments override MCCFR+ iterations, then CFR iterations.
let iterationArg index fallback =
    if fsi.CommandLineArgs.Length > index then
        int fsi.CommandLineArgs.[index]
    else
        fallback

let mccfrIterations = iterationArg 1 1_000_000
let cfrIterations = iterationArg 2 500

// MCCFR+ samples chance and the opponent. Its sparse sampled-delta log grows
// during warm-up and is then reused without steady-state allocation.
let struct (mccfrResult, mccfrProfileValue) =
    train SolverMode.MCCFRPlus mccfrIterations (mccfrIterations / 10) 0.006

let struct (cfrResult, cfrProfileValue) =
    train SolverMode.CFR cfrIterations 0 0.002

if mccfrIterations >= 1_000_000
   && not mccfrResult.Converged then
    failwith "MCCFR+ did not approach the known Mini Dudo value."

if cfrIterations >= 500
   && not cfrResult.Converged then
    failwith "Exhaustive CFR did not approach the known Mini Dudo value."

printfn "Mini Dudo (state size: %d bytes; expected value %.6f)" sizeof<State> expectedValue
printfn
    "MCCFR+   %9d iterations: training average %.6f (error %.6f), average profile %.6f (error %.6f)"
    mccfrResult.IterationsRun
    mccfrResult.MeanUtility0
    (abs (mccfrResult.MeanUtility0 - expectedValue))
    mccfrProfileValue
    (abs (mccfrProfileValue - expectedValue))
printfn
    "CFR       %9d iterations: training average %.6f (error %.6f), average profile %.6f (error %.6f)"
    cfrResult.IterationsRun
    cfrResult.MeanUtility0
    (abs (cfrResult.MeanUtility0 - expectedValue))
    cfrProfileValue
    (abs (cfrProfileValue - expectedValue))
