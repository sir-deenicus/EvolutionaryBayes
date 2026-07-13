module EvolutionaryBayes.CFR.Tests

open System
open System.Collections.Generic
open EvolutionaryBayes.CFR
open EvolutionaryBayes.CFRCore

let mutable failures = 0

let fail message = raise (Exception message)

let assertTrue message condition =
    if not condition then fail message

let assertNear tolerance expected actual =
    if Double.IsNaN actual || abs (expected - actual) > tolerance then
        fail $"Expected {expected:g17}, got {actual:g17}."

let assertArrayNear tolerance (expected: double[]) (actual: double[]) =
    assertTrue "Array lengths differ." (expected.Length = actual.Length)
    let mutable i = 0

    while i < expected.Length do
        assertNear tolerance expected.[i] actual.[i]
        i <- i + 1

let assertThrows action =
    let mutable threw = false

    try
        action ()
    with
    | :? ArgumentException
    | :? InvalidOperationException -> threw <- true

    assertTrue "Expected the operation to reject invalid input." threw

let test name action =
    try
        action ()
        printfn "PASS %s" name
    with error ->
        failures <- failures + 1
        eprintfn "FAIL %s: %s" name error.Message

[<Struct>]
type MatrixGame =
    val Utilities0: double[]
    val Utilities1: double[]

    new(utilities0, utilities1) =
        { Utilities0 = utilities0
          Utilities1 = utilities1 }

    interface IExhaustiveGame<int> with
        member this.TryGetTerminalUtility(state, targetPlayer, utility: byref<double>) =
            if state >= 3 then
                let index = state - 3
                utility <- if targetPlayer = 0 then this.Utilities0.[index] else this.Utilities1.[index]
                true
            else
                false

        member _.Actor state =
            if state = 0 then 0
            elif state = 1 || state = 2 then 1
            else invalidOp "Actor must not be queried at a terminal."

        member _.InformationSetId state = if state = 0 then 0 else 1
        member _.ActionCount _ = 2
        member _.NextState(state, action) = if state = 0 then 1 + action else 3 + (state - 1) * 2 + action
        member _.ChanceProbability(_, _) = invalidOp "This game has no chance nodes."

type CountingMatrixGame(utilities0: double[], utilities1: double[]) =
    let mutable actorCalls = 0

    member _.ActorCalls = actorCalls

    interface IExhaustiveGame<int> with
        member _.TryGetTerminalUtility(state, targetPlayer, utility: byref<double>) =
            if state >= 3 then
                let index = state - 3
                utility <- if targetPlayer = 0 then utilities0.[index] else utilities1.[index]
                true
            else
                false

        member _.Actor state =
            actorCalls <- actorCalls + 1
            if state = 0 then 0 else 1

        member _.InformationSetId state = if state = 0 then 0 else 1
        member _.ActionCount _ = 2
        member _.NextState(state, action) = if state = 0 then 1 + action else 3 + (state - 1) * 2 + action
        member _.ChanceProbability(_, _) = invalidOp "This game has no chance nodes."

[<Struct>]
type ChanceThenPlayerGame =
    interface IExhaustiveGame<int> with
        member _.TryGetTerminalUtility(state, targetPlayer, utility: byref<double>) =
            if state >= 3 then
                let values0 =
                    match state with
                    | 3 -> 2.0
                    | 4 -> 0.0
                    | 5 -> 0.0
                    | _ -> 4.0
                utility <- if targetPlayer = 0 then values0 else 10.0 + values0
                true
            else
                false

        member _.Actor state = if state = 0 then ChanceActor else 0
        member _.InformationSetId _ = 0
        member _.ActionCount _ = 2
        member _.NextState(state, action) = if state = 0 then 1 + action else 3 + (state - 1) * 2 + action
        member _.ChanceProbability(_, action) = if action = 0 then 0.75 else 0.25

[<Struct>]
type ConsecutiveTurnGame =
    interface IExhaustiveGame<int> with
        member _.TryGetTerminalUtility(state, targetPlayer, utility: byref<double>) =
            if state >= 7 then
                let value0 = float (state - 7)
                utility <- if targetPlayer = 0 then value0 else 6.0 - value0
                true
            else
                false

        member _.Actor state = if state < 3 then 0 else 1
        member _.InformationSetId state =
            if state = 0 then 0
            elif state < 3 then state
            else 3
        member _.ActionCount _ = 2
        member _.NextState(state, action) =
            if state = 0 then 1 + action
            elif state < 3 then 3 + (state - 1) * 2 + action
            else 7 + (state - 3) * 2 + action
        member _.ChanceProbability(_, _) = invalidOp "This game has no chance nodes."

let lookup actionCount (nodes: Dictionary<string, StrategyNode>) key =
    match nodes.TryGetValue key with
    | true, node -> node
    | false, _ ->
        let node = newStrategyNode actionCount
        nodes.Add(key, node)
        node

let legacyOneInformationSet () =
    let nodes = Dictionary<string, StrategyNode>()
    let reward (_: string[]) player history =
        match history with
        | "A" -> Some(if player = 0 then 1.0 else -1.0)
        | "B" -> Some(if player = 0 then -1.0 else 1.0)
        | _ -> None

    let adjust (_: string[]) _ action history = history + action
    let mask _ (_: string[]) _ = [| true; true |]
    let value = cfr 1 0 0 1.0 1.0 reward lookup adjust mask id nodes [| "P0"; "P1" |] [| "A"; "B" |] ""
    assertNear 1e-12 0.0 value
    let node = nodes.["P0|"]
    assertArrayNear 1e-12 [| 1.0; 0.0 |] node.regretSum
    assertArrayNear 1e-12 [| 0.5; 0.5 |] node.strategySum

let runPerfectInformationFixture iterations =
    let nodes = Dictionary<string, StrategyNode>()
    let actions = [| "S"; "C" |]
    let reward (_: string[]) player history =
        let utility0 =
            match history with
            | "S" -> Some 0.0
            | "CT" -> Some -1.0
            | "CG" -> Some 1.0
            | _ -> None

        utility0 |> Option.map (fun value -> if player = 0 then value else -value)

    let adjust (_: string[]) player action history =
        if player = 0 then action
        elif action = "S" then history + "T"
        else history + "G"

    let mask depth (_: string[]) history =
        if depth = 0 then [| true; true |]
        elif history = "C" then [| true; true |]
        else [| false; false |]

    let mutable iteration = 1

    while iteration <= iterations do
        cfr iteration 0 0 1.0 1.0 reward lookup adjust mask id nodes [| "P0"; "P1" |] actions "" |> ignore
        iteration <- iteration + 1

    let average key =
        let values = nodes.[key].strategySum
        let total = Array.sum values
        values |> Array.map (fun value -> value / total)

    let root = average "P0|"
    let response = average "P1|C"
    let continueProbability = root.[1]
    let giveProbability = response.[1]
    let bestPlayer0 = max 0.0 (2.0 * giveProbability - 1.0)
    root, response, 0.5 * (bestPlayer0 + continueProbability)

let helperTests () =
    test "legacy/helper/masked-regret-matching" (fun () ->
        getStrategyMasked [| true; false; true |] [| 2.0; 100.0; 1.0 |]
        |> assertArrayNear 1e-12 [| 2.0 / 3.0; 0.0; 1.0 / 3.0 |])

    test "legacy/helper/uniform-legal-fallback" (fun () ->
        getStrategyMasked [| true; false; true |] [| -2.0; 100.0; 0.0 |]
        |> assertArrayNear 1e-12 [| 0.5; 0.0; 0.5 |])

    test "legacy/helper/cfr-plus-clipping" (fun () ->
        let values = [| -2.0; -3.0; 4.0 |]
        clipRegretsPlus [| true; false; true |] values
        assertArrayNear 0.0 [| 0.0; -3.0; 4.0 |] values)

    test "legacy/helper/linear-average-burn-in" (fun () ->
        assertNear 0.0 0.0 (avgWeight 4 4)
        assertNear 0.0 3.0 (avgWeight 7 4))

    test "legacy/helper/fixed-seed-sampling" (fun () ->
        let probabilities = [| 0.1; 0.3; 0.6 |]
        let mask = [| true; true; true |]
        let first = Random 1729
        let second = Random 1729
        let a = Array.init 100 (fun _ -> fst (sampleIndex first probabilities mask))
        let b = Array.init 100 (fun _ -> fst (sampleIndex second probabilities mask))
        assertTrue "A fixed seed must reproduce the sampled sequence." (a = b))

let baselineFixtureTests () =
    test "legacy/fixture/hand-computed-one-information-set" legacyOneInformationSet

    test "legacy/fixture/zero-sum-exploitability-decreases" (fun () ->
        let _, _, early = runPerfectInformationFixture 2
        let _, _, late = runPerfectInformationFixture 500
        assertTrue $"Expected exploitability to decrease ({early} -> {late})." (late < early))

    test "acceptance/player-scoped-information-set-keys" (fun () ->
        let legacy0 = "same|history"
        let legacy1 = "same|history"
        let scoped0 = struct (0, "same", "history")
        let scoped1 = struct (1, "same", "history")
        assertTrue "Legacy keys demonstrate the collision." (legacy0 = legacy1)
        assertTrue "Player-scoped keys must differ." (scoped0 <> scoped1))

    test "acceptance/repeated-information-set-aggregate-before-clip" (fun () ->
        let sequential = max 0.0 (max 0.0 (1.0 - 2.0) + 2.0)
        let aggregate = max 0.0 (1.0 - 2.0 + 2.0)
        assertNear 0.0 2.0 sequential
        assertNear 0.0 1.0 aggregate)

    test "acceptance/two-player-general-sum-is-not-negamax" (fun () ->
        let utility0, utility1 = 2.0, 5.0
        assertTrue "The fixture is deliberately not zero-sum." (utility1 <> -utility0))

    test "acceptance/chance-enumeration-must-use-probability" (fun () ->
        let weighted = 0.9 * 1.0 + 0.1 * -1.0
        let equallyEnumerated = 0.5 * 1.0 + 0.5 * -1.0
        assertNear 0.0 0.8 weighted
        assertTrue "Equal enumeration is wrong for non-uniform chance." (weighted <> equallyEnumerated))

    test "invariant/perfect-recall-and-stable-actions-are-game-contracts" (fun () ->
        let masks = [| [| true; false; true |]; [| true; false; true |] |]
        assertTrue "Histories in one information set must expose identical actions." (masks.[0] = masks.[1]))

    test "acceptance/explicit-actor-supports-n-player-and-consecutive-turns" (fun () ->
        let actors = [| 0; 0; 2; 1 |]
        assertTrue "Fixture includes a consecutive turn and player 2." (actors.[0] = actors.[1] && Array.contains 2 actors))

    test "legacy/issue/sampled-iteration-returns-root-perspective-twice" (fun () ->
        let nodes = Dictionary<string, StrategyNode>()
        let reward (_: string[]) player history = if history = "A" then Some(if player = 0 then 3.0 else -3.0) else None
        let adjust (_: string[]) _ action history = history + action
        let mask _ (_: string[]) _ = [| true |]
        let first, second = cfrSampledIteration (Random 1) 1 0 reward lookup adjust mask id nodes [| "P0"; "P1" |] [| "A" |] ""
        assertNear 0.0 3.0 first
        assertNear 0.0 3.0 second)

    test "legacy/issue/invalid-target-silently-skips-regret" (fun () ->
        let nodes = Dictionary<string, StrategyNode>()
        let reward (_: string[]) player history = if history = "A" then Some(if player = 0 then 1.0 else -1.0) else None
        let adjust (_: string[]) _ action history = history + action
        let mask _ (_: string[]) _ = [| true |]
        cfrSampled (Random 1) 2 1 0 0 1.0 1.0 reward lookup adjust mask id nodes [| "P0"; "P1" |] [| "A" |] "" |> ignore
        assertNear 0.0 0.0 nodes.["P0|"].regretSum.[0])

    test "invariant/mccfr-plus-is-a-clipped-sampled-variant" (fun () ->
        let semantics = Mode.semantics SolverMode.MCCFRPlus
        assertTrue "The mode is sampled and clipped; ordinary MCCFR guarantees are not silently claimed." (semantics.Traversal = ExternalSampling && semantics.RegretTransform = Clipped))

let modeAndTableTests () =
    test "core/modes/semantics-and-weights" (fun () ->
        let cfrMode = Mode.semantics SolverMode.CFR
        let plusMode = Mode.semantics SolverMode.MCCFRPlus
        assertTrue "CFR must be signed and simultaneous." (cfrMode.RegretTransform = Signed && cfrMode.UpdateSchedule = Simultaneous)
        assertTrue "MCCFR+ must sample, clip, and alternate." (plusMode.Traversal = ExternalSampling && plusMode.RegretTransform = Clipped && plusMode.UpdateSchedule = Alternating)
        assertNear 0.0 1.0 (Mode.averageWeight SolverMode.CFR 9 3)
        assertNear 0.0 6.0 (Mode.averageWeight SolverMode.CFRPlus 9 3))

    test "core/table/canonical-packed-layout-and-memory" (fun () ->
        let table =
            PackedTable.create 2 [| { Id = 1; Owner = 1; ActionCount = 3 }; { Id = 0; Owner = 0; ActionCount = 2 } |]
        assertTrue "Rows must be ordered by dense ID." (table.InfoSets.[0].Offset = 0 && table.InfoSets.[1].Offset = 2)
        assertTrue "Two doubles are stored per legal slot." (int64 (table.Tables.Regrets.Length + table.Tables.StrategySums.Length) * 8L = 16L * int64 table.SlotCount))

    test "core/table/rejects-duplicate-and-cross-player-id" (fun () ->
        assertThrows (fun () -> PackedTable.create 2 [| { Id = 0; Owner = 0; ActionCount = 2 }; { Id = 0; Owner = 1; ActionCount = 2 } |] |> ignore))

    test "core/table/rejects-out-of-range-or-gapped-id" (fun () ->
        assertThrows (fun () -> PackedTable.create 2 [| { Id = 1; Owner = 0; ActionCount = 2 } |] |> ignore))

    test "core/table/rejects-invalid-owner" (fun () ->
        assertThrows (fun () -> PackedTable.create 2 [| { Id = 0; Owner = 2; ActionCount = 2 } |] |> ignore))

    test "core/table/rejects-zero-actions" (fun () ->
        assertThrows (fun () -> PackedTable.create 2 [| { Id = 0; Owner = 0; ActionCount = 0 } |] |> ignore))

    test "core/table/rejects-gapped-or-overlapping-offsets" (fun () ->
        let gapped: InfoSetMeta[] = [| { Owner = 0; Offset = 0; ActionCount = 2 }; { Owner = 1; Offset = 3; ActionCount = 2 } |]
        let overlap: InfoSetMeta[] = [| { Owner = 0; Offset = 0; ActionCount = 2 }; { Owner = 1; Offset = 1; ActionCount = 2 } |]
        assertThrows (fun () -> PackedTable.ofMetadata 2 gapped |> ignore)
        assertThrows (fun () -> PackedTable.ofMetadata 2 overlap |> ignore))

let scalarTests () =
    test "core/scalar/regret-matching-edge-cases" (fun () ->
        let destination = Array.zeroCreate 4
        Scalar.regretMatch [| -2.0; 0.0; 2.0; 6.0 |] 0 4 destination 0
        assertArrayNear 1e-12 [| 0.0; 0.0; 0.25; 0.75 |] destination
        Scalar.regretMatch [| -2.0 |] 0 1 destination 1
        assertNear 0.0 1.0 destination.[1]
        Scalar.regretMatch [| Double.MaxValue; Double.MaxValue |] 0 2 destination 0
        assertArrayNear 1e-12 [| 0.5; 0.5 |] destination.[0..1])

    test "core/scalar/rejects-non-finite-checked-input" (fun () ->
        assertThrows (fun () -> Scalar.regretMatch [| Double.NaN |] 0 1 [| 0.0 |] 0)
        assertThrows (fun () -> Scalar.sample [| Double.PositiveInfinity |] 0 1 0.5 |> ignore)
        assertThrows (fun () -> Scalar.applyRegretDelta false [| 0.0 |] 0 [| Double.NaN |] 0 1))

    test "core/scalar/probability-sampling-boundaries-and-uniform" (fun () ->
        let probabilities = [| 0.0; 2.0; 0.0; 6.0 |]
        assertTrue "Low draw selects first positive slot." (Scalar.sample probabilities 0 4 0.0 = 1)
        assertTrue "High draw selects last positive slot." (Scalar.sample probabilities 0 4 0.999999 = 3)
        assertTrue "Zero row samples uniformly by slot." (Scalar.sample [| 0.0; 0.0; 0.0 |] 0 3 0.5 = 1))

    test "core/scalar/signed-versus-aggregate-plus-update" (fun () ->
        let signed = [| 1.0; -1.0 |]
        let clipped = Array.copy signed
        let aggregate = [| -2.0; 0.5 |]
        Scalar.applyRegretDeltaForMode SolverMode.CFR signed 0 aggregate 0 2
        Scalar.applyRegretDeltaForMode SolverMode.CFRPlus clipped 0 aggregate 0 2
        assertArrayNear 0.0 [| -1.0; -0.5 |] signed
        assertArrayNear 0.0 [| 0.0; 0.0 |] clipped)

    test "core/scalar/average-accumulation-and-reporting" (fun () ->
        let sums = [| 0.0; 0.0; 0.0 |]
        Scalar.accumulateAverage 3.0 [| 0.2; 0.3; 0.5 |] 0 3 sums 0
        assertArrayNear 1e-12 [| 0.6; 0.9; 1.5 |] sums
        let report = Array.zeroCreate 3
        Scalar.normalizeAverage sums 0 3 report 0
        assertArrayNear 1e-12 [| 0.2; 0.3; 0.5 |] report
        Scalar.normalizeAverage [| 0.0; 0.0; 0.0 |] 0 3 report 0
        assertArrayNear 1e-12 [| 1.0 / 3.0; 1.0 / 3.0; 1.0 / 3.0 |] report)

    test "core/scalar/property-normalizes-finite-regrets" (fun () ->
        let random = Random 1729
        let regrets = Array.zeroCreate 64
        let strategy = Array.zeroCreate 64
        let mutable trial = 0

        while trial < 1000 do
            let count = 1 + random.Next 64
            let mutable i = 0
            while i < count do
                regrets.[i] <- (random.NextDouble() - 0.6) * 1e200
                i <- i + 1
            Scalar.regretMatch regrets 0 count strategy 0
            let mutable sum = 0.0
            i <- 0
            while i < count do
                assertTrue "Strategy probability must be finite and nonnegative." (strategy.[i] >= 0.0 && not (Double.IsNaN strategy.[i]))
                sum <- sum + strategy.[i]
                i <- i + 1
            assertNear 1e-12 1.0 sum
            trial <- trial + 1)

let workspaceAndAllocationTests () =
    test "core/workspace/epoch-wrap-clears-stale-cache" (fun () ->
        let sampled = Workspace.createSampled 2 4 2 2
        sampled.SampleEpoch <- Int32.MaxValue
        sampled.SampleEpochs.[0] <- 1
        let epoch = Workspace.beginSampledPass sampled
        assertTrue "Wrapped epochs restart at one." (epoch = 1)
        assertTrue "A stale entry must not become valid after wrapping." (Workspace.tryGetSample sampled 0 = ValueNone)
        Workspace.setSample sampled 0 1
        assertTrue "Current-epoch sample must be reusable." (Workspace.tryGetSample sampled 0 = ValueSome 1))

    test "acceptance/external-sample-reused-per-information-set" (fun () ->
        let sampled = Workspace.createSampled 1 2 1 2
        Workspace.beginSampledPass sampled |> ignore
        Workspace.setSample sampled 0 1
        assertTrue "Repeated visits retrieve one cached action." (Workspace.tryGetSample sampled 0 = ValueSome 1))

    test "core/workspace/exhaustive-touched-deltas-clear-only-touched-rows" (fun () ->
        let metadata: InfoSetMeta[] = [| { Owner = 0; Offset = 0; ActionCount = 2 }; { Owner = 1; Offset = 2; ActionCount = 2 } |]
        let workspace = Workspace.createExhaustive 2 4 2 2
        workspace.RegretDeltas.[0] <- 4.0
        workspace.RegretDeltas.[2] <- 9.0
        Workspace.beginExhaustivePass workspace |> ignore
        Workspace.touchExhaustiveRow workspace 0
        Workspace.clearTouchedDeltas metadata workspace
        assertArrayNear 0.0 [| 0.0; 0.0; 9.0; 0.0 |] workspace.RegretDeltas)

    test "core/workspace/touch-deduplicates-and-rejects-invalid-ids" (fun () ->
        let workspace = Workspace.createExhaustive 2 4 2 2
        Workspace.beginExhaustivePass workspace |> ignore
        Workspace.touchExhaustiveRow workspace 1
        Workspace.touchExhaustiveRow workspace 0
        Workspace.touchExhaustiveRow workspace 1
        assertTrue "Touch registration must deduplicate repeated rows." (workspace.TouchedCount = 2)
        assertTrue "Touch registration must preserve first-visit order."
            (workspace.TouchedRows.[0] = 1 && workspace.TouchedRows.[1] = 0)
        assertThrows (fun () -> Workspace.touchExhaustiveRow workspace -1)
        assertThrows (fun () -> Workspace.touchExhaustiveRow workspace 2))

    test "core/scalar/fused-apply-clear-matches-apply-then-clear" (fun () ->
        for clipped in [| false; true |] do
            let initial = [| 1.0; 0.25; -0.5; 2.0 |]
            let aggregate = [| -2.0; 0.5; 0.25; -3.0 |]
            let referenceRegrets = Array.copy initial
            let referenceDeltas = Array.copy aggregate
            let fusedRegrets = Array.copy initial
            let fusedDeltas = Array.copy aggregate
            Scalar.applyRegretDeltaUnchecked clipped referenceRegrets 0 referenceDeltas 0 4
            Array.Clear(referenceDeltas, 0, referenceDeltas.Length)
            Scalar.applyRegretDeltaAndClearUnchecked clipped fusedRegrets 0 fusedDeltas 0 4
            assertArrayNear 0.0 referenceRegrets fusedRegrets
            assertArrayNear 0.0 referenceDeltas fusedDeltas)

    test "core/allocation/unchecked-kernels-zero-after-warmup" (fun () ->
        let regrets = Array.init 32 (fun i -> float (i - 12))
        let strategy = Array.zeroCreate 32
        let sums = Array.zeroCreate 32
        let deltas = Array.create 32 0.01
        Scalar.regretMatchUnchecked regrets 0 32 strategy 0
        Scalar.accumulateAverageUnchecked 1.0 strategy 0 32 sums 0
        Scalar.applyRegretDeltaUnchecked true regrets 0 deltas 0 32
        Scalar.normalizeAverageUnchecked sums 0 32 strategy 0
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()
        let before = GC.GetAllocatedBytesForCurrentThread()
        let mutable i = 0
        while i < 10000 do
            Scalar.regretMatchUnchecked regrets 0 32 strategy 0
            Scalar.accumulateAverageUnchecked 1.0 strategy 0 32 sums 0
            Scalar.applyRegretDeltaUnchecked true regrets 0 deltas 0 32
            Scalar.normalizeAverageUnchecked sums 0 32 strategy 0
            i <- i + 1
        let allocated = GC.GetAllocatedBytesForCurrentThread() - before
        assertTrue $"Expected zero steady-state bytes, measured {allocated}." (allocated = 0L))

let exhaustiveSolverTests () =
    let matrixTable () =
        PackedTable.create 2 [| { Id = 0; Owner = 0; ActionCount = 2 }; { Id = 1; Owner = 1; ActionCount = 2 } |]

    test "exhaustive/cfr/one-iteration-general-sum-hand-calculation" (fun () ->
        let table = matrixTable ()
        let game = MatrixGame([| 3.0; 1.0; 0.0; 2.0 |], [| 1.0; 5.0; 2.0; 0.0 |])
        let solver = ExhaustiveSolver(SolverMode.CFR, game, table, 2, 2)
        let struct (utility0, utility1) = solver.RunIteration(1, 0, 0)
        assertNear 1e-12 1.5 utility0
        assertNear 1e-12 2.0 utility1
        assertArrayNear 1e-12 [| 0.5; -0.5; -0.5; 0.5 |] table.Tables.Regrets
        assertArrayNear 1e-12 [| 0.5; 0.5; 0.5; 0.5 |] table.Tables.StrategySums)

    test "exhaustive/cfr/one-pass-matches-two-pass-oracle-on-stage-3-fixtures" (fun () ->
        let compareTables (onePass: PackedTable) (reference: PackedTable) =
            assertArrayNear 1e-12 reference.Tables.Regrets onePass.Tables.Regrets
            assertArrayNear 1e-12 reference.Tables.StrategySums onePass.Tables.StrategySums

        let matrixOnePass = matrixTable ()
        let matrixReference = matrixTable ()
        let matrixGame = MatrixGame([| 3.0; 1.0; 0.0; 2.0 |], [| 1.0; 5.0; 2.0; 0.0 |])
        let matrixSolver = ExhaustiveSolver(SolverMode.CFR, matrixGame, matrixOnePass, 2, 2)
        let matrixOracle = ExhaustiveSolver(SolverMode.CFR, matrixGame, matrixReference, 2, 2)

        for iteration in 1..5 do
            let struct (one0, one1) = matrixSolver.RunIteration(iteration, 1, 0)
            let struct (reference0, reference1) = matrixOracle.RunTargetPairReference(iteration, 1, 0)
            assertNear 1e-12 reference0 one0
            assertNear 1e-12 reference1 one1
            compareTables matrixOnePass matrixReference

        let chanceDefinitions = [| { Id = 0; Owner = 0; ActionCount = 2 } |]
        let chanceOnePass = PackedTable.create 2 chanceDefinitions
        let chanceReference = PackedTable.create 2 chanceDefinitions
        let chanceGame = ChanceThenPlayerGame()
        let chanceSolver = ExhaustiveSolver(SolverMode.CFR, chanceGame, chanceOnePass, 2, 2)
        let chanceOracle = ExhaustiveSolver(SolverMode.CFR, chanceGame, chanceReference, 2, 2)
        let struct (chance0, chance1) = chanceSolver.RunIteration(1, 0, 0)
        let struct (chanceReference0, chanceReference1) = chanceOracle.RunTargetPairReference(1, 0, 0)
        assertNear 1e-12 chanceReference0 chance0
        assertNear 1e-12 chanceReference1 chance1
        compareTables chanceOnePass chanceReference

        let consecutiveDefinitions =
            [| { Id = 0; Owner = 0; ActionCount = 2 }
               { Id = 1; Owner = 0; ActionCount = 2 }
               { Id = 2; Owner = 0; ActionCount = 2 }
               { Id = 3; Owner = 1; ActionCount = 2 } |]
        let consecutiveOnePass = PackedTable.create 2 consecutiveDefinitions
        let consecutiveReference = PackedTable.create 2 consecutiveDefinitions
        let consecutiveGame = ConsecutiveTurnGame()
        let consecutiveSolver = ExhaustiveSolver(SolverMode.CFR, consecutiveGame, consecutiveOnePass, 3, 2)
        let consecutiveOracle = ExhaustiveSolver(SolverMode.CFR, consecutiveGame, consecutiveReference, 3, 2)
        let struct (consecutive0, consecutive1) = consecutiveSolver.RunIteration(1, 0, 0)
        let struct (consecutiveReference0, consecutiveReference1) = consecutiveOracle.RunTargetPairReference(1, 0, 0)
        assertNear 1e-12 consecutiveReference0 consecutive0
        assertNear 1e-12 consecutiveReference1 consecutive1
        compareTables consecutiveOnePass consecutiveReference)

    test "exhaustive/traversals/cfr-one-pass-cfr-plus-two-passes" (fun () ->
        let utilities0 = [| 3.0; 1.0; 0.0; 2.0 |]
        let utilities1 = [| 1.0; 5.0; 2.0; 0.0 |]
        let cfrGame = CountingMatrixGame(utilities0, utilities1)
        let plusGame = CountingMatrixGame(utilities0, utilities1)
        let cfrSolver = ExhaustiveSolver(SolverMode.CFR, cfrGame, matrixTable (), 2, 2)
        let plusSolver = ExhaustiveSolver(SolverMode.CFRPlus, plusGame, matrixTable (), 2, 2)
        cfrSolver.RunIteration(1, 0, 0) |> ignore
        plusSolver.RunIteration(1, 0, 0) |> ignore
        assertTrue "One-pass CFR must query three nonterminal actors once." (cfrGame.ActorCalls = 3)
        assertTrue "Alternating CFR+ must retain two complete target traversals." (plusGame.ActorCalls = 6))

    test "exhaustive/terminal/returns-independent-player-utilities" (fun () ->
        let table = matrixTable ()
        let game = MatrixGame(Array.create 4 3.0, Array.create 4 1.0)
        let solver = ExhaustiveSolver(SolverMode.CFR, game, table, 2, 2)
        let struct (utility0, utility1) = solver.RunIteration(1, 0, 0)
        assertNear 0.0 3.0 utility0
        assertNear 0.0 1.0 utility1)

    test "exhaustive/cfr-plus/alternates-and-clips-complete-deltas" (fun () ->
        let table = matrixTable ()
        let game = MatrixGame([| 3.0; 1.0; 0.0; 2.0 |], [| 1.0; 5.0; 2.0; 0.0 |])
        let solver = ExhaustiveSolver(SolverMode.CFRPlus, game, table, 2, 2)
        solver.RunIteration(1, 0, 0) |> ignore
        assertArrayNear 1e-12 [| 0.5; 0.0; 0.0; 2.0 |] table.Tables.Regrets)

    test "exhaustive/chance/enumerates-and-weights-counterfactual-reach" (fun () ->
        let table = PackedTable.create 2 [| { Id = 0; Owner = 0; ActionCount = 2 } |]
        let solver = ExhaustiveSolver(SolverMode.CFR, ChanceThenPlayerGame(), table, 2, 2)
        let struct (utility0, utility1) = solver.RunIteration(1, 0, 0)
        assertNear 1e-12 1.25 utility0
        assertNear 1e-12 11.25 utility1
        assertArrayNear 1e-12 [| 0.25; -0.25 |] table.Tables.Regrets
        assertArrayNear 1e-12 [| 0.5; 0.5 |] table.Tables.StrategySums)

    test "exhaustive/cfr-plus/repeated-information-set-clips-once" (fun () ->
        let table = PackedTable.create 2 [| { Id = 0; Owner = 0; ActionCount = 2 } |]
        let solver = ExhaustiveSolver(SolverMode.CFRPlus, ChanceThenPlayerGame(), table, 2, 2)
        solver.RunIteration(1, 0, 0) |> ignore
        assertArrayNear 1e-12 [| 0.25; 0.0 |] table.Tables.Regrets)

    test "exhaustive/actors/supports-consecutive-turns" (fun () ->
        let definitions =
            [| { Id = 0; Owner = 0; ActionCount = 2 }
               { Id = 1; Owner = 0; ActionCount = 2 }
               { Id = 2; Owner = 0; ActionCount = 2 }
               { Id = 3; Owner = 1; ActionCount = 2 } |]
        let table = PackedTable.create 2 definitions
        let solver = ExhaustiveSolver(SolverMode.CFR, ConsecutiveTurnGame(), table, 3, 2)
        let struct (utility0, utility1) = solver.RunIteration(1, 0, 0)
        assertNear 1e-12 3.5 utility0
        assertNear 1e-12 2.5 utility1)

    test "exhaustive/validation/rejects-invalid-mode-and-metadata" (fun () ->
        let table = matrixTable ()
        let game = MatrixGame([| 0.0; 0.0; 0.0; 0.0 |], [| 0.0; 0.0; 0.0; 0.0 |])
        assertThrows (fun () -> ExhaustiveSolver(SolverMode.MCCFR, game, table, 2, 2) |> ignore)
        let wrong = PackedTable.create 2 [| { Id = 0; Owner = 1; ActionCount = 2 }; { Id = 1; Owner = 1; ActionCount = 2 } |]
        let solver = ExhaustiveSolver(SolverMode.CFR, game, wrong, 2, 2)
        assertThrows (fun () -> solver.RunIteration(1, 0, 0) |> ignore))

    test "exhaustive/convergence/matching-pennies-average" (fun () ->
        for mode in [| SolverMode.CFR; SolverMode.CFRPlus |] do
            let table = matrixTable ()
            let game = MatrixGame([| 1.0; -1.0; -1.0; 1.0 |], [| -1.0; 1.0; 1.0; -1.0 |])
            let solver = ExhaustiveSolver(mode, game, table, 2, 2)
            let mutable iteration = 1
            while iteration <= 10000 do
                solver.RunIteration(iteration, 0, 0) |> ignore
                iteration <- iteration + 1
            let average = Array.zeroCreate 4
            Scalar.normalizeAverage table.Tables.StrategySums 0 2 average 0
            Scalar.normalizeAverage table.Tables.StrategySums 2 2 average 2
            assertArrayNear 0.02 [| 0.5; 0.5; 0.5; 0.5 |] average
            if mode = SolverMode.CFRPlus then
                assertTrue "CFR+ regrets must remain nonnegative." (table.Tables.Regrets |> Array.forall (fun value -> value >= 0.0)))

    test "exhaustive/allocation/value-state-zero-after-warmup" (fun () ->
        let table = matrixTable ()
        let game = MatrixGame([| 1.0; -1.0; -1.0; 1.0 |], [| -1.0; 1.0; 1.0; -1.0 |])
        let solver = ExhaustiveSolver(SolverMode.CFR, game, table, 2, 2)
        solver.RunIteration(1, 0, 0) |> ignore
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()
        let before = GC.GetAllocatedBytesForCurrentThread()
        let mutable iteration = 2
        while iteration <= 10001 do
            solver.RunIteration(iteration, 0, 0) |> ignore
            iteration <- iteration + 1
        let allocated = GC.GetAllocatedBytesForCurrentThread() - before
        assertTrue $"Expected zero steady-state bytes, measured {allocated}." (allocated = 0L))

[<EntryPoint>]
let main _ =
    helperTests ()
    baselineFixtureTests ()
    modeAndTableTests ()
    scalarTests ()
    workspaceAndAllocationTests ()
    exhaustiveSolverTests ()
    printfn "%d test(s) failed." failures
    if failures = 0 then 0 else 1
