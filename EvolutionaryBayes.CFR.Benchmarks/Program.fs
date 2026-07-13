module EvolutionaryBayes.CFR.Benchmarks

open System
open System.Collections.Generic
open System.Diagnostics
open System.Runtime.InteropServices
open EvolutionaryBayes.CFR
open EvolutionaryBayes.CFRCore

type Measurement =
    { Traversal: string
      Actions: int
      Depth: int
      Iterations: int
      Nodes: int64
      Elapsed: TimeSpan
      Allocated: int64 }

type CoreMeasurement =
    { Actions: int
      Iterations: int
      Elapsed: TimeSpan
      Allocated: int64 }

type LegacyMemoryMeasurement =
    { Actions: int
      Depth: int
      InfoSets: int
      NumericPayload: int64
      ManagedRetained: int64
      ConstructionAllocated: int64 }

type Stage3aMeasurement =
    { Solver: string
      NodeVisits: int64
      MedianMilliseconds: double
      MedianNodesPerSecond: double
      MedianAllocated: int64 }

type OptimizationMeasurement =
    { Variant: string
      MedianMilliseconds: double
      MedianAllocated: int64 }

[<Struct>]
type BenchmarkState =
    { Node: int
      Depth: int
      LastAction: int }

[<Struct>]
type BenchmarkGame =
    val Actions: int
    val TerminalDepth: int

    new(actions, terminalDepth) =
        { Actions = actions
          TerminalDepth = terminalDepth }

    interface IExhaustiveGame<BenchmarkState> with
        member this.TryGetTerminalUtility(state, targetPlayer, utility: byref<double>) =
            if state.Depth = this.TerminalDepth then
                utility <- if state.LastAction % 2 = targetPlayer then 1.0 else -1.0
                true
            else
                false

        member _.Actor state = state.Depth &&& 1
        member _.InformationSetId state = state.Node
        member this.ActionCount _ = this.Actions
        member this.NextState(state, action) =
            { Node = 1 + state.Node * this.Actions + action
              Depth = state.Depth + 1
              LastAction = action }
        member _.ChanceProbability(_, _) = invalidOp "This benchmark has no chance nodes."

let lookup actionCount (nodes: Dictionary<string, StrategyNode>) key =
    match nodes.TryGetValue key with
    | true, node -> node
    | false, _ ->
        let node = newStrategyNode actionCount
        nodes.Add(key, node)
        node

let run sampled actionCount depth iterations =
    let execute measured =
        let nodes = Dictionary<string, StrategyNode>()
        let random = Random 1729
        let actions = Array.init actionCount (fun i -> string (char (65 + i)))
        let contexts = [| "P0"; "P1" |]
        let mutable visited = 0L
        let reward (_: string[]) player (history: string) =
            visited <- visited + 1L
            if history.Length = depth then Some(if (history.[history.Length - 1] |> int) % 2 = player then 1.0 else -1.0)
            else None
        let adjust (_: string[]) _ action history = history + action
        let mask _ (_: string[]) _ = Array.create actionCount true
        let before = if measured then GC.GetAllocatedBytesForCurrentThread() else 0L
        let stopwatch = Stopwatch.StartNew()
        let mutable iteration = 1
        while iteration <= iterations do
            if sampled then
                cfrSampled random 0 iteration 0 0 1.0 1.0 reward lookup adjust mask id nodes contexts actions "" |> ignore
            else
                cfr iteration 0 0 1.0 1.0 reward lookup adjust mask id nodes contexts actions "" |> ignore
            iteration <- iteration + 1
        stopwatch.Stop()
        let allocated = if measured then GC.GetAllocatedBytesForCurrentThread() - before else 0L
        visited, stopwatch.Elapsed, allocated

    execute false |> ignore
    let nodes, elapsed, allocated = execute true
    { Traversal = if sampled then "sampled" else "exhaustive"
      Actions = actionCount
      Depth = depth
      Iterations = iterations
      Nodes = nodes
      Elapsed = elapsed
      Allocated = allocated }

let buildLegacyNodes actionCount depth =
    let nodes = Dictionary<string, StrategyNode>()
    let actions = Array.init actionCount (fun i -> string (char (65 + i)))
    let contexts = [| "P0"; "P1" |]
    let reward (_: string[]) player (history: string) =
        if history.Length = depth then
            Some(if (history.[history.Length - 1] |> int) % 2 = player then 1.0 else -1.0)
        else
            None
    let adjust (_: string[]) _ action history = history + action
    let mask _ (_: string[]) _ = Array.create actionCount true
    cfr 1 0 0 1.0 1.0 reward lookup adjust mask id nodes contexts actions "" |> ignore
    nodes

let measureLegacyMemory actionCount depth =
    let mutable warmupNodes = buildLegacyNodes actionCount depth
    GC.KeepAlive warmupNodes
    warmupNodes <- null
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let beforeRetained = GC.GetTotalMemory(false)
    let beforeAllocated = GC.GetAllocatedBytesForCurrentThread()
    let nodes = buildLegacyNodes actionCount depth
    let constructionAllocated = GC.GetAllocatedBytesForCurrentThread() - beforeAllocated
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let afterRetained = GC.GetTotalMemory(false)
    let retained = max 0L (afterRetained - beforeRetained)
    let numericPayload = int64 nodes.Count * int64 actionCount * 16L
    let result =
        { Actions = actionCount
          Depth = depth
          InfoSets = nodes.Count
          NumericPayload = numericPayload
          ManagedRetained = retained
          ConstructionAllocated = constructionAllocated }
    GC.KeepAlive nodes
    result

let revision (arguments: string[]) =
    arguments
    |> Array.tryFindIndex ((=) "--revision")
    |> Option.bind (fun index -> if index + 1 < arguments.Length then Some arguments.[index + 1] else None)
    |> Option.defaultValue "working-tree"

let runCore actionCount iterations =
    let regrets = Array.init actionCount (fun i -> float (i - actionCount / 3))
    let strategy = Array.zeroCreate actionCount
    let strategySums = Array.zeroCreate actionCount
    let deltas = Array.init actionCount (fun i -> if i % 2 = 0 then 0.001 else -0.001)

    let cycle () =
        Scalar.regretMatchUnchecked regrets 0 actionCount strategy 0
        Scalar.accumulateAverageUnchecked 1.0 strategy 0 actionCount strategySums 0
        Scalar.applyRegretDeltaUnchecked true regrets 0 deltas 0 actionCount
        Scalar.normalizeAverageUnchecked strategySums 0 actionCount strategy 0

    cycle ()
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
    let stopwatch = Stopwatch()
    let before = GC.GetAllocatedBytesForCurrentThread()
    stopwatch.Start()
    let mutable iteration = 0
    while iteration < iterations do
        cycle ()
        iteration <- iteration + 1
    stopwatch.Stop()
    let allocated = GC.GetAllocatedBytesForCurrentThread() - before
    { Actions = actionCount
      Iterations = iterations
      Elapsed = stopwatch.Elapsed
      Allocated = allocated }

let memoryPayload () =
    let infoSetCount = 250000
    let slotCount = 1000000
    let definitions =
        Array.init infoSetCount (fun id ->
            { Id = id
              Owner = id &&& 1
              ActionCount = 4 })
    let table = PackedTable.create 2 definitions
    let exhaustive = Workspace.createExhaustive infoSetCount slotCount 64 32
    let sampled = Workspace.createSampled infoSetCount 65536 64 32
    let doubleBytes (values: double[]) = int64 values.Length * 8L
    let intBytes (values: int[]) = int64 values.Length * 4L
    let commonBytes scratch =
        doubleBytes scratch.Strategies
        + doubleBytes scratch.Utilities
        + intBytes scratch.AverageEpochs
    let persistent = doubleBytes table.Tables.Regrets + doubleBytes table.Tables.StrategySums
    let metadata = int64 table.InfoSets.Length * int64 (Marshal.SizeOf<InfoSetMeta>())
    let exhaustiveOnly =
        doubleBytes exhaustive.RegretDeltas
        + intBytes exhaustive.TouchedRows
        + intBytes exhaustive.TouchedEpochs
    let sampledOnly =
        intBytes sampled.SampledActions
        + intBytes sampled.SampleEpochs
        + intBytes sampled.DeltaIndices
        + doubleBytes sampled.DeltaValues
    persistent, metadata, commonBytes exhaustive.Common, exhaustiveOnly, sampledOnly

let median (values: 'T[]) =
    Array.sortInPlace values
    values.[values.Length / 2]

let private stage3Definitions actions depth =
    let mutable levelCount = 1
    let mutable total = 0
    let mutable level = 0
    while level < depth do
        total <- total + levelCount
        levelCount <- levelCount * actions
        level <- level + 1
    Array.init total (fun id ->
        let mutable first = 0
        let mutable count = 1
        let mutable owner = 0
        while id >= first + count do
            first <- first + count
            count <- count * actions
            owner <- owner + 1
        { Id = id; Owner = owner &&& 1; ActionCount = actions })

let stage3NodeCount actions depth =
    let mutable level = 1L
    let mutable total = 0L
    let mutable d = 0
    while d <= depth do
        total <- total + level
        level <- level * int64 actions
        d <- d + 1
    total

let runStage3aComparison actions depth iterations repetitions =
    let referenceTimes = Array.zeroCreate repetitions
    let referenceRates = Array.zeroCreate repetitions
    let referenceAllocations = Array.zeroCreate repetitions
    let optimizedTimes = Array.zeroCreate repetitions
    let optimizedRates = Array.zeroCreate repetitions
    let optimizedAllocations = Array.zeroCreate repetitions
    let nodesPerTraversal = stage3NodeCount actions depth

    let run reference measured =
        let table = PackedTable.create 2 (stage3Definitions actions depth)
        let game = BenchmarkGame(actions, depth)
        let solver = ExhaustiveSolver(SolverMode.CFR, game, table, depth, actions)
        let root = { Node = 0; Depth = 0; LastAction = 0 }
        let execute iteration =
            if reference then solver.RunTargetPairReference(iteration, 0, root)
            else solver.RunIteration(iteration, 0, root)
        let mutable warmup = 1
        while warmup <= 100 do
            execute warmup |> ignore
            warmup <- warmup + 1
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()
        let before = GC.GetAllocatedBytesForCurrentThread()
        let started = Stopwatch.GetTimestamp()
        let mutable iteration = 101
        while iteration <= iterations + 100 do
            execute iteration |> ignore
            iteration <- iteration + 1
        let stopped = Stopwatch.GetTimestamp()
        let allocated = GC.GetAllocatedBytesForCurrentThread() - before
        let milliseconds = float (stopped - started) * 1000.0 / float Stopwatch.Frequency
        if measured then milliseconds, allocated else 0.0, 0L

    run true false |> ignore
    run false false |> ignore
    let mutable repetition = 0
    while repetition < repetitions do
        let recordReference () =
            let milliseconds, allocated = run true true
            referenceTimes.[repetition] <- milliseconds
            referenceAllocations.[repetition] <- allocated
            referenceRates.[repetition] <-
                float (2L * nodesPerTraversal * int64 iterations) / (milliseconds / 1000.0)
        let recordOptimized () =
            let milliseconds, allocated = run false true
            optimizedTimes.[repetition] <- milliseconds
            optimizedAllocations.[repetition] <- allocated
            optimizedRates.[repetition] <-
                float (nodesPerTraversal * int64 iterations) / (milliseconds / 1000.0)
        if repetition % 2 = 0 then
            recordReference ()
            recordOptimized ()
        else
            recordOptimized ()
            recordReference ()
        repetition <- repetition + 1

    [| { Solver = "Stage 3 two target passes"
         NodeVisits = 2L * nodesPerTraversal * int64 iterations
         MedianMilliseconds = median referenceTimes
         MedianNodesPerSecond = median referenceRates
         MedianAllocated = median referenceAllocations }
       { Solver = "Stage 3a one pass"
         NodeVisits = nodesPerTraversal * int64 iterations
         MedianMilliseconds = median optimizedTimes
         MedianNodesPerSecond = median optimizedRates
         MedianAllocated = median optimizedAllocations } |]

let runBoundaryComparison actions depth iterations repetitions =
    let definitions = stage3Definitions actions depth
    let metadata = (PackedTable.create 2 definitions).InfoSets
    let slotCount = metadata |> Array.sumBy (fun row -> row.ActionCount)

    let measure fused =
        let regrets = Array.zeroCreate slotCount
        let deltas = Array.create slotCount 0.001
        let started = Stopwatch.GetTimestamp()
        let before = GC.GetAllocatedBytesForCurrentThread()
        let mutable iteration = 0
        while iteration < iterations do
            let mutable rowIndex = 0
            while rowIndex < metadata.Length do
                let row = metadata.[rowIndex]
                if fused then
                    Scalar.applyRegretDeltaAndClearUnchecked false regrets row.Offset deltas row.Offset row.ActionCount
                else
                    Scalar.applyRegretDeltaUnchecked false regrets row.Offset deltas row.Offset row.ActionCount
                rowIndex <- rowIndex + 1
            if not fused then
                rowIndex <- 0
                while rowIndex < metadata.Length do
                    let row = metadata.[rowIndex]
                    Array.Clear(deltas, row.Offset, row.ActionCount)
                    rowIndex <- rowIndex + 1
            let mutable slot = 0
            while slot < deltas.Length do
                deltas.[slot] <- 0.001
                slot <- slot + 1
            iteration <- iteration + 1
        let allocated = GC.GetAllocatedBytesForCurrentThread() - before
        let stopped = Stopwatch.GetTimestamp()
        float (stopped - started) * 1000.0 / float Stopwatch.Frequency, allocated

    let separateTimes = Array.zeroCreate repetitions
    let separateAllocations = Array.zeroCreate repetitions
    let fusedTimes = Array.zeroCreate repetitions
    let fusedAllocations = Array.zeroCreate repetitions
    let mutable repetition = 0
    while repetition < repetitions do
        let record fused =
            let milliseconds, allocated = measure fused
            if fused then
                fusedTimes.[repetition] <- milliseconds
                fusedAllocations.[repetition] <- allocated
            else
                separateTimes.[repetition] <- milliseconds
                separateAllocations.[repetition] <- allocated
        if repetition % 2 = 0 then record false; record true
        else record true; record false
        repetition <- repetition + 1
    [| { Variant = "apply, then clear touched rows"
         MedianMilliseconds = median separateTimes
         MedianAllocated = median separateAllocations }
       { Variant = "fused apply-and-clear"
         MedianMilliseconds = median fusedTimes
         MedianAllocated = median fusedAllocations } |]

[<EntryPoint>]
let main arguments =
    let counts = [| 2; 3; 4; 8; 16; 32 |]
    let results = ResizeArray<Measurement>()
    for depth in [| 2; 3 |] do
        for actions in counts do
            let estimate = pown actions depth
            let iterations = max 1 (20000 / estimate)
            results.Add(run false actions depth iterations)
            results.Add(run true actions depth (max 2 (iterations * 2)))

    printfn "# CFR benchmark run"
    printfn ""
    printfn "- Source revision: `%s`" (revision arguments)
    printfn "- Build configuration: `Release`"
    printfn "- Runtime: `%s`" RuntimeInformation.FrameworkDescription
    printfn "- OS: `%s`" RuntimeInformation.OSDescription
    printfn "- Processor: `%s`" (Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER"))
    printfn "- Logical processors: `%d`" Environment.ProcessorCount
    printfn "- Seed: `1729`"
    printfn ""
    printfn "## Stage 0 legacy retained memory"
    printfn ""
    printfn "Managed retained bytes are the forced-GC delta while the legacy dictionary remains live; numeric payload is exact for the two double arrays in each StrategyNode."
    printfn ""
    printfn "| Actions | Depth | Information sets | Numeric array payload | Managed retained bytes | Construction allocated bytes |"
    printfn "| ---: | ---: | ---: | ---: | ---: | ---: |"
    for struct (actions, depth) in [| struct (2, 14); struct (3, 9); struct (4, 7); struct (8, 5); struct (16, 4); struct (32, 3) |] do
        let result = measureLegacyMemory actions depth
        printfn "| %d | %d | %d | %d | %d | %d |" result.Actions result.Depth result.InfoSets result.NumericPayload result.ManagedRetained result.ConstructionAllocated
    printfn ""
    printfn "## Baseline — Stage 1 legacy solver"
    printfn ""
    printfn "| Traversal | Actions | Depth | Iterations | Nodes | Elapsed ms | Nodes/s | Allocated bytes | Bytes/iteration |"
    printfn "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    for result in results do
        let seconds = max result.Elapsed.TotalSeconds 1e-9
        printfn "| %s | %d | %d | %d | %d | %.3f | %.0f | %d | %.1f |"
            result.Traversal result.Actions result.Depth result.Iterations result.Nodes result.Elapsed.TotalMilliseconds
            (float result.Nodes / seconds) result.Allocated (float result.Allocated / float result.Iterations)

    let persistent, metadata, common, exhaustiveOnly, sampledOnly = memoryPayload ()
    printfn ""
    printfn "## Stage 2 flat scalar core"
    printfn ""
    printfn "Memory case: 250,000 information sets, 1,000,000 legal slots, maximum depth 64, maximum 32 actions, sampled delta capacity 65,536."
    printfn ""
    printfn "| Component | Payload bytes |"
    printfn "| --- | ---: |"
    printfn "| Persistent regrets + strategy sums | %d |" persistent
    printfn "| Information-set metadata | %d |" metadata
    printfn "| Common traversal scratch | %d |" common
    printfn "| Exhaustive-only workspace | %d |" exhaustiveOnly
    printfn "| Sampled-only workspace | %d |" sampledOnly
    printfn "| Exhaustive total including persistent + metadata | %d |" (persistent + metadata + common + exhaustiveOnly)
    printfn "| Sampled total including persistent + metadata | %d |" (persistent + metadata + common + sampledOnly)
    printfn ""
    printfn "Each row cycle runs regret matching, average accumulation, clipped delta application, and average normalization."
    printfn ""
    printfn "| Actions | Iterations | Elapsed ms | Row cycles/s | Logical action-slots/s | Allocated bytes |"
    printfn "| ---: | ---: | ---: | ---: | ---: | ---: |"
    for actions in [| 1; 2; 3; 4; 8; 16; 32 |] do
        let iterations = max 10000 (20000000 / actions)
        let result = runCore actions iterations
        let seconds = max result.Elapsed.TotalSeconds 1e-9
        printfn "| %d | %d | %.3f | %.0f | %.0f | %d |"
            actions iterations result.Elapsed.TotalMilliseconds
            (float iterations / seconds) (float (int64 iterations * int64 actions * 4L) / seconds) result.Allocated
    printfn ""
    printfn "## Stage 3a exhaustive hot-path optimization"
    printfn ""
    printfn "Interleaved seven-run medians; 4 actions, depth 5, 500 measured iterations."
    printfn ""
    printfn "| Solver | Node visits | Median elapsed ms | Median nodes/s | Median allocated bytes |"
    printfn "| --- | ---: | ---: | ---: | ---: |"
    for result in runStage3aComparison 4 5 500 7 do
        printfn "| %s | %d | %.3f | %.0f | %d |" result.Solver result.NodeVisits result.MedianMilliseconds result.MedianNodesPerSecond result.MedianAllocated
    printfn ""
    printfn "Boundary microbenchmark: 341 rows, 1,364 slots, 20,000 boundaries; medians of seven interleaved runs. Both variants include restoring deltas for the next boundary."
    printfn ""
    printfn "| Variant | Median elapsed ms | Median allocated bytes |"
    printfn "| --- | ---: | ---: |"
    for result in runBoundaryComparison 4 5 20000 7 do
        printfn "| %s | %.3f | %d |" result.Variant result.MedianMilliseconds result.MedianAllocated
    0
