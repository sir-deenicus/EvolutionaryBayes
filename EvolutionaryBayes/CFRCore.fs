module EvolutionaryBayes.CFRCore

open System

type SolverMode =
    | CFR
    | CFRPlus
    | MCCFR
    | MCCFRPlus

type TraversalKind =
    | Exhaustive
    | ExternalSampling

type RegretTransform =
    | Signed
    | Clipped

type AveragingKind =
    | Uniform
    | Linear

type UpdateSchedule =
    | Simultaneous
    | Alternating

[<Struct>]
type ModeSemantics =
    { Traversal: TraversalKind
      RegretTransform: RegretTransform
      Averaging: AveragingKind
      UpdateSchedule: UpdateSchedule }

module Mode =
    let semantics mode =
        match mode with
        | CFR ->
            { Traversal = Exhaustive
              RegretTransform = Signed
              Averaging = Uniform
              UpdateSchedule = Simultaneous }
        | CFRPlus ->
            { Traversal = Exhaustive
              RegretTransform = Clipped
              Averaging = Linear
              UpdateSchedule = Alternating }
        | MCCFR ->
            { Traversal = ExternalSampling
              RegretTransform = Signed
              Averaging = Uniform
              UpdateSchedule = Simultaneous }
        | MCCFRPlus ->
            { Traversal = ExternalSampling
              RegretTransform = Clipped
              Averaging = Linear
              UpdateSchedule = Alternating }

    let averageWeight mode iteration burnIn =
        if iteration <= burnIn then
            0.0
        else
            match (semantics mode).Averaging with
            | Uniform -> 1.0
            | Linear -> float (iteration - burnIn)

[<Struct>]
type InfoSetDefinition =
    { Id: int
      Owner: int
      ActionCount: int }

[<Struct>]
type TrainingProgress =
    { IterationsRun: int
      IterationsCompleted: int
      MeanUtilities: double[] }

    member this.MeanUtility0 = this.MeanUtilities.[0]
    member this.MeanUtility1 = this.MeanUtilities.[1]

type TrainingStopReason =
    | IterationLimit = 0
    | ConvergenceTolerance = 1

[<Struct>]
type TrainingResult =
    { IterationsRun: int
      IterationsCompleted: int
      MeanUtilities: double[]
      StopReason: TrainingStopReason
      ConvergenceError: double voption
      ConvergenceChecks: int }

    member this.MeanUtility0 = this.MeanUtilities.[0]
    member this.MeanUtility1 = this.MeanUtilities.[1]

    member this.Converged =
        this.StopReason = TrainingStopReason.ConvergenceTolerance

[<Struct>]
type ConvergenceCheck =
    { Tolerance: double
      CheckEvery: int
      RequiredConsecutiveChecks: int }

module ConvergenceCheck =
    let private isFinite value =
        not (Double.IsNaN value || Double.IsInfinity value)

    let internal validate check =
        if not (isFinite check.Tolerance) || check.Tolerance < 0.0 then
            invalidArg "check" "Convergence tolerance must be finite and non-negative."

        if check.CheckEvery <= 0 then
            invalidArg "check" "Convergence check interval must be positive."

        if check.RequiredConsecutiveChecks <= 0 then
            invalidArg "check" "Required consecutive convergence checks must be positive."

    let create tolerance checkEvery =
        let check =
            { Tolerance = tolerance
              CheckEvery = checkEvery
              RequiredConsecutiveChecks = 1 }

        validate check
        check

    let withConsecutiveChecks requiredChecks check =
        let updated =
            { check with
                RequiredConsecutiveChecks = requiredChecks }

        validate updated
        updated

module Convergence =
    /// Error of one player's cumulative mean training utility from a known
    /// value. For general games, prefer an exploitability or NashConv measure.
    let utilityError player expectedValue =
        if player < 0 then
            invalidArg "player" "The player index cannot be negative."

        if Double.IsNaN expectedValue || Double.IsInfinity expectedValue then
            invalidArg "expectedValue" "Expected utility must be finite."

        fun (progress: TrainingProgress) ->
            if player >= progress.MeanUtilities.Length then
                invalidArg "player" "The player index is outside this training result."

            abs (progress.MeanUtilities.[player] - expectedValue)

[<Struct>]
type InfoSetMeta =
    { Owner: int
      Offset: int
      ActionCount: int }

type SolverTables =
    { Regrets: double[]
      StrategySums: double[] }

type PackedTable =
    { PlayerCount: int
      InfoSets: InfoSetMeta[]
      Tables: SolverTables
      SlotCount: int }

module PackedTable =
    let private validatePlayerCount playerCount =
        if playerCount <= 0 then
            invalidArg "playerCount" "Player count must be positive."

    let private checkedSlotCount (metadata: InfoSetMeta[]) =
        let mutable expectedOffset = 0L
        let mutable i = 0

        while i < metadata.Length do
            let row = metadata.[i]

            if row.ActionCount <= 0 then
                invalidArg "metadata" $"Information set {i} has no actions."

            if int64 row.Offset <> expectedOffset then
                invalidArg
                    "metadata"
                    $"Information set {i} starts at {row.Offset}; expected contiguous offset {expectedOffset}."

            expectedOffset <- expectedOffset + int64 row.ActionCount

            if expectedOffset > int64 Int32.MaxValue then
                invalidArg "metadata" "The packed table exceeds the maximum array length."

            i <- i + 1

        int expectedOffset

    let ofMetadata playerCount (metadata: InfoSetMeta[]) =
        validatePlayerCount playerCount

        if isNull metadata then
            nullArg "metadata"

        let mutable i = 0

        while i < metadata.Length do
            let owner = metadata.[i].Owner

            if owner < 0 || owner >= playerCount then
                invalidArg "metadata" $"Information set {i} has invalid owner {owner}."

            i <- i + 1

        let slotCount = checkedSlotCount metadata

        { PlayerCount = playerCount
          InfoSets = Array.copy metadata
          Tables =
            { Regrets = Array.zeroCreate slotCount
              StrategySums = Array.zeroCreate slotCount }
          SlotCount = slotCount }

    let create playerCount (definitions: InfoSetDefinition[]) =
        validatePlayerCount playerCount

        if isNull definitions then
            nullArg "definitions"

        let count = definitions.Length
        let seen = Array.zeroCreate<bool> count
        let owners = Array.zeroCreate<int> count
        let actionCounts = Array.zeroCreate<int> count
        let mutable i = 0

        while i < count do
            let definition = definitions.[i]

            if definition.Id < 0 || definition.Id >= count then
                invalidArg
                    "definitions"
                    $"Information-set ID {definition.Id} is outside the dense range 0..{count - 1}."

            if seen.[definition.Id] then
                invalidArg
                    "definitions"
                    $"Information-set ID {definition.Id} is duplicated, possibly across players."

            if definition.Owner < 0 || definition.Owner >= playerCount then
                invalidArg
                    "definitions"
                    $"Information set {definition.Id} has invalid owner {definition.Owner}."

            if definition.ActionCount <= 0 then
                invalidArg
                    "definitions"
                    $"Information set {definition.Id} must have at least one action."

            seen.[definition.Id] <- true
            owners.[definition.Id] <- definition.Owner
            actionCounts.[definition.Id] <- definition.ActionCount
            i <- i + 1

        let metadata = Array.zeroCreate<InfoSetMeta> count
        let mutable offset = 0L
        i <- 0

        while i < count do
            if not seen.[i] then
                invalidArg "definitions" $"Dense information-set ID {i} is missing."

            metadata.[i] <-
                { Owner = owners.[i]
                  Offset = int offset
                  ActionCount = actionCounts.[i] }

            offset <- offset + int64 actionCounts.[i]

            if offset > int64 Int32.MaxValue then
                invalidArg "definitions" "The packed table exceeds the maximum array length."

            i <- i + 1

        ofMetadata playerCount metadata

type TraversalScratch =
    { Strategies: double[]
      Utilities: double[]
      AverageEpochs: int[]
      mutable Epoch: int }

type ExhaustiveWorkspace =
    { Common: TraversalScratch
      RegretDeltas: double[]
      TouchedRows: int[]
      TouchedEpochs: int[]
      mutable RegretEpoch: int
      mutable TouchedCount: int }

type SampledWorkspace =
    { Common: TraversalScratch
      SampledActions: int[]
      SampleEpochs: int[]
      DeltaIndices: int[]
      DeltaValues: double[]
      mutable SampleEpoch: int
      mutable DeltaCount: int }

module Workspace =
    let private scratchLength maxDepth maxActionCount =
        if maxDepth <= 0 then
            invalidArg "maxDepth" "Maximum depth must be positive."

        if maxActionCount <= 0 then
            invalidArg "maxActionCount" "Maximum action count must be positive."

        let length = int64 maxDepth * int64 maxActionCount

        if length > int64 Int32.MaxValue then
            invalidArg "maxDepth" "Depth scratch exceeds the maximum array length."

        int length

    let createTraversal infoSetCount maxDepth maxActionCount =
        if infoSetCount < 0 then
            invalidArg "infoSetCount" "Information-set count cannot be negative."

        let length = scratchLength maxDepth maxActionCount

        { Strategies = Array.zeroCreate length
          Utilities = Array.zeroCreate length
          AverageEpochs = Array.zeroCreate infoSetCount
          Epoch = 0 }

    let private advanceEpoch (epochs: int[]) epoch =
        if epoch = Int32.MaxValue then
            Array.Clear(epochs, 0, epochs.Length)
            1
        else
            epoch + 1

    let beginAveragePass scratch =
        scratch.Epoch <- advanceEpoch scratch.AverageEpochs scratch.Epoch
        scratch.Epoch

    let createExhaustive infoSetCount slotCount maxDepth maxActionCount =
        if slotCount < 0 then
            invalidArg "slotCount" "Slot count cannot be negative."

        { Common = createTraversal infoSetCount maxDepth maxActionCount
          RegretDeltas = Array.zeroCreate slotCount
          TouchedRows = Array.zeroCreate infoSetCount
          TouchedEpochs = Array.zeroCreate infoSetCount
          RegretEpoch = 0
          TouchedCount = 0 }

    let beginExhaustivePass workspace =
        workspace.RegretEpoch <- advanceEpoch workspace.TouchedEpochs workspace.RegretEpoch
        workspace.TouchedCount <- 0
        workspace.RegretEpoch

    let touchExhaustiveRow workspace infoSetId =
        if infoSetId < 0 || infoSetId >= workspace.TouchedEpochs.Length then
            invalidArg "infoSetId" "Information-set ID is outside the workspace."

        if workspace.TouchedEpochs.[infoSetId] <> workspace.RegretEpoch then
            workspace.TouchedEpochs.[infoSetId] <- workspace.RegretEpoch
            workspace.TouchedRows.[workspace.TouchedCount] <- infoSetId
            workspace.TouchedCount <- workspace.TouchedCount + 1

    let clearTouchedDeltas (metadata: InfoSetMeta[]) workspace =
        let mutable touched = 0

        while touched < workspace.TouchedCount do
            let infoSetId = workspace.TouchedRows.[touched]
            let row = metadata.[infoSetId]
            Array.Clear(workspace.RegretDeltas, row.Offset, row.ActionCount)
            touched <- touched + 1

        workspace.TouchedCount <- 0

    let createSampled infoSetCount deltaCapacity maxDepth maxActionCount =
        if deltaCapacity < 0 then
            invalidArg "deltaCapacity" "Delta capacity cannot be negative."

        { Common = createTraversal infoSetCount maxDepth maxActionCount
          SampledActions = Array.zeroCreate infoSetCount
          SampleEpochs = Array.zeroCreate infoSetCount
          DeltaIndices = Array.zeroCreate deltaCapacity
          DeltaValues = Array.zeroCreate deltaCapacity
          SampleEpoch = 0
          DeltaCount = 0 }

    let beginSampledPass workspace =
        workspace.SampleEpoch <- advanceEpoch workspace.SampleEpochs workspace.SampleEpoch
        workspace.SampleEpoch

    let tryGetSample workspace infoSetId =
        if infoSetId < 0 || infoSetId >= workspace.SampleEpochs.Length then
            invalidArg "infoSetId" "Information-set ID is outside the workspace."

        if workspace.SampleEpochs.[infoSetId] = workspace.SampleEpoch then
            ValueSome workspace.SampledActions.[infoSetId]
        else
            ValueNone

    let setSample workspace infoSetId action =
        if infoSetId < 0 || infoSetId >= workspace.SampleEpochs.Length then
            invalidArg "infoSetId" "Information-set ID is outside the workspace."

        workspace.SampledActions.[infoSetId] <- action
        workspace.SampleEpochs.[infoSetId] <- workspace.SampleEpoch

    let resetDeltaLog workspace = workspace.DeltaCount <- 0

    let appendDelta workspace index value =
        if workspace.DeltaCount = workspace.DeltaIndices.Length then
            invalidOp "The sampled delta log capacity was exceeded."

        let position = workspace.DeltaCount
        workspace.DeltaIndices.[position] <- index
        workspace.DeltaValues.[position] <- value
        workspace.DeltaCount <- position + 1

    /// Drains a sampled delta log. Clipped updates are sorted and aggregated by
    /// slot so each cumulative regret is clipped exactly once per boundary.
    let applySampledDeltas clipped (regrets: double[]) workspace =
        let count = workspace.DeltaCount

        if clipped && count > 1 then
            Array.Sort(workspace.DeltaIndices, workspace.DeltaValues, 0, count)

        if clipped then
            let mutable position = 0

            while position < count do
                let index = workspace.DeltaIndices.[position]
                let mutable delta = workspace.DeltaValues.[position]
                position <- position + 1

                while position < count && workspace.DeltaIndices.[position] = index do
                    delta <- delta + workspace.DeltaValues.[position]
                    position <- position + 1

                let updated = regrets.[index] + delta
                regrets.[index] <- if updated < 0.0 then 0.0 else updated
        else
            let mutable position = 0

            while position < count do
                let index = workspace.DeltaIndices.[position]
                regrets.[index] <- regrets.[index] + workspace.DeltaValues.[position]
                position <- position + 1

        resetDeltaLog workspace

module Scalar =
    let inline private isFinite value =
        not (Double.IsNaN value || Double.IsInfinity value)

    let private validateRow arrayName (values: double[]) offset actionCount =
        if isNull values then
            nullArg arrayName

        if offset < 0 || actionCount <= 0 || offset > values.Length - actionCount then
            invalidArg arrayName "The requested row is outside the array."

    let private validateFiniteRow arrayName (values: double[]) offset actionCount =
        validateRow arrayName values offset actionCount
        let mutable i = 0

        while i < actionCount do
            if not (isFinite values.[offset + i]) then
                invalidArg arrayName "Rows passed to checked kernels must contain only finite values."

            i <- i + 1

    let private validateNonnegativeFiniteRow arrayName (values: double[]) offset actionCount =
        validateFiniteRow arrayName values offset actionCount
        let mutable i = 0

        while i < actionCount do
            if values.[offset + i] < 0.0 then
                invalidArg arrayName "Probability and average-strategy rows cannot contain negative values."

            i <- i + 1

    let regretMatchUnchecked
        (regrets: double[])
        regretOffset
        actionCount
        (strategy: double[])
        strategyOffset
        =
        let mutable maximum = 0.0
        let mutable scaledTotal = 0.0
        let mutable i = 0

        while i < actionCount do
            let regret = regrets.[regretOffset + i]

            if regret > 0.0 then
                if regret > maximum then
                    if maximum = 0.0 then
                        scaledTotal <- 1.0
                    else
                        scaledTotal <- scaledTotal * (maximum / regret) + 1.0

                    maximum <- regret
                else
                    scaledTotal <- scaledTotal + regret / maximum

            i <- i + 1

        if maximum = 0.0 then
            let probability = 1.0 / float actionCount
            i <- 0

            while i < actionCount do
                strategy.[strategyOffset + i] <- probability
                i <- i + 1
        else
            i <- 0

            while i < actionCount do
                let regret = regrets.[regretOffset + i]
                strategy.[strategyOffset + i] <-
                    if regret > 0.0 then (regret / maximum) / scaledTotal else 0.0

                i <- i + 1

    let regretMatch regrets regretOffset actionCount strategy strategyOffset =
        validateFiniteRow "regrets" regrets regretOffset actionCount
        validateRow "strategy" strategy strategyOffset actionCount
        regretMatchUnchecked regrets regretOffset actionCount strategy strategyOffset

    let sampleUnchecked (probabilities: double[]) offset actionCount draw =
        let mutable maximum = 0.0
        let mutable scaledTotal = 0.0
        let mutable i = 0

        while i < actionCount do
            let probability = probabilities.[offset + i]

            if probability > 0.0 then
                if probability > maximum then
                    if maximum = 0.0 then
                        scaledTotal <- 1.0
                    else
                        scaledTotal <- scaledTotal * (maximum / probability) + 1.0

                    maximum <- probability
                else
                    scaledTotal <- scaledTotal + probability / maximum

            i <- i + 1

        if maximum = 0.0 then
            min (actionCount - 1) (int (draw * float actionCount))
        else
            let target = draw * scaledTotal
            let mutable cumulative = 0.0
            let mutable selected = actionCount - 1
            let mutable lastPositive = selected
            let mutable found = false
            i <- 0

            while i < actionCount do
                let probability = probabilities.[offset + i]

                if probability > 0.0 then
                    lastPositive <- i
                    cumulative <- cumulative + probability / maximum

                    if not found && target < cumulative then
                        selected <- i
                        found <- true

                i <- i + 1

            if found then selected else lastPositive

    let sample probabilities offset actionCount draw =
        validateNonnegativeFiniteRow "probabilities" probabilities offset actionCount

        if not (isFinite draw) || draw < 0.0 || draw >= 1.0 then
            invalidArg "draw" "The random draw must be finite and in [0, 1)."

        sampleUnchecked probabilities offset actionCount draw

    let accumulateAverageUnchecked
        weight
        (strategy: double[])
        strategyOffset
        actionCount
        (strategySums: double[])
        sumOffset
        =
        if weight > 0.0 then
            let mutable i = 0

            while i < actionCount do
                strategySums.[sumOffset + i] <-
                    strategySums.[sumOffset + i] + weight * strategy.[strategyOffset + i]

                i <- i + 1

    let accumulateAverage weight strategy strategyOffset actionCount strategySums sumOffset =
        if not (isFinite weight) || weight < 0.0 then
            invalidArg "weight" "Average-strategy weight must be finite and nonnegative."

        validateNonnegativeFiniteRow "strategy" strategy strategyOffset actionCount
        validateNonnegativeFiniteRow "strategySums" strategySums sumOffset actionCount
        let mutable i = 0

        while i < actionCount do
            let updated = strategySums.[sumOffset + i] + weight * strategy.[strategyOffset + i]

            if not (isFinite updated) then
                invalidArg "weight" "Average-strategy accumulation overflowed."

            i <- i + 1

        accumulateAverageUnchecked weight strategy strategyOffset actionCount strategySums sumOffset

    let applyRegretDeltaUnchecked
        clipped
        (regrets: double[])
        regretOffset
        (deltas: double[])
        deltaOffset
        actionCount
        =
        let mutable i = 0

        while i < actionCount do
            let updated = regrets.[regretOffset + i] + deltas.[deltaOffset + i]
            regrets.[regretOffset + i] <- if clipped && updated < 0.0 then 0.0 else updated
            i <- i + 1

    let applyRegretDeltaAndClearUnchecked
        clipped
        (regrets: double[])
        regretOffset
        (deltas: double[])
        deltaOffset
        actionCount
        =
        let mutable i = 0

        while i < actionCount do
            let updated = regrets.[regretOffset + i] + deltas.[deltaOffset + i]
            regrets.[regretOffset + i] <- if clipped && updated < 0.0 then 0.0 else updated
            deltas.[deltaOffset + i] <- 0.0
            i <- i + 1

    let applyRegretDelta clipped regrets regretOffset deltas deltaOffset actionCount =
        validateFiniteRow "regrets" regrets regretOffset actionCount
        validateFiniteRow "deltas" deltas deltaOffset actionCount
        let mutable i = 0

        while i < actionCount do
            let updated = regrets.[regretOffset + i] + deltas.[deltaOffset + i]

            if not (isFinite updated) then
                invalidArg "deltas" "Regret update overflowed."

            i <- i + 1

        applyRegretDeltaUnchecked clipped regrets regretOffset deltas deltaOffset actionCount

    let applyRegretDeltaForMode mode regrets regretOffset deltas deltaOffset actionCount =
        let clipped = (Mode.semantics mode).RegretTransform = Clipped
        applyRegretDelta clipped regrets regretOffset deltas deltaOffset actionCount

    let normalizeAverageUnchecked
        (strategySums: double[])
        sumOffset
        actionCount
        (destination: double[])
        destinationOffset
        =
        let mutable maximum = 0.0
        let mutable scaledTotal = 0.0
        let mutable i = 0

        while i < actionCount do
            let value = strategySums.[sumOffset + i]

            if value > 0.0 then
                if value > maximum then
                    if maximum = 0.0 then
                        scaledTotal <- 1.0
                    else
                        scaledTotal <- scaledTotal * (maximum / value) + 1.0

                    maximum <- value
                else
                    scaledTotal <- scaledTotal + value / maximum

            i <- i + 1

        if maximum = 0.0 then
            let probability = 1.0 / float actionCount
            i <- 0

            while i < actionCount do
                destination.[destinationOffset + i] <- probability
                i <- i + 1
        else
            i <- 0

            while i < actionCount do
                let value = strategySums.[sumOffset + i]
                destination.[destinationOffset + i] <-
                    if value > 0.0 then (value / maximum) / scaledTotal else 0.0

                i <- i + 1

    let normalizeAverage strategySums sumOffset actionCount destination destinationOffset =
        validateNonnegativeFiniteRow "strategySums" strategySums sumOffset actionCount
        validateRow "destination" destination destinationOffset actionCount
        normalizeAverageUnchecked strategySums sumOffset actionCount destination destinationOffset

/// Optional output purification. This returns a new probability row and never
/// mutates solver tables or the supplied strategy.
module Strategy =
    let threshold cutoff (strategy: double[]) =
        if isNull strategy then
            nullArg "strategy"

        if strategy.Length = 0 then
            invalidArg "strategy" "A strategy must contain at least one action."

        if Double.IsNaN cutoff
           || Double.IsInfinity cutoff
           || cutoff < 0.0
           || cutoff > 1.0 then
            invalidArg "cutoff" "The threshold must be finite and in [0, 1]."

        let mutable total = 0.0
        let mutable largestIndex = 0
        let mutable largestProbability = -1.0
        let mutable i = 0

        while i < strategy.Length do
            let probability = strategy.[i]

            if Double.IsNaN probability
               || Double.IsInfinity probability
               || probability < 0.0
               || probability > 1.0 then
                invalidArg "strategy" "A strategy must contain finite probabilities in [0, 1]."

            total <- total + probability

            if probability > largestProbability then
                largestProbability <- probability
                largestIndex <- i

            i <- i + 1

        let tolerance = min 1e-9 (1e-12 * float strategy.Length)

        if abs (total - 1.0) > tolerance then
            invalidArg "strategy" "A strategy's probabilities must sum to one."

        let purified = Array.copy strategy

        if cutoff > 0.0 then
            let mutable retainedTotal = 0.0
            i <- 0

            while i < purified.Length do
                if purified.[i] < cutoff then
                    purified.[i] <- 0.0
                else
                    retainedTotal <- retainedTotal + purified.[i]

                i <- i + 1

            if retainedTotal = 0.0 then
                purified.[largestIndex] <- 1.0
            else
                i <- 0

                while i < purified.Length do
                    purified.[i] <- purified.[i] / retainedTotal
                    i <- i + 1

        purified

[<Literal>]
let ChanceActor = -1

/// Minimal contract shared by exhaustive and sampled traversal. Action indices
/// are local, dense legal slots in 0 .. ActionCount(state) - 1.
type IGame<'State> =
    abstract TerminalUtility: state: 'State * targetPlayer: int -> double voption
    abstract Actor: state: 'State -> int
    abstract InformationSetId: state: 'State -> int
    abstract ActionCount: state: 'State -> int
    abstract NextState: state: 'State * action: int -> 'State
    abstract ChanceProbability: state: 'State * action: int -> double

/// Allocation-free-after-construction exhaustive CFR. Two-player vanilla CFR
/// uses a one-pass specialization; other cases use target-player traversals.
type internal ExhaustiveSolver<'State, 'Game when 'Game :> IGame<'State>>
    (mode: SolverMode, game: 'Game, table: PackedTable, maxDepth: int, maxActionCount: int) =

    let semantics = Mode.semantics mode

    do
        if semantics.Traversal <> Exhaustive then
            invalidArg "mode" "ExhaustiveSolver supports only CFR and CFRPlus."

        if table.PlayerCount < 2 then
            invalidArg "table" "CFR requires at least two players."

        if maxDepth <= 0 then
            invalidArg "maxDepth" "Maximum depth must be positive."

        if maxActionCount <= 0 then
            invalidArg "maxActionCount" "Maximum action count must be positive."

    let workspace =
        Workspace.createExhaustive table.InfoSets.Length table.SlotCount maxDepth maxActionCount

    let validateProbability probability =
        if Double.IsNaN probability || Double.IsInfinity probability || probability < 0.0 then
            invalidOp "Chance probabilities must be finite and nonnegative."

    let rec traverse targetPlayer state ownReach externalReach depth =
        // This must precede every actor or information-set query: terminal
        // states are not required to define either value.
        match game.TerminalUtility(state, targetPlayer) with
        | ValueSome utility -> utility
        | ValueNone ->
            if depth >= maxDepth then
                invalidOp "The game exceeded the solver's configured maximum depth."

            let actor = game.Actor state
            let actionCount = game.ActionCount state

            if actionCount <= 0 || actionCount > maxActionCount then
                invalidOp "A nonterminal state exposed an invalid action count."

            if actor = ChanceActor then
                let mutable value = 0.0
                let mutable probabilitySum = 0.0
                let mutable action = 0

                while action < actionCount do
                    let probability = game.ChanceProbability(state, action)
                    validateProbability probability
                    probabilitySum <- probabilitySum + probability
                    let next = game.NextState(state, action)
                    value <- value + probability * traverse targetPlayer next ownReach (externalReach * probability) (depth + 1)
                    action <- action + 1

                if abs (probabilitySum - 1.0) > 1e-12 then
                    invalidOp "Chance probabilities must sum to one."

                value
            elif actor < 0 || actor >= table.PlayerCount then
                invalidOp "A nonterminal state returned an invalid actor."
            else
                let infoSetId = game.InformationSetId state

                if infoSetId < 0 || infoSetId >= table.InfoSets.Length then
                    invalidOp "A player state returned an invalid information-set ID."

                let row = table.InfoSets.[infoSetId]

                if row.Owner <> actor || row.ActionCount <> actionCount then
                    invalidOp "The game state disagrees with its information-set metadata."

                let scratchOffset = depth * maxActionCount
                Scalar.regretMatchUnchecked table.Tables.Regrets row.Offset actionCount workspace.Common.Strategies scratchOffset

                if actor = targetPlayer && workspace.Common.AverageEpochs.[infoSetId] <> workspace.Common.Epoch then
                    workspace.Common.AverageEpochs.[infoSetId] <- workspace.Common.Epoch
                    Scalar.accumulateAverageUnchecked
                        (ownReach)
                        workspace.Common.Strategies
                        scratchOffset
                        actionCount
                        table.Tables.StrategySums
                        row.Offset

                let mutable nodeValue = 0.0
                let mutable action = 0

                while action < actionCount do
                    let probability = workspace.Common.Strategies.[scratchOffset + action]
                    let next = game.NextState(state, action)
                    let actionValue =
                        if actor = targetPlayer then
                            traverse targetPlayer next (ownReach * probability) externalReach (depth + 1)
                        else
                            traverse targetPlayer next ownReach (externalReach * probability) (depth + 1)

                    workspace.Common.Utilities.[scratchOffset + action] <- actionValue
                    nodeValue <- nodeValue + probability * actionValue
                    action <- action + 1

                if actor = targetPlayer then
                    Workspace.touchExhaustiveRow workspace infoSetId
                    action <- 0

                    while action < actionCount do
                        workspace.RegretDeltas.[row.Offset + action] <-
                            workspace.RegretDeltas.[row.Offset + action]
                            + externalReach * (workspace.Common.Utilities.[scratchOffset + action] - nodeValue)

                        action <- action + 1

                nodeValue

    let rec traverseTwoPlayerCfr state reach0 reach1 chanceReach averageWeight depth =
        // Terminal detection remains before actor and information-set access.
        // A terminal must expose both independent utilities.
        match game.TerminalUtility(state, 0) with
        | ValueSome utility0 ->
            match game.TerminalUtility(state, 1) with
            | ValueSome utility1 -> struct (utility0, utility1)
            | ValueNone ->
                invalidOp "A terminal state did not expose both player utilities."
        | ValueNone ->
            if depth >= maxDepth then
                invalidOp "The game exceeded the solver's configured maximum depth."

            let actor = game.Actor state
            let actionCount = game.ActionCount state

            if actionCount <= 0 || actionCount > maxActionCount then
                invalidOp "A nonterminal state exposed an invalid action count."

            if actor = ChanceActor then
                let mutable nodeValue0 = 0.0
                let mutable nodeValue1 = 0.0
                let mutable probabilitySum = 0.0
                let mutable action = 0

                while action < actionCount do
                    let probability = game.ChanceProbability(state, action)
                    validateProbability probability
                    probabilitySum <- probabilitySum + probability
                    let next = game.NextState(state, action)
                    let struct (actionValue0, actionValue1) =
                        traverseTwoPlayerCfr
                            next
                            reach0
                            reach1
                            (chanceReach * probability)
                            averageWeight
                            (depth + 1)
                    nodeValue0 <- nodeValue0 + probability * actionValue0
                    nodeValue1 <- nodeValue1 + probability * actionValue1
                    action <- action + 1

                if abs (probabilitySum - 1.0) > 1e-12 then
                    invalidOp "Chance probabilities must sum to one."

                struct (nodeValue0, nodeValue1)
            elif actor < 0 || actor >= table.PlayerCount then
                invalidOp "A nonterminal state returned an invalid actor."
            else
                let infoSetId = game.InformationSetId state

                if infoSetId < 0 || infoSetId >= table.InfoSets.Length then
                    invalidOp "A player state returned an invalid information-set ID."

                let row = table.InfoSets.[infoSetId]

                if row.Owner <> actor || row.ActionCount <> actionCount then
                    invalidOp "The game state disagrees with its information-set metadata."

                let scratchOffset = depth * maxActionCount
                Scalar.regretMatchUnchecked table.Tables.Regrets row.Offset actionCount workspace.Common.Strategies scratchOffset

                if workspace.Common.AverageEpochs.[infoSetId] <> workspace.Common.Epoch then
                    workspace.Common.AverageEpochs.[infoSetId] <- workspace.Common.Epoch
                    let ownReach = if actor = 0 then reach0 else reach1
                    Scalar.accumulateAverageUnchecked
                        (averageWeight * ownReach)
                        workspace.Common.Strategies
                        scratchOffset
                        actionCount
                        table.Tables.StrategySums
                        row.Offset

                let mutable nodeValue0 = 0.0
                let mutable nodeValue1 = 0.0
                let mutable action = 0

                while action < actionCount do
                    let probability = workspace.Common.Strategies.[scratchOffset + action]
                    let next = game.NextState(state, action)
                    let struct (actionValue0, actionValue1) =
                        if actor = 0 then
                            traverseTwoPlayerCfr
                                next
                                (reach0 * probability)
                                reach1
                                chanceReach
                                averageWeight
                                (depth + 1)
                        else
                            traverseTwoPlayerCfr
                                next
                                reach0
                                (reach1 * probability)
                                chanceReach
                                averageWeight
                                (depth + 1)

                    workspace.Common.Utilities.[scratchOffset + action] <-
                        if actor = 0 then actionValue0 else actionValue1
                    nodeValue0 <- nodeValue0 + probability * actionValue0
                    nodeValue1 <- nodeValue1 + probability * actionValue1
                    action <- action + 1

                Workspace.touchExhaustiveRow workspace infoSetId
                let externalReach = chanceReach * (if actor = 0 then reach1 else reach0)
                let nodeValue = if actor = 0 then nodeValue0 else nodeValue1
                action <- 0

                while action < actionCount do
                    workspace.RegretDeltas.[row.Offset + action] <-
                        workspace.RegretDeltas.[row.Offset + action]
                        + externalReach * (workspace.Common.Utilities.[scratchOffset + action] - nodeValue)
                    action <- action + 1

                struct (nodeValue0, nodeValue1)

    let applyTouched clipped =
        let mutable touched = 0

        while touched < workspace.TouchedCount do
            let row = table.InfoSets.[workspace.TouchedRows.[touched]]
            Scalar.applyRegretDeltaAndClearUnchecked
                clipped
                table.Tables.Regrets
                row.Offset
                workspace.RegretDeltas
                row.Offset
                row.ActionCount
            touched <- touched + 1

        workspace.TouchedCount <- 0

    let beginTargetPass () =
        Workspace.beginAveragePass workspace.Common |> ignore

    let validateIteration iteration burnIn =
        if iteration <= 0 then
            invalidArg "iteration" "Iteration numbers are one-based and must be positive."

        if burnIn < 0 then
            invalidArg "burnIn" "Burn-in cannot be negative."

    let runGeneric iteration burnIn root (utilities: double[]) =
        let averageWeight = Mode.averageWeight mode iteration burnIn

        if semantics.UpdateSchedule = Simultaneous then
            Workspace.beginExhaustivePass workspace |> ignore
            let mutable targetPlayer = 0

            while targetPlayer < table.PlayerCount do
                beginTargetPass ()
                utilities.[targetPlayer] <-
                    traverse targetPlayer root averageWeight 1.0 0
                targetPlayer <- targetPlayer + 1

            applyTouched false
        else
            let mutable targetPlayer = 0

            while targetPlayer < table.PlayerCount do
                Workspace.beginExhaustivePass workspace |> ignore
                beginTargetPass ()
                utilities.[targetPlayer] <-
                    traverse targetPlayer root averageWeight 1.0 0
                applyTouched true
                targetPlayer <- targetPlayer + 1

    member _.Table = table
    member internal _.Workspace = workspace

    member _.RunIterationInto
        (iteration: int, burnIn: int, root: 'State, utilities: double[])
        =
        validateIteration iteration burnIn

        if isNull utilities then
            nullArg "utilities"

        if utilities.Length <> table.PlayerCount then
            invalidArg "utilities" "The utility buffer length must equal the player count."

        if table.PlayerCount = 2 then
            let averageWeight = Mode.averageWeight mode iteration burnIn

            if semantics.UpdateSchedule = Simultaneous then
                Workspace.beginExhaustivePass workspace |> ignore
                beginTargetPass ()
                let struct (utility0, utility1) =
                    traverseTwoPlayerCfr root 1.0 1.0 1.0 averageWeight 0
                applyTouched false
                utilities.[0] <- utility0
                utilities.[1] <- utility1
            else
                Workspace.beginExhaustivePass workspace |> ignore
                beginTargetPass ()
                utilities.[0] <- traverse 0 root averageWeight 1.0 0
                applyTouched true
                Workspace.beginExhaustivePass workspace |> ignore
                beginTargetPass ()
                utilities.[1] <- traverse 1 root averageWeight 1.0 0
                applyTouched true
        else
            runGeneric iteration burnIn root utilities

    member _.RunIteration(iteration: int, burnIn: int, root: 'State) =
        validateIteration iteration burnIn

        if table.PlayerCount <> 2 then
            invalidOp "Use RunIterationInto for games with more than two players."

        let averageWeight = Mode.averageWeight mode iteration burnIn

        if semantics.UpdateSchedule = Simultaneous then
            Workspace.beginExhaustivePass workspace |> ignore
            beginTargetPass ()
            let struct (utility0, utility1) =
                traverseTwoPlayerCfr root 1.0 1.0 1.0 averageWeight 0
            applyTouched false
            struct (utility0, utility1)
        else
            Workspace.beginExhaustivePass workspace |> ignore
            beginTargetPass ()
            let utility0 = traverse 0 root averageWeight 1.0 0
            applyTouched true
            Workspace.beginExhaustivePass workspace |> ignore
            beginTargetPass ()
            let utility1 = traverse 1 root averageWeight 1.0 0
            applyTouched true
            struct (utility0, utility1)

    member _.RunTargetPairReference(iteration: int, burnIn: int, root: 'State) =
        if mode <> CFR then
            invalidOp "The target-pair reference is defined only for vanilla CFR."

        if table.PlayerCount <> 2 then
            invalidOp "The target-pair reference is defined only for two-player games."

        validateIteration iteration burnIn

        let averageWeight = Mode.averageWeight mode iteration burnIn
        Workspace.beginExhaustivePass workspace |> ignore
        beginTargetPass ()
        let utility0 = traverse 0 root averageWeight 1.0 0
        beginTargetPass ()
        let utility1 = traverse 1 root averageWeight 1.0 0
        applyTouched false
        struct (utility0, utility1)

/// Allocation-free-after-construction external-sampling MCCFR. Two-player
/// games retain paired sampled passes; multiplayer games perform one exact
/// average-strategy sweep followed by one sampled regret pass per player.
type internal SampledSolver<'State, 'Game when 'Game :> IGame<'State>>
    (
        mode: SolverMode,
        game: 'Game,
        table: PackedTable,
        maxDepth: int,
        maxActionCount: int,
        deltaCapacity: int,
        random: Random
    ) =

    let semantics = Mode.semantics mode

    do
        if semantics.Traversal <> ExternalSampling then
            invalidArg "mode" "SampledSolver supports only MCCFR and MCCFRPlus."

        if table.PlayerCount < 2 then
            invalidArg "table" "MCCFR requires at least two players."

        if maxDepth <= 0 then
            invalidArg "maxDepth" "Maximum depth must be positive."

        if maxActionCount <= 0 then
            invalidArg "maxActionCount" "Maximum action count must be positive."

        if deltaCapacity <= 0 then
            invalidArg "deltaCapacity" "Sampled delta capacity must be positive."

        if isNull random then
            nullArg "random"

    let workspace =
        Workspace.createSampled table.InfoSets.Length deltaCapacity maxDepth maxActionCount

    // Only multiplayer MCCFR owns this extra workspace. The two-player fast
    // path retains the Stage 4 memory layout.
    let averageReaches =
        if table.PlayerCount > 2 then
            Array.create table.PlayerCount 1.0
        else
            Array.empty

    let validateProbability probability =
        if Double.IsNaN probability || Double.IsInfinity probability || probability < 0.0 then
            invalidOp "Chance probabilities must be finite and nonnegative."

    let nextDraw () =
        let draw = random.NextDouble()

        if Double.IsNaN draw || Double.IsInfinity draw || draw < 0.0 || draw >= 1.0 then
            invalidOp "The random source returned a value outside [0, 1)."

        draw

    let appendRowDelta offset actionCount scratchOffset nodeValue =
        let mutable action = 0

        while action < actionCount do
            Workspace.appendDelta
                workspace
                (offset + action)
                (workspace.Common.Utilities.[scratchOffset + action] - nodeValue)
            action <- action + 1

    let rec traverse targetPlayer averageWeight state depth =
        // Terminal states need not expose actor, information-set, or action data.
        match game.TerminalUtility(state, targetPlayer) with
        | ValueSome utility -> utility
        | ValueNone ->
            if depth >= maxDepth then
                invalidOp "The game exceeded the solver's configured maximum depth."

            let actor = game.Actor state
            let actionCount = game.ActionCount state

            if actionCount <= 0 || actionCount > maxActionCount then
                invalidOp "A nonterminal state exposed an invalid action count."

            let scratchOffset = depth * maxActionCount

            if actor = ChanceActor then
                let mutable probabilitySum = 0.0
                let mutable action = 0

                while action < actionCount do
                    let probability = game.ChanceProbability(state, action)
                    validateProbability probability
                    workspace.Common.Strategies.[scratchOffset + action] <- probability
                    probabilitySum <- probabilitySum + probability
                    action <- action + 1

                if abs (probabilitySum - 1.0) > 1e-12 then
                    invalidOp "Chance probabilities must sum to one."

                let sampledAction =
                    Scalar.sampleUnchecked
                        workspace.Common.Strategies
                        scratchOffset
                        actionCount
                        (nextDraw ())

                traverse
                    targetPlayer
                    averageWeight
                    (game.NextState(state, sampledAction))
                    (depth + 1)
            elif actor < 0 || actor >= table.PlayerCount then
                invalidOp "A nonterminal state returned an invalid actor."
            else
                let infoSetId = game.InformationSetId state

                if infoSetId < 0 || infoSetId >= table.InfoSets.Length then
                    invalidOp "A player state returned an invalid information-set ID."

                let row = table.InfoSets.[infoSetId]

                if row.Owner <> actor || row.ActionCount <> actionCount then
                    invalidOp "The game state disagrees with its information-set metadata."

                Scalar.regretMatchUnchecked
                    table.Tables.Regrets
                    row.Offset
                    actionCount
                    workspace.Common.Strategies
                    scratchOffset

                if actor = targetPlayer then
                    let mutable nodeValue = 0.0
                    let mutable action = 0

                    while action < actionCount do
                        let actionValue =
                            traverse
                                targetPlayer
                                averageWeight
                                (game.NextState(state, action))
                                (depth + 1)
                        workspace.Common.Utilities.[scratchOffset + action] <- actionValue
                        nodeValue <-
                            nodeValue
                            + workspace.Common.Strategies.[scratchOffset + action] * actionValue
                        action <- action + 1

                    appendRowDelta row.Offset actionCount scratchOffset nodeValue
                    nodeValue
                else
                    if workspace.Common.AverageEpochs.[infoSetId] <> workspace.Common.Epoch then
                        workspace.Common.AverageEpochs.[infoSetId] <- workspace.Common.Epoch
                        Scalar.accumulateAverageUnchecked
                            averageWeight
                            workspace.Common.Strategies
                            scratchOffset
                            actionCount
                            table.Tables.StrategySums
                            row.Offset

                    let sampledAction =
                        match Workspace.tryGetSample workspace infoSetId with
                        | ValueSome action -> action
                        | ValueNone ->
                            let action =
                                Scalar.sampleUnchecked
                                    workspace.Common.Strategies
                                    scratchOffset
                                    actionCount
                                    (nextDraw ())
                            Workspace.setSample workspace infoSetId action
                            action

                    traverse
                        targetPlayer
                        averageWeight
                        (game.NextState(state, sampledAction))
                        (depth + 1)

    let rec accumulateExactAverage averageWeight state depth =
        match game.TerminalUtility(state, 0) with
        | ValueSome _ -> ()
        | ValueNone ->
            if depth >= maxDepth then
                invalidOp "The game exceeded the solver's configured maximum depth."

            let actor = game.Actor state
            let actionCount = game.ActionCount state

            if actionCount <= 0 || actionCount > maxActionCount then
                invalidOp "A nonterminal state exposed an invalid action count."

            let scratchOffset = depth * maxActionCount

            if actor = ChanceActor then
                let mutable probabilitySum = 0.0
                let mutable action = 0

                while action < actionCount do
                    let probability = game.ChanceProbability(state, action)
                    validateProbability probability
                    probabilitySum <- probabilitySum + probability
                    accumulateExactAverage
                        averageWeight
                        (game.NextState(state, action))
                        (depth + 1)
                    action <- action + 1

                if abs (probabilitySum - 1.0) > 1e-12 then
                    invalidOp "Chance probabilities must sum to one."
            elif actor < 0 || actor >= table.PlayerCount then
                invalidOp "A nonterminal state returned an invalid actor."
            else
                let infoSetId = game.InformationSetId state

                if infoSetId < 0 || infoSetId >= table.InfoSets.Length then
                    invalidOp "A player state returned an invalid information-set ID."

                let row = table.InfoSets.[infoSetId]

                if row.Owner <> actor || row.ActionCount <> actionCount then
                    invalidOp "The game state disagrees with its information-set metadata."

                Scalar.regretMatchUnchecked
                    table.Tables.Regrets
                    row.Offset
                    actionCount
                    workspace.Common.Strategies
                    scratchOffset

                if workspace.Common.AverageEpochs.[infoSetId] <> workspace.Common.Epoch then
                    workspace.Common.AverageEpochs.[infoSetId] <- workspace.Common.Epoch
                    Scalar.accumulateAverageUnchecked
                        (averageWeight * averageReaches.[actor])
                        workspace.Common.Strategies
                        scratchOffset
                        actionCount
                        table.Tables.StrategySums
                        row.Offset

                let previousReach = averageReaches.[actor]
                let mutable action = 0

                while action < actionCount do
                    averageReaches.[actor] <-
                        previousReach * workspace.Common.Strategies.[scratchOffset + action]
                    accumulateExactAverage
                        averageWeight
                        (game.NextState(state, action))
                        (depth + 1)
                    action <- action + 1

                averageReaches.[actor] <- previousReach

    let beginTargetPass () =
        Workspace.beginSampledPass workspace |> ignore
        Workspace.beginAveragePass workspace.Common |> ignore

    let validateIteration iteration burnIn =
        if iteration <= 0 then
            invalidArg "iteration" "Iteration numbers are one-based and must be positive."

        if burnIn < 0 then
            invalidArg "burnIn" "Burn-in cannot be negative."

    let runMultiplayer iteration burnIn root (utilities: double[]) =
        let averageWeight = Mode.averageWeight mode iteration burnIn

        if averageWeight > 0.0 then
            let mutable player = 0

            while player < averageReaches.Length do
                averageReaches.[player] <- 1.0
                player <- player + 1

            Workspace.beginAveragePass workspace.Common |> ignore
            accumulateExactAverage averageWeight root 0

        Workspace.resetDeltaLog workspace
        let mutable targetPlayer = 0

        while targetPlayer < table.PlayerCount do
            Workspace.beginSampledPass workspace |> ignore
            utilities.[targetPlayer] <- traverse targetPlayer 0.0 root 0

            if semantics.UpdateSchedule = Alternating then
                Workspace.applySampledDeltas true table.Tables.Regrets workspace

            targetPlayer <- targetPlayer + 1

        if semantics.UpdateSchedule = Simultaneous then
            Workspace.applySampledDeltas false table.Tables.Regrets workspace

    member _.Table = table
    member internal _.Workspace = workspace

    member _.RunIterationInto
        (iteration: int, burnIn: int, root: 'State, utilities: double[])
        =
        validateIteration iteration burnIn

        if isNull utilities then
            nullArg "utilities"

        if utilities.Length <> table.PlayerCount then
            invalidArg "utilities" "The utility buffer length must equal the player count."

        if table.PlayerCount = 2 then
            let averageWeight = Mode.averageWeight mode iteration burnIn
            Workspace.resetDeltaLog workspace
            beginTargetPass ()
            utilities.[0] <- traverse 0 averageWeight root 0

            if semantics.UpdateSchedule = Alternating then
                Workspace.applySampledDeltas true table.Tables.Regrets workspace

            beginTargetPass ()
            utilities.[1] <- traverse 1 averageWeight root 0
            Workspace.applySampledDeltas
                (semantics.RegretTransform = Clipped)
                table.Tables.Regrets
                workspace
        else
            runMultiplayer iteration burnIn root utilities

    member _.RunIteration(iteration: int, burnIn: int, root: 'State) =
        validateIteration iteration burnIn

        if table.PlayerCount <> 2 then
            invalidOp "Use RunIterationInto for games with more than two players."

        let averageWeight = Mode.averageWeight mode iteration burnIn
        Workspace.resetDeltaLog workspace
        beginTargetPass ()
        let utility0 = traverse 0 averageWeight root 0

        if semantics.UpdateSchedule = Alternating then
            Workspace.applySampledDeltas true table.Tables.Regrets workspace

        beginTargetPass ()
        let utility1 = traverse 1 averageWeight root 0
        Workspace.applySampledDeltas
            (semantics.RegretTransform = Clipped)
            table.Tables.Regrets
            workspace
        struct (utility0, utility1)

/// Finite sequential general-sum CFR solver. Construction selects a direct
/// two-player fast path or the generic target-player schedule.
type Solver<'State, 'Game when 'Game :> IGame<'State>>
    (
        mode: SolverMode,
        game: 'Game,
        playerCount: int,
        definitions: InfoSetDefinition[],
        maxDepth: int,
        maxActionCount: int,
        sampledDeltaCapacity: int,
        random: Random
    ) =

    do
        if playerCount < 2 then
            invalidArg "playerCount" "CFR requires at least two players."

    let table = PackedTable.create playerCount definitions
    let semantics = Mode.semantics mode
    let mutable iterationsCompleted = 0
    let utilitySums = Array.zeroCreate playerCount
    let iterationUtilities = Array.zeroCreate playerCount

    let exhaustive =
        if semantics.Traversal = Exhaustive then
            ValueSome(ExhaustiveSolver(mode, game, table, maxDepth, maxActionCount))
        else
            ValueNone

    let sampled =
        if semantics.Traversal = ExternalSampling then
            ValueSome(
                SampledSolver(
                    mode,
                    game,
                    table,
                    maxDepth,
                    maxActionCount,
                    sampledDeltaCapacity,
                    random
                )
            )
        else
            ValueNone

    let runTwoPlayerIteration iteration burnIn root =
        match exhaustive with
        | ValueSome solver -> solver.RunIteration(iteration, burnIn, root)
        | ValueNone ->
            match sampled with
            | ValueSome solver -> solver.RunIteration(iteration, burnIn, root)
            | ValueNone -> invalidOp "The solver has no traversal implementation."

    let runMultiplayerIteration iteration burnIn root =
        match exhaustive with
        | ValueSome solver ->
            solver.RunIterationInto(iteration, burnIn, root, iterationUtilities)
        | ValueNone ->
            match sampled with
            | ValueSome solver ->
                solver.RunIterationInto(iteration, burnIn, root, iterationUtilities)
            | ValueNone -> invalidOp "The solver has no traversal implementation."

    let progress iterationsRun =
        let denominator = float iterationsCompleted
        let means = Array.zeroCreate playerCount
        let mutable player = 0

        while player < playerCount do
            means.[player] <- utilitySums.[player] / denominator
            player <- player + 1

        { IterationsRun = iterationsRun
          IterationsCompleted = iterationsCompleted
          MeanUtilities = means }

    let normalizedAverageProfile () =
        let profile = Array.zeroCreate table.SlotCount
        let mutable infoSetId = 0

        while infoSetId < table.InfoSets.Length do
            let row = table.InfoSets.[infoSetId]
            Scalar.normalizeAverageUnchecked
                table.Tables.StrategySums
                row.Offset
                row.ActionCount
                profile
                row.Offset
            infoSetId <- infoSetId + 1

        profile

    let evaluateAverageTarget (profile: double[]) targetPlayer root =

        let validateProbability probability =
            if Double.IsNaN probability
               || Double.IsInfinity probability
               || probability < 0.0 then
                invalidOp "Chance probabilities must be finite and nonnegative."

        let rec evaluate state depth =
            match game.TerminalUtility(state, targetPlayer) with
            | ValueSome utility -> utility
            | ValueNone ->
                if depth >= maxDepth then
                    invalidOp "The game exceeded the solver's configured maximum depth."

                let actor = game.Actor state
                let actionCount = game.ActionCount state

                if actionCount <= 0 || actionCount > maxActionCount then
                    invalidOp "A nonterminal state exposed an invalid action count."

                let mutable value = 0.0
                let mutable probabilitySum = 0.0
                let mutable action = 0

                if actor = ChanceActor then
                    while action < actionCount do
                        let probability = game.ChanceProbability(state, action)
                        validateProbability probability
                        probabilitySum <- probabilitySum + probability
                        let actionValue =
                            evaluate (game.NextState(state, action)) (depth + 1)
                        value <- value + probability * actionValue
                        action <- action + 1

                    if abs (probabilitySum - 1.0) > 1e-12 then
                        invalidOp "Chance probabilities must sum to one."
                elif actor < 0 || actor >= table.PlayerCount then
                    invalidOp "A nonterminal state returned an invalid actor."
                else
                    let currentInfoSetId = game.InformationSetId state

                    if currentInfoSetId < 0
                       || currentInfoSetId >= table.InfoSets.Length then
                        invalidOp "A player state returned an invalid information-set ID."

                    let row = table.InfoSets.[currentInfoSetId]

                    if row.Owner <> actor || row.ActionCount <> actionCount then
                        invalidOp "The game state disagrees with its information-set metadata."

                    while action < actionCount do
                        let probability = profile.[row.Offset + action]
                        let actionValue =
                            evaluate (game.NextState(state, action)) (depth + 1)
                        value <- value + probability * actionValue
                        action <- action + 1

                value

        evaluate root 0

    let validateSequence iteration =
        if iteration <> iterationsCompleted + 1 then
            invalidArg
                "iteration"
                $"Expected iteration {iterationsCompleted + 1}; training iterations must be sequential."

    let recordTwoPlayerIteration iteration burnIn root =
        validateSequence iteration
        let struct (utility0, utility1) as utilities =
            runTwoPlayerIteration iteration burnIn root
        iterationsCompleted <- iteration
        utilitySums.[0] <- utilitySums.[0] + utility0
        utilitySums.[1] <- utilitySums.[1] + utility1
        utilities

    let recordMultiplayerIteration iteration burnIn root =
        validateSequence iteration
        runMultiplayerIteration iteration burnIn root
        iterationsCompleted <- iteration
        let mutable player = 0

        while player < playerCount do
            utilitySums.[player] <-
                utilitySums.[player] + iterationUtilities.[player]
            player <- player + 1

    let recordIteration iteration burnIn root =
        if playerCount = 2 then
            recordTwoPlayerIteration iteration burnIn root |> ignore
        else
            recordMultiplayerIteration iteration burnIn root

    new
        (
            mode,
            game,
            definitions,
            maxDepth,
            maxActionCount,
            sampledDeltaCapacity,
            seed: int
        ) =
        Solver(
            mode,
            game,
            2,
            definitions,
            maxDepth,
            maxActionCount,
            sampledDeltaCapacity,
            Random(seed)
        )

    new
        (
            mode,
            game,
            definitions,
            maxDepth,
            maxActionCount,
            sampledDeltaCapacity,
            random: Random
        ) =
        Solver(
            mode,
            game,
            2,
            definitions,
            maxDepth,
            maxActionCount,
            sampledDeltaCapacity,
            random
        )

    new
        (
            mode,
            game,
            playerCount,
            definitions,
            maxDepth,
            maxActionCount,
            sampledDeltaCapacity,
            seed: int
        ) =
        Solver(
            mode,
            game,
            playerCount,
            definitions,
            maxDepth,
            maxActionCount,
            sampledDeltaCapacity,
            Random(seed)
        )

    member _.Mode = mode
    member _.PlayerCount = playerCount
    member _.IterationsCompleted = iterationsCompleted
    member internal _.Table = table

    /// Evaluate both players' expected utilities under the normalized average
    /// profile. This enumerates the complete game tree and is intended for
    /// reporting, convergence checks, and small-game correctness oracles.
    member _.EvaluateAverageProfile(root: 'State) =
        if playerCount <> 2 then
            invalidOp "Use EvaluateAverageProfileInto for games with more than two players."

        let profile = normalizedAverageProfile ()
        struct (
            evaluateAverageTarget profile 0 root,
            evaluateAverageTarget profile 1 root
        )

    /// Evaluate every player's utility under the normalized average profile.
    /// Reporting may allocate the one flat normalized profile; traversal does
    /// not allocate a utility vector at terminal states.
    member _.EvaluateAverageProfileInto(root: 'State, utilities: double[]) =
        if isNull utilities then
            nullArg "utilities"

        if utilities.Length <> playerCount then
            invalidArg "utilities" "The utility buffer length must equal the player count."

        let profile = normalizedAverageProfile ()
        let mutable player = 0

        while player < playerCount do
            utilities.[player] <- evaluateAverageTarget profile player root
            player <- player + 1

    /// Run one two-player iteration and return its utilities without
    /// allocating. Multiplayer callers use RunIterationInto or Train.
    member _.RunIteration(iteration: int, burnIn: int, root: 'State) =
        if playerCount <> 2 then
            invalidOp "Use RunIterationInto for games with more than two players."

        recordTwoPlayerIteration iteration burnIn root

    /// Run one iteration for any player count into caller-owned storage.
    member _.RunIterationInto
        (iteration: int, burnIn: int, root: 'State, utilities: double[])
        =
        if isNull utilities then
            nullArg "utilities"

        if utilities.Length <> playerCount then
            invalidArg "utilities" "The utility buffer length must equal the player count."

        if playerCount = 2 then
            let struct (utility0, utility1) =
                recordTwoPlayerIteration iteration burnIn root
            utilities.[0] <- utility0
            utilities.[1] <- utility1
        else
            recordMultiplayerIteration iteration burnIn root
            Array.Copy(iterationUtilities, utilities, playerCount)

    /// Run a fixed number of additional iterations and return cumulative
    /// mean utilities. The traversal remains allocation-free after warm-up.
    member this.Train(iterations: int, burnIn: int, root: 'State) =
        if iterations <= 0 then
            invalidArg "iterations" "Training iteration count must be positive."

        if burnIn < 0 then
            invalidArg "burnIn" "Burn-in cannot be negative."

        let mutable iterationsRun = 0

        while iterationsRun < iterations do
            recordIteration (iterationsCompleted + 1) burnIn root
            iterationsRun <- iterationsRun + 1

        let current = progress iterationsRun

        { IterationsRun = current.IterationsRun
          IterationsCompleted = current.IterationsCompleted
          MeanUtilities = current.MeanUtilities
          StopReason = TrainingStopReason.IterationLimit
          ConvergenceError = ValueNone
          ConvergenceChecks = 0 }

    /// Train until a caller-supplied non-negative error is within tolerance,
    /// or until maxIterations additional iterations have run. The error can
    /// measure exploitability, NashConv, or distance from a known game value.
    member this.TrainUntil
        (
            maxIterations: int,
            burnIn: int,
            root: 'State,
            check: ConvergenceCheck,
            measureError: TrainingProgress -> double
        ) =
        if maxIterations <= 0 then
            invalidArg "maxIterations" "Maximum training iteration count must be positive."

        if burnIn < 0 then
            invalidArg "burnIn" "Burn-in cannot be negative."

        if isNull (box measureError) then
            nullArg "measureError"

        ConvergenceCheck.validate check

        let mutable iterationsRun = 0
        let mutable checks = 0
        let mutable consecutiveChecks = 0
        let mutable lastError = ValueNone
        let mutable converged = false

        while iterationsRun < maxIterations && not converged do
            recordIteration (iterationsCompleted + 1) burnIn root
            iterationsRun <- iterationsRun + 1

            if iterationsRun % check.CheckEvery = 0
               || iterationsRun = maxIterations then
                let error = measureError (progress iterationsRun)

                if Double.IsNaN error || Double.IsInfinity error || error < 0.0 then
                    invalidOp "The convergence error must be finite and non-negative."

                checks <- checks + 1
                lastError <- ValueSome error

                if error <= check.Tolerance then
                    consecutiveChecks <- consecutiveChecks + 1
                    converged <-
                        consecutiveChecks >= check.RequiredConsecutiveChecks
                else
                    consecutiveChecks <- 0

        let current = progress iterationsRun

        { IterationsRun = current.IterationsRun
          IterationsCompleted = current.IterationsCompleted
          MeanUtilities = current.MeanUtilities
          StopReason =
            if converged then
                TrainingStopReason.ConvergenceTolerance
            else
                TrainingStopReason.IterationLimit
          ConvergenceError = lastError
          ConvergenceChecks = checks }

    member _.AverageStrategy(infoSetId: int) =
        if infoSetId < 0 || infoSetId >= table.InfoSets.Length then
            invalidArg "infoSetId" "Information-set ID is outside the solver table."

        let row = table.InfoSets.[infoSetId]
        let strategy = Array.zeroCreate row.ActionCount
        Scalar.normalizeAverageUnchecked
            table.Tables.StrategySums
            row.Offset
            row.ActionCount
            strategy
            0
        strategy
