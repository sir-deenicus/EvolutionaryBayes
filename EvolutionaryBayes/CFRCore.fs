module internal EvolutionaryBayes.CFRCore

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
