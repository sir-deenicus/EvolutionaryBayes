namespace EvolutionaryBayes

open System

/// Construction options shared by external-regret minimizers.
type ExternalRegretOptions =
    { /// Optional initial strategy. Values are normalized after validation.
      InitialStrategy: float[] option
      /// Optional random source used by Choose. Supplying one makes sampling reproducible.
      Random: Random option }

[<RequireQualifiedAccess>]
module ExternalRegretOptions =
    /// Uniform initialization with an independently created random source.
    let defaults =
        { InitialStrategy = None
          Random = None }

module internal ExternalRegretInternals =
    let isFinite value =
        not (Double.IsNaN value || Double.IsInfinity value)

    let normalizeInitialStrategy actionCount (initial: float[] option) =
        let values =
            match initial with
            | None -> Array.create actionCount 1.0
            | Some supplied ->
                if isNull supplied then
                    nullArg "options.InitialStrategy"

                if supplied.Length <> actionCount then
                    invalidArg
                        "options.InitialStrategy"
                        "The initial strategy length must equal the action count."

                Array.copy supplied

        let mutable maximum = 0.0

        for value in values do
            if not (isFinite value) then
                invalidArg "options.InitialStrategy" "Initial strategy values must be finite."

            if value < 0.0 then
                invalidArg "options.InitialStrategy" "Initial strategy values must be nonnegative."

            if value > maximum then
                maximum <- value

        if maximum = 0.0 then
            invalidArg "options.InitialStrategy" "The initial strategy must have positive total mass."

        let mutable scaledTotal = 0.0

        for value in values do
            scaledTotal <- scaledTotal + value / maximum

        for i = 0 to values.Length - 1 do
            values.[i] <- (values.[i] / maximum) / scaledTotal

        values

    let prepare (options: ExternalRegretOptions) (actions: 'Action[]) =
        if isNull (box options) then
            nullArg (nameof options)

        if isNull actions then
            nullArg (nameof actions)

        if actions.Length = 0 then
            invalidArg (nameof actions) "At least one action is required."

        let random =
            match options.Random with
            | None -> Random()
            | Some supplied when isNull supplied -> nullArg "options.Random"
            | Some supplied -> supplied

        Array.copy actions,
        normalizeInitialStrategy actions.Length options.InitialStrategy,
        random

    let snapshot (actions: 'Action[]) (probabilities: float[]) =
        Array.init actions.Length (fun i -> actions.[i], probabilities.[i])

/// Shared log-space exponential reweighting used by MWUA and finite
/// replicator dynamics. This remains in MWUA.fs because MWUA owns the
/// numerical primitive; consumers provide their own state and semantics.
module internal ExponentialWeightsInternals =
    let logWeightsOf (probabilities: float[]) =
        probabilities
        |> Array.map (fun probability ->
            if probability = 0.0 then Double.NegativeInfinity
            else log probability)

    /// Applies p_i' proportional to p_i * exp(rate * score_i).
    ///
    /// logWeights is persistent caller-owned state. candidates and
    /// probabilities are caller-owned output workspace. A positive
    /// minimumUnnormalizedWeight preserves finite positive support when a
    /// normalized exponential would underflow; zero log weights remain zero.
    let reweight
        rate
        (scores: float[])
        minimumUnnormalizedWeight
        (logWeights: float[])
        (candidates: float[])
        (probabilities: float[])
        =
        let length = logWeights.Length

        if scores.Length <> length
           || candidates.Length <> length
           || probabilities.Length <> length
           || length = 0 then
            invalidArg (nameof scores) "Exponential-weight arrays must have the same positive length."

        if not (ExternalRegretInternals.isFinite rate) || rate <= 0.0 then
            invalidArg (nameof rate) "The exponential reweighting rate must be finite and positive."

        if not (ExternalRegretInternals.isFinite minimumUnnormalizedWeight)
           || minimumUnnormalizedWeight < 0.0 then
            invalidArg
                (nameof minimumUnnormalizedWeight)
                "The minimum unnormalized weight must be finite and nonnegative."

        let mutable referenceScore = Double.NegativeInfinity

        for i = 0 to length - 1 do
            if not (ExternalRegretInternals.isFinite scores.[i]) then
                invalidArg (nameof scores) "Exponential-weight scores must be finite."

            if Double.IsNaN logWeights.[i] || Double.IsPositiveInfinity logWeights.[i] then
                invalidArg (nameof logWeights) "Log weights must be finite or negative infinity."

            if not (Double.IsNegativeInfinity logWeights.[i])
               && scores.[i] > referenceScore then
                referenceScore <- scores.[i]

        if not (ExternalRegretInternals.isFinite referenceScore) then
            invalidArg (nameof logWeights) "At least one log weight must have positive support."

        let mutable maximumCandidate = Double.NegativeInfinity

        for i = 0 to length - 1 do
            if Double.IsNegativeInfinity logWeights.[i] then
                candidates.[i] <- Double.NegativeInfinity
            else
                let relativeScore = scores.[i] - referenceScore
                let scaledScore = rate * relativeScore
                let candidate = logWeights.[i] + scaledScore

                if not (ExternalRegretInternals.isFinite relativeScore)
                   || not (ExternalRegretInternals.isFinite scaledScore)
                   || not (ExternalRegretInternals.isFinite candidate) then
                    invalidArg
                        (nameof scores)
                        "The rate and score range produced a non-finite exponential update."

                candidates.[i] <- candidate

                if candidate > maximumCandidate then
                    maximumCandidate <- candidate

        let minimumLogOffset =
            if minimumUnnormalizedWeight = 0.0 then
                Double.NegativeInfinity
            else
                log minimumUnnormalizedWeight

        let mutable total = 0.0

        for i = 0 to length - 1 do
            let value =
                if Double.IsNegativeInfinity candidates.[i] then
                    0.0
                else
                    let offset = candidates.[i] - maximumCandidate

                    if offset < minimumLogOffset then
                        minimumUnnormalizedWeight
                    else
                        exp offset

            probabilities.[i] <- value
            total <- total + value

        if not (ExternalRegretInternals.isFinite total) || total <= 0.0 then
            invalidOp "The exponential update produced invalid total weight."

        // Commit only after the complete candidate state has been validated.
        for i = 0 to length - 1 do
            probabilities.[i] <- probabilities.[i] / total

            logWeights.[i] <-
                if Double.IsNegativeInfinity candidates.[i] then
                    Double.NegativeInfinity
                else
                    candidates.[i] - maximumCandidate

type internal IExternalRegretKernel =
    abstract Learn: utilities: float[] * strategy: float[] -> unit
    abstract Reset: strategy: float[] -> unit

/// A mutable online learner that minimizes external regret over a fixed action set.
///
/// Learn receives full-information utility feedback: the supplied function is
/// evaluated exactly once for every action. LearnBatch is sequential and is
/// equivalent to calling Learn on each element in enumeration order.
/// Instances are not thread-safe.
type ExternalRegretMinimizer<'Action> internal
    (
        actions: 'Action[],
        initialStrategy: float[],
        random: Random,
        kernel: IExternalRegretKernel
    ) =

    let strategy = Array.copy initialStrategy
    let strategySums = Array.zeroCreate strategy.Length
    let utilities = Array.zeroCreate strategy.Length
    let playedStrategy = Array.zeroCreate strategy.Length
    let mutable rounds = 0L

    /// The strategy that will be used by the next call to Choose.
    member _.Strategy =
        ExternalRegretInternals.snapshot actions strategy

    /// The mean of the strategies used in completed learning rounds.
    /// Before the first round, this returns the current initial strategy.
    member _.AverageStrategy =
        if rounds = 0L then
            ExternalRegretInternals.snapshot actions strategy
        else
            let denominator = float rounds
            Array.init actions.Length (fun i -> actions.[i], strategySums.[i] / denominator)

    /// The number of successfully completed learning rounds.
    member _.Rounds = rounds

    /// Samples one action from the current strategy without advancing the learner.
    member _.Choose() =
        let draw = random.NextDouble()

        if not (ExternalRegretInternals.isFinite draw) || draw < 0.0 || draw >= 1.0 then
            invalidOp "The random source returned a value outside [0, 1)."

        let mutable cumulative = 0.0
        let mutable index = 0

        while index < strategy.Length - 1 && cumulative + strategy.[index] <= draw do
            cumulative <- cumulative + strategy.[index]
            index <- index + 1

        actions.[index]

    /// Performs one full-information update.
    member _.Learn(utility: 'Action -> float) =
        if isNull (box utility) then
            nullArg (nameof utility)

        if rounds = Int64.MaxValue then
            invalidOp "The learner cannot represent another completed round."

        // Evaluate and validate all caller code before changing learner state.
        for i = 0 to actions.Length - 1 do
            let value = utility actions.[i]

            if not (ExternalRegretInternals.isFinite value) then
                invalidArg (nameof utility) "Utilities must be finite."

            utilities.[i] <- value

        // The average records the strategy used before this feedback arrived.
        Array.Copy(strategy, playedStrategy, strategy.Length)
        kernel.Learn(utilities, strategy)

        for i = 0 to strategySums.Length - 1 do
            strategySums.[i] <- strategySums.[i] + playedStrategy.[i]

        rounds <- rounds + 1L

    /// Performs a sequence of full-information updates in enumeration order.
    member this.LearnBatch(feedback: seq<'Action -> float>) =
        if isNull (box feedback) then
            nullArg (nameof feedback)

        for utility in feedback do
            this.Learn utility

    /// Restores the initial strategy and clears all learning history.
    member _.Reset() =
        kernel.Reset strategy
        Array.Clear(strategySums, 0, strategySums.Length)
        rounds <- 0L

type private ExponentialWeightsKernel(initialStrategy: float[], learningRate: float) =
    let initialLogWeights = ExponentialWeightsInternals.logWeightsOf initialStrategy

    let logWeights = Array.copy initialLogWeights
    let candidateLogWeights = Array.zeroCreate initialStrategy.Length
    let nextStrategy = Array.zeroCreate initialStrategy.Length

    interface IExternalRegretKernel with
        member _.Learn(utilities, strategy) =
            ExponentialWeightsInternals.reweight
                learningRate
                utilities
                0.0
                logWeights
                candidateLogWeights
                nextStrategy

            Array.Copy(nextStrategy, strategy, strategy.Length)

        member _.Reset(strategy) =
            Array.Copy(initialStrategy, strategy, strategy.Length)
            Array.Copy(initialLogWeights, logWeights, logWeights.Length)

type private RegretMatchingKernel(initialStrategy: float[], plus: bool) =
    let regrets = Array.zeroCreate initialStrategy.Length
    let candidateRegrets = Array.zeroCreate initialStrategy.Length
    let nextStrategy = Array.zeroCreate initialStrategy.Length

    interface IExternalRegretKernel with
        member _.Learn(utilities, strategy) =
            // Scaling first prevents the expected-utility sum from overflowing
            // when callers use large but finite utilities.
            let mutable scale = 0.0

            for value in utilities do
                let magnitude = abs value
                if magnitude > scale then scale <- magnitude

            let expectedUtility =
                if scale = 0.0 then
                    0.0
                else
                    let mutable sum = 0.0
                    let mutable compensation = 0.0

                    for i = 0 to utilities.Length - 1 do
                        let term = strategy.[i] * (utilities.[i] / scale)
                        let corrected = term - compensation
                        let updated = sum + corrected
                        compensation <- (updated - sum) - corrected
                        sum <- updated

                    scale * (max -1.0 (min 1.0 sum))

            let mutable maximumPositive = 0.0

            for i = 0 to regrets.Length - 1 do
                let delta = utilities.[i] - expectedUtility
                let updated = regrets.[i] + delta

                if not (ExternalRegretInternals.isFinite delta)
                   || not (ExternalRegretInternals.isFinite updated) then
                    invalidArg
                        "utility"
                        "The utility range or cumulative regret exceeded the finite numeric range."

                let candidate = if plus && updated < 0.0 then 0.0 else updated
                candidateRegrets.[i] <- candidate

                if candidate > maximumPositive then
                    maximumPositive <- candidate

            if maximumPositive = 0.0 then
                Array.Copy(initialStrategy, nextStrategy, nextStrategy.Length)
            else
                let mutable scaledTotal = 0.0

                for candidate in candidateRegrets do
                    if candidate > 0.0 then
                        scaledTotal <- scaledTotal + candidate / maximumPositive

                for i = 0 to nextStrategy.Length - 1 do
                    nextStrategy.[i] <-
                        if candidateRegrets.[i] > 0.0 then
                            (candidateRegrets.[i] / maximumPositive) / scaledTotal
                        else
                            0.0

            Array.Copy(candidateRegrets, regrets, regrets.Length)
            Array.Copy(nextStrategy, strategy, strategy.Length)

        member _.Reset(strategy) =
            Array.Clear(regrets, 0, regrets.Length)
            Array.Copy(initialStrategy, strategy, strategy.Length)

/// Exponential multiplicative weights (Hedge) over a fixed action set.
[<RequireQualifiedAccess>]
module Mwua =
    /// Creates a learner with advanced initialization and sampling options.
    let createWith
        (options: ExternalRegretOptions)
        learningRate
        (actions: 'Action[])
        =
        if not (ExternalRegretInternals.isFinite learningRate) || learningRate <= 0.0 then
            invalidArg (nameof learningRate) "The learning rate must be finite and positive."

        let ownedActions, initialStrategy, random =
            ExternalRegretInternals.prepare options actions

        ExternalRegretMinimizer(
            ownedActions,
            initialStrategy,
            random,
            ExponentialWeightsKernel(initialStrategy, learningRate)
        )

    /// Creates an exponentially weighted learner with a uniform initial strategy.
    let create learningRate (actions: 'Action[]) =
        createWith ExternalRegretOptions.defaults learningRate actions

/// Full-information regret matching over a fixed action set.
[<RequireQualifiedAccess>]
module RegretMatching =
    let private make plus (options: ExternalRegretOptions) (actions: 'Action[]) =
        let ownedActions, initialStrategy, random =
            ExternalRegretInternals.prepare options actions

        ExternalRegretMinimizer(
            ownedActions,
            initialStrategy,
            random,
            RegretMatchingKernel(initialStrategy, plus)
        )

    /// Creates a standard regret-matching learner with advanced options.
    let createWith options (actions: 'Action[]) =
        make false options actions

    /// Creates a standard regret-matching learner with a uniform fallback strategy.
    let create (actions: 'Action[]) =
        createWith ExternalRegretOptions.defaults actions

    /// Creates a regret-matching+ learner with advanced options.
    let createPlusWith options (actions: 'Action[]) =
        make true options actions

    /// Creates a regret-matching+ learner with a uniform fallback strategy.
    let createPlus (actions: 'Action[]) =
        createPlusWith ExternalRegretOptions.defaults actions

/// A sampled action and the probability with which it was selected.
///
/// Choices are issued by BanditRegretMinimizer. A choice must be returned to
/// the same learner exactly once through Learn before another choice is made.
[<Sealed>]
type BanditChoice<'Action> internal (action: 'Action, probability: float, index: int) =
    /// The sampled action to evaluate in the environment.
    member _.Action = action

    /// The probability with which Action was sampled.
    member _.Probability = probability

    member internal _.Index = index

/// An EXP3 learner over a fixed action set.
///
/// Choose samples one action from the current exploration mixture. Learn
/// accepts only that action's observed reward, which must be in [0, 1], and
/// applies the corresponding importance-weighted update. LearnBatch performs
/// the same Choose/evaluate/Learn loop for each reward function in enumeration
/// order and evaluates each function only at its sampled action.
/// Instances are not thread-safe.
type BanditRegretMinimizer<'Action> internal
    (
        actions: 'Action[],
        initialStrategy: float[],
        random: Random,
        learningRate: float,
        exploration: float
    ) =

    let explorationShare = exploration / float actions.Length

    let initialLogWeights =
        let values = initialStrategy |> Array.map log
        let maximum = Array.max values
        values |> Array.map (fun value -> value - maximum)

    let logWeights = Array.copy initialLogWeights
    let candidateLogWeights = Array.zeroCreate actions.Length
    let strategy = Array.zeroCreate actions.Length
    let candidateStrategy = Array.zeroCreate actions.Length
    let strategySums = Array.zeroCreate actions.Length
    let mutable rounds = 0L
    let mutable pendingChoice: BanditChoice<'Action> option = None

    let setInitialStrategy () =
        for i = 0 to strategy.Length - 1 do
            strategy.[i] <-
                (1.0 - exploration) * initialStrategy.[i] + explorationShare

    let cancelIfPending choice =
        match pendingChoice with
        | Some pending when Object.ReferenceEquals(pending, choice) ->
            pendingChoice <- None
        | _ -> ()

    do setInitialStrategy ()

    /// The distribution from which the next action will be sampled.
    member _.Strategy =
        ExternalRegretInternals.snapshot actions strategy

    /// The mean of the sampling distributions used in completed rounds.
    /// Before the first round, this returns the current initial distribution.
    member _.AverageStrategy =
        if rounds = 0L then
            ExternalRegretInternals.snapshot actions strategy
        else
            let denominator = float rounds
            Array.init actions.Length (fun i -> actions.[i], strategySums.[i] / denominator)

    /// The number of successfully completed learning rounds.
    member _.Rounds = rounds

    /// Samples one action and reserves it as the learner's current round.
    member _.Choose() =
        if pendingChoice.IsSome then
            invalidOp "Learn or Reset must consume the outstanding bandit choice before Choose is called again."

        if rounds = Int64.MaxValue then
            invalidOp "The learner cannot represent another completed round."

        let draw = random.NextDouble()

        if not (ExternalRegretInternals.isFinite draw) || draw < 0.0 || draw >= 1.0 then
            invalidOp "The random source returned a value outside [0, 1)."

        let mutable cumulative = 0.0
        let mutable index = 0

        while index < strategy.Length - 1 && cumulative + strategy.[index] <= draw do
            cumulative <- cumulative + strategy.[index]
            index <- index + 1

        let choice = BanditChoice(actions.[index], strategy.[index], index)
        pendingChoice <- Some choice
        choice

    /// Applies one sampled-action reward in [0, 1].
    member _.Learn(choice: BanditChoice<'Action>, observedReward: float) =
        if isNull (box choice) then
            nullArg (nameof choice)

        match pendingChoice with
        | None ->
            invalidArg (nameof choice) "The learner has no outstanding choice."
        | Some pending when not (Object.ReferenceEquals(pending, choice)) ->
            invalidArg (nameof choice) "The choice is stale or belongs to another learner."
        | Some _ -> ()

        if not (ExternalRegretInternals.isFinite observedReward)
           || observedReward < 0.0
           || observedReward > 1.0 then
            invalidArg (nameof observedReward) "Observed rewards must be finite values in [0, 1]."

        let estimatedReward = observedReward / choice.Probability
        let increment = learningRate * estimatedReward
        let selectedCandidate = logWeights.[choice.Index] + increment

        if not (ExternalRegretInternals.isFinite estimatedReward)
           || not (ExternalRegretInternals.isFinite increment)
           || not (ExternalRegretInternals.isFinite selectedCandidate) then
            invalidArg
                (nameof observedReward)
                "The learning rate, exploration, and reward produced a non-finite EXP3 update."

        let mutable maximumCandidate = selectedCandidate

        for i = 0 to logWeights.Length - 1 do
            let candidate =
                if i = choice.Index then selectedCandidate else logWeights.[i]

            candidateLogWeights.[i] <- candidate

            if candidate > maximumCandidate then
                maximumCandidate <- candidate

        let mutable total = 0.0

        for i = 0 to candidateLogWeights.Length - 1 do
            let shifted = candidateLogWeights.[i] - maximumCandidate

            if not (ExternalRegretInternals.isFinite shifted) then
                invalidArg
                    (nameof observedReward)
                    "The EXP3 log-weight range exceeded the finite numeric range."

            candidateLogWeights.[i] <- shifted
            let weight = exp shifted
            candidateStrategy.[i] <- weight
            total <- total + weight

        if not (ExternalRegretInternals.isFinite total) || total <= 0.0 then
            invalidOp "The EXP3 update produced an invalid normalization total."

        for i = 0 to candidateStrategy.Length - 1 do
            let exploitationProbability = candidateStrategy.[i] / total
            candidateStrategy.[i] <-
                (1.0 - exploration) * exploitationProbability + explorationShare

        // Mutation begins only after the complete update has been validated.
        for i = 0 to strategy.Length - 1 do
            strategySums.[i] <- strategySums.[i] + strategy.[i]

        Array.Copy(candidateLogWeights, logWeights, logWeights.Length)
        Array.Copy(candidateStrategy, strategy, strategy.Length)
        rounds <- rounds + 1L
        pendingChoice <- None

    /// Runs sampled-action feedback sequentially in enumeration order.
    /// Each function is evaluated exactly once, for the action sampled in its round.
    member this.LearnBatch(rewardFunctions: seq<'Action -> float>) =
        if isNull (box rewardFunctions) then
            nullArg (nameof rewardFunctions)

        for reward in rewardFunctions do
            if isNull (box reward) then
                nullArg (nameof rewardFunctions)

            let choice = this.Choose()

            try
                this.Learn(choice, reward choice.Action)
            with _ ->
                // The choice is local to this loop, so cancel a failed round and
                // leave the learner usable. Earlier successful rounds remain.
                cancelIfPending choice
                reraise ()

    member internal _.Cancel(choice: BanditChoice<'Action>) =
        cancelIfPending choice

    /// Restores the initial exploration mixture and clears all learning history.
    /// Any outstanding choice becomes invalid.
    member _.Reset() =
        Array.Copy(initialLogWeights, logWeights, logWeights.Length)
        Array.Clear(strategySums, 0, strategySums.Length)
        setInitialStrategy ()
        rounds <- 0L
        pendingChoice <- None

/// Exponential weights for exploration and exploitation (EXP3).
[<RequireQualifiedAccess>]
module Exp3 =
    /// Creates EXP3 with advanced initialization and sampling options.
    let createWith
        (options: ExternalRegretOptions)
        learningRate
        exploration
        (actions: 'Action[])
        =
        if not (ExternalRegretInternals.isFinite learningRate) || learningRate <= 0.0 then
            invalidArg (nameof learningRate) "The learning rate must be finite and positive."

        if not (ExternalRegretInternals.isFinite exploration)
           || exploration <= 0.0
           || exploration > 1.0 then
            invalidArg (nameof exploration) "Exploration must be finite and in (0, 1]."

        let ownedActions, initialStrategy, random =
            ExternalRegretInternals.prepare options actions

        if exploration / float ownedActions.Length = 0.0 then
            invalidArg
                (nameof exploration)
                "Exploration is too small to assign positive floating-point mass to every action."

        if initialStrategy |> Array.exists (fun probability -> probability <= 0.0) then
            invalidArg
                "options.InitialStrategy"
                "EXP3 requires positive initial probability for every action."

        BanditRegretMinimizer(
            ownedActions,
            initialStrategy,
            random,
            learningRate,
            exploration
        )

    /// Creates EXP3 with a uniform initial strategy.
    let create learningRate exploration (actions: 'Action[]) =
        createWith ExternalRegretOptions.defaults learningRate exploration actions
