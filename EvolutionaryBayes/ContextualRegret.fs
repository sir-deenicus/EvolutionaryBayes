namespace EvolutionaryBayes

open System
open System.Collections.Generic

module private ContextualFactory =
    let snapshotActions (actions: 'Action[]) =
        if isNull actions then null else Array.copy actions

    let snapshotOptions (options: ExternalRegretOptions) =
        if isNull (box options) then
            options
        else
            let initialStrategy =
                options.InitialStrategy
                |> Option.map (fun values ->
                    if isNull values then null else Array.copy values)

            { options with InitialStrategy = initialStrategy }

    // Construct one learner eagerly so invalid algorithm parameters fail at the
    // contextual constructor rather than at the first observed context.
    let validated (create: unit -> 'Learner) =
        let mutable first = Some(create ())

        fun () ->
            match first with
            | Some learner ->
                first <- None
                learner
            | None -> create ()

/// Independent full-information external-regret learners indexed by exact context.
///
/// An unseen context receives a fresh learner. Learn applies its feedback even
/// on that first encounter, and later calls for an equal context reuse the same
/// learner. LearnBatch processes (context, feedback) pairs sequentially.
/// Instances are not thread-safe.
type ExactContextExternalRegretMinimizer<'Context, 'Action when 'Context: equality> internal
    (createLearner: unit -> ExternalRegretMinimizer<'Action>) =

    let learners = Dictionary<'Context, ExternalRegretMinimizer<'Action>>()

    let validateContext context =
        if isNull (box context) then
            nullArg (nameof context)

    let getOrCreate context =
        validateContext context

        match learners.TryGetValue context with
        | true, learner -> learner
        | false, _ ->
            let learner = createLearner ()
            learners.Add(context, learner)
            learner

    /// The number of distinct contexts encountered by a state-reading or learning call.
    member _.ContextCount = learners.Count

    /// The strategy for an exact context. An unseen context is initialized.
    member _.Strategy(context: 'Context) =
        (getOrCreate context).Strategy

    /// The mean strategy played in completed rounds for an exact context.
    member _.AverageStrategy(context: 'Context) =
        (getOrCreate context).AverageStrategy

    /// The number of completed rounds for an exact context.
    member _.Rounds(context: 'Context) =
        (getOrCreate context).Rounds

    /// Samples from the learner belonging to the exact context.
    member _.Choose(context: 'Context) =
        (getOrCreate context).Choose()

    /// Applies one full-information update to the learner for the exact context.
    member _.Learn(context: 'Context, utility: 'Action -> float) =
        (getOrCreate context).Learn utility

    /// Applies (context, full-information feedback) pairs in enumeration order.
    member this.LearnBatch(feedback: seq<'Context * ('Action -> float)>) =
        if isNull (box feedback) then
            nullArg (nameof feedback)

        for context, utility in feedback do
            this.Learn(context, utility)

    /// Resets every initialized context while retaining the exact-context table.
    member _.Reset() =
        for learner in learners.Values do
            learner.Reset()

/// A sampled contextual action and the probability with which it was selected.
///
/// Return a choice to the same learner exactly once through Learn before
/// requesting another choice.
[<Sealed>]
type ContextualBanditChoice<'Context, 'Action> internal
    (context: 'Context, action: 'Action, probability: float) =

    /// The context supplied when the action was selected.
    member _.Context = context

    /// The sampled action to evaluate in the environment.
    member _.Action = action

    /// The probability with which Action was sampled.
    member _.Probability = probability

/// Independent EXP3 learners indexed by exact context.
///
/// Choose creates or reuses the learner for its exact context. Learn routes the
/// resulting opaque choice back to that same learner, so the first observation
/// is never dropped or applied to a different context. Instances are not
/// thread-safe.
type ExactContextBanditRegretMinimizer<'Context, 'Action when 'Context: equality> internal
    (createLearner: unit -> BanditRegretMinimizer<'Action>) =

    let learners = Dictionary<'Context, BanditRegretMinimizer<'Action>>()
    let mutable pending: (ContextualBanditChoice<'Context, 'Action> * BanditRegretMinimizer<'Action> * BanditChoice<'Action>) option = None

    let validateContext context =
        if isNull (box context) then
            nullArg (nameof context)

    let getOrCreate context =
        validateContext context

        match learners.TryGetValue context with
        | true, learner -> learner
        | false, _ ->
            let learner = createLearner ()
            learners.Add(context, learner)
            learner

    let cancelIfPending choice =
        match pending with
        | Some (expected, learner, innerChoice) when Object.ReferenceEquals(expected, choice) ->
            learner.Cancel innerChoice
            pending <- None
        | _ -> ()

    /// The number of distinct contexts encountered by a state-reading or learning call.
    member _.ContextCount = learners.Count

    /// The EXP3 sampling strategy for an exact context. An unseen context is initialized.
    member _.Strategy(context: 'Context) =
        (getOrCreate context).Strategy

    /// The mean EXP3 sampling strategy for an exact context.
    member _.AverageStrategy(context: 'Context) =
        (getOrCreate context).AverageStrategy

    /// The number of completed EXP3 rounds for an exact context.
    member _.Rounds(context: 'Context) =
        (getOrCreate context).Rounds

    /// Samples one action for a context and reserves it as the current round.
    member _.Choose(context: 'Context) =
        if pending.IsSome then
            invalidOp "Learn or Reset must consume the outstanding contextual choice before Choose is called again."

        let learner = getOrCreate context
        let innerChoice = learner.Choose()
        let choice = ContextualBanditChoice(context, innerChoice.Action, innerChoice.Probability)
        pending <- Some(choice, learner, innerChoice)
        choice

    /// Applies one sampled-action reward in [0, 1] to the learner that issued the choice.
    member _.Learn(choice: ContextualBanditChoice<'Context, 'Action>, observedReward: float) =
        if isNull (box choice) then
            nullArg (nameof choice)

        match pending with
        | None ->
            invalidArg (nameof choice) "The learner has no outstanding contextual choice."
        | Some (expected, _, _) when not (Object.ReferenceEquals(expected, choice)) ->
            invalidArg (nameof choice) "The choice is stale or belongs to another learner."
        | Some (_, learner, innerChoice) ->
            learner.Learn(innerChoice, observedReward)
            pending <- None

    /// Runs exact-context sampled-action feedback in enumeration order.
    member this.LearnBatch(feedback: seq<'Context * ('Action -> float)>) =
        if isNull (box feedback) then
            nullArg (nameof feedback)

        for context, reward in feedback do
            if isNull (box reward) then
                nullArg (nameof feedback)

            let choice = this.Choose context

            try
                this.Learn(choice, reward choice.Action)
            with _ ->
                cancelIfPending choice
                reraise ()

    /// Resets every initialized context and invalidates any outstanding choice.
    member _.Reset() =
        for learner in learners.Values do
            learner.Reset()

        pending <- None

/// Exact-context exponential multiplicative weights.
[<RequireQualifiedAccess>]
module ContextualMwua =
    /// Creates exact-context MWUA with advanced per-context initialization options.
    let createWith options learningRate (actions: 'Action[]) =
        let ownedOptions = ContextualFactory.snapshotOptions options
        let ownedActions = ContextualFactory.snapshotActions actions
        let factory =
            ContextualFactory.validated (fun () -> Mwua.createWith ownedOptions learningRate ownedActions)

        ExactContextExternalRegretMinimizer<'Context, 'Action>(factory)

    /// Creates exact-context MWUA with a uniform initial strategy per context.
    let create learningRate (actions: 'Action[]) =
        createWith ExternalRegretOptions.defaults learningRate actions

/// Exact-context full-information regret matching.
[<RequireQualifiedAccess>]
module ContextualRegretMatching =
    /// Creates exact-context standard regret matching with advanced per-context options.
    let createWith options (actions: 'Action[]) =
        let ownedOptions = ContextualFactory.snapshotOptions options
        let ownedActions = ContextualFactory.snapshotActions actions
        let factory =
            ContextualFactory.validated (fun () -> RegretMatching.createWith ownedOptions ownedActions)

        ExactContextExternalRegretMinimizer<'Context, 'Action>(factory)

    /// Creates exact-context standard regret matching with a uniform fallback per context.
    let create (actions: 'Action[]) =
        createWith ExternalRegretOptions.defaults actions

    /// Creates exact-context regret matching+ with advanced per-context options.
    let createPlusWith options (actions: 'Action[]) =
        let ownedOptions = ContextualFactory.snapshotOptions options
        let ownedActions = ContextualFactory.snapshotActions actions
        let factory =
            ContextualFactory.validated (fun () -> RegretMatching.createPlusWith ownedOptions ownedActions)

        ExactContextExternalRegretMinimizer<'Context, 'Action>(factory)

    /// Creates exact-context regret matching+ with a uniform fallback per context.
    let createPlus (actions: 'Action[]) =
        createPlusWith ExternalRegretOptions.defaults actions

/// Exact-context EXP3 sampled-action learning.
[<RequireQualifiedAccess>]
module ContextualExp3 =
    /// Creates exact-context EXP3 with advanced per-context initialization options.
    let createWith options learningRate exploration (actions: 'Action[]) =
        let ownedOptions = ContextualFactory.snapshotOptions options
        let ownedActions = ContextualFactory.snapshotActions actions
        let factory =
            ContextualFactory.validated (fun () -> Exp3.createWith ownedOptions learningRate exploration ownedActions)

        ExactContextBanditRegretMinimizer<'Context, 'Action>(factory)

    /// Creates exact-context EXP3 with a uniform initial strategy per context.
    let create learningRate exploration (actions: 'Action[]) =
        createWith ExternalRegretOptions.defaults learningRate exploration actions

/// A deterministic contextual policy that recommends one action for a context.
type ContextualPolicy<'Context, 'Action> = 'Context -> 'Action

/// EXP4 over a fixed set of deterministic contextual policies.
///
/// Each policy is evaluated once by Choose. Their weighted recommendations are
/// mixed with uniform exploration to sample an action. Learn accepts only the
/// sampled reward in [0, 1] and importance-weights it back to every policy that
/// recommended the sampled action. Policies should be pure and deterministic
/// for a given context. Instances are not thread-safe.
type ContextualBanditRegretMinimizer<'Context, 'Action> internal
    (
        actions: 'Action[],
        policies: ContextualPolicy<'Context, 'Action>[],
        initialPolicyStrategy: float[],
        random: Random,
        learningRate: float,
        exploration: float
    ) =

    let comparer = EqualityComparer<'Action>.Default
    let explorationShare = exploration / float actions.Length

    let initialLogWeights =
        initialPolicyStrategy
        |> Array.map (fun probability ->
            if probability = 0.0 then Double.NegativeInfinity
            else log probability)

    let logWeights = Array.copy initialLogWeights
    let candidateLogWeights = Array.zeroCreate policies.Length
    let policyStrategy = Array.copy initialPolicyStrategy
    let candidatePolicyStrategy = Array.zeroCreate policies.Length
    let policyStrategySums = Array.zeroCreate policies.Length
    let pendingRecommendations = Array.zeroCreate policies.Length
    let pendingActionStrategy = Array.zeroCreate actions.Length
    let inspectionRecommendations = Array.zeroCreate policies.Length
    let inspectionActionStrategy = Array.zeroCreate actions.Length
    let mutable rounds = 0L
    let mutable pending: (ContextualBanditChoice<'Context, 'Action> * int) option = None

    let findActionIndex recommendation =
        let mutable index = 0

        while index < actions.Length && not (comparer.Equals(actions.[index], recommendation)) do
            index <- index + 1

        if index = actions.Length then -1 else index

    let computeActionStrategy
        (context: 'Context)
        (recommendations: int[])
        (actionStrategy: float[])
        =
        Array.Clear(actionStrategy, 0, actionStrategy.Length)

        // Caller policy code is evaluated completely before a choice becomes pending.
        for policyIndex = 0 to policies.Length - 1 do
            let recommendation = policies.[policyIndex] context
            let actionIndex = findActionIndex recommendation

            if actionIndex < 0 then
                invalidArg
                    "policies"
                    "Every contextual policy recommendation must belong to the configured action set."

            recommendations.[policyIndex] <- actionIndex
            actionStrategy.[actionIndex] <- actionStrategy.[actionIndex] + policyStrategy.[policyIndex]

        for actionIndex = 0 to actionStrategy.Length - 1 do
            actionStrategy.[actionIndex] <-
                (1.0 - exploration) * actionStrategy.[actionIndex] + explorationShare

    let sampleActionIndex (strategy: float[]) =
        let draw = random.NextDouble()

        if not (ExternalRegretInternals.isFinite draw) || draw < 0.0 || draw >= 1.0 then
            invalidOp "The random source returned a value outside [0, 1)."

        let mutable cumulative = 0.0
        let mutable index = 0

        while index < strategy.Length - 1 && cumulative + strategy.[index] <= draw do
            cumulative <- cumulative + strategy.[index]
            index <- index + 1

        index

    let cancelIfPending choice =
        match pending with
        | Some (expected, _) when Object.ReferenceEquals(expected, choice) -> pending <- None
        | _ -> ()

    /// The current exponential-weights distribution over contextual policies.
    member _.PolicyStrategy =
        ExternalRegretInternals.snapshot policies policyStrategy

    /// The mean policy distribution used in completed rounds.
    /// Before the first round, this returns the configured initial distribution.
    member _.AveragePolicyStrategy =
        if rounds = 0L then
            ExternalRegretInternals.snapshot policies policyStrategy
        else
            let denominator = float rounds
            Array.init policies.Length (fun i -> policies.[i], policyStrategySums.[i] / denominator)

    /// The explored action distribution induced by the current policies for a context.
    /// This does not reserve a choice or advance the learner.
    member _.Strategy(context: 'Context) =
        computeActionStrategy context inspectionRecommendations inspectionActionStrategy
        ExternalRegretInternals.snapshot actions inspectionActionStrategy

    /// The number of successfully completed learning rounds.
    member _.Rounds = rounds

    /// Samples one action for a context and reserves it as the current round.
    member _.Choose(context: 'Context) =
        if pending.IsSome then
            invalidOp "Learn or Reset must consume the outstanding contextual choice before Choose is called again."

        if rounds = Int64.MaxValue then
            invalidOp "The learner cannot represent another completed round."

        computeActionStrategy context pendingRecommendations pendingActionStrategy
        let actionIndex = sampleActionIndex pendingActionStrategy
        let choice =
            ContextualBanditChoice(context, actions.[actionIndex], pendingActionStrategy.[actionIndex])

        pending <- Some(choice, actionIndex)
        choice

    /// Applies the sampled action's observed reward in [0, 1].
    member _.Learn(choice: ContextualBanditChoice<'Context, 'Action>, observedReward: float) =
        if isNull (box choice) then
            nullArg (nameof choice)

        let selectedActionIndex =
            match pending with
            | None ->
                invalidArg (nameof choice) "The learner has no outstanding contextual choice."
            | Some (expected, _) when not (Object.ReferenceEquals(expected, choice)) ->
                invalidArg (nameof choice) "The choice is stale or belongs to another learner."
            | Some (_, actionIndex) -> actionIndex

        if not (ExternalRegretInternals.isFinite observedReward)
           || observedReward < 0.0
           || observedReward > 1.0 then
            invalidArg (nameof observedReward) "Observed rewards must be finite values in [0, 1]."

        let estimatedReward = observedReward / choice.Probability
        let increment = learningRate * estimatedReward

        if not (ExternalRegretInternals.isFinite estimatedReward)
           || not (ExternalRegretInternals.isFinite increment) then
            invalidArg
                (nameof observedReward)
                "The learning rate, exploration, and reward produced a non-finite EXP4 update."

        let mutable maximumCandidate = Double.NegativeInfinity

        for policyIndex = 0 to logWeights.Length - 1 do
            let candidate =
                if Double.IsNegativeInfinity logWeights.[policyIndex] then
                    Double.NegativeInfinity
                elif pendingRecommendations.[policyIndex] = selectedActionIndex then
                    logWeights.[policyIndex] + increment
                else
                    logWeights.[policyIndex]

            if Double.IsNaN candidate || Double.IsPositiveInfinity candidate then
                invalidArg
                    (nameof observedReward)
                    "The EXP4 policy log-weight update exceeded the finite numeric range."

            candidateLogWeights.[policyIndex] <- candidate

            if candidate > maximumCandidate then
                maximumCandidate <- candidate

        if not (ExternalRegretInternals.isFinite maximumCandidate) then
            invalidOp "The EXP4 update removed all positive policy weight."

        let mutable total = 0.0

        for policyIndex = 0 to candidateLogWeights.Length - 1 do
            let weight =
                if Double.IsNegativeInfinity candidateLogWeights.[policyIndex] then
                    0.0
                else
                    exp (candidateLogWeights.[policyIndex] - maximumCandidate)

            candidatePolicyStrategy.[policyIndex] <- weight
            total <- total + weight

        if not (ExternalRegretInternals.isFinite total) || total <= 0.0 then
            invalidOp "The EXP4 update produced an invalid policy normalization total."

        // Mutation begins only after the complete importance-weighted update is valid.
        for policyIndex = 0 to policyStrategy.Length - 1 do
            policyStrategySums.[policyIndex] <-
                policyStrategySums.[policyIndex] + policyStrategy.[policyIndex]

            policyStrategy.[policyIndex] <- candidatePolicyStrategy.[policyIndex] / total
            logWeights.[policyIndex] <- candidateLogWeights.[policyIndex] - maximumCandidate

        rounds <- rounds + 1L
        pending <- None

    /// Runs (context, sampled-action reward function) pairs in enumeration order.
    member this.LearnBatch(feedback: seq<'Context * ('Action -> float)>) =
        if isNull (box feedback) then
            nullArg (nameof feedback)

        for context, reward in feedback do
            if isNull (box reward) then
                nullArg (nameof feedback)

            let choice = this.Choose context

            try
                this.Learn(choice, reward choice.Action)
            with _ ->
                cancelIfPending choice
                reraise ()

    /// Restores the initial policy distribution and invalidates any outstanding choice.
    member _.Reset() =
        Array.Copy(initialLogWeights, logWeights, logWeights.Length)
        Array.Copy(initialPolicyStrategy, policyStrategy, policyStrategy.Length)
        Array.Clear(policyStrategySums, 0, policyStrategySums.Length)
        rounds <- 0L
        pending <- None

/// Exponential weights for exploration and exploitation with policy advice (EXP4).
[<RequireQualifiedAccess>]
module Exp4 =
    /// Creates EXP4 with advanced policy initialization and sampling options.
    /// InitialStrategy, when supplied, is a distribution over policies.
    let createWith
        (options: ExternalRegretOptions)
        learningRate
        exploration
        (actions: 'Action[])
        (policies: ContextualPolicy<'Context, 'Action>[])
        =
        if not (ExternalRegretInternals.isFinite learningRate) || learningRate <= 0.0 then
            invalidArg (nameof learningRate) "The learning rate must be finite and positive."

        if not (ExternalRegretInternals.isFinite exploration)
           || exploration <= 0.0
           || exploration > 1.0 then
            invalidArg (nameof exploration) "Exploration must be finite and in (0, 1]."

        if isNull actions then
            nullArg (nameof actions)

        if actions.Length = 0 then
            invalidArg (nameof actions) "At least one action is required."

        let ownedActions = Array.copy actions
        let comparer = EqualityComparer<'Action>.Default

        for i = 0 to ownedActions.Length - 1 do
            for j = i + 1 to ownedActions.Length - 1 do
                if comparer.Equals(ownedActions.[i], ownedActions.[j]) then
                    invalidArg (nameof actions) "EXP4 requires distinct action values."

        let ownedPolicies, initialPolicyStrategy, random =
            ExternalRegretInternals.prepare options policies

        for policy in ownedPolicies do
            if isNull (box policy) then
                invalidArg (nameof policies) "Contextual policies cannot be null."

        if exploration / float ownedActions.Length = 0.0 then
            invalidArg
                (nameof exploration)
                "Exploration is too small to assign positive floating-point mass to every action."

        ContextualBanditRegretMinimizer(
            ownedActions,
            ownedPolicies,
            initialPolicyStrategy,
            random,
            learningRate,
            exploration
        )

    /// Creates EXP4 with a uniform initial distribution over policies.
    let create
        learningRate
        exploration
        (actions: 'Action[])
        (policies: ContextualPolicy<'Context, 'Action>[])
        =
        createWith ExternalRegretOptions.defaults learningRate exploration actions policies
