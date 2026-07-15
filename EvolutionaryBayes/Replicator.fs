namespace EvolutionaryBayes

open System
open System.Collections.Generic

/// Finite-strategy deterministic replicator dynamics.
[<RequireQualifiedAccess>]
module Replicator =
    let private isFinite value =
        not (Double.IsNaN value || Double.IsInfinity value)

    /// A normalized finite population over distinct named strategies.
    ///
    /// Population values are immutable. Strategies are shallow-copied at
    /// construction, and every public array is a fresh snapshot.
    [<Sealed>]
    type Population<'Strategy> internal (strategies: 'Strategy[], shares: float[]) =
        let comparer = EqualityComparer<'Strategy>.Default

        member internal _.StrategyValues = strategies
        member internal _.ShareValues = shares

        /// The number of strategies in the population.
        member _.Count = strategies.Length

        /// A fresh snapshot of the named, normalized population shares.
        member _.Distribution =
            Array.init strategies.Length (fun i -> strategies.[i], shares.[i])

        /// A fresh snapshot of the strategy names in their fixed order.
        member _.Strategies = Array.copy strategies

        /// A fresh snapshot of the normalized shares in strategy order.
        member _.Shares = Array.copy shares

        /// Returns the current share of a configured strategy.
        member _.Share(strategy: 'Strategy) =
            let mutable index = 0

            while index < strategies.Length
                  && not (comparer.Equals(strategies.[index], strategy)) do
                index <- index + 1

            if index = strategies.Length then
                invalidArg (nameof strategy) "The strategy does not belong to this population."

            shares.[index]

    /// Population-dependent fitness evaluated for one named strategy.
    type Fitness<'Strategy> = Population<'Strategy> -> 'Strategy -> float

    let private validatePopulation (population: Population<'Strategy>) =
        if isNull (box population) then
            nullArg (nameof population)

    let private validateFitness (fitness: Fitness<'Strategy>) =
        if isNull (box fitness) then
            nullArg (nameof fitness)

    let private validateTimeStep timeStep =
        if not (isFinite timeStep) || timeStep <= 0.0 then
            invalidArg (nameof timeStep) "The time step must be finite and positive."

    let private evaluateFitness
        (fitness: Fitness<'Strategy>)
        (population: Population<'Strategy>)
        =
        validatePopulation population
        validateFitness fitness

        let strategies = population.StrategyValues
        let values = Array.zeroCreate strategies.Length

        for i = 0 to strategies.Length - 1 do
            let value = fitness population strategies.[i]

            if not (isFinite value) then
                invalidArg (nameof fitness) "Fitness values must be finite."

            values.[i] <- value

        values

    let private copyPopulation (population: Population<'Strategy>) =
        Population(Array.copy population.StrategyValues, Array.copy population.ShareValues)

    /// Creates a finite population and normalizes its nonnegative shares.
    /// Strategy names must be distinct under their default equality comparer.
    let create (distribution: ('Strategy * float)[]) =
        if isNull distribution then
            nullArg (nameof distribution)

        if distribution.Length = 0 then
            invalidArg (nameof distribution) "At least one strategy is required."

        let strategies = Array.zeroCreate<'Strategy> distribution.Length
        let shares = Array.zeroCreate distribution.Length
        let seen = HashSet<'Strategy>(EqualityComparer<'Strategy>.Default)
        let mutable maximumShare = 0.0

        for i = 0 to distribution.Length - 1 do
            let strategy, share = distribution.[i]

            if not (seen.Add strategy) then
                invalidArg (nameof distribution) "Strategy names must be distinct."

            if not (isFinite share) then
                invalidArg (nameof distribution) "Population shares must be finite."

            if share < 0.0 then
                invalidArg (nameof distribution) "Population shares must be nonnegative."

            strategies.[i] <- strategy
            shares.[i] <- share

            if share > maximumShare then
                maximumShare <- share

        if maximumShare = 0.0 then
            invalidArg (nameof distribution) "The population must have positive total mass."

        // Scaling before summing avoids overflow for large finite input masses.
        let mutable scaledTotal = 0.0

        for share in shares do
            scaledTotal <- scaledTotal + share / maximumShare

        for i = 0 to shares.Length - 1 do
            shares.[i] <- (shares.[i] / maximumShare) / scaledTotal

        Population(strategies, shares)

    /// Computes dx_i/dt = x_i (f_i(x) - meanFitness(x)).
    /// The returned array is an independent snapshot in population order.
    let derivative
        (fitness: Fitness<'Strategy>)
        (population: Population<'Strategy>)
        =
        let values = evaluateFitness fitness population
        let shares = population.ShareValues
        let mutable referenceIndex = 0

        while shares.[referenceIndex] = 0.0 do
            referenceIndex <- referenceIndex + 1

        let referenceFitness = values.[referenceIndex]
        let centeredFitness = Array.zeroCreate values.Length
        let mutable centeredMean = 0.0
        let mutable compensation = 0.0

        // Centering makes a common fitness shift algebraically irrelevant and
        // improves the numeric range of the weighted mean. Fitness on absent
        // strategies cannot affect the derivative and is excluded here.
        for i = 0 to values.Length - 1 do
            if shares.[i] > 0.0 then
                let centered = values.[i] - referenceFitness

                if not (isFinite centered) then
                    invalidArg
                        (nameof fitness)
                        "The finite fitness range is too wide to form relative fitness values."

                centeredFitness.[i] <- centered
                let term = shares.[i] * centered
                let corrected = term - compensation
                let updated = centeredMean + corrected
                compensation <- (updated - centeredMean) - corrected
                centeredMean <- updated

        Array.init values.Length (fun i ->
            let rate =
                if shares.[i] = 0.0 then
                    0.0
                else
                    let relative = centeredFitness.[i] - centeredMean
                    let candidate = shares.[i] * relative

                    if not (isFinite relative) || not (isFinite candidate) then
                        invalidArg
                            (nameof fitness)
                            "The fitness values produced a non-finite replicator derivative."

                    candidate

            population.StrategyValues.[i], rate)

    /// Applies one positivity-preserving exponential replicator step.
    /// Zero-share strategies remain absent and positive-share strategies remain
    /// positive within the representable floating-point range.
    let step
        timeStep
        (fitness: Fitness<'Strategy>)
        (population: Population<'Strategy>)
        =
        validateTimeStep timeStep
        let values = evaluateFitness fitness population
        let shares = population.ShareValues
        let logWeights = ExponentialWeightsInternals.logWeightsOf shares
        let candidates = Array.zeroCreate shares.Length
        let nextShares = Array.zeroCreate shares.Length
        let minimumPositive = Double.Epsilon * float shares.Length

        ExponentialWeightsInternals.reweight
            timeStep
            values
            minimumPositive
            logWeights
            candidates
            nextShares

        Population(Array.copy population.StrategyValues, nextShares)

    /// Applies the same time step sequentially for the requested number of steps.
    /// Zero steps returns an independent population value without evaluating fitness.
    let run
        steps
        timeStep
        (fitness: Fitness<'Strategy>)
        (population: Population<'Strategy>)
        =
        if steps < 0 then
            invalidArg (nameof steps) "The step count must be nonnegative."

        validateTimeStep timeStep
        validateFitness fitness
        validatePopulation population

        let mutable current = copyPopulation population

        for _ = 1 to steps do
            current <- step timeStep fitness current

        current
