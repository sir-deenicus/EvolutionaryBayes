#load "MWUA.fs"
#load "Replicator.fs"

open EvolutionaryBayes

type Action =
    | Go
    | Stop

let fitness (population: Replicator.Population<Action>) action =
    let goShare = population.Share Go

    match action with
    | Go -> -100.0 * goShare + (1.0 - goShare)
    | Stop -> 0.0

let initial = Replicator.create [| Go, 0.5; Stop, 0.5 |]

// This is the instantaneous continuous-time velocity, useful for analysis or
// an ODE solver. Replicator.step and Replicator.run do not consume it.
let derivative = Replicator.derivative fitness initial
let next = Replicator.step 0.01 fitness initial
let evolved = Replicator.run 5_000 0.01 fitness initial

printfn "Initial derivative: %A" derivative
printfn "After one step:     %A" next.Distribution
printfn "Stable mixture:     %A" evolved.Distribution
