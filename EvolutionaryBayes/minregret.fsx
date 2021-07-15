#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\netstandard2.1\Prelude.dll"
#r @"bin\Debug\netstandard2.1\EvolutionaryBayes.dll"

open Prelude
open Prelude.Math
open EvolutionaryBayes.RegretMinimization
 
type Dir = Stop | Go

let reward =
    function
    | (Go, Go) -> -100.
    | (Go, Stop) -> 1.
    | (Stop, Go) -> 0.
    | (Stop, Stop) -> 0.

let expert1 =
    RegretLearner<_, _>(reward, [| Go; Stop |], minreward = -100., maxreward = 100.)

let expert2 =
    RegretLearner<_, _>(reward, [| Go; Stop |], minreward = -100., maxreward = 100.)

for _ in 0 .. 999999 do
    let action1 = expert1.Sample()
    let action2 = expert2.Sample()

    expert1.Learn(observation = action1)
    expert2.Learn(action2)

expert1.NormalizedRegret

expert1.ActionWeights

expert1.Sample()
expert1.LearnedActionWeights()
expert2.LearnedActionWeights()

 