#I @"C:\Users\cybernetic\source\repos\DiffSharp\src\DiffSharp\bin\x64\Debug\netstandard2.0"
#r "netstandard.dll"
#r "DiffSharp.dll"
#r @"C:\Users\cybernetic\Code\Libs\MathNet\lib\net40\MathNet.Numerics.dll"
#r @"bin\x64\Debug\netcoreapp2.1\EvolutionaryBayes.dll"
#r @"bin\x64\Debug\netcoreapp2.1\SimpleHMC.dll"
#r @"C:\Users\cybernetic\Code\Libs\net4+\Prelude\Prelude.dll"

open DiffSharp.AD.Float32
open EvolutionaryBayes.ProbMonad
open SimpleHMC
open Helpers
open Prelude.Common
open Prelude.Math
open System 
open SimpleHMC
open EvolutionaryBayes.Distributions

let sample hd = SimpleHMC.sample None 100 simpleEuclideanKineticEnergy hd


let lik =
    observe toDV (fun param y ->
        LogDensities.multiNormal y (toDM [ [ 1.; 0.95 ]
                                           [ 0.95; 1. ] ]
                                     |> DM.inverse) param) [  [ 5.f; 15.f ]
                                                              [ 10.f; 12.f ]
                                                              [ 3.f; 9.f ] ] //; 10.f; 4.f ]

let samples = sample 0.1f 10 10000 lik (always ((toDV [ 0.; 0. ])))

let p2 = sample 0.1f 20 5000 (LogDensities.studentT (D 2.f) (D 1.f) (2.f)) (always ((toDV [ 0.5 ])))

let p3 = sample 0.01f 20 10000 (LogDensities.truncatedNormal 0.f 10.f (DV.ofNum 0.f) (D 1.f)) (always ((toDV [ 0.5 ])))
let p4 = sample 0.01f 20 10000 (LogDensities.boundSamples 0.f 10.f (LogDensities.normal (DV.ofNum 0.f) (D 1.f))) (always ((toDV [ 0.5 ])))

//not the most correct thing to do.  
let lik2 =
    observe toDV (fun (param : DV) y ->
        let v = param.[..1]
        
        let vars =
            param.[2..] |> DV.toArray |> Array.map float32

        let ma =
            array2D [ [ vars.[1]; vars.[0] ]
                      [ vars.[0]; vars.[2] ] ]
            |> Array2D.map D

        let m = DM.ofArray2D ma
        LogDensities.multiNormal v (DM.inverse m) y) [ [ 5.f; 15.f ]
                                                       [ 10.f; 12.f ]
                                                       [ 3.f; 9.f ] ] //; 10.

let samples2 =
    sample 0.1f 25 5000 lik2 (always ((toDV [ 0.; 0.; 0.5; 2.; 2. ])))

let density param =
    LogDensities.normal (DV.ofNum 0.f) (D 1.f) param //prior
    + observe DV.ofNum (fun param y -> LogDensities.normal param (1.f) y) //likelihood
          [ 5.f; 8.f; 7.f ] param //; 10.f; 4.f ]

 
let p0 = sample 0.1f 10 5000 density (Samplers.sampleScalar (normal 0. 1.))

let lik1 =
    observe DV.ofNum (fun param y -> LogDensities.normal param (1.f) y) [ 5.f; 8.f; 7.f ]  //; 10.f; 4.f ]

let p = sample 0.1f 10 5000 lik1 (Samplers.sampleScalar (normal 0. 1.))

let ps =
    p3
    |> Sampling.roundAndGroupSamplesWith
           (fun v -> round 1 (float (float32 v.[0])))
    |> Array.sortByDescending snd
    |> Array.map (fun (x, y) -> (string x) + "," + (string y))



let zs =
    samples2
    |> List.map (fun v -> (float32 v.[0] |> string) + "," + (v.[1] |> float32 |> string))
            
let inline roundf n = float >> round n

let zs2 =
    samples2
    |> Sampling.roundAndGroupSamplesWith (
            DV.toArray
            >> Array.skip 2
            >> Array.take 1
            >> Array.map float32 
            >> Array.map (roundf 0 >> bucketRange 0 5.))
    |> Array.sortByDescending snd
    |> Array.map (fun (x, p) ->
           Array.concat [| x; [| p |] |]
           |> Strings.joinToStringWith ",")

System.IO.File.WriteAllLines(pathCombine DocumentsFolder "ttt.txt", ps)
