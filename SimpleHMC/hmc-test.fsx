#I @"C:\Users\cybernetic\source\repos\DiffSharp\src\DiffSharp\bin\x64\Debug\netstandard2.0"
#r "netstandard.dll"
#r "DiffSharp.dll"
#r @"C:\Users\cybernetic\Code\Libs\MathNet\lib\net40\MathNet.Numerics.dll"
#r @"bin\Debug\netcoreapp2.1\EvolutionaryBayes.dll"
#r @"bin\x64\Debug\netcoreapp2.1\SimpleHMC.dll"
#r @"C:\Users\cybernetic\Code\Libs\net4+\Prelude\Prelude.dll"

open DiffSharp.AD.Float32
open EvolutionaryBayes.ProbMonad
open SimpleHMC
open Helpers
open Prelude.Common
open Prelude.Math
open System

let normal (mu : D) (sigma : D) (xv : DV) =
    log
        (1.f / (sqrt (2.f * float32 System.Math.PI * sigma ** 2.f))
         * exp (-(xv.[0] - mu) ** 2.f / (D 2.f * sigma ** 2.f)))


let numToDV x = toDV [|x|]

let lik1 =
    observe (fun param y -> Densities.multiNormal param (toDM [ [ 1.f ] ]) y)
        (List.map numToDV [ 5.f; 8.f; 7.f ]) //; 10.f; 4.f ]

let lik =
    observe (fun param y ->
        Densities.multiNormal param (toDM [ [ 1.; 0.95 ]
                                            [ 0.95; 1. ] ]
                                     |> DM.inverse) y) [ toDV [ 5.f; 15.f ]
                                                         toDV [ 10.f; 12.f ]
                                                         toDV [ 3.f; 9.f ] ] //; 10.f; 4.f ]

let samples = sample 0.1f 10 1000 lik (always ((toDV [ 0.; 0. ])))

let lik2 =
    observe (fun (param : DV) y ->
        let v = param.[..1]
        let cov = param.[2..] |> DV.unitDV
        Densities.multiNormal v (let m = DM.AddDiagonal(DM.zeroCreate 2 2, cov)
                                 m |> DM.inverse) y) [ toDV [ 5.f; 15.f ]
                                                       toDV [ 10.f; 12.f ]
                                                       toDV [ 3.f; 9.f ] ] //; 10.

let samples2 =
    sample (0.01f) 20 5000 lik2 (always ((toDV [ 0.; 0.; 0.5; 0.5 ])))
     
let p = sample (0.1f) 10 1000 lik1 (Samplers.normal 0. 1.)
  
let ps =
    (List.rev p).[100..]
    |> Sampling.roundAndGroupSamplesWith
           (fun v -> round 1 (float (float32 v.[0])))
    |> Array.sortByDescending snd
    |> Array.map (fun (x, y) -> (string x) + "," + (string y))

let inline roundf n = float >> round n


let zs =
    samples2
    |> List.map (fun v -> (float32 v.[0] |> string) + "," + (v.[1] |> float32 |> string))
            

let zs2 =
    samples2
    |> Sampling.roundAndGroupSamplesWith (
            DV.toArray
            >> Array.skip 0
            >> Array.take 2
            >> Array.map float32
            >> id
            >> Array.map (roundf 1))
    |> Array.sortByDescending snd
    |> Array.map (fun (x, p) ->
           Array.concat [| x; [| p |] |]
           |> Strings.joinToStringWith ",")

System.IO.File.WriteAllLines(pathCombine DocumentsFolder "ttt.txt", zs)
