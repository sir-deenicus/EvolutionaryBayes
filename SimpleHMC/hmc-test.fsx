#I @"C:\Users\cybernetic\source\repos\DiffSharp\src\DiffSharp\bin\x64\Debug\netstandard2.0"
#r @"netstandard.dll"
#r @"DiffSharp.dll"
#r @"C:\Users\cybernetic\Code\Libs\MathNet\lib\net40\MathNet.Numerics.dll"
#r @"bin\Debug\netcoreapp2.1\EvolutionaryBayes.dll"
#r @"bin\x64\Debug\netcoreapp2.1\SimpleHMC.dll"
#r @"C:\Users\cybernetic\Code\Libs\net4+\Prelude\Prelude.dll"

open DiffSharp.AD.Float32
open EvolutionaryBayes.ProbMonad
open SimpleHMC
open Helpers
open Prelude.Common
open EvolutionaryBayes

let n = Samplers.normal 0. 1.

n.Sample() 

let multiNormal (mu:DV) sigma (x:DV) =
    let s = sigma |> DM.inverse
    (-((x - mu) * s * (x - mu)) / (D 2.f))

let hmc2 n hdelta hsteps (x0:DV) (f:DV->D) =
    let u x = -(f x) // potential energy
    let k p = (p * p) / D 2.f // kinetic energy
    let hamilton x p = u x + k p
    let x = ref x0
    [|for i in 1..n do
        let p = DV.init x0.Length (fun _ -> rndn())
        let x', p' = leapFrog u k hdelta hsteps (!x, p)
        if rnd() < float32 (exp ((hamilton !x p) - (hamilton x' p'))) then x := x'
        yield !x|]

let normal (mu : D) (sigma : D) (xv : DV) = 
    1.f/(sqrt(2.f * float32 System.Math.PI * sigma ** 2.f)) * exp(-(xv.[0] - mu) ** 2.f / (D 2.f * sigma ** 2.f))

let p = hmc2 10000 (D 0.01f) 10 (toDV [0.f]) f

let samples = 
    multiNormal (toDV [0.; 0.]) (toDM [[1.; 0.3]; [0.3; 1.]])
    |> hmc2 10000 (D 0.1f) 10 (toDV [0.; 0.])

let zs = samples  |> List.map (fun v -> (float32 v.[0] |> string)  + "," + (v.[1] |> float32 |> string ))

System.IO.File.WriteAllLines (pathCombine DocumentsFolder "ttt.txt", zs)

let lik =
    observe (fun x y -> Densities.normal (D x) (D 1.f) y) [ 5.f]//; 10.f; 4.f ] 

let lik2 =
    observe (fun (x:DV) y -> Densities.normal x.[0] x.[1] y) [ 5.f]//; 10.f; 4.f ] 

let lik =
    observe (fun x y -> Densities.multiNormal (toDV x) (toDM [[1.; 0.95]; [0.95; 1.]] |> DM.inverse) y) [ [5.f; 15.f]; [10.f ; 12.f] ; [3.f ; 9.f]]//; 10.f; 4.f ] 

let samples = sample (0.1f) 10 10000 (Densities.multiNormal (toDV [10.; 0.]) (toDM [[1.; 0.93]; [0.95; 1.]] |> DM.inverse)) (always ((toDV [0.; 0.])))
let samples = sample (0.1f) 10 10000 lik (always ((toDV [0.; 0.])))

let p = sample (0.001f) 30 1000 (Densities.normal (D 0.f) (D 1.f )) (always ((toDV [0.])))

let f =  normal (D 5.f) (D 1.f)
let p = sample 0.1f 10 10000 f n
p.[100..] 
|> Sampling.roundAndGroupSamplesWith (fun v -> round 1 (float(float32 v.[0])))
|> Array.sortByDescending snd

let ps = samples  |> Array.map (fun v -> float32 v.[0], float32 v.[1])
let qz = grad (fun (x,y)  -> x + D 1.f) 