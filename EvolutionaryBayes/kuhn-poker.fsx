#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\netstandard2.1\Prelude.dll"
#r @"bin/Debug/net5.0/EvolutionaryBayes.dll"

open Prelude.Common
open EvolutionaryBayes.CFR
open Prelude.Math

type Cards =
    | J = 1
    | Q = 2
    | K = 3
    
let inline reward (contexts: _ []) player history =
    let oppontent = 1 - player
    let won = contexts[player] > contexts[oppontent]

    match history with
    | "CC" -> Some(if won then 1. else -1.)
    | "BC" -> Some 1.
    | "BB"
    | "CBB" -> if won then Some 2. else Some -2.
    | "CBC" -> Some 1.
    | _ -> None
  
let adjustState _ _ action (history:string) = 
    history + action
 
let getActionMask _ _ _ = [|true; true|]  


let iterations = 1000000
  
let cards1 = [| Cards.J; Cards.Q; Cards.K|]


let nodeMap = Dict<string, StrategyNode>()

let utils = ResizeArray<float>()

for i in 1..iterations do
    let playercards = Array.shuffle cards1 |> Array.take 2
    let util = cfr 0 1. 1. reward basicLookUp adjustState getActionMask string nodeMap playercards [| "B"; "C" |] ""
    
    utils.Add(util) 
 
//Average game value:
utils |> Seq.average

[| for KeyValue (k, s) in nodeMap do
       let _, avg = getAvgStrategyForced s
       k, Array.map (round 4) avg |]
|> Array.sortBy fst
|> Array.map (fun (k, ps) ->
    let k2 =
        if (k.Length - 1) % 2 = 0 then "P1 " + k
        else "P2 " + k
    Array.concat [| [| k2 |]; Array.map string ps |])
|> makeTable "\n" [| ""; "B"; "C" |] ""