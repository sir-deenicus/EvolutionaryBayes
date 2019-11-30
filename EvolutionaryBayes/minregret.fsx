#r @"bin\Release\net47\Prelude.dll"
#r @"bin\Release\net47\EvolutionaryBayes.dll"

open EvolutionaryBayes.RegretMinimization

type Dir = Stop | Go

let rewardD = function
   | (Go, Go) -> -100.
   | (Go, Stop) -> 1.
   | (Stop, Go) -> 0.
   | (Stop, Stop) -> 0. 

let experts = Expert<int,_,_>( rewardD, [|Go;Stop|], minreward = -100., maxreward = 100.)        

for i in 0..1 do experts.New i
 
experts.Forget()
 
for _ in 0..99999 do     
    let c = experts.SampleActionAt 0 
    let c2 = experts.SampleActionAt 1 
     
    experts.Learn 0 c
    experts.Learn 1 c2
    
experts.[0].WeightedActions experts.Actions 
experts.WeightedActionsFor 1 
  

