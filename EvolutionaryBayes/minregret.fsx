#r @"bin\Release\net47\Prelude.dll"
#r @"bin\Release\net47\EvolutionaryBayes.dll"

open EvolutionaryBayes.RegretMinimization

type Dir = Stop | Go

let rewardD = function
   | (Go, Go) -> -100.
   | (Go, Stop) -> 1.
   | (Stop, Go) -> 0.
   | (Stop, Stop) -> 0. 

let experts = RegretLearner<int,_,_>( rewardD, [|Go;Stop|], minreward = -100., maxreward = 100.)        

experts.New [0..1]
 
experts.Forget()
 
for _ in 0..999999 do     
    let c = experts.SampleActionOf 0 
    let c2 = experts.SampleActionOf 1 
     
    experts.Learn(0,c)
    experts.Learn(1,c2)
    
experts.[0].WeightedActions experts.Actions 
experts.WeightedActionsFor 1 

experts.WeightedActions()
  

