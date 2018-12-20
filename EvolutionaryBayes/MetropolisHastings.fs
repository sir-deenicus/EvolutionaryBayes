namespace EvolutionaryBayes

open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions

module SimpleMCMC = 
    let internal iterate n (lik : 'a -> float) (proposal : Distribution<_>) 
        (chain : list<float * 'a>) =
        let rec loop newChain (p, chainState) n =
            if n = 0 then newChain
            else 
                let nextDist =
                    dist { 
                        let! candidate = proposal
                        let q = lik candidate
                        let acp = min 1. (q / p)
                        let! accept = bernoulli acp
                        return (if accept then (q, candidate)
                                else (p, chainState))
                    }                
                let next = nextDist.Sample()
                loop (next :: newChain) next (n - 1)
        if (n <= 0) then (chain)
        else loop chain (List.head chain) n    
    let sample likelihood (prior : Distribution<_>) n =
        let x = prior.Sample()
        let initial = [ likelihood x, x ]
        let chain = iterate n likelihood prior initial
        List.map snd chain

module MetropolisHastings =
    let internal iterate n perturb (lik : 'a -> float) (proposal : Distribution<_>) initial =
        let rec loop newChain (p, chainState) n =
            if n = 0 then newChain
            else 
                let nextDist =
                    dist { 
                        let! meth = bernoulli 0.5
                        if meth then 
                            let! candidate = proposal
                            let p' = lik candidate 
                            let! accept = bernoulli (min 1. (p' / p))
                            if accept then return (p', candidate)
                            else return (p, chainState)
                        else 
                            let state' = perturb chainState
                            let p' = lik state' 
                            let! accept' = bernoulli (min 1. (p' / p))
                            return (if accept' then (p', state')
                                    else (p, chainState)) }                
                let next = nextDist.Sample()
                loop (snd next :: newChain) next (n - 1)
        if (n <= 0) then ([snd initial])
        else loop [snd initial] initial n    
    let sample likelihood perturb (prior : Distribution<_>) n =
        let x = prior.Sample()
        iterate n perturb likelihood prior (likelihood x, x) 
