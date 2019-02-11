namespace EvolutionaryBayes

open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions

module MetropolisHastings =
    let internal iterate transitionp n perturb (lik : 'a -> float) (proposal : Distribution<_>) (initial: float * 'a) =
        let rec loop newChain (p, chainState) n =
            if n = 0 then newChain
            else 
                let nextDist =
                    dist { 
                        let! meth = bernoulli transitionp
                        if not meth then 
                            let! candidate = proposal
                            let p' = lik candidate 
                            let! accept = bernoulli (min 1. (logdiv p' p))
                            if accept then return (p', candidate)
                            else return (p, chainState)
                        else 
                            let state' = perturb chainState
                            let p' = lik state' 
                            let! accept' = bernoulli (min 1. (logdiv p' p))
                            return (if accept' then (p', state')
                                    else (p, chainState)) }                
                let next = nextDist.Sample()
                loop (snd next :: newChain) next (n - 1)
        if (n <= 0) then ([snd initial])
        else loop [snd initial] initial n    
    let sample transitionp likelihood perturb (prior : Distribution<_>) (n:int) =
        let x = prior.Sample()
        iterate transitionp n perturb likelihood prior (likelihood x, x) 
