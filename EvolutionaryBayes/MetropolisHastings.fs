namespace EvolutionaryBayes

open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions

module MetropolisHastings =
    let internal iterate attenuate T0 transitionp n perturb (lik : 'a -> float) (proposal : Distribution<_>) (initial: float * 'a) =
        let rec loop T newChain (p, chainState) n =
            if n = 0 then newChain
            else 
                let randomperturb = (bernoulli transitionp).Sample()
                let nextDist =
                    dist { 
                        if not randomperturb then 
                            let! candidate = proposal
                            let p' = lik candidate  
                            let! accept = bernoulli (min 1. (logdivT T p' p))
                            if accept then return (p', candidate)
                            else return (p, chainState)
                        else 
                            let state' = perturb chainState
                            let p' = lik state' 
                            let! accept' = bernoulli (min 1. (logdiv p' p))
                            return (if accept' then (p', state')
                                    else (p, chainState)) }                
                let next = nextDist.Sample()
                let T' = if randomperturb then T else max 1. (T * attenuate)
                loop T' (snd next :: newChain) next (n - 1)
        if (n <= 0) then ([snd initial])
        else loop T0 [snd initial] initial n   
        
    let internal iterate2 attenuate T0 n perturb (lik : 'a -> float) (initial: float * 'a) =
        let rec loop T newChain (p, chainState) n =
            if n = 0 then newChain
            else 
                let state' = perturb chainState
                let p' = lik state' 
                let accept' = bernoulli (min 1. (logdivT T p' p))
                let next = if accept'.Sample() then (p', state')
                           else (p, chainState) 
                let T' = max 1. (T * attenuate)
                loop T' (snd next :: newChain) next (n - 1)
        if (n <= 0) then ([snd initial])
        else loop T0 [snd initial] initial n    

    let sampleT attentuate T transitionp likelihood perturb (prior : Distribution<_>) (n:int) =
        let x = prior.Sample()
        iterate attentuate T transitionp n perturb likelihood prior (likelihood x, x)
      
    let sample2 attentuate T likelihood perturb (n:int) init = 
        iterate2 attentuate T n perturb likelihood (likelihood init, init)

    let sample transitionp likelihood perturb (prior : Distribution<_>) (n:int) =
        sampleT 1. 1. transitionp likelihood perturb prior n