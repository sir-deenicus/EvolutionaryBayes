﻿namespace EvolutionaryBayes

open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions
open Helpers

module MetropolisHastings =
    let internal iterate n (lik : 'a -> float)
        (proposal : Distribution<_>) (initial : float * 'a) =
        let rec loop newChain (p, currentState) n =
            if n = 0 then newChain
            else
                let nextDist =
                    dist { 
                        let! candidate = proposal
                        let p' = lik candidate
                        let! accept = bernoulli (min 1. (logdiv p' p))
                        if accept then return (p', candidate)
                        else return (p, currentState) 
                    }

                let next = nextDist.Sample()
                
                loop (snd next :: newChain) next (n - 1)
        if (n <= 0) then ([ snd initial ])
        else loop [ snd initial ] initial n

    let iterate2 attenuate T0 n perturb (lik : 'a -> float)
        (initial : float * 'a) =
        let rec loop T newChain (p, currentstate) n =
            if n = 0 then newChain
            else
                let state' = perturb currentstate
                let p' = lik state'
                let accept' = bernoulli (min 1. (logdivT T p' p))

                let next =
                    if accept'.Sample() then (p', state')
                    else (p, currentstate)

                let T' = max 1. (T * attenuate)
                loop T' (snd next :: newChain) next (n - 1)
        if (n <= 0) then ([ snd initial ])
        else loop T0 [ snd initial ] initial n

    let sampleBasic likelihood (prior : Distribution<_>) (n : int) =
        let x = prior.Sample()
        iterate n likelihood prior (likelihood x, x)

    let sample attentuate T likelihood perturb (n : int) init =
        iterate2 attentuate T n perturb likelihood (likelihood init, init)

///The perturbation/mutation step can be a sampler whose parameters are the current state. 
type MCMC<'data, 'sample>
        (mutator, loglikelihood, init, ?nsamples, ?Temperature, ?attenuate) =

    let T = defaultArg Temperature 1.
    let atten = defaultArg attenuate 1.
    let numsamples = defaultArg nsamples 100_000 
    let samples = ResizeArray<'sample>() 
    let mutable defaultData : option<'data> = None

    member __.Samples = Seq.toArray samples

    member __.DefaultData with get() = defaultData and set d = defaultData <- d

    member __.Sample(?points : 'data list, ?samplecount) =
        samples.Clear()
        let points' =
            match points with
            | None ->
                match defaultData with
                | None -> failwith "cannot sample"
                | Some d -> [ d ]
            | Some d -> d
        samples.AddRange
            (MetropolisHastings.sample atten T (loglikelihood points') mutator
                 (defaultArg samplecount numsamples) init)