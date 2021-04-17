namespace EvolutionaryBayes

open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions
open Prelude 

module MetropolisHastings =
    let internal iterateSimple n (lik : 'a -> float)
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

    let iterate attenuate T0 n perturb (likelihood : 'a -> float)
        (initial : float * 'a) =
        let rec loop T newChain (p, currentstate) n =
            if n = 0 then newChain
            else
                let state' = perturb currentstate
                let p' = likelihood state'
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
        iterateSimple n likelihood prior (likelihood x, x)

    let sample attentuate T likelihood perturb (n : int) init =
        iterate attentuate T n perturb likelihood (likelihood init, init)
         

///The perturbation/mutation step can be a sampler whose parameters are the current state. 
type MCMC<'data, 'sample when 'sample: equality>
        (mutator, loglikelihood, init, ?nsamples, ?Temperature, ?attenuate) =

    let T = defaultArg Temperature 1.
    let atten = defaultArg attenuate 1.
    let numsamples = defaultArg nsamples 100_000 
    let samples = ResizeArray<'sample>() 
    let mutable observations = ResizeArray<'data>()  

    member __.ClearSamples() = samples.Clear()

    member __.Samples = Seq.toArray samples 

    member __.GetDistribution(?groupWith) =
        SampleSummarize.roundAndGroupSamplesWith (defaultArg groupWith id) samples
        |> categorical2
        
    member __.SampleN(n, ?groupWith) =   
        SampleSummarize.roundAndGroupSamplesWith (defaultArg groupWith id) samples
        |> Sampling.discreteSampleN n        

    member __.Observations = observations
    
    member d.RunChain(?samplecount) = 
        samples.AddRange
            (MetropolisHastings.sample atten T (loglikelihood observations) mutator
                    (defaultArg samplecount numsamples) init) 
        d.Samples