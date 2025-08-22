namespace EvolutionaryBayes

open EvolutionaryBayes.ProbMonad
open EvolutionaryBayes.Distributions
open Prelude 

module MetropolisHastings =
    let internal iterateSimple n (lik : 'a -> float)
        (proposal : Distribution<_>) (initial : 'a * float) =
        let rec loop newChain (currentState, p) n =
            if n = 0 then newChain
            else
                let nextDist =
                    dist { 
                        let! candidate = proposal
                        let p' = lik candidate
                        let! accept = bernoulli (min 1. (logdiv p' p))
                        if accept then return (candidate, p')
                        else return (currentState, p) 
                    }

                let next = nextDist.Sample()
                
                loop (fst next :: newChain) next (n - 1)
        if (n <= 0) then ([ fst initial ])
        else loop [ fst initial ] initial n

    let iterate attenuate T0 n perturb (likelihood : 'a -> float)
        (initial : 'a * float) =
        let rec loop T newChain (currentstate, p) n =
            if n = 0 then newChain
            else
                let state' = perturb currentstate
                let p' = likelihood state'
                let accept' = bernoulli (min 1. (logdivT T p' p))

                let next =
                    if accept'.Sample() then (state', p')
                    else (currentstate, p)

                let T' = max 1. (T * attenuate)
                loop T' (fst next :: newChain) next (n - 1)
        if (n <= 0) then ([ fst initial ])
        else loop T0 [ fst initial ] initial n

    let sampleBasic likelihood (prior : Distribution<_>) (n : int) =
        let x = prior.Sample()
        iterateSimple n likelihood prior (x, likelihood x)

    let sample attentuate T likelihood perturb (n : int) init =
        iterate attentuate T n perturb likelihood (init, likelihood init)
         

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

    /// <summary>
    /// Gets the distribution of the samples, optionally grouped by a provided function.
    /// </summary>
    /// <param name="groupWith">An optional function to group the samples with. Defaults to the identity function if not provided.</param>
    /// <returns>The distribution of the samples as a categorical distribution.</returns>
    member __.GetDistribution(?groupWith) =
        SampleSummarize.roundAndGroupSamplesWith (defaultArg groupWith id) samples
        |> categorical2
        
    /// <summary>
    /// Samples a specified number of elements from the distribution.
    /// </summary>
    /// <param name="n">The number of samples to draw.</param>
    /// <param name="groupWith">An optional function to group the samples with. Defaults to the identity function if not provided.</param>
    /// <returns>A sequence of sampled elements.</returns>
    member __.SampleN(n, ?groupWith) =   
        SampleSummarize.roundAndGroupSamplesWith (defaultArg groupWith id) samples
        |> Sampling.discreteSampleN n        

    member __.Observations = observations
    
    member d.RunChain(?samplecount) = 
        samples.AddRange
            (MetropolisHastings.sample atten T (loglikelihood observations) mutator
                    (defaultArg samplecount numsamples) init) 
        d.Samples