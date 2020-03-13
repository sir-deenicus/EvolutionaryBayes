module EvolutionaryBayes.ParticleFilters

open EvolutionaryBayes.ProbMonad
open System
open Prelude.Common
open Prelude.Math
open Helpers
open EvolutionaryBayes
open System.Collections.Generic
open EvolutionaryBayes.Distributions
open Helpers.Sampling
 
let reweightWith likelihood samples =
    Array.countBy id samples
    |> normalizeWeights
    |> Array.map (fun (x, p) -> x, p * likelihood x)
    |> normalizeWeights

let weightWithT T likelihood samples =
    samples
    |> Array.map (fun x -> x, (likelihood x) ** (1./T))
    |> normalizeWeights

let weightWith likelihood samples = weightWithT 1. likelihood samples

let importanceSamples (likelihood : 'a -> float) (n : int)
    (prior : Distribution<_>) = prior.SampleN n |> reweightWith likelihood

let importanceSamplesArray (likelihood : 'a -> float) (n : int)
    (samples) = discreteSampleN n samples |> reweightWith likelihood

let recursiveImportanceSample T attenuate mutateprob mutate (likelihood : 'a -> float)
    (numparticles : int) (numsteps : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood numparticles prior 

    let rec loop T samples dist =
        if samples = 0 then dist
        else
            discreteSampleN numparticles dist
            |> Array.map (fun x ->
                   if random.NextDouble() < mutateprob then mutate choices x
                   else x)
            |> weightWithT T likelihood 
            |> importanceSamplesArray likelihood numparticles 
            |> loop (max 1. (T * attenuate)) (samples - 1)
    loop T numsteps choices

///Inspired by evolution (see papers on regret minimization's connection to evolution), this
///method maintains a memory where each element is a set of samples weighted by their average
///performance. The head of this list is the most recent generation. Worst performing generations
///are evicted when memory size limit is reached. Hopefully, by then their descendants can be found
///in later generations. At each step, a generation is sampled and then a member of that generation.
///Because the memory size is bounded there is a fixed upper bound on total samples with the population
///mixing and hopefully drifting towards the most fit.
///If the prior is too strong (such as an always or with too little variation), this will 
///end up dominating the evolution, acting like a strong gravitational attractor causing
///exploration to stick too close to the initial state. Forget prior is for when the prior is too certain.
let evolveSequence T atten forgetPrior mutateprob maxsize mem mutate
    (likelihood : 'a -> float) (samplesPerGen : int) (numSteps : int)
    (prior : Distribution<_>) =

    let rec loop T steps mem =
        if steps = 0 then mem
        else
            let population =  
                match mem with
                | [] -> prior
                | _ ->
                    distBuilder (fun () ->
                        let history =
                            discreteSample //sample a generation
                                (normalizeWeights [| for (m, p) in mem -> m, p ** (1. / T) |]) 

                        let hypothesis =
                            discreteSample //sample a hypothesis from selected generation
                                (Array.map (fun (x, p) -> x, p ** (1. / T)) history
                                |> normalizeWeights) 
                        hypothesis)

            let samples =
                population.SampleN samplesPerGen
                |> Array.map (fun sample ->
                       if random.NextDouble() < mutateprob then
                           mutate population sample
                       else sample)
                |> weightWithT T likelihood
                |> importanceSamplesArray likelihood samplesPerGen

            let w = Array.averageBy snd samples

            let memory =
                let mem' = (samples, w) :: mem
                if mem'.Length > maxsize then
                    List.sortByDescending snd mem' |> List.take maxsize
                else mem'
            loop (max 1. (T * atten)) (steps - 1) memory
    
    let guesses = importanceSamples likelihood samplesPerGen prior
    let mem' = 
        if forgetPrior then mem
        else (guesses, Array.averageBy snd guesses) :: mem
    loop T numSteps mem'

///The purpose of this function is to take a set of weighted samples and inject or remember it into
///existing memory by drawing the most probable samples and comparing against an existing population.
///This is done for just 3 steps, which should be enough to get good mixing into existing pop of
///fittest members of this sample.
let remember likelihood mutate maxsize mem (samples : _ []) =
    Distributions.categorical2 samples
    |> evolveSequence 1. 1. false 0.1 maxsize mem mutate likelihood samples.Length 3 

let inline testPath (paths : Dict<_,_>) x =
    match paths.tryFind x with
    | Some -1. -> true, -1.
    | Some r -> false, r
    | None -> false, 1.

let rec propagateUp maxWeight isreward (paths : Dict<_, _>) attenuateUp r =
    function
    | _ when r < 0.01 -> ()
    | [] -> ()
    | (_ :: path) ->
        paths.ExpandElseAdd path (fun v ->
            if v = -1. || v >= maxWeight then v
            else max 0. (if isreward then v + r
                         else v * r)) (if isreward then min maxWeight (1. + r)
                                       else r)
        propagateUp maxWeight isreward paths attenuateUp (r * attenuateUp) path

type PathGuide<'a when 'a : equality>(?attenutate, ?priorPaths, ?maxPropagatorWeight) =
    let paths = defaultArg priorPaths (Dict<'a list,_>())
    let atten = defaultArg attenutate 0.5
    let propweight = defaultArg maxPropagatorWeight 1.
    member __.PropagateUp isreward r path = propagateUp propweight isreward paths atten r path
    member __.TestPath path = testPath paths path  

type PopulationSampler<'a when 'a : equality>(generator : Distribution<'a>, scorer, ?mutate,?temperature, ?attenuate, ?popMutatePF, ?popMutate) =

    let mutate = defaultArg mutate id

    let mutateOnPopulation = defaultArg popMutate (fun _ x -> mutate x)

    let mutateOnPopulationPF = defaultArg popMutatePF (fun _ x -> mutate x)

    let T, atten = defaultArg temperature 1., defaultArg attenuate 1.

    member __.RecursiveImportanceSample(?mutateprob, ?samplespergen, ?generations) =
        let mp = defaultArg mutateprob 0.4
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        recursiveImportanceSample T atten mp mutateOnPopulationPF scorer samplespergen gens generator
        |> categorical2

    member __.SampleChain n =
        MetropolisHastings.sample atten T scorer mutate n (generator.Sample())
        |> Sampling.roundAndGroupSamplesWith id
        |> categorical2

    ///If the prior is too strong (such as an always or other with too little variation), this will 
    ///end up dominating the evolution, acting like a strong gravitational attractor causing
    ///exploration to stick to close the initial state. Forget prior is for when the prior is too certain.
    member __.EvolveSequence(?mutateprob, ?maxpoplen, ?samplespergen,
                             ?generations, ?forgetPrior) =
        let mp = defaultArg mutateprob 0.65
        let maxpopsize = defaultArg maxpoplen 100
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        let forgetInit = defaultArg forgetPrior false
        let pops =
            evolveSequence T atten forgetInit mp maxpopsize [] mutateOnPopulation scorer
                samplespergen gens generator
            |> List.toArray
            |> Array.normalizeWeights
            |> categorical2
        dist { let! pop = pops 
               return (discreteSample pop) }

    member __.SampleFrom n (dist : Distribution<_>) =
        dist.SampleN(n)
        |> Array.map (fun x -> x, scorer x)
        |> Array.normalizeWeights
        |> Helpers.Sampling.compactMapSamplesAvg id
        |> Array.sortByDescending snd

    member m.SampleFromRaw n (dist : Distribution<_>) =
        m.SampleFrom n dist
        |> Array.map (fun (x, _) -> x, scorer x)
        |> Array.sortByDescending snd
