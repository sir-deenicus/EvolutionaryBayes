module EvolutionaryBayes.ParticleFilters

open EvolutionaryBayes.ProbMonad
open System
open Prelude.Common
open Prelude.Math
open Helpers
open EvolutionaryBayes
open System.Collections.Generic
open EvolutionaryBayes.Distributions
  
let inline cdf (prob : _ []) =
    let cd = Array.create prob.Length prob.[0] 
    for i in 1..prob.Length - 1 do
        let j = (prob.Length - i - 1) 
        let (x, p) = prob.[i]
        cd.[j] <- x, snd cd.[j+1] + p
    cd
      
let getDiscreteSample (pcdf : ('a * float) []) =
    let k, pcdlen = 
        random.NextDouble() * (snd pcdf.[0]), pcdf.Length - 1
        
    let rec cummProb idx =
        if k > snd pcdf.[idx] then cummProb (idx - 1)
        else idx, pcdf.[idx] 
    let i, (x,_) = cummProb pcdlen 
    x
    
let discreteSample p = cdf p |> getDiscreteSample

let discreteSampleN n items = [|for _ in 1..n -> discreteSample items|]
  
let inline normalizeWeights data =
    let sum = Array.sumBy snd data |> float
    [|for (x,p) in data -> x, float p / sum|]
 
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

let sequenceSamples mutateprob mutate (likelihood : 'a -> float)
    (numparticles : int) (numsteps : int) (prior : Distribution<_>) =
    let choices = importanceSamples likelihood numparticles prior 

    let rec loop samples dist =
        if samples = 0 then dist
        else
            discreteSampleN numparticles dist
            |> Array.map (fun x ->
                   if random.NextDouble() < mutateprob then mutate choices x
                   else x)
            |> weightWith likelihood 
            |> importanceSamplesArray likelihood numparticles 
            |> loop (samples - 1)
    loop numsteps choices

///Inspired by evolution (see papers on regret minimization's connection to evolution), this
///method maintains a memory where each element is a set of samples weighted by their average
///performance. The head of this list is the most recent generation. Worst performing generations
///are evicted when memory size limit is reached. Hopefully, by then their descendants can be found
///in later generations. At each step, a generation is sampled and then a member of that generation.
///Because the memory size is bounded there is a fixed upper bound on total samples with the population
///mixing and hopefully drifting towards the most fit.
let evolveSequence T atten mutateprob maxsize mem mutate
    (likelihood : 'a -> float) (samplesPerGen : int) (numSteps : int)
    (prior : Distribution<_>) =
    let choices = importanceSamples likelihood samplesPerGen prior
    let dist' = choices, Array.averageBy snd choices

    let rec loop T steps mem =
        if steps = 0 then mem
        else
            let population =  //tends to not explore as well so use Temperature
                distBuilder (fun () ->
                    let history =
                        discreteSample //sample a generation
                            (Array.normalizeWeights [| for (m, p) in mem -> m, p ** (1. / T) |]) 

                    let hypothesis =
                        discreteSample //sample a hypothesis from selected generation
                            (Array.map (fun (x, p) -> x, p ** (1. / T)) history) 
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
    loop T numSteps (dist' :: mem)

///The purpose of this function is to take a set of weighted samples and inject or remember it into
///existing memory by drawing the most probable samples and comparing against an existing population.
///This is done for just 3 steps, which should be enough to get good mixing into existing pop of
///fittest members of this sample.
let remember likelihood mutate maxsize mem (samples : _ []) =
    Distributions.categorical2 samples
    |> evolveSequence 1. 1. 0.1 maxsize mem mutate likelihood samples.Length 3


let inline testPath (paths : Dict<_,_>) k =
    match paths.tryFind k with
    | Some -1. -> true, -1.
    | Some x -> false, x
    | None -> false, 1.

let rec propagateUp (paths : Dict<_,_>) r =
    function
    | _ when r < 0.01 -> ()
    | [] -> ()
    | (_ :: path) ->
        paths.ExpandElseAdd path (fun v ->
            if v = -1. then v
            else max 0. (v + v * r)) (1. + r)
        propagateUp paths (r * 0.5) path

type SequenceSampler<'a when 'a : equality>(generator : Distribution<'a>, mutate, scorer, ?temperature, ?attenuate, ?popMutatePF, ?popMutate) =

    let mutateOnPopulation = defaultArg popMutate (fun _ x -> mutate x)

    let mutateOnPopulationPF = defaultArg popMutatePF (fun _ x -> mutate x)

    let T, atten = defaultArg temperature 1., defaultArg attenuate 1.

    member __.SequenceSamples(?mutateprob, ?samplespergen, ?generations) =
        let mp = defaultArg mutateprob 0.4
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50
        sequenceSamples mp mutateOnPopulationPF scorer samplespergen gens generator

    member __.SampleChain n =
        MetropolisHastings.sample2 atten T scorer mutate n (generator.Sample())
        |> Sampling.roundAndGroupSamplesWith id
        |> categorical2

    member __.EvolveSequence(?mutateprob, ?maxpopsize, ?samplespergen,
                             ?generations) =
        let mp = defaultArg mutateprob 0.4
        let maxpopsize = defaultArg maxpopsize 250
        let samplespergen = defaultArg samplespergen 500
        let gens = defaultArg generations 50

        let pops =
            evolveSequence T atten mp maxpopsize [] mutateOnPopulation scorer
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
        |> Helpers.Sampling.compactMapSamples id
        |> Array.sortByDescending snd

    member __.SampleFromRaw n (dist : Distribution<_>) =
        dist.SampleN(n)
        |> Array.map (fun x -> x, scorer x)
        |> Array.normalizeWeights
        |> Helpers.Sampling.compactMapSamples id
        |> Array.map (fun (x, _) -> x, scorer x)
        |> Array.sortByDescending snd
