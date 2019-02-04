﻿module SimpleHMC

//Hamiltonian monte carlo, hacky start
open DiffSharp.AD.Float32
open EvolutionaryBayes.ProbMonad

//contains code from http://diffsharp.github.io/DiffSharp/examples-hamiltonianmontecarlo.html
let normal = MathNet.Numerics.Distributions.Normal(0.,1.)
let Rnd = new System.Random()
// Uniform random number ~U(0, 1)
let rnd() = Rnd.NextDouble() |> float32

// Leapfrog integrator
// u: potential energy function
// k: kinetic energy function
// d: integration step size
// steps: number of integration steps
// (x0, p0): initial position and momentum vectors
let leapFrog (u: DV -> D) (k: DV -> D) (d: D) steps (x0,p0) =
    let hd = d / 2.f
    [1..steps]
    |> List.fold (fun (x,p) _ -> 
           let p' = p - hd * grad u x
           let x' = x + d * grad k p'
           x',p' - hd * grad u x') (x0,p0)

// Hamiltonian Monte Carlo
// hdelta: step size for Hamiltonian dynamics
// hsteps: number of steps for Hamiltonian dynamics
// x0: initial state
// f: target distribution function
let hmc hdelta hsteps (f: DV -> D) (x0: DV) =
    let u x = -log(f x) // potential energy
    let k p = (p * p) / D 2.f // kinetic energy
    let hamiltonian x p = u x + k p
    let p = DV.init x0.Length (fun _ -> normal.Sample())
    let x',p' = leapFrog u k (D hdelta) hsteps (x0,p)
    if rnd() < float32(exp((hamiltonian x0 p) - (hamiltonian x' p'))) then x'
    else x0

let sample hdelta hsteps n (f: DV -> D) (prior:Distribution<_>) = 
    let rec loop c s (x:DV) =
        if c = 0 then s
        else
        let x' = hmc hdelta hsteps f x
        loop (c-1) (x'::s) x'
    let parts = n / 10
    [|for i in 0..parts do
        let init = prior.Sample() 
        yield! loop 10 [] init|]
    

let observe pdf l y = List.fold (fun s x -> s * pdf x y) (D 1.f) l

module Densities =
    let rec factorial acc (x: D) =
        if x = D 1.f then x
        else factorial (acc * x) (x - 1.f)
    
    // Multivariate normal distribution (any dimension)
    // mu: mean vector
    // sigma: covariance matrix
    // x: variable vector
    let multiNormal (mu: DV) (sigmaInverse: DM) (x: DV) =
        let xm = x - mu
        exp(-(xm * sigmaInverse * xm) / D 2.f)
    
    let normal (mu: D) (sigma: D) (xv: DV) =
        let x = xv.[0]
        exp(-(x - mu) ** 2.f / (D 2.f * sigma ** 2.f))
    
    let lognormal (mu: D) (sigma: D) (xv: DV) = normal mu sigma (log xv)
    
    let gamma (shape) (rate) (xv: DV) =
        let x = xv.[0]
        x ** (shape - 1.f) * exp(-rate * x)
    
    let poisson lambda (kvec: DV) =
        let k = kvec.[0]
        lambda ** k / (factorial (D 1.f) k)
    
    let exponential λ (x: DV) = exp(-λ * x.[0])
    
    let studentT loc scale deg (xv: DV) =
        let x = xv.[0]
        (D 1.f + D 1.f / deg * ((x - loc) / scale) ** 2.f) 
        ** -((deg + 1.f) / 2.f)

let n = dist { return! EvolutionaryBayes.Distributions.bernoulli 0. } 

let lik = observe (fun x y -> Densities.normal (D x) (D 1.f) y) [ 5.f; 10.f; 4.f ]