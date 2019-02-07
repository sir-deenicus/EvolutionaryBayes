module SimpleHMC

//Hamiltonian monte carlo, hacky start
open DiffSharp.AD.Float32
open EvolutionaryBayes.ProbMonad
open MathNet.Numerics

//contains code from http://diffsharp.github.io/DiffSharp/examples-hamiltonianmontecarlo.html
let Rnd = new System.Random() 

/// Leapfrog integrator
/// u: potential energy function
/// k: kinetic energy function
/// d: integration step size
/// steps: number of integration steps
/// (x0, p0): initial position and momentum vectors
let leapFrog (u : DV -> D) (k : DV -> D) (d : D) steps (x0, p0) =
    let hd = d / 2.f
    [ 1..steps ]
    |> List.fold (fun (x, p) _ ->
           let p' = p - hd * grad u x
           let x' = x + d * grad k p'
           x', p' - hd * grad u x') (x0, p0)

/// Hamiltonian Monte Carlo
/// hdelta: step size for Hamiltonian dynamics
/// hsteps: number of steps for Hamiltonian dynamics
/// x0: initial state
/// f: target distribution function 
let sample hdelta hsteps n (f : DV -> D) (prior : Distribution<_>) =
    let u x = -(f x) // potential energy
    let k p = (p * p) / D 2.f // kinetic energy
    let hamiltonian x p = u x + k p
    let normal = Distributions.Normal(0., 1.)
    // Uniform random number ~U(0, 1)
    let rnd() = Rnd.NextDouble() |> float32 
    let hmc (x0:DV) =
        let p = DV.init x0.Length (fun _ -> normal.Sample() |> float32 |> D)
        let x', p' = leapFrog u k (D hdelta) hsteps (x0, p)
        let acceptprob = float32 (exp ((hamiltonian x0 p) - (hamiltonian x' p'))) 
        if rnd() <= min 1.f acceptprob then x'
        else x0  
    let rec loop c s (x : DV) =
        if c = 0 then s
        else
            let x' = hmc x
            loop (c - 1) (x' :: s) x' 
    let init = prior.Sample()
    loop n [] init 

//let observe log_pdf observations parameters = List.fold (fun s x -> s + log_pdf parameters x) (D 0.f) observations

let observe (log_pdf : DV -> 'b -> D) (observations : 'b list)
    (parameters : DV) = List.sumBy (log_pdf parameters) observations

module Densities =
    let rec factorial acc (x : D) =
        if x = D 1.f then x
        else factorial (acc * x) (x - 1.f)

    /// Multivariate normal distribution (any dimension)
    /// mu: mean vector
    /// sigma: covariance matrix
    /// x: variable vector
    let multiNormal (mu : DV) (sigmaInverse : DM) (x : DV) =
        let xm = x - mu
        (-(xm * sigmaInverse * xm) / D 2.f) 

    let lognormal (mu : DV) (sigmaInverse : DM) (x : DV) = multiNormal mu sigmaInverse (log x)

    let gamma (shape) (rate:DV) (x : DV) = 
        log ((DV.sum x) ** (shape - 1.f) * exp (-rate * x))

    let poisson lambda (k : DV) = 
        let k' = DV.sum k
        log (lambda ** k' / (factorial (D 1.f) k'))

    let exponential (λ:DV) (x : DV) = (-λ * x)

    let studentT loc scale degr (x : DV) =
        let x' = DV.sum x
        let deg = D degr
        log
            ((D 1.f + D 1.f / deg * ((x' - loc) / scale) ** 2.f)
             ** -((deg + 1.f) / 2.f))

module Samplers =
    open MathNet.Numerics.Distributions

    let normal m s =
        let n = Normal(m, s)
        { new Distribution<DV> with
              member d.Sample() =
                  [| n.Sample()
                     |> float32
                     |> D |]
                  |> DV.ofArray }
