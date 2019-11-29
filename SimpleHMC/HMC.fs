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


let simpleEuclideanKineticEnergy (p:DV) = (p * p) / D 2.f 

/// Hamiltonian Monte Carlo
/// hdelta: step size for Hamiltonian dynamics
/// hsteps: number of steps for Hamiltonian dynamics
/// x0: initial state
/// f: target distribution function 
let sample samplelen warmup k hdelta hsteps numsamples (f : DV -> D)
    (prior : Distribution<_>) =
    let u x = -(f x) // potential energy
    let hamiltonian x p = u x + k p
    let normal = Distributions.Normal(0., 1.)
    let slen = defaultArg samplelen -1
    // Uniform random number ~U(0, 1)
    let rnd() = Rnd.NextDouble() |> float32

    let hmc (x0 : DV) =
        let p =
            DV.init x0.Length (fun _ -> normal.Sample() |> float32 |> D)

        let x', p' = leapFrog u k (D hdelta) hsteps (x0, p)
        let acceptprob =
            float32 (exp ((hamiltonian x0 p) - (hamiltonian x' p')))
        if rnd() <= min 1.f acceptprob then x'
        else x0

    let rec loop n steps chains curchain (x : DV) =
        if steps = 0 then List.concat ((List.rev curchain).[warmup..] :: chains)
        else
            let x' = hmc x
            let n', x'', chain, chains' =
                if slen = -1 then 0, x', x' :: curchain, []
                else if n > slen then
                        let p = prior.Sample()
                        0, p, [ p ], (List.rev curchain).[warmup..] :: chains
                     else n + 1, x', x' :: curchain, chains
            loop n' (steps - 1) chains' chain x''

    let init = prior.Sample()
    loop 0 numsamples [] [] init


let observe (map : 'b -> DV) (log_pdf : DV -> DV -> D) (observations : 'b list)
    (parameters : DV) = List.sumBy (log_pdf parameters) (List.map map observations)

let inline density likelihood prior param = prior param + likelihood param 

module DV =
    let inline ofNum x = DV.create 1 (D x) 
    let inline ofD (x:D) = DV.create 1 x 

module D =
    let inline toFloat x = float(float32 x)

  
module LogDensities = 
    open MathNet.Numerics.Distributions
    open System

    /// Multivariate normal distribution (any dimension)
    /// mu: mean vector
    /// sigma: covariance matrix
    /// x: variable vector
    let multiNormal (mu : DV) (sigmaInverse : DM) (x : DV) =
        let xm = x - mu
        (-(xm * sigmaInverse * xm) / D 2.f) 

    let gamma (shape) (rate:DV) (x : DV) = 
        log ((DV.sum x) ** (shape - 1.f) * exp (-rate * x))

    let multiLognormal (mu : DV) (sigmaInverse : DM) (x : DV) = multiNormal mu sigmaInverse (exp x)
     
    let inline normal (mu) (stdev) = multiNormal (mu) (toDM [[stdev]])

    let inline lognormal shape scale x = normal shape scale (exp x)

    let studentT loc scale degr (x : DV) =
        let x' = DV.sum x
        let deg = D degr
        log
            ((D 1.f + D 1.f / deg * ((x' - loc) / scale) ** 2.f)
             ** -((deg + 1.f) / 2.f))

    let truncated cdf a b f x =
        let g (xv : DV) =
            let x = DV.sum xv
            if x <= (D a) || x > D b then log ((x - x) + Single.Epsilon)
            else f xv
        g x / (D(cdf b) - D(cdf a))

    let boundSamples minval maxval f x = 
        if DV.min x <= (D minval) || DV.max x > D maxval then 
            let xs = DV.sum x
            log ((xs - xs) + Single.Epsilon)
        else f x 

    let truncatedNormal a b (m : DV) (s : D) (x : DV) =
        truncated
            (fun x ->
            Normal(D.toFloat m.[0], D.toFloat s).CumulativeDistribution(float x)
            |> float32) a b (normal m s) x


module Samplers =
    open MathNet.Numerics.Distributions
    let sample (toDV:'a->DV) (d:Distribution<_>) = dist { let! x = d in return toDV(x) } 
    let sampleScalar (d:Distribution<float>) = dist { let! x = d in return DV.ofNum(float32 x) }
    let inline sampleVec (d:Distribution<_>) = dist { let! x = d in return toDV (x) }

module DM =
    let fillOffDiagonals (m : DM) (v : DV) =
        let cols, rows = m.Cols, m.Rows
        let ma = DM.toArray2D m |> Array2D.map float32
        let mutable i = 0
        toDV [| for c in 0..cols - 1 do
                    for r in 0..rows - 1 ->
                        if c <> r then
                            let x = v.[i]
                            i <- i + 1
                            x
                        else (D ma.[r, c]) |]
        |> DV.toDM rows


