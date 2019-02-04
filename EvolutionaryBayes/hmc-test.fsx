

let n = dist { return! EvolutionaryBayes.Distributions.bernoulli 0. }

let lik =
    observe (fun x y -> Densities.normal (D x) (D 1.f) y) [ 5.f; 10.f; 4.f ]

