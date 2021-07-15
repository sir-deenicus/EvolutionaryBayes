I have a strong interest in probabilistic programming and believe it plays a key role in most paths leading to more intelligent and helpful machines. I've worked on a Hansei Port, a couple probability monads, worked with [Infer.Net](http://infer.Net) and Stan plus learned a great deal from WebPPL materials. Each of these has its advantages and downsides:

**Hansei** 

**Pro**: Fast when space is constrained, shares many advantages of logic based programming, can be viewed as Enhanced Sequence Comprehensions Augmented with a backtracking, weight tracking search operator.  

That last is important and I believe this form of control flow is the most under-used. Unlike sampling based approaches, sampling in this regime is simply lazily unfolding a search tree—recursion is natural and constraints are most easily applied here. Because of its iterative nature, generated samples (depending on search procedure) need not not be revisited.

**Con**: Discrete and so can be very memory intensive. Computation time can blow up in large spaces. One must be very careful to avoid such blow ups.

**Infer.Net**

**Pro**: Fast, scalable (arguably the most scalable bayesian approach that is not a variational approach—even its variational message passing seems more principled than run of the mill variational bayes).  Although you’re restricted in the class of computations you can represent, it is nonetheless a wide class. Indeed, this approach to probabilistic programming is probably not sufficiently well explored.

**Con**: Infer.net operates on and extends graphical models and works by hand defined primitive distributions. If your problem does not fit within this framework then you're stuck.

**Stan and HMC**

**Pro:** HMC is a highly principled approach to probabilistic programming, effective, built in diagnostic tools, gradients allow scaling to larger spaces. State of the art research augmenting basic HMC (not just NUTs).

**Con:** Persnickety (: to keep good properties, operations violating volume preservation or reversibility etc. are out, attempts to account for these violations tend to be fishy). Need to be differentiable, cannot easily do discrete mixture models or non-parametric unbounded growth models. Stan specifically, is very heavy, with a *complex* install workflow requiring an entire C++ compiler. Weak to multimodality ([https://arxiv.org/abs/1808.03230](https://arxiv.org/abs/1808.03230)) but note that *everything is weak multimodality, including brains (possible exceptions are planet-scale inference processes ala Evolution).*

*Minor*: Parameters must be tuned (work-arounds include NUTs but also, using hyper gradients on log probability of random subsample of data penalized by a term to encourage exploration).

**WebPPL or Probability Monads ala Gordon, Scibior and Ghahramani**

Pro: fully flexible, can easily write arbitrary models.

Con: For both, flexibility comes with a steep price: inference can be quite costly. For the monadic approach, extra costs as running interpreters via the Free monad is *slow.*

# EvoBayes

EvoBayes starting point is the observation that most Bayesian approaches tend to blow up on needed computational resources, are (one or more of) slow, heavy or with a cumbersome interface. Meanwhile, simulated annealing, which is not so different from Metropolis Hastings, is surprisingly fast and scales to quite hostile combinatorial spaces. A similar observation can be made about the ease of use of evolutionary or genetic algorithm packages, which are commonly deployed while also maintaining a population (their flexibility being of interest more so than their optimality). This caused me to consider a method combining all these approaches. 

EvoBayes is a a light-weight dsl for composing fast samplers and determining likelihoods.  It is a bit less principled/more forgiving from a bayesian perspective than all the methods mentioned above.

```F#
    let game2 a b c d =
        dist {
            let! p = beta a b
            let! q = beta c d
            let! p1res = bernoulliChoice "P1 win" "P2 win" p
            let! p2res = bernoulliChoice "P2 win" "P1 win" q
            if p1res = p2res then return p1res
            else 
                let tiebreak =
                    if p > q then "P1 win"
                    elif p = q then "can't say"
                    else "P2 win"
                return tiebreak
        }
    
    (game2 6. 10. 2. 10.).SampleN(1_000_000)
    |> Sampling.roundAndGroupSamplesWith id
    |> Array.sortByDescending snd
```
EvoBayes supports F# computation expressions, and while you can do simple sampling based probabilistic programming with it, its core purpose is as a DSL for sequencing statements while allowing for easy nesting and composition of samplers. *Note*: Although recursion is supported, it can blow the stack  (after around a few thousand iterations) as this is not a discrete probability monad meant to build lazy search trees. Recursive structures are best built by mutations in (MCMC, particle filter) sampler iterations.  

These types of priors are generators and should be thought of as a way to encode your beliefs about a good place to start searching. 

I believe a good part of the potential of computers as tools for thought is in the ability to create declarative domain languages. Procedural thinking—which surprisingly, when viewed in a certain way, includes mathematics where you really need to understand the ins and outs of a problem—places more pressure on correctly working out all the steps of the problem.

For example, in math we must carefully write out the probabilities, carry out long calculations, remembering what terms to sum. We must carefully work out say, a probability tree on paper or a whiteboard. Whereas here, we use our declarative syntax to write out our problem in a manner close to language. This specification can then either be executed as a random sampling simulation or as a mathematical expression or as a probability tree.  With a DSL, we describe the problem in a way that allows us to think close to natively in the language of the problem.

## Metropolis Hastings

EvoBayes is focused on the likelihood and the concept of perturbations. It supports a perturbation or transition operator that behaves much as what one would find in simulated annealing. In its Metropolis Hastings implementation, it is necessary to provide a prior to go with the transition operators and likelihood computations.

Ultimately, constructing a prior that can also compute likelihoods with computation expressions is probably not the best way to go (given the intended use cases), also likely requiring the use of symbols and some effort adjusting the distributions to accept them. But that's not all since getting something that can be both sampled from and computes likelihoods seems very tricky--there seems no obvious way to apply the needed arbitrary projection to the intermediate tensor products.

It is much easier instead to implement a way for base distributions to compute likelihoods and then manually compute the prior and the sum for the likelihood. With included helpers it should not take much more work than doing so with computation expressions.

## Particle Filter

In addition to the MCMC/simulated annealing hybrid, the Particle Filter section is inspired by genetic algorithms, actual evolution and particle filters. The issue of priors is here not large and the generator based priors do act more typically bayesian.

*Quoting a code comment:*

> Inspired by evolution (see papers on regret minimization's connection to evolution), this method maintains a memory where each element is a set of samples weighted by their average performance. The head of this list is the most recent generation. Worst performing generations are evicted when memory size limit is reached. Hopefully, by then their descendants can be found in later generations. At each step, a generation is sampled and then a member of that generation. Because the memory size is bounded there is a fixed upper bound on total samples with the population mixing and hopefully drifting towards the most fit.
>

The most useful aspect of this approach is it allows any method that generates samples to be capable of incremental online learning that resists catastrophic forgetting from drift.

## Transitions/Perturbations/Mutations:

Transitions and perturbations are a key idea in EvoBayes. Consider the following example where one of the weights are randomly perturbed.
```F#
    (fun (m, b) ->
        let ps = [| m; b |]
        let i = random.Next(0, ps.Length)
        ps.[i] <- ps.[i] + random.NextDouble(-0.1, 0.1)
        ps.[0], ps.[1])
```
As I mentioned before, this aspect is not too dissimilar to simulated annealing. The advantage now is a nice DSL and a maintained population to capture uncertainty. Here we’re sampling one directional change at a time, but while it is possible to perturb all dimensions at once, I have noticed that this makes sampling less effective across all dimensions. In the above, a random uniform is added to the present state. This can also be seen as conditioning on a uniform, but we can use arbitrary distributions for exploration, which eases tuning variance or constraining what values the parameters will take. This is particularly useful for more complex matrix and vector distributions.

```
(fun (m, b) ->
    let ps = [| m; b |]
    let i = random.Next(0, ps.Length)
    ps.[i] <- Normal(ps.[i], 1.).Sample()
    ps.[0], ps.[1])
```

# Limitations

This library is meant to be a more flexible and robust alternative to simulated annealing and genetic programming with a bayesian derived approach. If you need to properly explore high dimensional spaces and aren’t doing anything crazy like interactively sampling recursively built symbolic expressions, then you are better off using a HMC package. If Control flow on discrete structures and recursive search is paramount then consider Hansei. Finally, for large complex models that are more powerful than graphical models but still not fully recursive, consider Infer.net.

However, one area where this library can be useful is doing lightweight probabilistic programming on lower end devices. For example, this library is fully compatible with Android devices.

# Regret Learning



In addition to bayesian methods, the library supports a multiplicative weights update algorithm—which are related to evolution—for regret minimization based learning. There can be experts per context, placing 

```
let reward =
    function
    | (Go, Go) -> -100.
    | (Go, Stop) -> 1.
    | (Stop, Go) -> 0.
    | (Stop, Stop) -> 0.

let experts =
    RegretLearner<int, _, _>(reward, [| Go; Stop |], minreward = -100., maxreward = 10.)

experts.New [ 0; 1 ] 

for _ in 0 .. 999999 do
    let action1 = experts.SampleActionOf(key = 0)
    let action2 = experts.SampleActionOf(1)

    experts.Learn(key = 0, observation = action1)
    experts.Learn(1, action2)

```

