I have a strong interest in probabilistic programming and believe it plays a key role in most paths leading to more intelligent and helpful machines. I've worked on a Hansei Port, a couple probability monads, worked with [Infer.Net](http://infer.Net) and Stan plus learned a great deal from WebPPL materials. Each of these has its advantages and downsides:

**Hansei** 

**Pro**: Fast when space is constrained, shares many advantages of logic based programming, can be viewed as Enhanced Sequence Comprehensions Augmented with a backtracking, weight tracking search operator.  

That last is important and I believe this form of control flow is the most under-used. Unlike sampling based approaches, sampling in this regime is simply lazily unfolding a tree—recursion is natural and constraints are most easily applied here. Because of its iterative nature, generated samples are not repeated.

**Con**: Discrete and so can be very memory intensive. Computation time can blow up in large spaces. One must be very careful to avoid such blow ups, hard to use in an online manner.

**Infer.Net**

**Pro**: Fast, scalable (arguably the most scalable bayesian approach that is not a ~~hack~~ variational approach)

**Con**: Infer.net operates on and extends graphical models and works by hand defined primitive distributions. If your problem does not fit within this framework then you're stuck.

**Stan and HMC**

**Pro:** HMC is a highly principled approach to probabilistic programming, effectively, built in tools diagnostics, gradients allow scaling to large spaces. State of the art research augmenting basic HMC (not just NUTs).

**Con:** Persnickety (: to keep good properties, operations violating volume preservation or reversibility etc. are out, attempts to account for these violations tend to be fishy). Need to be differentiable, cannot easily do discrete mixture models or non-parametric unbounded growth models. Stan specifically, is very heavy, with a *complex* install workflow requiring an entire C++ compiler. Weak to multimodality ([https://arxiv.org/abs/1808.03230](https://arxiv.org/abs/1808.03230)) but note that *everything is weak multimodality, including brains (possible exceptions are planetscale resources ala Evolution).*

*Minor*: Parameters must be tuned (work-arounds include NUTs but also, using hyper gradients on log probability of random subsample of data penalized by a term to encourage exploration).

**WebPPL or Probability Monads ala Gordon, Scibior and Ghahramani**

Pro: fully flexible, can easily write arbitrary models.

Con: For both, flexibility comes with a steep price: inference can be quite costly. For the monadic approach, extra costs as running interpreters via the Free monad is *slow.*

# **EvoBayes**
EvoBayes starting point is the observation that most Bayesian approaches are (one or more) tend to blow up computational resources, slow, heavy or with a cumbersome interface. Meanwhile, simulated annealing, which is not so different from Metropolis Hastings is surprisingly fast and scales to quite hostile combinatorial spaces. A similar observation can be made about evolutionary or genetic algorithms, which are more commonly deployed while also maintaining a population. This caused me to consider a method combining all these approaches. EvoBayes is as such, a light-weight dsl for composing fast samplers and determining likelihoods. Its downside is it is less principled from a bayesian perspective than all the methods mentioned above.
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
EvoBayes supports F# computation expressions, and while you can do simple sample based probabilistic programming with it, its core purpose is as a DSL for sequencing statements while allowing for easy nesting and composition. These types of priors are generators and should be thought of as a way to encode your beliefs about a good place to start searching. 

## Metropolis Hastings

EvoBayes is focused on the likelihood and the concept of perturbations. It supports a perturbation or transition operator that behaves much as what one would find in simulated annealing. In its Metropolis Hastings implementation, it is neccessary to provide a prior to go with the transition operators and likelihood computations.

Ultimately, constructing a prior that can also compute likelihoods with computation expressions is probably not the best way to go (given the intended use cases), also likely requiring the use of symbols and a lot of effort adjusting the distributions to accept them. But that's not all since getting something that can be both sampled from and computes likelihoods seems very tricky--there seems no obvious way to apply the needed arbitrary projection to the intermediate tensor products.

It is much easier instead to implement a way for base distributions to compute likelihoods and then manually compute the prior and the sum for the likelihood. With included helpers it should not take much more work than doing so with the aid of a computation expression.

## Particle Filter

In addition to the MCMC/simulated annealing hybrid, the Particle Filter section is inspired by genetic algorithms, actual evolution and particle filters. The issue of priors is here not large and the generator based priors do act more typically bayesian.

*Quoting a code comment:*

> Inspired by evolution (see papers on regret minimization's connection to evolution), this method maintains a memory where each element is a set of samples weighted by their average performance. The head of this list is the most recent generation. Worst performing generations are evicted when memory size limit is reached. Hopefully, by then their descendants can be found in later generations. At each step, a generation is sampled and then a member of that generation. Because the memory size is bounded there is a fixed upper bound on total samples with the population mixing and hopefully drifting towards the most fit.

The most useful aspect of this approach is allows any method that generates samples to be capable of incremental online learning that resists catastrophic forgetting from drift.

## Transitions/Perturbations/Mutations:

Transitions and perturbations are a key idea in EvoBayes. Consider the following example where one of the weights are randomly perturbed.
```F#
    (fun (m, b) ->
        let ps = [| m; b |]
        let i = random.Next(0, ps.Length)
        ps.[i] <- ps.[i] + random.NextDouble(-0.1, 0.1)
        ps.[0], ps.[1])
```
As I mentioned before, this aspect is not too dissimilar to simulated annealing. The advantage now is a nice DSL and a maintained population to capture uncertainty. 

**Regret**

Of course, as the dimensionality of the problem grows, exploration becomes an issue. One method to alleviate this is to have min-regret learners at each array index such that something other than uniform sampling is used to decide which weight to adjust. One can even imagine a tree based approach where the selected index propagates an attenuated reward up a tree which selects nodes randomly till a leaf (index) is reached.

## **Hamiltonian Monte Carlo**

If EvoBayes is a lax, many things go as long as they works empirically wild west, Hamiltonian monte carlo is a highly mathematically principled approach to probabilistic programming. This framework also includes a small implementation of it based on the Diffsharp sample. Missing are: adaptation, mass matrices, diagnostic tools. 

**Priors** 

HMC requires priors. To support this, you can write your prior as a sum, as shown below while the density functions adds it to the log likelihood. *Code below is illustrative:*
```F#
    let lik =
        observe toDV (fun (param : DV) y ->
            let v = param.[..1]
            
            let vars =
                param.[2..] |> DV.toArray |> Array.map float32
    
            let ma =
                array2D [ [ vars.[1]; vars.[0] ]
                          [ vars.[0]; vars.[2] ] ]
                |> Array2D.map D
    
            let m = DM.ofArray2D ma
            LogDensities.multiNormal v (DM.inverse m) y) [ [ 5.f; 15.f ]
                                                           [ 10.f; 12.f ]
                                                           [ 3.f; 9.f ] ]  
    let fprior (param:DV) =
        LogDensities.normal (DV.ofNum 0.f) (D 1.f) (DV.ofNum param.[0])
        + LogDensities.normal (DV.ofNum param.[0]) (D 1.f) param.[1]
        
    let f = density lik fprior
    let p0 = sample 0.1f 10 5000 f (Samplers.sampleScalar (normal 0. 1.))
```
**Hyper-parameters (*not implemented yet)***

To get good results, HMC requires parameter tuning. I believe one way to do this is with hyper-gradients. After a small number of samples are drawn, their mean is taken and used to compute the log probability of some validation sample. The loss is then used to tune the gradients to adjust gradient step size and number of leap frog steps. A similar thing can be done for a mass matrix. Indeed, a wilder idea would be to search for a matrix that random projects up according to some fixed matrix into the matrix or diagonal of appropriate dimension. That sort of unmotivated hail mary is done all the time in neural networks and I think is part of the community's strength.

**Evolutionary Methods**

In addition, to hyper-gradients, EvoBayes could be used to search for starting settings of these parameters. An even wilder application would be to search over a limited grammar to define new kinetic energy functions (thus the structure of the manifold over which inference is done).

**A place for Neural Networks**

Most people's idea of combining probabilistic programming and deep learning is by putting a distribution on parameter weights. My opinion is this is not the most ideal combination of both ideas. You're still doing optimization and if you were not making some trade-off then integration would be easy in the first place (eg, Variational autoencoders are limited and don't generalize well, normalizing flows have stability issues, bayesian neural nets have scalability and generalization issues, graph bayesian neural nets—are you sure you Infer.net is not suitable to your problem—variational methods are biased and don't do well at the tail).

One place Neural nets could work is as density estimators. A small neural net could be trained on the required interval and it would stand in place for any density of interest. This would widely increase the class of available models.

**Online Learning**

The particle filters of EvoBayes could also be used for the HMC sampler. The samples can be fed and then kept live using the evolutionary sequential sampler. The HMC sampler accepts the ability to use a generative prior. With such starting points, assuming not from scratch, burn-in is no longer required and the generative prior can be used to improve exploration while the particle filter increases or remembers and forgets (appropriately) samples online. 

You can sample say, 1000 samples  from 100 start point. Each of the 100, are started from an existing sample which will be surviving samples from the higher density locations. This approach I suspect, can be made to scale very well across computers.

## Tasks

- [x]  Implement Variation on Particle Filter @Feb 6, 2019
    - [x]  Use Memory and History, setting a basis for online learning @Feb 6, 2019
    - [x]  Implement Variation on Particle Independent SMC @Feb 6, 2019

        *(Note: Trivially follows from implementation of Metropolis and Sequenced Sampler)*

        Also, as an alternative to history, to save on memory, emphasize the importance of the *ability to remove rather than just add/grow in mutate!*

- [x]  Fix log probability @Feb 4, 2019
- [x]  More sampling distributions @Feb 6, 2019
- [x]  HMC priors @Feb 9, 2019
- [x]  test HMC @Feb 5, 2019
    - [x]  fix log densities @Feb 8, 2019
    - [x]  Fix HMC @Feb 6, 2019
        - [x]  Further fix/refine HMC @Feb 8, 2019
- [x]  Complete Write up @Feb 10, 2019
