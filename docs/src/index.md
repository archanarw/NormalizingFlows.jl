# NormalizingFlows.jl

`NormalizingFlows.jl` is a julia package that allows you to construct and train a special kind of neural network called a *normalizing flow*.

A normalizing flow transforms a simple distribution into a complex distribution by applying a sequence of transformations. Each transformation is a special kind of neural network that is invertible and differentiable. By composing many of these transformations together, we can construct arbitrarily complex probability distributions that: 
- can efficiently sampled from
- allow efficient computation of the density of any value in the domain of the distribution

There are many potential use cases for a normalizing flow. One particularly important use case is training a generative model, i.e., updating the parameters of a normalizing flow so that it models a given dataset.


## Outline

```@contents
```