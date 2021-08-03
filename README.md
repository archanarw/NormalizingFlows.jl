# NormalizingFlows.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://archanarw.github.io/NormalizingFlows.jl/)

`NormalizingFlows.jl` is a julia package that allows you to construct and train a special kind of neural network called a *normalizing flow*.

A normalizing flow transforms a simple distribution into a complex distribution by applying a sequence of transformations. Each transformation is a special kind of neural network that is invertible and differentiable. By composing many of these transformations together, we can construct arbitrarily complex probability distributions that: 
- can efficiently sampled from
- allow efficient computation of the density of any value in the domain of the distribution

There are many potential use cases for a normalizing flow. One particularly important use case is training a generative model, i.e., updating the parameters of a normalizing flow so that it models a given dataset.


## Installation

```julia
(@v1.5) pkg> add https://github.com/archanarw/NormalizingFlows.jl
```

## Available Types of Normalizing Flows
See [here](https://arxiv.org/pdf/1912.02762.pdf) for a review on normalizing flows.

* Autoregressive Flows –⁠ 
Autoregressive flows specifies the transformation to have the following form:

```math
z'ᵢ = τ(zᵢ; hᵢ) 
hᵢ = cᵢ(z_{<i})
```

where τ is termed the transformer and cᵢ is the i-th conditioner, where i = 1, ..., D and D is the size of z.

`AffineLayer` has a tranformer of the form τ(zᵢ; hᵢ) = αᵢzᵢ + βᵢ where hᵢ = {αᵢ, βᵢ} and a masked conditioner.

* Linear Flows –⁠
They are of the form: z' = Wz where W is a D×D invertible matrix that parameterizes the transformation.

`LinearFlows` has three layers: Conv1x1 layer (permutation layer), Normalization layer and Affine Coupling layer.

* Residual Flows -
⁠ They are of the form: z₀ = z + g_φ(z) where g_φ is a function that outputs a D-dimensional translation vector, parameterized by φ.
The two types of residual flows implemented are:

`PlanarFlows`- Here, the function g_{φ} is a one-layer neural network with a single hidden unit:
```math 
z₀ = z + vσ(wᵀz + b)
``` 
where σ is a differentiable activation function such as the hyperbolic tangent.

`RadialFlows`- The function g_{φ} is such that
```math
z' = z + \\frac{β}{α + r(z)} (z - z₀)
 r = ||z - z₀||
```
where ||·|| is the Euclidean norm.

## Usage

```julia
julia> using NormalizingFlows
```

### Examples

Examples of contruction of the models, training and sampling on MNIST dataset can be viewed [here](examples).
