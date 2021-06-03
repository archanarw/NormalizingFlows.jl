# NormalizingFlows.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://archanarw.github.io/NormalizingFlows.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://archanarw.github.io/NormalizingFlows.jl/dev)

Normalizing Flows transform a simple distribution into a complex distribution by applying a series of neural networks to the base distribution. A sample from a complex (multi-modal) distribution can be obtained from a simple base distribution (e.g. Uniform) using normalizing flows. In a normalizing flow model, the transformation is invertible and differentiable, thus allowing efficient computation of the density of any value in the domain of the distribution. Given any dataset, the density can be computed and used for further statistical inference.

The main idea of flow-based modeling is to express x as a transformation T of a real vector u sampled from pᵤ(u):

```math
x = T(u), u ∼ pᵤ(u)
```
The package `NormalizingFlows.jl` implements training and generation of samples for the following normalizing flow models - 
- Affine Autoregressive Flows
- Linear Flows
- Planar Flows
- Radial Flows

## Installation

```julia
(@v1.5) pkg> clone(https://github.com/archanarw/NormalizingFlows.jl)
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
