# NormalizingFlows.jl

<!-- [![](https://img.shields.io/badge/docs-dev-blue.svg)]() -->

Normalizing Flow transforms a simple distribution to a complex distribution by applying a series of neural networks. Density estimation and statistical inference can be done using Normalizing Flow. Given the samples, the density function from which the samples were generated can be retrieved and further used for statistical inference.

The main idea of flow-based modeling is to express $ x$ as a transformation $ T$ of a real vector $ u$ sampled from $p_{u}(u)$:

$$x = T(u), u ∼ p_{u}(u)$$


<!-- ## Installation -->

```julia
(@v1.5) pkg> clone(https://github.com/archanarw/NormalizingFlows.jl)
```

## Available Types of [Normalizing Flows](https://arxiv.org/pdf/1912.02762.pdf)

* Autoregressive Flows –⁠ 
Autoregressive flows specifies the transformation to have the following form:

$z'_{i} = τ (z_{i}; h_{i})$ 
$h_{i} = c_{i}(z_{<i})$

where $τ$ is termed the transformer and $c_{i}$ is the $i$-th conditioner, where $i = 1, ..., D$ and $D$ is the size of $ z$.

`AffineLayer` has a tranformer of the form $τ(z_{i}; h_{i}) = α_{i}z_{i} + β_{i}$ where $h_{i}$ = {$α_{i}$, $β_{i}$} and a masked conditioner.

* Linear Flows –⁠
They are of the form: $z' = Wz$ where $W $ is a $D×D$ invertible matrix that parameterizes the transformation.

`LinearFlows` has three layers: Conv1x1 layer (permutation layer), Normalization layer and Affine Coupling layer.

* Residual Flows -
⁠ They are of the form: $z_{0} = z + g_{φ}(z)$ where g_{φ} is a function that outputs a D-dimensional translation vector, parameterized by φ.
The two types of residual flows implemented are:
- `PlanarFlows`: Here, the function g_{φ} is a one-layer neural network with a single hidden unit: $z_{0} = z + vσ(w^{T}z + b)$ where σ is a differentiable activation function such as the hyperbolic tangent.
- `RadialFlows`: The function g_{φ} is such that
$z' = z + \frac{β}{α + r(z)} (z - z_{0})$, $r = ||z - z_{0}||$ where $ ||·||$ is the Euclidean norm.

## Usage

```julia
julia> using NormalizingFlows
```

<!-- ### Construction and Evaluation -->
