# NormalizingFlows.jl

Normalizing Flow transforms a simple distribution to a complex distribution by applying a series of neural networks. Density estimation and statistical inference can be done using Normalizing Flow. Given the samples, the density function from which the samples were generated can be retrieved and further used for statistical inference.

The main idea of flow-based modeling is to express x as a transformation T of a real vector u sampled from pᵤ(u):

```math
x = T(u), u ∼ pᵤ(u)
```

## Outline

```@contents
```