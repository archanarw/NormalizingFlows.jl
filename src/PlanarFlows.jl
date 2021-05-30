using Flux, Distributions
using LinearAlgebra
import Flux.params, Base.eltype

export PlanarFlow, sample, params

"""
The parameters of Planar Flows are as follows:
- `v` ∈ ℝᴰ
- `w` ∈ ℝᴰ
- `b` ∈ ℝ

The transformation is -
z′ = z + vσ(wᵀz + b) where σ is an activation function such as tanh, 
        z is the input to the flow and z′ is the output.

This flow can be interpreted as expanding/contracting
the space in the direction perpendicular to the hyperplane wᵀz + b = 0
"""
struct PlanarFlow{T}
    v::Array{T, 1}
    w::Array{T, 1}
    b::T
end

function PlanarFlow(rng::AbstractRNG, T, D)
    v = rand(rng, T, D)
    w = rand(rng, T, D)
    b = rand(rng, T)
    return PlanarFlow(v,w,b)
end

PlanarFlow(D) = PlanarFlow(Random._GLOBAL_RNG, Float64, D)

(P::PlanarFlow)(z) = z .+ P.v .* tanh.((P.w)'*z .+ P.b)

params(P::PlanarFlow) = Flux.params(P.v, P.w, [P.b])
eltype(P::PlanarFlow) = eltype(P.v)

"""
    `sample(pᵤ, P)`
# Inputs - 
- pᵤ : Base distribution which may be from the package Distributions
    or any distribution which can be sampled using `rand`
- P : Planar flow
"""
function sample(rng::AbstractRNG, pᵤ, P::PlanarFlow)
    T = eltype(P)
    u = convert.(T, rand(rng, pᵤ, size(P.v)))
    return P(u)
end

sample(pᵤ, P::PlanarFlow) = sample(Random._GLOBAL_RNG, pᵤ, P)