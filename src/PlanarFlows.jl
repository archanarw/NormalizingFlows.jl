using Flux, Distributions
using LinearAlgebra
import Flux.params

export PlanarFlow, sample, params

struct PlanarFlow
    v
    w
    b
end

function PlanarFlow(rng::AbstractRNG, T, D)
    v = rand(rng, T, D)
    w = rand(rng, T, D)
    b = rand(rng, T, D)
    return PlanarFlow(v,w,b)
end

PlanarFlow(D) = PlanarFlow(Random._GLOBAL_RNG, Float64, D)

(P::PlanarFlow)(z) = P.v .* σ.((P.w)'*z .+ P.b)

params(P::PlanarFlow) = Flux.params(P.v, P.w, P.b)

"""
`sample(pᵤ, A)`
# Inputs - 
- pᵤ : Base distribution which may be from the package Distributions
    or any distribution which can be sampled using `rand`
- P : Planar flow
"""
function sample(rng::AbstractRNG, pᵤ, P::PlanarFlow)
    u = rand(rng, pᵤ, size(P.v))
    return P(u)
end

sample(pᵤ, P::PlanarFlow) = sample(Random._GLOBAL_RNG, pᵤ, P)