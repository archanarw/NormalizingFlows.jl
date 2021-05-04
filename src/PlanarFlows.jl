using Flux, Distributions
using LinearAlgebra
import Flux.params

export PlanarFlow, sample, params

struct PlanarFlow
    v
    w
    b
end

function PlanarFlow(D)
    v = rand(D)
    w = rand(D)
    b = rand(D)
    return PlanarFlow(v,w,b)
end

(P::PlanarFlow)(z) = z .+ P.v .* σ.((P.w)'*z .+ P.b)

params(P::PlanarFlow) = Flux.params(P.v, P.w, P.b)

"""
`sample(pᵤ, A)`
# Inputs - 
- pᵤ : Base distribution which may be from the package Distributions
    or any distribution which can be sampled using `rand`
- P : Planar flow
"""
function sample(pᵤ, P::PlanarFlow)
    u = rand(pᵤ, size(P.v))
    return P(u)
end