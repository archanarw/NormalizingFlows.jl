using Flux, Distributions
using LinearAlgebra
import Flux.params

export RadialFlow, sample, params

"""
The parameters of radial flow are:
- `α` α ∈ (0, +∞)
- `β` ∈ ℝ
- z₀ ∈ ℝᴰ

It takes the following form:
z′ = z + β/(α + r(z))*(z - z₀) where r(z) = ||z-z₀||

The above transformation can be thought of as a 
contraction/expansion radially with center z₀
"""
struct RadialFlow{T}
    z₀::Array{T}
    α::T
    β::T
end

function RadialFlow(rng::AbstractRNG, T, D)
    z₀ = rand(rng, T, D)
    α = exp(rand(rng, T))
    β = rand(rng, T)
    return RadialFlow(z₀, α, β)
end

RadialFlow(D) = RadialFlow(Random._GLOBAL_RNG, Float64, D)

(R::RadialFlow)(z) = z .+ (R.β ./ (R.α .+ norm(z))) .* (z .- R.z₀)

params(R::RadialFlow) = Flux.params(R.z₀, [R.α], [R.β])

"""
    `sample(pᵤ, R)`
# Inputs - 
- pᵤ : Base distribution which may be from the package Distributions
    or any distribution which can be sampled using `rand`
- R : Radial flow
"""
function sample(rng::AbstractRNG, pᵤ, R::RadialFlow)
    u = rand(rng, pᵤ, size(R.z₀))
    return R(u)
end

sample(pᵤ, R::RadialFlow) = sample(Random._GLOBAL_RNG, pᵤ, R)