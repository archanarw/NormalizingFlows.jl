using Flux, Distributions, ForwardDiff, LinearAlgebra, Random
import Flux.params

export Conditioner, AffineLayer, params, sample, expected_pdf

#Implementing the transformer τ
#τ(z_i, h_i) = α_i*z_i + β_i where α_i must be non-zero, h_i = {α_i, β_i}, h_i = c(s_i)
τ(z, h) = exp(h.α[1])*z + h.β
inverse_τ(z,h) = inv(exp(h.α[1]))*z - h.β/exp(h.α)

"Returns lower triangular square matrix of size k whose values are 1"
function lower_ones(T, k::Integer)
    a = Array{T,2}(undef, k, k)
    [a[i,j] = i < j ? zero(T) : one(T) for i = 1:k, j = 1:k]
end

# Implementing the conditioner using Masked Autoregressive flows
# h_i = c(s_i) where s_i is vector of z_i where i < D and h1 = c(s1), s1 is the initial condition
# h_i = {α_i, β_i}, where both are real.
"Conditioner is a feedforward neural network such that  Conditioner: z -> h"
struct Conditioner
    W #DXD matrix
    b #Vector of size D
end

function Conditioner(rng::AbstractRNG, K::Integer, T)
    m::Array{T,2} = rand(rng, K, K)
    mask = lower_ones(T, K)
    m = m.*mask
    Conditioner(m, rand(rng, K))
end

Conditioner(K::Integer) = Conditioner(Random.GLOBAL_RNG, K, Float64)

(c::Conditioner)(z) = s.W*z .+ s.b

struct AffineLayer
    c::Conditioner
end

function f(A::AffineLayer, transform, z)
    return [transform(z[i], (α = A.c.W[i,i], β = A.c.b[i])) for i in 1:length(z)]
end

(A::AffineLayer)(z) = f(A,τ,z)

params(A::AffineLayer) = Flux.params(A.c.W,A.c.b)

# Sampling from the model
"""
`sample(pᵤ, A)`
# Inputs - 
- pᵤ : Base distribution which may be from the package Distributions
or any distribution which can be sampled using `rand`
- A : Affine layer
"""
function sample(rng::AbstractRNG, pᵤ, A::AffineLayer)
    l = size(A.c.b)
    u = rand(rng, pᵤ, l)
    return f(A, inverse_τ, u)
end

sample(pᵤ, A) = sample(Random.GLOBAL_RNG, pᵤ, A)

# pdf of the distribution after applying transform
#p_x = p_u(T^-1(x))|det J_T^-1(x)|
"""
`expected_pdf(data, p_u, A)`

# Inputs 
- `z` : Value whose probability density is estimated
- `pᵤ` : Base distribution which may be from the package Distributions
or any distribution which can be sampled using `rand`
- A : Affine layer

# Returns the probability density of `z` wrt to the distribution given by A
"""
function expected_pdf(z, pᵤ, A)
    j = sum(log.(abs.(diag(A.c.W)))) #Jacobian
    t = f(A, τ, z)
    return pdf(pᵤ, t)*abs(det(j))
end