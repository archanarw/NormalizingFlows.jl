using Flux, Distributions, ForwardDiff, LinearAlgebra, Random, StatsBase
import Flux.params, Base.eltype, StatsBase.sample, Distributions.pdf, Distributions.logpdf

export lower_ones, AffineLayer, params, sample, pdf, τ, inverse_τ, apply_transform

#Implementing the transformer τ
#τ(z_i, h_i) = α_i*z_i + β_i where α_i must be non-zero, h_i = {α_i, β_i}, h_i = c(s_i)
τ(z, h) = exp(h.α[1])*z + h.β
inverse_τ(z,h) = inv(exp(h.α[1]))*z - h.β*inv(exp(h.α))

"Returns lower triangular square matrix of size k whose values are 1"
function lower_ones(T, k::Integer)
    return [i < j ? zero(T) : one(T) for i = 1:k, j = 1:k]
end

function islower_ones(T, k)
    arr = lower_ones(T, k)
    for i in 1:size(arr,1)
        for j in 1:size(arr,2)
            if i<j
                if arr[i,j] != zero(T)
                    return false
                end
            else
                if arr[i,j] != one(T)
                    return false
                end     
            end
        end
        return true
    end
    # All elements of `arr` above the diagonal should be zero
    # The remaining elements of `arr` are 1
    # `arr` is a square matrix
    # type of each element in `arr` is T
    # size of `arr` is kxk
end

# Implementing the conditioner using Masked Autoregressive flows
# h_i = c(s_i) where s_i is vector of z_i where i < D and h1 = c(s1), s1 is the initial condition
# h_i = {α_i, β_i}, where both are real.

"""
Affine Layer with parameters:
- `W`: Weight of the conditioner
- `b`: Bias of the conditioner

`AffineLayer` implements the conditioner and the transformation to the input `z`
"""
struct AffineLayer{U <: AbstractArray{<: Real}, V <: AbstractArray{<: Real}}
    W::U #DXD matrix
    b::V #Vector of size D
end

function AffineLayer(rng::AbstractRNG, K::Integer, T)
    m::AbstractArray{T,2} = rand(rng, K, K)
    mask = lower_ones(T, K)
    m = m.*mask
    AffineLayer(m, rand(rng, T, K))
end

AffineLayer(K::Integer) = AffineLayer(Random.GLOBAL_RNG, K, Float64)

function apply_transform(A::AffineLayer, transform, z)
    return [transform(z[i], (α = (transpose(A.W[:,i]))*z, β = A.b[i])) for i in 1:length(z)]
end

(A::AffineLayer)(z) = apply_transform(A,τ,z)

params(A::AffineLayer) = Flux.params(A.W, A.b)
eltype(A::AffineLayer) = eltype(A.b)

# Sampling from the model
"""
    `sample(pᵤ, A)`

# Inputs - 
- pᵤ : Base distribution which may be from the package Distributions
or any distribution which can be sampled using `rand`
- A : Affine layer
"""
function StatsBase.sample(rng::AbstractRNG, pᵤ, A::AffineLayer)
    l = size(A.b)
    u = rand(rng, pᵤ, l)
    return A(u)
end

StatsBase.sample(pᵤ, A::AffineLayer) = sample(Random.GLOBAL_RNG, pᵤ, A)

# pdf of the distribution after applying transform
#p_x = p_u(T^-1(x))|det J_T^-1(x)|
"""
    `pdf(z, pᵤ, A)`

# Inputs 
- `z` : Value at which probability density is estimated
- `pᵤ` : Base distribution which may be from the package Distributions
or any distribution whose density can be evaluated using `pdf`
- A : Affine layer

# Returns the probability density of `z` wrt to the distribution given by A
"""
function Distributions.pdf(z, pᵤ, A)
    j = prod([exp((transpose(A.W[:,i]))*z) for i in 1:length(z)]) #Jacobian
    t = apply_transform(A, inverse_τ, z)
    return pdf(pᵤ, t)/abs(det(j))
end

"""
    `logpdf(z, pᵤ, A)`

# Inputs 
- `z` : Value at which logpdf is estimated
- `pᵤ` : Base distribution which may be from the package Distributions
or any distribution whose density can be evaluated using `pdf`
- A : Affine layer

# Returns the probability density of `z` wrt to the distribution given by A
"""
function Distributions.logpdf(z, pᵤ, A)
    j = sum([(transpose(A.W[:,i]))*z for i in 1:length(z)]) #log absolute Jacobian determinant
    t = apply_transform(A, inverse_τ, z)
    return logpdf(pᵤ, t)+ j
end