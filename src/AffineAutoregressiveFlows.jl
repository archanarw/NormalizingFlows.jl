using Flux, Distributions, ForwardDiff, LinearAlgebra, Random
import Flux.params

export train!, τ, inverse_τ, Conditioner, sample, expected_pdf, AffineLayer, f

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
# Conditioner(feedforward neural network):z -> h
struct Conditioner
    W #DXD matrix
    b #Vector of size D
end

function Conditioner(rng::AbstractRNG, K::Integer)
    m = rand(rng, K)
    mask = lower_ones(Float64, K)
    m = m.*mask
    Conditioner(m, rand(rng, K))
end

Conditioner(K::Integer) = Conditioner(Random.GLOBAL_RNG, K)

(c::Conditioner)(z) = s.W*z .+ s.b

struct AffineLayer
    c::Conditioner
end

function f(A::AffineLayer, transform, z)
    return [transform(z[i], (α = A.c.W[i,i], β = A.c.b[i])) for i in 1:length(z)]
end

(A::AffineLayer)(z) = f(A,τ,z)

"""
`train!(rng::AbstractRNG, ps, data, p_u, opt, model)`

# Inputs - 
- `rng`
- `ps`: Parameters of the conditioner
- `data` : The training data
- `pᵤ` : Base distribution
- `opt` : Optimizer
- `model` : AffineLayer
"""
function train!(rng::AbstractRNG, ps, data, pᵤ, opt, model)
    for (x,y) in data
        u = rand(rng, pᵤ, size(x))
        g = gradient(Flux.params(ps[Int(y[1]+1)][1], ps[Int(y[1]+1)][2])) do
            Flux.Losses.kldivergence(abs.(model[Int(y[1]+1)](x)), abs.(u))
        end
        # Flux.update!(opt, ps[Int(y[1]+1)], g)
        ps[Int(y[1]+1)] = Flux.params(ps[Int(y[1]+1)] .+ (0.001 .* g))
        model[Int(y[1]+1)] = AffineLayer(Conditioner(ps[Int(y[1]+1)][1], ps[Int(y[1]+1)][2]))
    end
end

train!(ps, data, pᵤ, opt, model) = train!(Random.GLOBAL_RNG, ps, data, pᵤ, opt, model)

# Sampling from the model
"""
`sample(pᵤ, A)`
# Inputs - 
- pᵤ : Base distribution
- A : Affine layer
"""
function sample(pᵤ, A)
    l = size(A.c.b)
    u = rand(pᵤ, l)
    return f(A, inverse_τ, u)
end

# pdf of the distribution after applying transform
#p_x = p_u(T^-1(x))|det J_T^-1(x)|
"""
`expected_pdf(data, p_u, A)`

# Inputs 
- `z` : Value whose probability density is estimated
- `pᵤ` : Base distribution
- A : Affine layer

# Returns the probability density of `z` wrt to the distribution given by A
"""
function expected_pdf(z, pᵤ, A)
    j = sum(log.(abs.(diag(A.c.W)))) #Jacobian
    t = f(A, τ, z)
    return pdf(pᵤ, t)*abs(det(j))
end
