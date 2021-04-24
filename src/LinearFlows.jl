using Flux, Distributions, Random
using LinearAlgebra

export train_glow!, Actnorm, affinecouplinglayer, sample_glow, glow

τ(z, h) = exp(h.α[1])*z + h.β

function lower_ones(k::T) where T <: Real
    a = Array{T,2}(undef, k, k)
    [a[i,j] = i < j ? zero(T) : one(T) for i = 1:k, j = 1:k]
end

struct Conditioner
    W #DXD matrix
    b #Vector of size D
end

function Conditioner(rng::AbstractRNG, K::Integer)
    m = rand(rng, K)
    mask = lower_ones(K)
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

# GLOW has 3 layers:
# First - Actnorm(Autoregressive layer with batch normalization)
# Second - Conv1x1 layer (W = PLU, where P is a fixed permutation layer, L and U are learnt)
# Third - Coupling layer
struct Actnorm
    m
    sd
end

function Actnorm(data)
    channels = size(data, ndims(data))
    # m = [mean(data[:,:,i,:]) for i in 1:channels]
    # sd = [std(data[:,:,i,:]) for i in 1:channels]
    m = mean(data)
    sd = std(data)
    return Actnorm(m,sd)
end

(AN::Actnorm)(data) = (data .- AN.m)/AN.sd

function affinecouplinglayer(z, A::AffineLayer)
    D = size(z,1)
    d = div(D,2)
    z′ = z[1:d]
    z′ = vcat(z′, A(z[1:d]))
end

#Actnorm before glow
#glow includes permutation and affine coupling
function glow(channels, A::AffineLayer)
    conv1x1 = Flux.Conv((1,1), channels => channels, relu)
    # Flatten after this
    coupling = z -> affinecouplinglayer(z,A)
    return Chain(conv1x1, flatten, coupling, softmax)
end

function train_glow!(ps, data, p_u, opt, model, AN)
    data = AN(data)
    for (x,y) in data
        u = pdf(p_u, abs.(model[Int(y[1]+1)](x)))
        g = gradient(ps[Int(y[1]+1)]) do
            Flux.Losses.kldivergence(u, abs.(x))
        end
        Flux.update!(opt, ps[Int(y[1]+1)], g)
    end
end

function sample_glow(p_u, m, size)
    u = rand(p_u, size)
    return m(u)
end