using Flux, Distributions, Random
import Flux.params

export affinecouplinglayer, sample, GLOW

"""
Coupling layer:
It transforms the input as:
zᵢ′ = zᵢ for i < d
zᵢ′ = A(zᵢ) for i >= d where d = D/2
# Inputs - 
- z : input from trainig data
- A : AffineLayer of the required size d, i.e., D/2 where
    D is the size of input data
"""
function affinecouplinglayer(z, A::AffineLayer)
    D = size(z,1)
    d = div(D,2)
    z′ = z[1:d]
    z′ = vcat(z′, A(z[(d+1):end]))
    return z′
end

"""
GLOW has 3 layers:
First - Conv1x1 layer (for ease of finding inverse : W = PLU, 
                      where P is a fixed permutation layer, L and U are learnt)
Second - Actnorm(Autoregressive layer with batch normalization)
Third - Coupling layer
"""
struct GLOW
    conv
    B
    A::AffineLayer
end

function GLOW(rng::AbstractRNG, T, channels, D)
    conv1x1 = Flux.Conv((1,1), channels => channels, relu)
    # Flatten after this
    d = div(D,2)
    A = AffineLayer(rng, d, T)
    return GLOW(conv1x1, BatchNorm(channels), A)
end

GLOW(channels, D) = GLOW(Random._GLOBAL_RNG, Float64, channels, D)

#Actnorm before glow
#glow includes permutation and affine coupling
function (L::GLOW)(z)
    conv1x1 = L.conv
    # Flatten after this
    A = L.A
    coupling = z -> affinecouplinglayer(z,A)
    return Chain(conv1x1, L.B, flatten, coupling, softmax)(z)
end

params(L::GLOW) = Flux.params(L.conv, L.B, L.A.W, L.A.b)

# Sampling from the model
"""
`sample(pᵤ, A)`
# Inputs - 
- pᵤ : Base distribution which may be from the package Distributions
or any distribution which can be sampled using `rand`
- L : Linear Flow
"""
function sample(rng::AbstractRNG, pᵤ, channels, L::GLOW)
    l = Int(sqrt(size(L.A.b, 1)*2))
    u = rand(rng, pᵤ, l, l, channels, 1)
    return L(u)
end

sample(pᵤ, L::GLOW) = sample(Random.GLOBAL_RNG, pᵤ, L)