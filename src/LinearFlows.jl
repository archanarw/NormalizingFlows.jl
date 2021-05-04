using Flux, Distributions, Random
using LinearAlgebra

export Actnorm, affinecouplinglayer, sample, glow

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


function sample(p_u, m, size)
    u = rand(p_u, size)
    return m(u)
end