using Flux, Distributions
using LinearAlgebra

#Multiplying with same P will undo the permutation
# function permutation_layer(m, n)
#     k = m
#     p = Array(I(k))
#     r = rand(1:k,2)
#     p_ = zeros(k)'
#     for i in 1:k
#         if i == r[1]
#             p_ = vcat(p_, p[r[2],:]')
#         elseif i == r[2]
#             p_ = vcat(p_, p[r[1],:]')
#         else
#             p_ = vcat(p_, p[i,:]')
#         end
#     end
#     return p_[2:end,:]
# end

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

function affinecouplinglayer(z, A::affinelayer)
    D = size(z,1)
    d = div(D,2)
    z′ = z[1:d]
    z′ = vcat(z′, A(z[1:d]))
end

#Actnorm before glow
#glow includes permutation and affine coupling
function glow(channels, A::affinelayer)
    conv1x1 = Flux.Conv((1,1), channels => channels, relu)
    # Flatten after this
    coupling = z -> affinecouplinglayer(z,A)
    return Chain(conv1x1, flatten, coupling, softmax)
end

function train_glow(ps, data, p_u, opt, model, AN)
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

