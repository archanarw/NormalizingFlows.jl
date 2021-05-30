using NormalizingFlows, Random
using Test
  
rng = MersenneTwister(0)
# Conditioner(rng, k, T)
# Weight of the conditioner must be of size kxk
# Bias must be of size k
# Each element in the conditioner must be of type T
# Weight matrix must be lower triangular

function is_lower(A)
    k = size(A, 1)
    if size(A, 2) != k
        return false
    end
    for i in 1:k
        for j in (i+1):k
            if A[i,j] != zero(eltype(A))
                return false
            end
        end
    end
    return true
end

@test size(AffineLayer(rng, 3, Float32).W) == (3,3)
@test size(AffineLayer(rng, 3, Float32).b) == (3,)
@test eltype(AffineLayer(rng, 3, Float32).W) == Float32
@test is_lower(AffineLayer(rng, 3, Float32).W)

# (A::AffineLayer)(z)
# Must return an array of τ(zᵢ) for each element in zᵢ in z
model = AffineLayer(rng, 3, Float32)
z = randn(rng, 3)
@test size.(NormalizingFlows.params(model)) == [(3, 3), (3,)]
@test size(model(z)) == size(z)

##################################################################

# function loss_kl(x, u)
#     Flux.Losses.kldivergence(abs.(x), abs.(eval_model(u)))
# end

# function train!(rng::AbstractRNG, ps, data, loss, pᵤ, opt)
#     for i in size(data, 2)
#         x = data[:,i]
#         u::Array{Float32,1} = rand(rng, pᵤ, size(x))
#         g = gradient(ps) do
#             @show loss(x, u)
#         end
#         Flux.update!(opt, ps, g)
#     end
# end

# train!(data, loss, pᵤ, opt, model) = train!(Random.GLOBAL_RNG, data, loss, pᵤ, opt, model)

# function eval_model(x)
#     out = model(x)
#     Flux.reset!(simple_rnn)
#     out
# end

# model = Chain(Flux.GRU(784, 784), Dense(784, 784))

# ps = Flux.params(model)
# opt = ADAM(0.001)

# for i in 1:50
#     train!(rng, ps, x_0, loss_kl, pᵤ, opt)
# end
# s = convert.(Float32, rand(rng, pᵤ, 784))
# s = model(s)