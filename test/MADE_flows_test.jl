using NormalizingFlows, Flux
using MLDatasets, Plots
using Distributions, Test
using Random
rng = MersenneTwister(1234)

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

@test size(Conditioner(rng, 3, Float32).W) == (3,3)
@test size(Conditioner(rng, 3, Float32).b) == (3,)
@test eltype(Conditioner(rng, 3, Float32).W) == Float32
@test is_lower(Conditioner(rng, 3, Float32).W)

# (A::AffineLayer)(z)
# Must return an array of τ(zᵢ) for each element in zᵢ in z
model = AffineLayer(Conditioner(rng, 3, Float32))
z = randn(rng, 3)
@test size.(NormalizingFlows.params(model)) == [(3, 3), (3,)]
@test size(model(z)) == size(z)


xtrain, ytrain = MLDatasets.MNIST.traindata(Float32);
	
xtrain = Flux.flatten(xtrain);

train_loader = Flux.Data.DataLoader((xtrain, ytrain));

t = [(x,y) for (x,y) in train_loader]
sort!(t, by = x-> x[2])
t1 = []
for (x,y) in t
    if y == [1]
        break
    end
    push!(t1, (x,y))
end

x_0 = t1[1][1]
for (x,y) in t1[2:end]
    x_0 = hcat(x_0, x)
end

opt = Flux.ADAM(0.001)
pᵤ = Uniform(0,1)
model = NormalizingFlows.AffineLayer(Conditioner(rng, 784, Float32))

l = Flux.Losses.crossentropy(abs.(model(rand(rng, pᵤ, 784))), abs.(xtrain[:,1]))

for i in 1:30
    train!(rng, x_0, loss_kl, pᵤ, opt, model)
end

s = NormalizingFlows.sample(rng, pᵤ, model)

@test abs.(Flux.Losses.crossentropy(abs.(s), abs.(xtrain[:,1]))) < abs.(l)

# heatmap(reshape(abs.(s), 28, 28))

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