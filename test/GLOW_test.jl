using Flux, NormalizingFlows, Distributions
using MLDatasets, Plots
using Random, Test

rng = MersenneTwister(0)

xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)

xtrain = xtrain |> Flux.unsqueeze(3)

# train_loader = Flux.Data.DataLoader((xtrain, ytrain))

opt = Flux.ADAM(0.001)
pᵤ = Uniform(0,1)

model = GLOW(rng, Float32, 1, 900)

for i in 1:30
    train!(rng, xtrain, loss_kl, pᵤ, opt, model)
end