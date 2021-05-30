using Flux, NormalizingFlows, Distributions
using MLDatasets, Plots
using Random

rng = MersenneTwister(0)

#GLOW Training
xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)

xtrain = xtrain |> Flux.unsqueeze(3)

# train_loader = Flux.Data.DataLoader((xtrain, ytrain))

opt = Flux.ADAM(0.001)
pᵤ = Uniform(0,1)

model = GLOW(rng, Float32, 1, 28^2)

for i in 1:10
    train!(rng, xtrain, loss_kl, pᵤ, opt, model)
end

s = NormalizingFlows.sample(rng, pᵤ, 1, model)
heatmap(reshape(abs.(s), 28, 28))