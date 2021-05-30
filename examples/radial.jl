using Flux, NormalizingFlows, Distributions
using MLDatasets, Plots
using Random

rng = MersenneTwister(0)

xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtrain = Flux.flatten(xtrain)
train_loader = Flux.Data.DataLoader((xtrain, ytrain))

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
    x_0 = reduce(hcat, [x_0, x])
end

opt = Flux.ADAM(0.001)
pᵤ = Uniform(0,1)
model = RadialFlow(rng, Float32, 28^2)

for i in 1:20
    train!(rng, x_0, loss_kl, pᵤ, opt, model)
end

s = NormalizingFlows.sample(rng, pᵤ, model)
heatmap(reshape(abs.(s), 28, 28))