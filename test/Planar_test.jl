using Flux, NormalizingFlows, Distributions
using MLDatasets, Plots
using Random, Test

rng = MersenneTwister(0)

xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtest, ytest = MLDatasets.MNIST.testdata(Float32)
	
xtrain = Flux.flatten(xtrain)
xtest = Flux.flatten(xtest)

train_loader = Flux.Data.DataLoader((xtrain, ytrain))
test_loader = Flux.Data.DataLoader((xtest, ytest))

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
model = PlanarFlow(28^2)

@test size.(NormalizingFlows.params(model)) == [(784,),(784,),(784,)]

for i in 1:50
    train!(x_0, loss_kl, pᵤ, opt, model)
end

s = NormalizingFlows.sample(pᵤ, model)
heatmap(reshape(abs.(s), 28, 28))