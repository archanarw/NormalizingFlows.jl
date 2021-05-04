using NormalizingFlows, Flux
using MLDatasets, Plots
using Distributions, Test

using Random
rng = MersenneTwister(0)

#Conditioner
@test size(Conditioner(rng, 3, Float32).W) == (3,3)
@test size(Conditioner(rng, 3, Float32).b) == (3,)
@test typeof(Conditioner(rng, 3, Float32).W) == Array{Float32,2}


xtrain, ytrain = MLDatasets.MNIST.traindata(Float32);
xtest, ytest = MLDatasets.MNIST.testdata(Float32);
	
xtrain = Flux.flatten(xtrain);
xtest = Flux.flatten(xtest);

train_loader = Flux.Data.DataLoader((xtrain, ytrain));
test_loader = Flux.Data.DataLoader((xtest, ytest));

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

opt = Flux.Descent(0.001)
pᵤ = Uniform(0,1)
model = NormalizingFlows.AffineLayer(Conditioner(rng, 784, Float64))

#AffineLayer
@test size.(NormalizingFlows.params(model)) == [(784, 784), (784,)]

for i in 1:50
    NormalizingFlows.train!(rng, x_0, loss_kl, pᵤ, opt, model)
end

s = NormalizingFlows.sample(rng, pᵤ, model)

#Sampling
@test NormalizingFlows.expected_pdf(s, pᵤ, model) == 759.9282996555975.*ones(784)

# heatmap(reshape(abs.(s), 28, 28))