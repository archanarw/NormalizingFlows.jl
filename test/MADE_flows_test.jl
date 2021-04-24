using NormalizingFlows, Flux
using MLDatasets, Plots
using Distributions

# d = Flux.flatten(randn(3,2,10))
# labels = zeros(10)
# data = Flux.Data.DataLoader((d,labels))
# c = Conditioner(6)
# model = []
# model = vcat(model, affinelayer(c))
# opt = Flux.ADAM(0.001)
# p_u = Uniform(0,1)
# ps = Flux.Params[]
# ps = vcat(ps, Flux.params(c.W,c.b))
# for i in 1:30
#     train(ps, data, p_u, opt)
# end
# s = sample(p_u, model[1])
# expected_pdf(s, p_u, model[1])

xtrain, ytrain = MLDatasets.MNIST.traindata(Float32);
xtest, ytest = MLDatasets.MNIST.testdata(Float32);
	
xtrain = Flux.flatten(xtrain);
xtest = Flux.flatten(xtest);

# ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9);

xtrain, ytrain = xtrain[:,1:500], ytrain[1:500];

train_loader = Flux.Data.DataLoader((xtrain, ytrain));
test_loader = Flux.Data.DataLoader((xtest, ytest));

opt = Flux.ADAM(0.001)
pᵤ = Uniform(0,1)
labels = unique(ytrain)
ps = Flux.Params[]
model = []
for label in sort(labels)
    m = NormalizingFlows.AffineLayer(Conditioner(784))
    ps = vcat(ps, Flux.params(m.c.W, m.c.b))
    model = vcat(model, m)
end

for i in 1:20
    NormalizingFlows.train!(ps, train_loader, pᵤ, opt, model)
    println(i)
end

s = NormalizingFlows.sample(pᵤ, model[8])
heatmap(reshape(s, 28, 28))