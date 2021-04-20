xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtest, ytest = MLDatasets.MNIST.testdata(Float32)

# ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9);

xtrain, ytrain = xtrain[:,:,1:5000], ytrain[1:5000]

train_loader = Flux.Data.DataLoader((xtrain, ytrain))
test_loader = Flux.Data.DataLoader((xtest, ytest))

opt = Flux.ADAM(0.001)
p_u = Uniform(0,1)
labels = unique(ytrain)
ps = Flux.Params[]
model = []
D = div(*(size(xtrain[:,:,1])...), size(xtrain[:,:,1],ndims(xtrain[:,:,1])))
d = div(D,2)
AN = Actnorm(xtrain)
A = affinelayer(Conditioner(d,d))
model = glow(1,A)
for label in sort(labels)
    ps = vcat(ps, Flux.params(A))
    model = vcat(model, m)
end

# for i in 1:30
#     train(ps, train_loader, p_u, opt, model)
# end