using MLDatasets, ImageCore

xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtest, ytest = MLDatasets.MNIST.testdata(Float32)
	
xtrain = Flux.flatten(xtrain)
xtest = Flux.flatten(xtest)

# ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9);

xtrain, ytrain = xtrain[:,1:5000], ytrain[1:5000]

train_loader = Flux.Data.DataLoader((xtrain, ytrain))
test_loader = Flux.Data.DataLoader((xtest, ytest))

opt = Flux.ADAM(0.001)
p_u = Uniform(0,1)
labels = unique(ytrain)
ps = Flux.Params[]
model = []
for label in sort(labels)
    P = PlanarFlow(size(xtrain,1))
    ps = vcat(ps, Flux.params(P.v, P.w, P.b))
    model = vcat(model, P)
end

for i in 1:30
    train_planar(ps, train_loader, p_u, opt, model)
end

s = sample_planar(p_u, model[1])
MNIST.convert2image(s)