using Flux, Distributions, Random, StatsBase

export train!, loss_kl

"""
`train!(rng::AbstractRNG, data, loss, p_u, opt, model)`

The function updates `ps`, the parameters of AffineLayer, so that pᵤ may be transformed 
to the distribution of each of the labelled inputs.

# Inputs - 
- `rng`
- `data` : The training data, sequence of samples of the target distribution
- `loss` : A function which takes the model, input, `x` and `u`, sample of base distribution
        and returns the divergence between the model applied to `x` and base distribution, which must be 
        minimized.
- `pᵤ` : Base distribution which may be from the package Distributions
        or any distribution which can be sampled using `rand`
- `opt` : Flux Optimizer
- `model` : The neural network used, i.e., one among AffineLayer, GLOW and Planar
"""
function train!(rng::AbstractRNG, data, loss, pᵤ, opt, model)
    ps = params(model)
    for i in 1:size(data, 2)
        x = data[:,i]
        u = oftype(x, rand(rng, pᵤ, size(x)...))
        g = gradient(() -> loss(model, x, u), Flux.params(ps[:]))
        Flux.update!(opt, ps[:], g)
    end
end

function train!(rng::AbstractRNG, data, loss, pᵤ, opt, model::GLOW)
    ps = params(model)
    for i in 1:size(data, 4)
        x = data[:,:,:,i] |> Flux.unsqueeze(3)
        u = oftype(x, rand(rng, pᵤ, size(x)...))
        g = gradient(() -> loss(model, Flux.flatten(x), u), Flux.params(ps[:]))
        Flux.update!(opt, ps[:], g)
    end
end

train!(data, loss, pᵤ, opt, model) = train!(Random.GLOBAL_RNG, data, loss, pᵤ, opt, model)

function train_labelled_data!(rng::AbstractRNG, data_loader, pᵤ, opt, model)
    train = [(x,y) for (x,y) in data_loader]
    sort!(train, by = x -> x[2])
    curr = 0
    while (curr != train[end][2]+1)
        t = []
        for (x,y) in train
            if y == [curr+1]
                curr+=1
                break
            end
            push!(t, (x,y))
        end
        x_curr = t[1][1]
        for (x_,y_) in t[2:end]
            x_curr = hcat(x_curr, x_)
        end
        train!(rng, x_curr, pᵤ, opt, model[curr+1])
    end
end

train_labelled_data!(data_loader, pᵤ, opt, model) = train_labelled_data!(Random.GLOBAL_RNG, data_loader, pᵤ, opt, model)

"KL Divergence"
function loss_kl(model, x, u)
    oftype(x[1], StatsBase.kldivergence(abs.(x), abs.(model(u))))
end