using Flux, Distributions

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
    for i in size(data, 2)
        x = data[:,i]
        u = rand(rng, pᵤ, size(x))
        g = gradient(ps) do
            @show loss(model, x, u)
        end
        Flux.update!(opt, ps, g)
    end
end

"KL Divergence"
function loss_kl(model, x, u)
    Flux.Losses.kldivergence(abs.(model(x)), abs.(u))
end

train!(data, loss, pᵤ, opt, model) = train!(Random.GLOBAL_RNG, data, loss, pᵤ, opt, model)

# function train_labelled_data!(rng::AbstractRNG, ps, data_loader, pᵤ, opt, model)
#    sort!(data_loader, by = x -> x[2])
#    for y in 1:data_loader[]
# end