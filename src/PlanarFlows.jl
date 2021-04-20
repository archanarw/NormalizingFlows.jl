using Flux, Distributions
using LinearAlgebra

struct PlanarFlow
    v
    w
    b
end

function PlanarFlow(D)
    v = rand(D)
    w = rand(D)
    b = rand(D)
    return PlanarFlow(v,w,b)
end

(P::PlanarFlow)(z) = z .+ P.v .* σ.((P.w)'*z .+ P.b)

function train_planar(ps, data, p_u, opt, model)
    for (x,y) in data
        x′ = model[Int(y[1]+1)](rand(p_u, size(x,1)))
        g = gradient(ps[Int(y[1]+1)]) do
            Flux.Losses.kldivergence(x, x′)
        end
        Flux.update!(opt, ps[Int(y[1]+1)], g)
    end
end

function sample_planar(p_u, P)
    u = rand(p_u, size(P.v))
    return P(u)
end