using Flux, Distributions, ForwardDiff, LinearAlgebra
import Flux.params

export train, τ, inverse_τ, Conditioner, sample, expected_pdf, affinelayer, f

#Implementing the transformer τ
#τ(z_i, h_i) = α_i*z_i + β_i where α_i must be non-zero, h_i = {α_i, β_i}, h_i = c(s_i)
τ(z, h) = exp(h.α[1])*z + h.β
inverse_τ(z,h) = inv(exp(h.α[1]))*z - h.β/exp(h.α)

"Returns lower triangular square matrix of size k whose values are 1"
function lower_ones(m::Integer, n::Integer)
    k = min(m,n)
    a = [[1] zeros(k-1)']
    for i in 2:k
        v = hcat(ones(i)', zeros(k-i)')
        a = vcat(a, v)
    end
    if(m<n)
        a = hcat(a, zeros(m, n-m))
    elseif(m>n)
        a = vcat(a, ones(m-n, n))
    end
    return a
end

# Implementing the conditioner using Masked Autoregressive flows
# h_i = c(s_i) where s_i is vector of z_i where i < D and h1 = c(s1), s1 is the initial condition
# h_i = {α_i, β_i}, where both are real.
# Conditioner(feedforward neural network):z -> h
struct Conditioner
    W #DXD matrix
    b #Vector of size D
end

function Conditioner(K::Integer, L::Integer)
    m = rand(L,K)
    mask = lower_ones(L,K)
    m = m.*mask
    Conditioner(m, rand(L))
end

(c::Conditioner)(z) = s.W*z .+ s.b

struct affinelayer
    c::Conditioner
end

function f(A::affinelayer, transform, z)
    # l, k = size(A.c.W)
    # T = [transform(z[1,j],(α = A.c.W[j,1], β = A.c.b[j])) for j in 1:l]'
    # for i in 2:k
    #     t = [transform(z[i,j],(α = A.c.W[j,i], β = A.c.b[j])) for j in 1:l]'
    #     T = vcat(T, t)
    # end
    # return T
    return [transform(z[i], (α = A.c.W[i,i], β = A.c.b[i])) for i in 1:length(z)]
end

Flux.params(m::affinelayer) = Flux.params(m.c.W, m.c.b)

(A::affinelayer)(z) = f(A,τ,z)

function train(ps, data, p_u, opt, model)
    for (x,y) in data
        u′ = model[Int(y[1]+1)](x)
        u = rand(p_u, size(u′))
        g = gradient(ps[Int(y[1]+1)]) do
            Flux.Losses.kldivergence(u′, u)
        end
        Flux.update!(opt, ps[Int(y[1]+1)], g)
    end
end

# function train(ps, data, p_u, opt, m)
#     for (x,y) in data
#         u′ = m(x)
#         u = rand(p_u, size(u′))
#         g = gradient(ps) do
#             Flux.Losses.kldivergence(u′,u)
#         end
#         Flux.update!(opt, ps, g)
#     end
# end

# Sampling from the model
function sample(p_u, A)
    l = size(A.c.b)
    u = rand(p_u, l)
    return f(A, inverse_τ, u)
end

# pdf of the distribution after applying transform
"p_x = p_u(T^-1(x))|det J_T^-1(x)|"
function expected_pdf(data, p_u, A)
    j = sum(log.(abs.(diag(A.c.W))))
    t = f(A, τ, data)
    return pdf(p_u, t)*abs(det(j))
end

