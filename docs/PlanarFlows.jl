### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 1e648e40-bd72-11eb-025e-2d0414ed36bd
md"## Normalizing Flows: Planar Flows"

# ╔═╡ a7a00dcd-d1aa-4041-87e7-66673c295dfd
md"Planar flows use functions of form

$f(z) = z + u h(w^{T} z + b)$"

# ╔═╡ 92ba968c-296b-484e-9bd3-56d309150d18
md"where $u, w ∈ \mathbb{R}^d$, $b \in \mathbb{R}$, and $h$ is a function such as the sigmoid function."

# ╔═╡ fd6bf6d5-db03-4b5a-b7e8-fd8dc97579e1
md"The Jacobian is then given by:

$\frac{δf(z)}{δz} = I + u h'(w^{T} z + b)w^{T}$"

# ╔═╡ dc964e09-7c1f-46d0-bead-ee45f1a799cb
md"And,

$det(\frac{δf(z)}{δz}) = 1 + h'(w^{T} z + b)w^{T}u$"

# ╔═╡ b781a822-01bd-4e18-b524-24e69179abea
md"which is required to calculate the density of the distribution."

# ╔═╡ 9b551eac-84fd-4ce1-8b37-0221200b3602
md"###### Model:"

# ╔═╡ 6f8e995f-1f7d-43dd-aca0-6888204099e7
md"###### Sampling:"

# ╔═╡ d93f4477-d81f-4c12-b747-ee498dd89f05


# ╔═╡ ecb2fab7-1fbb-4448-a57f-6683aaf3dbcc
begin
	using NormalizingFlows
	P = PlanarFlow(4)
end

# ╔═╡ fe2ec09e-312b-4f76-809f-2c9acd2ba7cc
begin
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
	model = PlanarFlow(rng, Float32, 28^2)

	for i in 1:20
		train!(rng, x_0, loss_kl, pᵤ, opt, model)
	end

	s = NormalizingFlows.sample(rng, pᵤ, model)
	heatmap(reshape(abs.(s), 28, 28))
end

# ╔═╡ Cell order:
# ╟─1e648e40-bd72-11eb-025e-2d0414ed36bd
# ╟─a7a00dcd-d1aa-4041-87e7-66673c295dfd
# ╟─92ba968c-296b-484e-9bd3-56d309150d18
# ╟─fd6bf6d5-db03-4b5a-b7e8-fd8dc97579e1
# ╟─dc964e09-7c1f-46d0-bead-ee45f1a799cb
# ╟─b781a822-01bd-4e18-b524-24e69179abea
# ╟─9b551eac-84fd-4ce1-8b37-0221200b3602
# ╠═ecb2fab7-1fbb-4448-a57f-6683aaf3dbcc
# ╠═fe2ec09e-312b-4f76-809f-2c9acd2ba7cc
# ╟─6f8e995f-1f7d-43dd-aca0-6888204099e7
# ╠═d93f4477-d81f-4c12-b747-ee498dd89f05