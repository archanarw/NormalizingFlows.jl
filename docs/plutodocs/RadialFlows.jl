### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 0789eb59-99c4-4051-a327-57dd0a6bdb7a
begin
	using NormalizingFlows
	P = RadialFlow(4)
end

# ╔═╡ 1c9982d8-61b6-4a3d-a44e-e953a2b6163a
begin
	using Flux, Distributions
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

	x_0 = [x for (x, y) in t1]

	opt = Flux.ADAM(0.001)
	pᵤ = Uniform(0,1)
	model = RadialFlow(rng, Float32, 28^2)

	for i in 1:20
		train!(rng, x_0, loss_kl, pᵤ, opt, model)
	end

	s = NormalizingFlows.sample(rng, pᵤ, model)
	heatmap(reshape(abs.(s), 28, 28))
end

# ╔═╡ 19b66242-c111-11eb-2a7f-c1f51fc654c0
md"## Normalizing Flows: Radial Flows"

# ╔═╡ 43efe91a-2b82-4c94-be3f-3bd8d9b196cc
md"Radial flows use functions of form

$z' = z + \frac{β}{α + r(z)} (z - z_{0})$"

# ╔═╡ e8f4cb10-677c-4083-a4d0-06b5fbacbc78
md"where $α ∈ (0, ∞), β ∈ \mathbb{R}$, $z_{0} \in \mathbb{R}^{D}$, and $r = ||z-z_{0}||$ where $ ||·||$ is the Euclidean norm."

# ╔═╡ a36d2864-e198-4154-a4a4-6807896ef8a3
md" The above transformation can be thought of as a contraction/expansion radially
with center $z_{0}$."

# ╔═╡ 9dc75a44-7fd6-4281-bad6-4d271866a682
md"The Jacobian of radial flow is given by:

$J_{f_{\phi}}(z) = (1+ \frac{β}{α + r(z)})I - \frac{β}{r(z)(α + r(z))^{2}}(z - z_{0})(z - z_{0})^{T}$"

# ╔═╡ 58742bd0-eefe-421b-b3d1-4f48464071e5
md"And the determinant of the Jacobian is as follows:

$det J_{f_{\phi}}(z) = (1 + \frac{αβ}{(α + r(z))^{2}})(1+ \frac{β}{α + r(z)})^{D-1}$"

# ╔═╡ 3e542f27-e74c-4e1f-9c19-643960465405
md"###### Model:"

# ╔═╡ ceac6461-8c93-4bcd-9641-17e02e532351
md"###### Training:"

# ╔═╡ b85b60f1-2acf-47aa-9d09-81ecd12c82e0
md"###### Sampling:"

# ╔═╡ c435355c-6a6c-4562-95ca-bc01ca9cc1fb
md"`sample(pᵤ, R)` takes the base distribution and radial flow model as inputs and outputs a sample of the density from which the input samples were generated."

# ╔═╡ Cell order:
# ╟─19b66242-c111-11eb-2a7f-c1f51fc654c0
# ╟─43efe91a-2b82-4c94-be3f-3bd8d9b196cc
# ╟─e8f4cb10-677c-4083-a4d0-06b5fbacbc78
# ╟─a36d2864-e198-4154-a4a4-6807896ef8a3
# ╟─9dc75a44-7fd6-4281-bad6-4d271866a682
# ╟─58742bd0-eefe-421b-b3d1-4f48464071e5
# ╟─3e542f27-e74c-4e1f-9c19-643960465405
# ╠═0789eb59-99c4-4051-a327-57dd0a6bdb7a
# ╟─ceac6461-8c93-4bcd-9641-17e02e532351
# ╠═1c9982d8-61b6-4a3d-a44e-e953a2b6163a
# ╟─b85b60f1-2acf-47aa-9d09-81ecd12c82e0
# ╟─c435355c-6a6c-4562-95ca-bc01ca9cc1fb
