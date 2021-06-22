### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ ecb2fab7-1fbb-4448-a57f-6683aaf3dbcc
begin
	using NormalizingFlows, Random
	rng = Random.seed!(123)
	P = PlanarFlow(rng, Float64, 4)
end

# ╔═╡ 20eae2af-c95f-4d2b-81ab-5576ffd2d1ba
begin
	using Distributions, Plots
	function image_sine(n)
		x = sin.(π/n:(π/n):π)
		arr = x
		for i in 1:(n-1)
			arr = hcat(arr, x)
		end
		return arr
	end
end

# ╔═╡ 1e648e40-bd72-11eb-025e-2d0414ed36bd
md"## Normalizing Flows: Planar Flows"

# ╔═╡ a7a00dcd-d1aa-4041-87e7-66673c295dfd
md"Planar flows use functions of form

$f(z) = z + v h(w^{T} z + b)$"

# ╔═╡ 92ba968c-296b-484e-9bd3-56d309150d18
md"where $v, w ∈ \mathbb{R}^D$, $b \in \mathbb{R}$, and $h$ is an activation function such as the hyperbolic tangent."

# ╔═╡ 6aa83819-e523-4566-a20c-b256e0cecf0f
md" This flow can be interpreted as expanding/contracting
the space in the direction perpendicular to the hyperplane $wᵀz + b = 0$."

# ╔═╡ fd6bf6d5-db03-4b5a-b7e8-fd8dc97579e1
md"The Jacobian is then given by:

$\frac{δf(z)}{δz} = I + v h'(w^{T} z + b)w^{T}$"

# ╔═╡ dc964e09-7c1f-46d0-bead-ee45f1a799cb
md"And,

$det(\frac{δf(z)}{δz}) = 1 + h'(w^{T} z + b)w^{T}v$"

# ╔═╡ b781a822-01bd-4e18-b524-24e69179abea
md"which is required to calculate the density of the distribution."

# ╔═╡ 9b551eac-84fd-4ce1-8b37-0221200b3602
md"###### Model:"

# ╔═╡ 8a2b8d59-f8d4-472e-807f-b7c279d16706
md"###### Dataset:"

# ╔═╡ 30b079a8-4bc7-4f6b-873c-3c6b2983c2b2
n = 10

# ╔═╡ dc3d6759-dacc-4d18-9b2c-31c48c70afe7
begin
	using Flux
	opt = Flux.Descent(0.01)
	model = NormalizingFlows.PlanarFlow(rng, Float64, n*n)
end

# ╔═╡ 373b09d6-2843-4aa9-8ac0-52126fb2c775
heatmap(image_sine(n))

# ╔═╡ ba435d82-0ee9-4c98-ab01-c083e54b5fed
begin
	data = reshape(image_sine(n), n*n, 1)
	for i in 1:500
		data = hcat(data, reshape(image_sine(n), n*n, 1))
	end
end

# ╔═╡ 0f904fa5-f47a-48bb-955b-a1694df38b28
md"The base distribution looks like this:"

# ╔═╡ fb577e73-e2f6-4991-8b00-651e2792fedf
md"###### Training:"

# ╔═╡ 811d30d6-9aae-4e72-950a-c188697454d7
begin
	p = Uniform(0, 1)
	heatmap(rand(p, 10, 10))
end

# ╔═╡ 86f30a55-f740-486e-897b-811d0e1bbd98
begin
	for i in 1:50
    	train!(data, loss_kl, p, opt, model)
	end
end

# ╔═╡ 6f8e995f-1f7d-43dd-aca0-6888204099e7
md"###### Sampling:"

# ╔═╡ d93f4477-d81f-4c12-b747-ee498dd89f05
md"`sample(pᵤ, P)` takes the base distribution and Planar flow model as inputs and outputs a sample of the density from which the input samples were generated."

# ╔═╡ 49aeaf2d-a9d4-48fc-96b4-e509e27d0001
begin
	s = NormalizingFlows.sample(rng, p, model)
	heatmap((reshape(abs.(s), n, n)))
end

# ╔═╡ Cell order:
# ╟─1e648e40-bd72-11eb-025e-2d0414ed36bd
# ╟─a7a00dcd-d1aa-4041-87e7-66673c295dfd
# ╟─92ba968c-296b-484e-9bd3-56d309150d18
# ╟─6aa83819-e523-4566-a20c-b256e0cecf0f
# ╟─fd6bf6d5-db03-4b5a-b7e8-fd8dc97579e1
# ╟─dc964e09-7c1f-46d0-bead-ee45f1a799cb
# ╟─b781a822-01bd-4e18-b524-24e69179abea
# ╟─9b551eac-84fd-4ce1-8b37-0221200b3602
# ╠═ecb2fab7-1fbb-4448-a57f-6683aaf3dbcc
# ╟─8a2b8d59-f8d4-472e-807f-b7c279d16706
# ╠═20eae2af-c95f-4d2b-81ab-5576ffd2d1ba
# ╠═30b079a8-4bc7-4f6b-873c-3c6b2983c2b2
# ╠═373b09d6-2843-4aa9-8ac0-52126fb2c775
# ╠═ba435d82-0ee9-4c98-ab01-c083e54b5fed
# ╟─0f904fa5-f47a-48bb-955b-a1694df38b28
# ╟─fb577e73-e2f6-4991-8b00-651e2792fedf
# ╠═811d30d6-9aae-4e72-950a-c188697454d7
# ╠═dc3d6759-dacc-4d18-9b2c-31c48c70afe7
# ╠═86f30a55-f740-486e-897b-811d0e1bbd98
# ╟─6f8e995f-1f7d-43dd-aca0-6888204099e7
# ╟─d93f4477-d81f-4c12-b747-ee498dd89f05
# ╠═49aeaf2d-a9d4-48fc-96b4-e509e27d0001
