### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 0789eb59-99c4-4051-a327-57dd0a6bdb7a
begin
	using NormalizingFlows, Random
	rng = Random.seed!(123)
	R = RadialFlow(rng, Float64, 4)
end

# ╔═╡ 13ada439-52d5-47f7-bf06-3d78f8ae3558
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

# ╔═╡ 19b66242-c111-11eb-2a7f-c1f51fc654c0
md"## Radial Flows"

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

# ╔═╡ 96f08283-f755-4386-a600-4928602b8bf8
md"###### Dataset:"

# ╔═╡ 51bfe441-0cea-48f3-906a-2cd7d7fb9385
n = 10 #Number of rows/columns in the image

# ╔═╡ 64ea179b-ef26-48f5-9e40-36a49ed9f948
begin
	using Flux
	opt = Flux.Descent(0.01)
	model = NormalizingFlows.RadialFlow(rng, Float64, n*n)
end

# ╔═╡ 8a5824e3-4d6c-4745-84f5-560e0ff4c8ef
heatmap(image_sine(n))

# ╔═╡ 71fc4b3c-53e4-4b5a-ad6a-70aef01c1b63
begin
	data = reshape(image_sine(n), n*n, 1)
	for i in 1:2000
		data = hcat(data, reshape(image_sine(n), n*n, 1))
	end
end

# ╔═╡ ceac6461-8c93-4bcd-9641-17e02e532351
md"###### Training:"

# ╔═╡ 06dd3688-53e9-4630-98ab-637faac37a27
begin
	p = Uniform(0, 1)
	heatmap(rand(p, 10, 10))
end

# ╔═╡ 1f05b288-b663-47a2-b1e3-4b52b7a3cd95
begin
	for i in 1:50
    	train!(data, loss_kl, p, opt, model)
	end
end

# ╔═╡ b85b60f1-2acf-47aa-9d09-81ecd12c82e0
md"###### Sampling:"

# ╔═╡ c435355c-6a6c-4562-95ca-bc01ca9cc1fb
md"`sample(pᵤ, R)` takes the base distribution and radial flow model as inputs and outputs a sample of the density from which the input samples were generated."

# ╔═╡ 171f4c02-3f01-4222-bd86-9dac62cd5a2e
begin
	s = NormalizingFlows.sample(rng, p, model)
	heatmap((reshape(abs.(s), n, n)))
end

# ╔═╡ Cell order:
# ╟─19b66242-c111-11eb-2a7f-c1f51fc654c0
# ╟─43efe91a-2b82-4c94-be3f-3bd8d9b196cc
# ╟─e8f4cb10-677c-4083-a4d0-06b5fbacbc78
# ╟─a36d2864-e198-4154-a4a4-6807896ef8a3
# ╟─9dc75a44-7fd6-4281-bad6-4d271866a682
# ╟─58742bd0-eefe-421b-b3d1-4f48464071e5
# ╟─3e542f27-e74c-4e1f-9c19-643960465405
# ╠═0789eb59-99c4-4051-a327-57dd0a6bdb7a
# ╟─96f08283-f755-4386-a600-4928602b8bf8
# ╠═13ada439-52d5-47f7-bf06-3d78f8ae3558
# ╠═51bfe441-0cea-48f3-906a-2cd7d7fb9385
# ╠═8a5824e3-4d6c-4745-84f5-560e0ff4c8ef
# ╠═71fc4b3c-53e4-4b5a-ad6a-70aef01c1b63
# ╟─ceac6461-8c93-4bcd-9641-17e02e532351
# ╠═06dd3688-53e9-4630-98ab-637faac37a27
# ╠═64ea179b-ef26-48f5-9e40-36a49ed9f948
# ╠═1f05b288-b663-47a2-b1e3-4b52b7a3cd95
# ╟─b85b60f1-2acf-47aa-9d09-81ecd12c82e0
# ╟─c435355c-6a6c-4562-95ca-bc01ca9cc1fb
# ╠═171f4c02-3f01-4222-bd86-9dac62cd5a2e
