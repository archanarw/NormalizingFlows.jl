### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ a80c1858-f83d-4b96-8c2a-8a8bb908c40c
begin
	using NormalizingFlows, Random, Flux
	rng = Random.seed!(123)
	L = GLOW(rng, Float64, 1, 6)
end

# ╔═╡ fad7133c-cfb7-4965-a0d2-93acf9bcb091
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

# ╔═╡ 77469ac0-9aeb-11eb-3992-4b8337e3e5ff
md"### Linear Flows"

# ╔═╡ 0c17d6b5-37b0-4343-8a3e-302c838c1ea3
md"Normalizing flows of the type:"

# ╔═╡ ff85f833-8993-4f8b-a62e-e3136c4d588b
md" 
$z' = Wz$ 

where $W $ is a $D×D$ invertible matrix that parameterizes the transformation."

# ╔═╡ 41ea3fc3-d4ca-44e1-8295-96eb2ca9bb23
md"GLOW is a type of Linear Flow which has 3 layers - Normalization (activation normalization), transformation by $W$ and autoregressive coupling layer."

# ╔═╡ 5d5fb897-bf5c-4bda-baa4-e8126bd1164c
md" ###### ActNorm"

# ╔═╡ 24253d32-95f7-495c-b7b0-ed0628beeddb
md" This layer is a form of data dependent initialization, it performs an affine normalization similar to batch normalization, whose bias and scale are trainable (after initialization)."

# ╔═╡ 8e8e23d3-ea5d-4a24-bc2f-e73d3e86b4c5
md"###### Permutation layer (W)"

# ╔═╡ cb3921f7-7a11-4945-90ab-364e4f77ea9c
md" $W$ is an invertible $1$x$1$ convolution layer here, which is similar to having a permutation layer."

# ╔═╡ 2e8e417a-073e-46ea-890e-6e1deb453a8b
md" ###### Coupling"

# ╔═╡ 1978987e-09f6-42cf-bcd3-588574d6738f
md"
- $z'_{≤d} = z_{≤d}$
- $(h_{d+1}, . . . , h_{D}) = F(z_{≤d})$
- $z'_{i} = τ(z_{i}; h_{i}) for i > d.$"

# ╔═╡ a3e7a0eb-9928-4960-93a1-85217771ac02
md" The Jacobian of the above transformation is $W$, making the Jacobian determinant equal to $det(W)$."

# ╔═╡ 3c719e90-a605-472d-81e9-5406dabdad41
md" It takes $O(n³)$ to find determinant of $W$, which can be reduced to $O(n)$ with
$W = PLU$ where $P$ is the permutation matrix (which is fixed), $L$ and $U$ is learnt.
"

# ╔═╡ 55f64b70-d9d4-4b07-8c09-63a16215e527
md"##### Model: "

# ╔═╡ a0aa6ded-817c-4966-9e2b-1350d88ba196
md"##### Building the dataset:"

# ╔═╡ 1eb5be81-53e4-4921-b90c-8eef4dd029e4
n = 10 #Number of rows/columns in the image

# ╔═╡ 14ab944e-1d1e-4941-bcd2-850908fec46e
heatmap(image_sine(n))

# ╔═╡ 88ef798a-2a59-4b25-81bc-8a03dcf9c8e2
begin
	data = image_sine(n) |> Flux.unsqueeze(3)
	for i in 1:500
		data = cat(data, image_sine(n), dims = 3)
	end
end

# ╔═╡ e25b0215-3a85-465c-aec7-4b51bd0b435b
size(data)

# ╔═╡ ae45acf2-b2e5-4562-8ed2-0b59cf126dd8
begin
	p = Uniform(0, 1)
	heatmap(rand(p, 10, 10))
end

# ╔═╡ 998ae05f-65bd-48b2-a825-a79a7b82c5e5
md"##### Training:"

# ╔═╡ e4cadfc7-1073-44e4-919f-28155821a946
begin
	opt = Flux.Descent(0.01)
	model = GLOW(rng, Float64, 1, n*n)
	d = data |> Flux.unsqueeze(3)
	for i in 1:10
    	train!(rng, d, loss_kl, p, opt, model)
	end
end

# ╔═╡ 9f9e5557-e86e-4d7b-b1ef-598fc0f698ba
md"##### Sampling:"

# ╔═╡ f28fa39c-66ca-432b-bb99-de3084006f85
md"`sample(pᵤ, L)` takes the base distribution and GLOW layer as inputs and outputs a sample of the density from which the input samples were generated."

# ╔═╡ df8ac971-6dc1-4ed9-a1a0-3b947cde372f
begin
	s = NormalizingFlows.sample(rng, p, 1, model)
	heatmap(reshape(abs.(s), n, n))
end

# ╔═╡ Cell order:
# ╟─77469ac0-9aeb-11eb-3992-4b8337e3e5ff
# ╟─0c17d6b5-37b0-4343-8a3e-302c838c1ea3
# ╟─ff85f833-8993-4f8b-a62e-e3136c4d588b
# ╟─41ea3fc3-d4ca-44e1-8295-96eb2ca9bb23
# ╟─5d5fb897-bf5c-4bda-baa4-e8126bd1164c
# ╟─24253d32-95f7-495c-b7b0-ed0628beeddb
# ╟─8e8e23d3-ea5d-4a24-bc2f-e73d3e86b4c5
# ╟─cb3921f7-7a11-4945-90ab-364e4f77ea9c
# ╟─2e8e417a-073e-46ea-890e-6e1deb453a8b
# ╟─1978987e-09f6-42cf-bcd3-588574d6738f
# ╟─a3e7a0eb-9928-4960-93a1-85217771ac02
# ╟─3c719e90-a605-472d-81e9-5406dabdad41
# ╟─55f64b70-d9d4-4b07-8c09-63a16215e527
# ╠═a80c1858-f83d-4b96-8c2a-8a8bb908c40c
# ╟─a0aa6ded-817c-4966-9e2b-1350d88ba196
# ╠═fad7133c-cfb7-4965-a0d2-93acf9bcb091
# ╠═1eb5be81-53e4-4921-b90c-8eef4dd029e4
# ╠═14ab944e-1d1e-4941-bcd2-850908fec46e
# ╠═88ef798a-2a59-4b25-81bc-8a03dcf9c8e2
# ╠═e25b0215-3a85-465c-aec7-4b51bd0b435b
# ╠═ae45acf2-b2e5-4562-8ed2-0b59cf126dd8
# ╟─998ae05f-65bd-48b2-a825-a79a7b82c5e5
# ╠═e4cadfc7-1073-44e4-919f-28155821a946
# ╟─9f9e5557-e86e-4d7b-b1ef-598fc0f698ba
# ╟─f28fa39c-66ca-432b-bb99-de3084006f85
# ╠═df8ac971-6dc1-4ed9-a1a0-3b947cde372f
