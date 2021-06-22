### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ b9d9fc20-194d-4644-a5fb-508147d5a1d9
using NormalizingFlows

# ╔═╡ 6090dde1-7d21-4507-ad95-b3d1b2d0de6c
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

# ╔═╡ 7a63e583-e9cc-4077-b862-bd8d68eae27b
md"## Autoregressive Flows"

# ╔═╡ 856d139a-1fb7-432d-b10d-f331e9b9ec47
md" Notation:
- $ f_{ϕ}$ is the model (for simplicity, instead of $Tₖ$ or $T⁻¹ₖ$)
- $ z$ in the input to the transform, $z'$ is the output, irrespective of $Tₖ$ or $T⁻¹ₖ$"

# ╔═╡ b7f36d06-c0d4-4f5a-891d-035634a2ea18
md" Autoregressive flows specifies $f_{ϕ}$ to have the following form:

$z'_{i} = τ (z_{i}; h_{i})$ 
$h_{i} = c_{i}(z_{<i})$

where $τ$ is termed the transformer and $c_{i}$ is the $i$-th conditioner, where $i = 1, ..., D$ and $D$ is the size of $z$.
"

# ╔═╡ 6459d722-c203-4bfb-b1fb-fca1d5d7efd5
md"Here the conditioner determines the parameters of the transformer, and in turn, can modify the transformer’s behavior. The conditioner does not need to be a bijection."

# ╔═╡ 3bd7d23f-959a-4187-9d2f-cd6b99a9e775
md"The forward evaluation is:

$zₖ = Tₖ(zₖ₋₁) for  k = 1, . . . , K$
 where:

$z₀ = u$ 

where $u$ is sampled from base distribution

$z_{K} = x$ 

where $x$ is expected to be a sample of the same distribution as that of the input.
"

# ╔═╡ 3eaae25e-f1b9-4c45-b64e-f561383bb8e9
md"
The main idea of flow-based modeling is to express $x$ as a transformation $T$ of a real vector $ u$ sampled from $p_{u}(u)$:

$x = T(u), u ∼ p_{u}(u)$
"

# ╔═╡ 459f27ea-4f31-4d2a-8334-955b9f4f2ae8
md" The PDF of $x$ is

$p_{x}(x) = p_{u}(T_{-1}(x)) |det(J_{T}₋₁(x))|$
"

# ╔═╡ 7ec7cfb5-c4ae-4634-92dd-0f15a8b16841
md" The Jacobian of the transform is a lower-triangular matrix (because each $zᵢ'$ does not depend on $zₖ'$ where $k≮i$) whose diagonal elements are the derivatives of the transformer for each of the $ D$ elements of $z$."

# ╔═╡ ba986db8-3dc6-4eaa-a7bf-530cbcf8064e
md"#### Implementing the transformer"

# ╔═╡ d1aa48df-34fa-4d78-a21e-64e7ab151ce1
md" Types of transformers that may be implemented are: 
- Affine
- Combination-based
- Integration-based
- Spline-based"

# ╔═╡ 69fe62b8-d7a9-4855-954d-c65579596fef
md"###### Affine Transformer"

# ╔═╡ 80e41d0a-98fe-442b-8a1a-74330be9e8c5
md" 

$τ(z_{i}; h_{i}) = α_{i}z_{i} + β_{i}$ 

where $h_{i}$ = {$α_{i}$, $β_{i}$}"

# ╔═╡ 4bc92e70-02e1-4402-998f-e218db839438
md" and 

$log |det J_{fᵩ}| = ∑|αᵢ|$"

# ╔═╡ b36968e5-2ee1-4fe5-9986-a015ccaac8aa
md"#### Implementing the Conditioner"

# ╔═╡ 5a47c4f1-1320-4786-ba96-3e839ccb9e8c
md"###### Using RNN"

# ╔═╡ bcb0befe-7fca-4649-a11d-2e982ec1faf1
md" $hᵢ = c(s_{i})$ where $s_{1}$ = initial state and
$s_{i} = RNN(z_{i-1}, s_{i-1})$ for $i = 2, ..., D$"

# ╔═╡ 3a29b119-4952-4c70-8555-b75ed744a4b9
md"###### Masked Conditioners"

# ╔═╡ e0d68d17-2e12-4393-9179-b432a3ff2a5d
md" This approach uses a single, typically feedforward neural network that takes in $z$ and outputs the entire sequence $(h_{1}, h_{2}. . ., h_{D})$ in one pass."

# ╔═╡ 9946212b-0ca1-46f6-9eb9-b1d39f8fdadb
md"To construct such a network, one takes an arbitrary neural network and removes connections until there is no path from input $z_{i}$ to output $(h_{i}, . . . , h_{D})$, i.e., remove connections is by multiplying each weight matrix elementwise with a binary matrix of the same size."

# ╔═╡ c397b6fc-e91c-4f0a-8180-8b6bc03f79f8
md"###### Model"

# ╔═╡ 32c961c5-4d11-422f-a6cd-e87b90c3069f
md" Conditioner"

# ╔═╡ 9840535f-daba-491c-b035-c3d12a20d68e
md"Affine Layer"

# ╔═╡ 2df53653-8353-4928-9245-640c3cf51d30
md"It is the layer that applies the transformation to the input, the parameters of which come from the conditioner."

# ╔═╡ 4838b626-14ae-4b2d-b6b2-88a1daa62736
md"Transformer :- 

$τ(z_{i}, h_{i}) = α_{i}*z_{i} + β_{i}$ where $α_{i}$ must be non-zero, $h_{i}$ = {$α_{i}$, $β_{i}$}, $h_{i} = c(s_{i}), i = 1, ... ,D$
"

# ╔═╡ d06a4a43-6be9-48fb-bd77-b9d989faf2e7
A = NormalizingFlows.AffineLayer(5)

# ╔═╡ 7ae0927c-2338-4902-9de9-fd74caa75652
md"Values after applying tranformation to all elements -"

# ╔═╡ 1fa27f83-29e7-48b5-9461-1e19741bc824
begin
	z = rand(5)
	A(z)
end

# ╔═╡ ee32779d-425c-4ebc-93b3-ef6243a53bcf
md"The function `apply_transform` applies the transformer($T$) or inverse($T'$) to the values of $z$ -"

# ╔═╡ 3843a662-7123-4cc9-99ac-11936fca38f1
apply_transform(A, inverse_τ, z)

# ╔═╡ 71ade19b-b4f8-493d-97db-5160d7fdd741
md"Training"

# ╔═╡ 05e07cc5-4b34-47ec-b3be-2390554e88f4
md"`train!(rng, data, pᵤ, opt, model)` is used to train the parameters of the conditioner so that the density function may be learnt."

# ╔═╡ 512a4768-9481-422b-aae2-978f2c57bcdf
md"For each data point ($z$) in `data`, the divergence between induced distribution,  $p_{x}(x, θ)$ and target distribution, $p_{x}^{*}(x)$ is minimized. "

# ╔═╡ cc3166d9-427d-42d7-a094-ed245292433e
n = 10 #Number of rows/columns in the image

# ╔═╡ 2e73e793-bfe5-4de6-a9d8-59cc65758904
begin
	using Flux
	opt = Flux.Descent(0.01)
	model = NormalizingFlows.AffineLayer(n*n)
end

# ╔═╡ 6bc33096-a59d-42ed-aa3a-304f387f6526
heatmap(image_sine(n))

# ╔═╡ 4ec794f9-93d3-4ecc-a2eb-3fdd1af70d5d
begin
	data = reshape(image_sine(n), n*n, 1)
	for i in 1:1000
		data = hcat(data, reshape(image_sine(n), n*n, 1))
	end
end

# ╔═╡ 1bfd644e-2350-47f0-a719-e75a296e97fc
md"The base distribution is:"

# ╔═╡ aba2407e-2c12-420c-8779-10839a6bbc5b
begin
	p = Uniform(0, 1)
	heatmap(rand(p, 10, 10))
end

# ╔═╡ 04ecc79b-7a19-483a-a698-66432b612cf8
begin
	for i in 1:50
    	train!(data, loss_kl, p, opt, model)
	end
end

# ╔═╡ e47cf87b-537a-4082-b8c2-223511a48cc6
md"Sampling"

# ╔═╡ 5a0153e4-f1e5-426b-843a-5b21ee7a92d4
md"`sample(pᵤ, A)` takes the base distribution and affine layer as inputs and outputs a sample of the density from which the input samples were generated."

# ╔═╡ f9e00b08-aab4-4dba-bf6f-196e95d2bbf2
begin
	s = NormalizingFlows.sample(p, model)
	heatmap((reshape(abs.(s), n, n)))
end

# ╔═╡ 539512c8-1a64-4899-98f8-e6655f478ee9
md"Esimating the density"

# ╔═╡ d400578f-f326-4d09-87f9-05ef698cd6ae
md"`pdf(z, pᵤ, A)` takes the base distribution, affine layer and the value whose density is to be esitmated and outputs the probability density of `z` according to the density function generated."

# ╔═╡ Cell order:
# ╟─7a63e583-e9cc-4077-b862-bd8d68eae27b
# ╟─856d139a-1fb7-432d-b10d-f331e9b9ec47
# ╟─b7f36d06-c0d4-4f5a-891d-035634a2ea18
# ╟─6459d722-c203-4bfb-b1fb-fca1d5d7efd5
# ╟─3bd7d23f-959a-4187-9d2f-cd6b99a9e775
# ╟─3eaae25e-f1b9-4c45-b64e-f561383bb8e9
# ╟─459f27ea-4f31-4d2a-8334-955b9f4f2ae8
# ╟─7ec7cfb5-c4ae-4634-92dd-0f15a8b16841
# ╟─ba986db8-3dc6-4eaa-a7bf-530cbcf8064e
# ╟─d1aa48df-34fa-4d78-a21e-64e7ab151ce1
# ╟─69fe62b8-d7a9-4855-954d-c65579596fef
# ╟─80e41d0a-98fe-442b-8a1a-74330be9e8c5
# ╟─4bc92e70-02e1-4402-998f-e218db839438
# ╟─b36968e5-2ee1-4fe5-9986-a015ccaac8aa
# ╟─5a47c4f1-1320-4786-ba96-3e839ccb9e8c
# ╟─bcb0befe-7fca-4649-a11d-2e982ec1faf1
# ╟─3a29b119-4952-4c70-8555-b75ed744a4b9
# ╟─e0d68d17-2e12-4393-9179-b432a3ff2a5d
# ╟─9946212b-0ca1-46f6-9eb9-b1d39f8fdadb
# ╟─c397b6fc-e91c-4f0a-8180-8b6bc03f79f8
# ╟─32c961c5-4d11-422f-a6cd-e87b90c3069f
# ╠═b9d9fc20-194d-4644-a5fb-508147d5a1d9
# ╟─9840535f-daba-491c-b035-c3d12a20d68e
# ╟─2df53653-8353-4928-9245-640c3cf51d30
# ╟─4838b626-14ae-4b2d-b6b2-88a1daa62736
# ╠═d06a4a43-6be9-48fb-bd77-b9d989faf2e7
# ╟─7ae0927c-2338-4902-9de9-fd74caa75652
# ╠═1fa27f83-29e7-48b5-9461-1e19741bc824
# ╟─ee32779d-425c-4ebc-93b3-ef6243a53bcf
# ╠═3843a662-7123-4cc9-99ac-11936fca38f1
# ╟─71ade19b-b4f8-493d-97db-5160d7fdd741
# ╟─05e07cc5-4b34-47ec-b3be-2390554e88f4
# ╟─512a4768-9481-422b-aae2-978f2c57bcdf
# ╠═6090dde1-7d21-4507-ad95-b3d1b2d0de6c
# ╠═cc3166d9-427d-42d7-a094-ed245292433e
# ╠═6bc33096-a59d-42ed-aa3a-304f387f6526
# ╠═4ec794f9-93d3-4ecc-a2eb-3fdd1af70d5d
# ╟─1bfd644e-2350-47f0-a719-e75a296e97fc
# ╠═aba2407e-2c12-420c-8779-10839a6bbc5b
# ╠═2e73e793-bfe5-4de6-a9d8-59cc65758904
# ╠═04ecc79b-7a19-483a-a698-66432b612cf8
# ╟─e47cf87b-537a-4082-b8c2-223511a48cc6
# ╟─5a0153e4-f1e5-426b-843a-5b21ee7a92d4
# ╠═f9e00b08-aab4-4dba-bf6f-196e95d2bbf2
# ╟─539512c8-1a64-4899-98f8-e6655f478ee9
# ╟─d400578f-f326-4d09-87f9-05ef698cd6ae
