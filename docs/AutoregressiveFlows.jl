### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ ea9afae0-9954-11eb-155e-5f4c0043af56
md"## Normalizing Flows: Autoregressive Flows"

# ╔═╡ 856d139a-1fb7-432d-b10d-f331e9b9ec47
md" Notation:
- fᵩ is the model (for simplicity, instead of Tₖ or T⁻¹ₖ)
- z in the input to the tramsform, z′ is the output, irrespective of Tₖ or T⁻¹ₖ"

# ╔═╡ 3bd7d23f-959a-4187-9d2f-cd6b99a9e775
md"
z₀ = u and zₖ = x, the forward evaluation is:
zₖ = Tₖ(zₖ₋₁) for k = 1, . . . , K
(here u is sampled from base distribution and x is expected to be the distribution the input is a sample of)
"

# ╔═╡ 3eaae25e-f1b9-4c45-b64e-f561383bb8e9
md"
The main idea of flow-based modeling is to express x as a transformation T of a real vector u sampled from pᵤ(u):
x = T(u) where u ∼ pᵤ(u).
"

# ╔═╡ 459f27ea-4f31-4d2a-8334-955b9f4f2ae8
md"
And, pₓ(x) = pᵤ(T₋₁(x)) |det(JT₋₁(x))|
"

# ╔═╡ b7f36d06-c0d4-4f5a-891d-035634a2ea18
md" Autoregressive flows specifies fᵩ to have the following form:
z′ᵢ = τ (zᵢ; hᵢ) where hᵢ = cᵢ(z<ᵢ),
where τ is termed the transformer and cᵢ the i-th conditioner.
"

# ╔═╡ 7ec7cfb5-c4ae-4634-92dd-0f15a8b16841
md" The Jacobian of the transform is a lower-triangular matrix (because each zᵢ′ does not depend on zₖ′ where k≮i) whose diagonal elements are the derivatives of the
transformer for each of the D elements of z."

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

# ╔═╡ a0303dcf-a909-42e1-b931-c0408f527511
md" τ(zᵢ; hᵢ) = αᵢzᵢ + βᵢ where hᵢ = {αᵢ,βᵢ}"

# ╔═╡ 4bc92e70-02e1-4402-998f-e218db839438
md" And log|det Jfᵩ| = ∑|αᵢ|"

# ╔═╡ 5b637af1-a2a5-4852-905d-185be9afc125
md"###### Combination-based transformers"

# ╔═╡ 4aa89bef-a853-44fd-a32a-f6e9a45c30ba
md" τ(zᵢ; hᵢ) = wᵢ₀ + ∑ᴷₖ₌₁ wᵢₖσ(αᵢₖzᵢ + βᵢₖ) where hᵢ = {wᵢ₀, . . . , wᵢₖ, αᵢₖ, βᵢₖ}"

# ╔═╡ b36968e5-2ee1-4fe5-9986-a015ccaac8aa
md"#### Implementing the Conditioner"

# ╔═╡ 5a47c4f1-1320-4786-ba96-3e839ccb9e8c
md"###### Using RNN"

# ╔═╡ bcb0befe-7fca-4649-a11d-2e982ec1faf1
md" hᵢ = c(sᵢ) where s₁ = initial state and
sᵢ = RNN(zᵢ₋₁, sᵢ₋₁) for i > 1"

# ╔═╡ 3a29b119-4952-4c70-8555-b75ed744a4b9
md"###### Masked Conditioners"

# ╔═╡ e0d68d17-2e12-4393-9179-b432a3ff2a5d
md" This approach uses a single, typically feedforward neural network that takes in z and outputs the entire sequence (h₁, h₂. . .) in one pass."

# ╔═╡ 9946212b-0ca1-46f6-9eb9-b1d39f8fdadb
md"To construct such a network, one takes an arbitrary neural network and removes connections until there is no path from input zᵢ to outputs (h₁, . . . , hᵢ), i.e., remove connections is by multiplying each weight matrix elementwise with a binary matrix of the same size."

# ╔═╡ d7596004-a071-479d-ba08-7ffeb159eaf7
md" Sampling:
- u~p(u)
- xᵢ = τ(u, hᵢ)"

# ╔═╡ a8fdb6c5-744d-4850-802e-8ffcc8bdb6ec
md"#### Coupling Layers"

# ╔═╡ 6d0bd4e1-6ca3-43ae-bb95-7edd70996989
md"
The idea is to choose an index k (a common choice is D/2 rounded
to an integer) and design the conditioner such that:
• Parameters (h₁, . . . , hₖ) are constants, i.e. not a function of z.
• Parameters (hₖ\_+₁, . . . , hD) are functions of z≤k only, i.e. they don’t depend on z>k.
"

# ╔═╡ 19b44213-d9dd-4834-a9b5-6c3431528dad
md" zᵢ′ = zᵢ for i≤k and zᵢ′ = τ(zᵢ, hᵢ) for i>k"

# ╔═╡ Cell order:
# ╟─ea9afae0-9954-11eb-155e-5f4c0043af56
# ╟─856d139a-1fb7-432d-b10d-f331e9b9ec47
# ╟─3bd7d23f-959a-4187-9d2f-cd6b99a9e775
# ╟─3eaae25e-f1b9-4c45-b64e-f561383bb8e9
# ╟─459f27ea-4f31-4d2a-8334-955b9f4f2ae8
# ╟─b7f36d06-c0d4-4f5a-891d-035634a2ea18
# ╟─7ec7cfb5-c4ae-4634-92dd-0f15a8b16841
# ╟─ba986db8-3dc6-4eaa-a7bf-530cbcf8064e
# ╟─d1aa48df-34fa-4d78-a21e-64e7ab151ce1
# ╟─69fe62b8-d7a9-4855-954d-c65579596fef
# ╟─a0303dcf-a909-42e1-b931-c0408f527511
# ╟─4bc92e70-02e1-4402-998f-e218db839438
# ╟─5b637af1-a2a5-4852-905d-185be9afc125
# ╟─4aa89bef-a853-44fd-a32a-f6e9a45c30ba
# ╟─b36968e5-2ee1-4fe5-9986-a015ccaac8aa
# ╟─5a47c4f1-1320-4786-ba96-3e839ccb9e8c
# ╟─bcb0befe-7fca-4649-a11d-2e982ec1faf1
# ╟─3a29b119-4952-4c70-8555-b75ed744a4b9
# ╟─e0d68d17-2e12-4393-9179-b432a3ff2a5d
# ╟─9946212b-0ca1-46f6-9eb9-b1d39f8fdadb
# ╟─d7596004-a071-479d-ba08-7ffeb159eaf7
# ╟─a8fdb6c5-744d-4850-802e-8ffcc8bdb6ec
# ╟─6d0bd4e1-6ca3-43ae-bb95-7edd70996989
# ╟─19b44213-d9dd-4834-a9b5-6c3431528dad
