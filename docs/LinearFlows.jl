### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 77469ac0-9aeb-11eb-3992-4b8337e3e5ff
md"### Normalizing Flows: Linear Flows"

# ╔═╡ ff85f833-8993-4f8b-a62e-e3136c4d588b
md" z′ = Wz where W is a D×D invertible matrix that parameterizes the transformation"

# ╔═╡ a3e7a0eb-9928-4960-93a1-85217771ac02
md" The Jacobian of the above transformation is simply W, making the Jacobian determinant equal to det(W)"

# ╔═╡ 3c719e90-a605-472d-81e9-5406dabdad41
md" It takes O(n³) to find determinant of W, which can be reduced to O(n) with
W = PLU
where P is the permutation matrix (which is fixed), L and U is learnt.
"

# ╔═╡ 6627ee55-1a22-4039-8df4-3b00a90c8a4b
md"
There must be 3 layers - Normalation(activation normalization), W and autoregressive coupling"

# ╔═╡ 5d5fb897-bf5c-4bda-baa4-e8126bd1164c
md" ###### ActNorm"

# ╔═╡ 24253d32-95f7-495c-b7b0-ed0628beeddb
md" This layer is a form of data dependent initialization, it performs an affine normalization similar to batch normalization, whose bias and scale are trainable (after initialization)."

# ╔═╡ 8e8e23d3-ea5d-4a24-bc2f-e73d3e86b4c5
md"###### Permutation layer (W)"

# ╔═╡ cb3921f7-7a11-4945-90ab-364e4f77ea9c
md" W is an invertible 1x1 convolution"

# ╔═╡ 2e8e417a-073e-46ea-890e-6e1deb453a8b
md" ###### Coupling"

# ╔═╡ 1978987e-09f6-42cf-bcd3-588574d6738f
md"
- z′≤d = z≤d
- (h\_d+1, . . . , h\_D) = F(z≤d)
- z′ᵢ = τ(zᵢ; hᵢ) for i > d."

# ╔═╡ Cell order:
# ╟─77469ac0-9aeb-11eb-3992-4b8337e3e5ff
# ╟─ff85f833-8993-4f8b-a62e-e3136c4d588b
# ╟─a3e7a0eb-9928-4960-93a1-85217771ac02
# ╟─3c719e90-a605-472d-81e9-5406dabdad41
# ╟─6627ee55-1a22-4039-8df4-3b00a90c8a4b
# ╟─5d5fb897-bf5c-4bda-baa4-e8126bd1164c
# ╟─24253d32-95f7-495c-b7b0-ed0628beeddb
# ╟─8e8e23d3-ea5d-4a24-bc2f-e73d3e86b4c5
# ╟─cb3921f7-7a11-4945-90ab-364e4f77ea9c
# ╟─2e8e417a-073e-46ea-890e-6e1deb453a8b
# ╟─1978987e-09f6-42cf-bcd3-588574d6738f
