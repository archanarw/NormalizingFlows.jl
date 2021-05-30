var documenterSearchIndex = {"docs":
[{"location":"#Documentation","page":"Documentation","title":"Documentation","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"","category":"page"},{"location":"#Functions","page":"Documentation","title":"Functions","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"Modules = [NormalizingFlows]","category":"page"},{"location":"#NormalizingFlows.AffineLayer","page":"Documentation","title":"NormalizingFlows.AffineLayer","text":"Affine Layer with parameters:\n\nW: Weight of the conditioner\nb: Bias of the conditioner\n\nAffineLayer implements the conditioner and the transformation to the input z\n\n\n\n\n\n","category":"type"},{"location":"#NormalizingFlows.GLOW","page":"Documentation","title":"NormalizingFlows.GLOW","text":"GLOW has 3 layers: First - Conv1x1 layer (for ease of finding inverse : W = PLU,                        where P is a fixed permutation layer, L and U are learnt) Second - Normalization Third - Affine Coupling layer\n\n\n\n\n\n","category":"type"},{"location":"#NormalizingFlows.PlanarFlow","page":"Documentation","title":"NormalizingFlows.PlanarFlow","text":"The parameters of Planar Flows are as follows:\n\nv ∈ ℝᴰ\nw ∈ ℝᴰ\nb ∈ ℝ\n\nThe transformation is - z′ = z + vσ(wᵀz + b) where σ is an activation function such as tanh,          z is the input to the flow and z′ is the output.\n\nThis flow can be interpreted as expanding/contracting the space in the direction perpendicular to the hyperplane wᵀz + b = 0\n\n\n\n\n\n","category":"type"},{"location":"#NormalizingFlows.RadialFlow","page":"Documentation","title":"NormalizingFlows.RadialFlow","text":"The parameters of radial flow are:\n\nα α ∈ (0, +∞)\nβ ∈ ℝ\nz₀ ∈ ℝᴰ\n\nIt takes the following form: z′ = z + β/(α + r(z))*(z - z₀) where r(z) = ||z-z₀||\n\nThe above transformation can be thought of as a  contraction/expansion radially with center z₀\n\n\n\n\n\n","category":"type"},{"location":"#NormalizingFlows.affinecouplinglayer-Tuple{Any,AffineLayer}","page":"Documentation","title":"NormalizingFlows.affinecouplinglayer","text":"Coupling layer: It transforms the input as:\n\nzᵢ′ = zᵢ for i < d\nzᵢ′ = A(zᵢ) for i >= d where d = D/2\n\nInputs -\n\nz : input from trainig data\nA : AffineLayer of the required size d, i.e., D/2 where   D is the size of input data\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.expected_pdf-Tuple{Any,Any,Any}","page":"Documentation","title":"NormalizingFlows.expected_pdf","text":"`expected_pdf(data, p_u, A)`\n\nInputs\n\nz : Value whose probability density is estimated\npᵤ : Base distribution which may be from the package Distributions\n\nor any distribution which can be sampled using rand\n\nA : Affine layer\n\nReturns the probability density of z wrt to the distribution given by A\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.loss_kl-Tuple{Any,Any,Any}","page":"Documentation","title":"NormalizingFlows.loss_kl","text":"KL Divergence\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.lower_ones-Tuple{Any,Integer}","page":"Documentation","title":"NormalizingFlows.lower_ones","text":"Returns lower triangular square matrix of size k whose values are 1\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.sample-Tuple{Random.AbstractRNG,Any,AffineLayer}","page":"Documentation","title":"NormalizingFlows.sample","text":"`sample(pᵤ, A)`\n\nInputs -\n\npᵤ : Base distribution which may be from the package Distributions\n\nor any distribution which can be sampled using rand\n\nA : Affine layer\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.sample-Tuple{Random.AbstractRNG,Any,Any,GLOW}","page":"Documentation","title":"NormalizingFlows.sample","text":"`sample(pᵤ, L)`\n\nInputs -\n\npᵤ : Base distribution which may be from the package Distributions\n\nor any distribution which can be sampled using rand\n\nL : Linear Flow\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.sample-Tuple{Random.AbstractRNG,Any,PlanarFlow}","page":"Documentation","title":"NormalizingFlows.sample","text":"`sample(pᵤ, P)`\n\nInputs -\n\npᵤ : Base distribution which may be from the package Distributions   or any distribution which can be sampled using rand\nP : Planar flow\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.sample-Tuple{Random.AbstractRNG,Any,RadialFlow}","page":"Documentation","title":"NormalizingFlows.sample","text":"`sample(pᵤ, R)`\n\nInputs -\n\npᵤ : Base distribution which may be from the package Distributions   or any distribution which can be sampled using rand\nR : Radial flow\n\n\n\n\n\n","category":"method"},{"location":"#NormalizingFlows.train!-Tuple{Random.AbstractRNG,Any,Any,Any,Any,Any}","page":"Documentation","title":"NormalizingFlows.train!","text":"train!(rng::AbstractRNG, data, loss, p_u, opt, model)\n\nThe function updates ps, the parameters of AffineLayer, so that pᵤ may be transformed  to the distribution of each of the labelled inputs.\n\nInputs -\n\nrng\ndata : The training data, sequence of samples of the target distribution\nloss : A function which takes the model, input, x and u, sample of base distribution       and returns the divergence between the model applied to x and base distribution, which must be        minimized.\npᵤ : Base distribution which may be from the package Distributions       or any distribution which can be sampled using rand\nopt : Flux Optimizer\nmodel : The neural network used, i.e., one among AffineLayer, GLOW and Planar\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"Modules = [NormalizingFlows]","category":"page"}]
}
