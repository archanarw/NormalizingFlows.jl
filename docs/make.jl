using Documenter, NormalizingFlows
makedocs(sitename = "NormalizingFlows Documentation", modules = [NormalizingFlows])

deploydocs(
    repo = "github.com/archanarw/NormalizingFlows.jl.git",
)