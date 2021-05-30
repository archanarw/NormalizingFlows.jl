using Documenter, NormalizingFlows
makedocs(
    modules = [NormalizingFlows],
    sitename = "NormalizingFlows.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingstarted.md"
        "Functions" => "functions.md"
    ]
)
deploydocs(
    repo = "github.com/archanarw/NormalizingFlows.jl.git",
)