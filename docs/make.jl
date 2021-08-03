using Documenter, NormalizingFlows
makedocs(
    modules = [NormalizingFlows],
    sitename = "NormalizingFlows.jl",
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gettingstarted.md",
        "Functions" => "functions.md"
    ]
)
deploydocs(
    repo = "github.com/archanarw/NormalizingFlows.jl.git",
)
