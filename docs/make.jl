using Documenter
using DocStringExtensions
using Hypatia

makedocs(
    sitename = "Hypatia",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Hypatia],
    pages = [
        "Home" => "index.md",
        "Modeling" => "modeling.md",
        "Solving" => "solving.md",
        "API" => "api.md",
        ],
    )

deploydocs(
    repo = "github.com/chriscoey/Hypatia.jl.git",
    target = "build",
    push_preview = true,
    )
