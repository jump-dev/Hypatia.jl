using Documenter
using DocStringExtensions
using Hypatia

makedocs(
    sitename = "Hypatia",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Hypatia],
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
        ],
    )

deploydocs(
    repo = "github.com/chriscoey/Hypatia.jl.git",
    push_preview = true,
    )
