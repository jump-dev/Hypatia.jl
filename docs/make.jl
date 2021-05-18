using Documenter
using DocStringExtensions
using Hypatia

makedocs(
    sitename = "Hypatia",
    format = Documenter.HTML(),
    modules = [Hypatia]
    )

deploydocs(
    repo = "github.com/chriscoey/Hypatia.jl.git",
    push_preview = true,
    )
