#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

using Documenter
using DocStringExtensions
using Hypatia

makedocs(
    sitename = "Hypatia",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Hypatia, Hypatia.Cones, Hypatia.Models, Hypatia.Solvers, Hypatia.PolyUtils],
    pages = [
        "Home" => "index.md",
        "Cones" => "cones.md",
        "Modeling" => "modeling.md",
        "Solving" => "solving.md",
        "Examples" => "examples.md",
        "Benchmarks" => "benchmarks.md",
        "API reference" => [
            "api/hypatia.md",
            "api/cones.md",
            "api/models.md",
            "api/solvers.md",
            "api/polyutils.md",
        ],
    ],
    # linkcheck = true, # test URLs
)

deploydocs(
    repo = "github.com/chriscoey/Hypatia.jl.git",
    target = "build",
    push_preview = true,
)
