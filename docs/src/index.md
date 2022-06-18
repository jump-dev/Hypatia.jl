# Hypatia

Hypatia is a highly-customizable interior point solver for generic conic optimization problems, written in [Julia](https://julialang.org/).
It is licensed under the MIT License; see [LICENSE](https://github.com/chriscoey/Hypatia.jl/blob/master/LICENSE.md).

See the [README](https://github.com/chriscoey/Hypatia.jl/blob/master/README.md) for a quick-start introduction.
For more information about Hypatia's algorithms and cones, please see our [working paper](https://arxiv.org/abs/2005.01136) and [cones reference](https://github.com/chriscoey/Hypatia.jl/wiki/files/coneref.pdf).

## Installation

To use Hypatia, install [Julia](https://julialang.org/downloads/), then at the Julia REPL, type:

```julia
using Pkg
Pkg.add("Hypatia")
using Hypatia
```

Hypatia is an experimental solver and a work in progress, and may not run with older releases of Julia.
Default options/parameters are not well-tuned, so we encourage you to experiment with these.

## Usage

Hypatia can be accessed through a low-level native Julia interface or through open-source modeling tools such as [JuMP](https://github.com/jump-dev/JuMP.jl) and [Convex.jl](https://github.com/jump-dev/Convex.jl).
The native interface is more expressive, allowing Hypatia to solve conic models expressed with generic real floating point types and structured matrices or linear operators, for example.
However, it is typically sufficient and more convenient to use JuMP.
