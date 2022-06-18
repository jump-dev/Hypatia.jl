<img src="https://github.com/chriscoey/Hypatia.jl/wiki/hypatia_logo.png" alt="Hypatia logo" width="358"/>

[![Build Status](https://github.com/chriscoey/Hypatia.jl/workflows/CI/badge.svg)](https://github.com/chriscoey/Hypatia.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/chriscoey/Hypatia.jl/branch/master/graph/badge.svg?token=x7G2wQeKJF)](https://codecov.io/gh/chriscoey/Hypatia.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriscoey.github.io/Hypatia.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriscoey.github.io/Hypatia.jl/dev)

Hypatia is a highly-customizable open source interior point solver for generic conic optimization problems, written in [Julia](https://julialang.org/).
It is licensed under the MIT License (see [LICENSE](https://github.com/chriscoey/Hypatia.jl/blob/master/LICENSE.md)).

For more information on Hypatia, please see:
  - [documentation](https://chriscoey.github.io/Hypatia.jl/dev) for Hypatia's conic form, predefined cones, and interfaces
  - [cones reference](https://github.com/chriscoey/Hypatia.jl/wiki/files/coneref.pdf) for cone definitions and oracles
  - [examples folder](https://github.com/chriscoey/Hypatia.jl/tree/master/examples) for applied examples and instances
  - [benchmarks folder](https://github.com/chriscoey/Hypatia.jl/tree/master/benchmarks) for scripts used to run and analyze various computational benchmarks

and preprints of our papers:
  - [Solving natural conic formulations with Hypatia.jl](https://arxiv.org/abs/2005.01136) for computational arguments for expanding the class of cones recognized by conic solvers
  - [Performance enhancements for a generic conic interior point algorithm](https://arxiv.org/abs/2107.04262) for a description of Hypatia's algorithm and our enhanced stepping procedures
  - [Sum of squares generalizations for conic sets](https://arxiv.org/abs/2103.11499) for barriers and computational techniques for our generalized polynomial sum of squares cones
  - [Conic optimization with spectral functions on Euclidean Jordan algebras](https://arxiv.org/abs/2103.04104) for barriers and computational techniques for many of our epigraph/hypograph cones

and corresponding [raw results CSV files](https://github.com/chriscoey/Hypatia.jl/wiki) generated by our run scripts in the benchmarks folder.

### Installation

To use Hypatia, install [Julia](https://julialang.org/downloads/), then at the Julia REPL, type:
```julia
using Hypatia
using Pkg;
Pkg.add("Hypatia");
```
Hypatia is an experimental solver and a work in progress, and may not run with older releases of Julia.
Default options/parameters are not well-tuned, so we encourage you to experiment with these.

### Usage

Hypatia can be accessed through a low-level native Julia interface or through open-source modeling tools such as [JuMP](https://github.com/jump-dev/JuMP.jl) and [Convex.jl](https://github.com/jump-dev/Convex.jl).
The native interface is more expressive, allowing Hypatia to solve conic models expressed with generic real floating point types and structured matrices or linear operators, for example.
However, it is typically sufficient and more convenient to use JuMP.

Using JuMP, we can model a simple D-optimal experiment design problem and call Hypatia:
```julia
using LinearAlgebra
using JuMP
using Hypatia

# setup JuMP model
opt = Hypatia.Optimizer(verbose = false)
model = Model(() -> opt)
@variable(model, x[1:3] >= 0)
@constraint(model, sum(x) == 5)
@variable(model, hypo)
@objective(model, Max, hypo)
V = rand(2, 3)
Q = V * diagm(x) * V'
aff = vcat(hypo, [Q[i, j] for i in 1:2 for j in 1:i]...)
@constraint(model, aff in MOI.RootDetConeTriangle(2))

# solve and query solution
optimize!(model)
termination_status(model)
objective_value(model)
value.(x)
```
See our [D-optimal design](https://github.com/chriscoey/Hypatia.jl/blob/master/examples/doptimaldesign/JuMP.jl) example for more information and references.

Many more examples using the native interface or JuMP can be found in the [examples folder](https://github.com/chriscoey/Hypatia.jl/tree/master/examples).

### Contributing

Comments, questions, suggestions, and improvements/extensions to the code or documentation are welcomed.
Please reach out on [Discourse](https://discourse.julialang.org/c/domain/opt), or submit an issue or contribute a PR on our [Github repo](https://github.com/chriscoey/Hypatia.jl).
If contributing code, try to maintain consistent style and add docstrings or comments for clarity.
New examples are welcomed and should be implemented similarly to the [existing examples](https://github.com/chriscoey/Hypatia.jl/tree/master/examples).

### Acknowledgements

This work has been partially funded by the National Science Foundation under grant OAC-1835443 and the Office of Naval Research under grant N00014-18-1-2079.

### Citing Hypatia

If you find Hypatia solver useful, please cite our [solver paper](https://arxiv.org/abs/2005.01136):
```bibtex
@misc{coey2021solving,
    title={Solving natural conic formulations with Hypatia.jl}, 
    author={Chris Coey and Lea Kapelevich and Juan Pablo Vielma},
    year={2021},
    eprint={2005.01136},
    archivePrefix={arXiv},
    primaryClass={math.OC}
}
```

If you find aspects of Hypatia's IPM implementation useful, please cite our [algorithm paper](https://arxiv.org/abs/2107.04262):
```bibtex
@misc{coey2021performance,
    title={Performance enhancements for a generic conic interior point algorithm}, 
    author={Chris Coey and Lea Kapelevich and Juan Pablo Vielma},
    year={2021},
    eprint={2107.04262},
    archivePrefix={arXiv},
    primaryClass={math.OC}
}
```
