<img src="https://github.com/chriscoey/Hypatia.jl/wiki/hypatia_logo.png" alt="Hypatia logo" width="360"/>

[![Build Status](https://github.com/chriscoey/Hypatia.jl/workflows/CI/badge.svg)](https://github.com/chriscoey/Hypatia.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/chriscoey/Hypatia.jl/branch/master/graph/badge.svg?token=x7G2wQeKJF)](https://codecov.io/gh/chriscoey/Hypatia.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriscoey.github.io/Hypatia.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chriscoey.github.io/Hypatia.jl/dev)

# Hypatia.jl

Hypatia is a highly-customizable interior point solver for generic conic optimization problems, written in Julia.
It is licensed under the MIT License; see [LICENSE](https://github.com/chriscoey/Hypatia.jl/blob/master/LICENSE.md).

To learn how to model and solve conic problems with Hypatia, see the many applied examples in the [examples folder](https://github.com/chriscoey/Hypatia.jl/tree/master/examples).
For more information about conic optimization, Hypatia's algorithms, and proper cones, please see our [working paper](https://arxiv.org/abs/2005.01136) and our [cones reference](https://github.com/chriscoey/Hypatia.jl/wiki/files/coneref.pdf).
Hypatia is an experimental solver and a work in progress, and may not run with older releases of Julia.
If you have trouble using Hypatia or wish to make improvements, please submit an issue or contribute a PR.
Default options/parameters are not well-tuned, so we encourage you to experiment with these.

Here is a simple example from D-optimal experiment design that sets up a JuMP model and calls Hypatia:
```julia
using LinearAlgebra
using JuMP
using Hypatia

# setup JuMP model
model = Model(Hypatia.Optimizer)
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

This work has been partially funded by the National Science Foundation under grant OAC-1835443 and the Office of Naval Research under grant N00014-18-1-2079.
