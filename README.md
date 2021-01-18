# Hypatia.jl

[![Build Status](https://github.com/chriscoey/Hypatia.jl/workflows/CI/badge.svg)](https://github.com/chriscoey/Hypatia.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/chriscoey/Hypatia.jl/branch/master/graph/badge.svg?token=x7G2wQeKJF)](https://codecov.io/gh/chriscoey/Hypatia.jl)

Hypatia is a highly-customizable open source interior point solver for generic conic optimization problems, written in Julia.
For more information, please see our [working paper](https://arxiv.org/abs/2005.01136) and our [cones reference](https://github.com/chriscoey/Hypatia.jl/wiki/files/coneref.pdf).
We plan to set up proper documentation in this repo soon.

Hypatia is an experimental solver and a work in progress, and may not run with older releases of Julia.
If you have trouble using Hypatia or wish to make improvements, please submit an issue or contribute a PR.
Default options/parameters are not well-tuned, so we encourage you to experiment with these.

To learn how to model using exotic cones in Hypatia, look through the examples folder.
Our examples are set up using either [JuMP](https://github.com/jump-dev/JuMP.jl) or Hypatia's native interface.
Modeling with JuMP is generally more user-friendly, though it may make sense to try the more-expressive native interface for large dense or structured models.

Here is a simple example (from D-optimal experiment design) that sets up a JuMP model and calls Hypatia:
```julia
using LinearAlgebra
using JuMP
using Hypatia

# setup model
V = rand(2, 3)
model = Model(Hypatia.Optimizer)
@variable(model, x[1:3] >= 0)
@constraint(model, sum(x) == 5)
@variable(model, hypo)
@objective(model, Max, hypo)
Q = V * diagm(x) * V'
@constraint(model, vcat(hypo, [Q[i, j] for i in 1:2 for j in 1:i]...) in MOI.RootDetConeTriangle(2))

# solve
optimize!(model)
termination_status(model)
objective_value(model)
value.(x)
```

This work has been partially funded by the National Science Foundation under grant OAC-1835443 and the Office of Naval Research under grant N00014-18-1-2079.
