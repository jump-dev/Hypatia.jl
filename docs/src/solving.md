# Solving conic models

Hypatia can be accessed through a low-level native Julia interface or through open-source modeling tools such as [JuMP](https://github.com/jump-dev/JuMP.jl) and [Convex.jl](https://github.com/jump-dev/Convex.jl).
The native interface is more expressive, allowing Hypatia to solve conic models expressed with generic real floating point types and structured matrices or linear operators, for example.
However, it is typically sufficient and more convenient to use JuMP.

## Native interface

```@meta
CurrentModule = Hypatia.Solvers
```

Hypatia's [`Solvers`](@ref) module provides a [`Solver`](@ref) type with low-level functions for solving models and querying solve information and conic certificates; see [Solvers module](@ref).

Below is a simple example of a spectral norm optimization problem:

```julia
using LinearAlgebra
import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

T = BigFloat
(Xn, Xm) = (3, 4)
dim = Xn * Xm
c = vcat(one(T), zeros(T, dim))
A = hcat(zeros(T, dim, 1), Matrix{T}(I, dim, dim))
b = rand(T, dim)
G = -one(T) * I
h = vcat(zero(T), rand(T, dim))
cones = [Cones.EpiNormSpectral{T, T}(Xn, Xm)]
model = Hypatia.Models.Model{T}(c, A, b, G, h, cones);
```

Now we call optimize and query the solution:

```julia
julia> solver = Solvers.Solver{T}(verbose = true);


julia> Solvers.load(solver, model);


julia> Solvers.solve(solver);

 iter        p_obj        d_obj |  abs_gap    x_feas    z_feas |      tau       kap        mu |  dir_res  step     alpha
    0   2.0000e+00   0.0000e+00 | 4.00e+00  5.00e-01  6.14e-01 | 1.00e+00  1.00e+00  1.00e+00 |
    1   2.5147e+00   2.2824e+00 | 1.07e+00  2.24e-01  2.74e-01 | 6.71e-01  7.44e-01  3.14e-01 | 3.45e-77  co-a  7.00e-01
    2   3.0958e+00   3.0966e+00 | 3.39e-01  7.40e-02  9.08e-02 | 6.08e-01  2.70e-01  1.01e-01 | 1.73e-77  co-a  7.00e-01
...
   33   3.2962e+00   3.2962e+00 | 1.33e-30  1.86e-30  2.29e-30 | 2.21e-01  1.88e-30  3.50e-31 | 4.85e-50  co-a  5.00e-01
   34   3.2962e+00   3.2962e+00 | 2.29e-31  2.77e-31  3.40e-31 | 2.23e-01  1.56e-31  5.28e-32 | 2.70e-48  co-a  8.50e-01
   35   3.2962e+00   3.2962e+00 | 7.32e-32  1.21e-31  1.49e-31 | 2.04e-01  1.15e-31  1.93e-32 | 2.52e-49  co-a  6.00e-01
optimal solution found; terminating

status is Optimal after 35 iterations and 1.417 seconds

julia> Solvers.get_status(solver)
Optimal::Status = 3

julia> Solvers.get_primal_obj(solver)
3.296219213377718379486912616497183695150874915748434424139285907044225666610375

julia> Solvers.get_dual_obj(solver)
3.29621921337771837948691261649702242580041815859477664034990154727149325280482

julia> Solvers.get_x(solver)
13-element Vector{BigFloat}:
 3.296219213377718379486912616497183695150874915748434424139285907044225666610375
 0.2014951884389319019635492500560556305705409904059068283971229702024509946542355
 0.2974304558864173403751380894865665103907350340298129119333915978313596204101965
...
 0.5076818262444516526146124557277599761994226912376378984924714206880970861448005
 0.4719189060586783692091058031965586401783925037880741775651825386863356120491953
 0.6377366379371803537625028513405673487028387050401566067448589228065218202592247
```

## MathOptInterface and JuMP

```@meta
CurrentModule = Hypatia
```

[JuMP](https://github.com/jump-dev/JuMP.jl) is generally more user-friendly than Hypatia's native interface, though it may make sense to try the more-expressive native interface for large dense or structured models.
Hypatia exports MathOptInterface wrappers for Hypatia's solver (see [`Optimizer`](@ref)) and predefined cones (see [MathOptInterface cones](@ref)).

Below is a simple example from D-optimal experiment design:

```julia
using LinearAlgebra
using JuMP
using Hypatia

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
```

The model is now:

```julia
julia> model
A JuMP Model
Maximization problem with:
Variables: 4
Objective function type: VariableRef
`AffExpr`-in-`MathOptInterface.EqualTo{Float64}`: 1 constraint
`Vector{AffExpr}`-in-`MathOptInterface.RootDetConeTriangle`: 1 constraint
`VariableRef`-in-`MathOptInterface.GreaterThan{Float64}`: 3 constraints
Model mode: AUTOMATIC
CachingOptimizer state: EMPTY_OPTIMIZER
Solver name: Hypatia
Names registered in the model: hypo, x
```

Now we call optimize and query the solution:

```julia
julia> optimize!(model)


julia> termination_status(model)
OPTIMAL::TerminationStatusCode = 1

julia> objective_value(model)
1.4303650845824805

julia> value.(x)
3-element Vector{Float64}:
 2.499999987496876
 2.499999992300735
 2.0202389761081463e-8
```
