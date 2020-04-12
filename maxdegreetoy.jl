# maxdegree kwarg in matrix SOS doesn't agree with scalar SOS, or definition is unclear

using SumOfSquares
using DynamicPolynomials
using MosekTools

n = 1
@polyvar x[1:n]
dom = @set(x[1] >= 0)

# scalar case
model = SOSModel(Mosek.Optimizer)
@variable(model, p, Poly(monomials(x, 0:4)))
@constraint(model, p >= 0, domain = dom, maxdegree = 0)
@constraint(model, p == 6x[1])
@objective(model, Min, coefficients(p)[1])
optimize!(model)
@show termination_status(model) # optimal

#
model = SOSModel(Mosek.Optimizer)
@variable(model, p, Poly(monomials(x, 0:4)))
@constraint(model, p * ones(1, 1) in PSDCone(), domain = dom, maxdegree = 0)
@constraint(model, p == 6x[1])
@objective(model, Min, coefficients(p)[1])
optimize!(model)
@show termination_status(model) # infeasible

model = SOSModel(Mosek.Optimizer)
@variable(model, p, Poly(monomials(x, 0:4)))
@constraint(model, p * ones(1, 1) in PSDCone(), domain = dom, maxdegree = 1)
@constraint(model, p == 6x[1])
@objective(model, Min, coefficients(p)[1])
optimize!(model)
@show termination_status(model) # infeasible

model = SOSModel(Mosek.Optimizer)
@variable(model, p, Poly(monomials(x, 0:4)))
@constraint(model, p * ones(1, 1) in PSDCone(), domain = dom, maxdegree = 2)
@constraint(model, p == 6x[1])
@objective(model, Min, coefficients(p)[1])
optimize!(model)
@show termination_status(model) # optimal

using SumOfSquares
using DynamicPolynomials
@polyvar x[1:2]
model = SOSModel()
@variable(model, p, Poly(monomials(x, 0:1)))
S = [p p; p p]
@constraint(model, Symmetric(S) in PSDCone())
