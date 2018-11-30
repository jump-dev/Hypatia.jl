#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using JuMP, PolyJuMP, Hypatia, MultivariatePolynomials, DynamicPolynomials, Random
using MathOptInterface
MOI = MathOptInterface

function scaletest(X, y)
    r = 2
    (npoints, n) = size(X)

    @polyvar x[1:n]

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, p, PolyJuMP.Poly(monomials(x, 0:r)))

    @variable(model, z)
    @objective(model, Min, z / npoints)
    @constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))

    return (model, p)
end

for n in [10, 100, 1000, 10_000]
    X = rand(n, 2)
    y = sum(X, dims=2)
    (model, p) = scaletest(X, y)
    JuMP.optimize!(model)
end
