#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using JuMP, PolyJuMP, Hypatia, MultivariatePolynomials, DynamicPolynomials, Random
using MathOptInterface
MOI = MathOptInterface
using MathOptInterfaceMosek

function scaletest(X, y)
    r = 6
    (npoints, n) = size(X)

    @polyvar x[1:n]

    # model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    model = Model(with_optimizer(MosekOptimizer))
    @variable(model, p, PolyJuMP.Poly(monomials(x, 0:r)))

    @variable(model, z)
    @objective(model, Min, z / npoints)
    println("adding soc")
    @constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))

    return (model, p)
end

for n in [20_000]
    X = rand(n, 4)
    y = sum(X, dims=2)
    println("building")
    (model, p) = scaletest(X, y)
    println("solving")
    JuMP.optimize!(model)
end
