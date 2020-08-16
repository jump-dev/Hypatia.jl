#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing its log-determinant or root-determinant (equivalent optimal solutions with different optimal objective values)
=#

import DynamicPolynomials

struct CentralPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int # number of polyvars
    halfdeg::Int # half degree of random polynomials
    logdet_obj::Bool # use logdet, else rootdet
end

function build(inst::CentralPolyMatJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, halfdeg) = (inst.n, inst.halfdeg)

    DynamicPolynomials.@polyvar x[1:n]
    basis = DynamicPolynomials.monomials(x, 0:halfdeg)
    L = length(basis)
    poly_half = randn(L, L) / L * basis
    poly_rand = poly_half' * poly_half

    model = JuMP.Model()
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)

    # objective hypograph
    JuMP.@variable(model, Q[1:L, 1:L], Symmetric)
    v1 = [Q[i, j] for i in 1:L for j in 1:i] # vectorized Q
    if inst.logdet_obj
        JuMP.@constraint(model, vcat(hypo, 1, v1) in MOI.LogDetConeTriangle(L))
    else
        JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(L))
    end

    # coefficients equal
    poly_Q = basis' * Q * basis
    JuMP.@constraint(model, DynamicPolynomials.coefficients(poly_Q - poly_rand) .== 0)

    return model
end

insts[CentralPolyMatJuMP]["minimal"] = [
    ((1, 2, false),),
    ((1, 2, false), StandardConeOptimizer),
    ((2, 2, true),),
    ((2, 2, true), StandardConeOptimizer),
    ]
insts[CentralPolyMatJuMP]["fast"] = [
    ((1, 10, true),),
    ((1, 10, true), StandardConeOptimizer),
    ((1, 15, false),),
    ((1, 15, false), StandardConeOptimizer),
    ((2, 3, true),),
    ((2, 3, true), StandardConeOptimizer),
    ((2, 3, false),),
    ((2, 3, false), StandardConeOptimizer),
    ((2, 6, true),),
    ((2, 5, true), StandardConeOptimizer),
    ((2, 7, false),),
    ((2, 6, false), StandardConeOptimizer),
    ((3, 2, true),),
    ((3, 2, false),),
    ((3, 4, true),),
    ((3, 4, false),),
    ((7, 2, true),),
    ((7, 2, true), StandardConeOptimizer),
    ((7, 2, false),),
    ((7, 2, false), StandardConeOptimizer),
    ]
insts[CentralPolyMatJuMP]["slow"] = [
    ((1, 20, false),),
    ((2, 3, false),),
    ((2, 10, false),),
    ((2, 8, false), StandardConeOptimizer),
    ((3, 4, true), StandardConeOptimizer),
    ((3, 4, false), StandardConeOptimizer),
    ((3, 5, true),),
    ((3, 5, false),),
    ((6, 3, true),),
    ((6, 3, false),),
    ]
