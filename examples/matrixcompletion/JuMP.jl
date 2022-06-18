#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
see description in native.jl
=#

using SparseArrays

struct MatrixCompletionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    nuclear_obj::Bool # use nuclear norm objective, else spectral norm
    symmetric::Bool # use symmetric matrix, else rectangular matrix
    use_EF::Bool # in symmetric case, construct EF for spectral/nuclear cone
    nrow::Int # number of rows in matrix
    ncol::Int # not used in symmetric case
    sparsity::Real # fraction of values that are known
end

function build(inst::MatrixCompletionJuMP{T}) where {T <: Float64}
    @assert 0 < inst.sparsity < 1
    nrow = inst.nrow
    @assert nrow >= 1
    symmetric = inst.symmetric
    nuclear_obj = inst.nuclear_obj

    if symmetric
        ncol = nrow
        len = Cones.svec_length(nrow)
        U = triu!(sprandn(T, nrow, nrow, inst.sparsity))
        (rows, cols, Avals) = findnz(U)
    else
        ncol = inst.ncol
        @assert ncol >= nrow
        len = nrow * ncol
        (rows, cols, Avals) = findnz(sprandn(T, nrow, ncol, inst.sparsity))
    end
    num_unknown = len - length(Avals)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)
    JuMP.@variable(model, x[1:num_unknown])
    JuMP.@constraint(model, vcat(1, x) in MOI.GeometricMeanCone(1 + length(x)))

    X = Matrix{JuMP.AffExpr}(undef, nrow, ncol)
    for (r, c, a) in zip(rows, cols, Avals)
        X[r, c] = a
    end
    k = 1
    for c in 1:ncol, r in 1:(symmetric ? c : nrow)
        if !isassigned(X, r, c)
            X[r, c] = x[k]
            k += 1
        end
    end
    @assert k == length(x) + 1

    if symmetric
        LinearAlgebra.copytri!(X, 'U')
        if inst.use_EF
            if nuclear_obj
                # EF for symmetric nuclear norm (L1 of eigvals)
                JuMP.@variable(model, X1[1:nrow, 1:nrow], PSD)
                JuMP.@variable(model, X2[1:nrow, 1:nrow], PSD)
                eq = [X[i, j] - X1[i, j] + X2[i, j] for j in 1:nrow for i in 1:j]
                JuMP.@constraint(model, eq .== 0)
                JuMP.@constraint(model, t >= tr(X1) + tr(X2))
            else
                # EF for symmetric spectral norm (Linf of eigvals)
                tI = t * Matrix(I, nrow, nrow)
                JuMP.@constraint(model, Symmetric(tI - X) in JuMP.PSDCone())
                JuMP.@constraint(model, Symmetric(tI + X) in JuMP.PSDCone())
            end
        else
            K = Hypatia.EpiNormSpectralTriCone{T, T}(1 + len, nuclear_obj)
            Xvec = Vector{JuMP.AffExpr}(undef, len)
            Cones.smat_to_svec!(Xvec, X, sqrt(T(2)))
            JuMP.@constraint(model, vcat(t, Xvec) in K)
        end
    else
        K = (nuclear_obj ? MOI.NormNuclearCone : MOI.NormSpectralCone)(nrow, ncol)
        JuMP.@constraint(model, vcat(t, vec(X)) in K)
    end

    # save for use in tests
    model.ext[:x_var] = x
    model.ext[:X_var] = X

    return model
end

function test_extra(inst::MatrixCompletionJuMP{T}, model::JuMP.Model) where {T}
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective and feasibility
    tol = eps(T)^0.2
    x_opt = JuMP.value.(model.ext[:x_var])
    x_geom = exp(sum(log, x_opt) / length(x_opt))
    @test x_geom <= 1 + tol
    X_opt = JuMP.value.(model.ext[:X_var])
    s = (inst.symmetric ? abs.(eigvals(Symmetric(X_opt, :U))) : svdvals(X_opt))
    snorm = (inst.nuclear_obj ? sum(s) : maximum(s))
    @test JuMP.objective_value(model) ≈ snorm atol = tol rtol = tol
    return
end
