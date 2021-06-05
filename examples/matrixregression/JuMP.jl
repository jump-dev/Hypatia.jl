#=
see description in native.jl

allows objective to minimize frobenius norm or nuclear norm of residual matrix,
plus regularization; similar to
https://arxiv.org/ftp/arxiv/papers/1405/1405.1207.pdf
=#

using SparseArrays

struct MatrixRegressionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    Y::Matrix{T}
    X::Matrix{T}
    nuc_obj::Bool # use nuclear norm loss, else squared loss
    lam_fro::Real # penalty on Frobenius norm
    lam_nuc::Real # penalty on nuclear norm
    lam_las::Real # penalty on l1 norm
    lam_glr::Real # penalty on penalty on row group l1 norm
    lam_glc::Real # penalty on penalty on column group l1 norm
end

function MatrixRegressionJuMP{Float64}(
    n::Int,
    m::Int,
    p::Int,
    args...;
    A_max_rank::Int = div(m, 2) + 1,
    A_sparsity::Real = max(0.2, inv(sqrt(m * p))),
    Y_noise::Real = 0.01,
    )
    @assert p >= m
    @assert 1 <= A_max_rank <= m
    @assert 0 < A_sparsity <= 1
    A_left = sprandn(p, A_max_rank, A_sparsity)
    A_right = sprandn(A_max_rank, m, A_sparsity)
    A = 10 * A_left * A_right
    X = randn(n, p)
    Y = X * A + Y_noise * randn(n, m)
    return MatrixRegressionJuMP{Float64}(Y, X, args...)
end

function MatrixRegressionJuMP{Float64}(n::Int, m::Int)
    @assert n >= m >= 1
    X = randn(n, m)
    Y = randn(n, m)
    return MatrixRegressionJuMP{Float64}(Y, X, true, 0.1, 0, 0, 0, 0)
end

function build(inst::MatrixRegressionJuMP{T}) where {T <: Float64}
    (Y, X) = (inst.Y, inst.X)
    @assert min(inst.lam_fro, inst.lam_nuc, inst.lam_las,
        inst.lam_glr, inst.lam_glc) >= 0
    (data_n, data_m) = size(Y)
    data_p = size(X, 2)
    @assert size(X, 1) == data_n
    @assert data_p >= data_m
    @assert !inst.nuc_obj || data_m <= data_n
    model = JuMP.Model()
    JuMP.@variable(model, A[1:data_p, 1:data_m])
    JuMP.@variable(model, loss)
    loss_mat = Y - X * A

    if inst.nuc_obj
        JuMP.@constraint(model, vcat(loss, vec(loss_mat')) in
            MOI.NormNuclearCone(data_m, data_n))
    else
        if data_n > data_p
            # dimension reduction via QR
            F = qr(X, ColumnNorm())
            loss_mat = (F.Q' * Y)[1:data_p, :] - F.R[1:data_p, 1:data_p] *
                F.P' * A
        end
        JuMP.@constraint(model, vcat(loss, 1, vec(loss_mat) / sqrt(data_n)) in
            JuMP.RotatedSecondOrderCone())
    end

    obj = one(T) * loss
    if !iszero(inst.lam_fro)
        JuMP.@variable(model, t_fro)
        JuMP.@constraint(model, vcat(t_fro, inst.lam_fro * vec(A)) in
            JuMP.SecondOrderCone())
        obj += t_fro
    end
    if !iszero(inst.lam_nuc)
        JuMP.@variable(model, t_nuc)
        JuMP.@constraint(model, vcat(t_nuc, inst.lam_nuc * vec(A)) in
            MOI.NormNuclearCone(data_p, data_m))
        obj += t_nuc
    end
    if !iszero(inst.lam_las)
        JuMP.@variable(model, t_las)
        JuMP.@constraint(model, vcat(t_las, inst.lam_las * vec(A)) in
            MOI.NormOneCone(data_p * data_m + 1))
        obj += t_las
    end
    if !iszero(inst.lam_glr)
        JuMP.@variable(model, t_glr[1:data_p])
        JuMP.@constraint(model, [i = 1:data_p], vcat(t_glr[i],
            inst.lam_glr * A[i, :]) in JuMP.SecondOrderCone())
        obj += sum(t_glr)
    end
    if !iszero(inst.lam_glc)
        JuMP.@variable(model, t_glc[1:data_m])
        JuMP.@constraint(model, [i = 1:data_m], vcat(t_glc[i],
            inst.lam_glc * A[:, i]) in JuMP.SecondOrderCone())
        obj += sum(t_glc)
    end
    JuMP.@objective(model, Min, obj)

    model.ext[:A_var] = A # save for use in tests

    return model
end

function test_extra(inst::MatrixRegressionJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective value is correct
    tol = eps(T)^0.2
    (Y, X) = (inst.Y, inst.X)
    A_opt = JuMP.value.(model.ext[:A_var])
    loss_mat = Y - X * A_opt
    if inst.nuc_obj
        loss = sum(svdvals(loss_mat))
    else
        loss = sum(abs2, loss_mat) / (2 * size(Y, 1))
    end
    obj_result = loss +
        inst.lam_fro * norm(vec(A_opt), 2) +
        inst.lam_nuc * sum(svdvals(A_opt)) +
        inst.lam_las * norm(vec(A_opt), 1) +
        inst.lam_glr * sum(norm, eachrow(A_opt)) +
        inst.lam_glc * sum(norm, eachcol(A_opt))
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol
    return
end
