#=
helpers for constructing extended formulations in JuMP
=#

# add a separable spectral cone constraint, possibly extended
function add_sepspectral(
    h::Cones.SepSpectralFun,
    csqr::Type{<:Cones.ConeOfSquares{T}},
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    use_standard_cones::Bool,
    ) where {T <: Real}
    if use_standard_cones
        extend_sepspectral(h, csqr, d, aff, model)
    else
        JuMP.@constraint(model, aff in
            Hypatia.EpiPerSepSpectralCone{T}(h, csqr, d))
    end
    return
end

# construct a standard cone EF for a separable spectral vector cone constraint
function extend_sepspectral(
    h::Cones.SepSpectralFun,
    ::Type{Cones.VectorCSqr{T}},
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    ) where {T <: Real}
    @assert d >= 1
    @assert d == length(aff) - 2

    # perspective nonnegative NOTE could be redundant
    JuMP.@constraint(model, aff[2] >= 0)

    # linear constraint
    x = JuMP.@variable(model, [1:d])
    JuMP.@constraint(model, aff[1] >= sum(x))

    # 3-dim constraints
    for i in 1:d
        aff_i = vcat(x[i], aff[2], aff[2 + i])
        extend_atom_jump(h, aff_i, model)
    end

    return
end

# construct a standard cone EF for a separable spectral matrix cone constraint
# TODO support complex MatrixCSqr
function extend_sepspectral(
    h::Cones.SepSpectralFun,
    ::Type{Cones.MatrixCSqr{T, T}},
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    ) where {T <: Real}
    @assert d >= 1
    @assert Cones.svec_length(d) == length(aff) - 2

    # symmetric matrix (use upper triangle) of aff[2:end]
    W = zeros(JuMP.AffExpr, d, d)
    Cones.svec_to_smat!(W, aff[3:end], sqrt(2))

    # eigenvalue ordering
    λ = JuMP.@variable(model, [1:d])
    JuMP.@constraint(model, [i in 1:(d - 1)], λ[i] >= λ[i + 1])
    JuMP.@constraint(model, tr(W) == sum(λ))

    # PSD constraints
    for i in 1:d
        Z_i = JuMP.@variable(model, [1:d, 1:d], PSD)
        s_i = JuMP.@variable(model)
        JuMP.@constraint(model, sum(λ[1:i]) - i * s_i - tr(Z_i) >= 0)
        mat_i = Symmetric(Z_i - W + s_i * Matrix(I, d, d), :U)
        JuMP.@SDconstraint(model, mat_i >= 0)
    end

    # vector separable spectral constraint
    vec_aff = vcat(aff[1], aff[2], λ)
    extend_sepspectral(h, Cones.VectorCSqr{T}, d, vec_aff, model)

    return
end

#=
InvSSF:
(x > 0, y > 0, z > 0) : x > y * inv(z / y) = y^2 / z
↔ x * z > y^2
↔ (x, z, sqrt(2) * y) ∈ JuMP.RotatedSecondOrderCone
=#
function extend_atom_jump(
    ::Cones.InvSSF,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert length(aff) == 3
    aff_new = vcat(aff[1], aff[3], sqrt(2) * aff[2])
    JuMP.@constraint(model, aff_new in JuMP.RotatedSecondOrderCone())
    return
end

#=
NegLogSSF:
(x, y > 0, z > 0) : x > y * -log(z / y)
↔ y * exp(-x / y) < z
↔ (-x, y, z) ∈ MOI.ExponentialCone
=#
function extend_atom_jump(
    ::Cones.NegLogSSF,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert length(aff) == 3
    aff_new = vcat(-aff[1], aff[2], aff[3])
    JuMP.@constraint(model, aff_new in MOI.ExponentialCone())
    return
end

#=
NegEntropySSF:
(x, y > 0, z > 0) : x > z * log(z / y) = -z * log(y / z)
↔ z * exp(-x / z) < y
↔ (-x, z, y) ∈ MOI.ExponentialCone
=#
function extend_atom_jump(
    ::Cones.NegEntropySSF,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert length(aff) == 3
    aff_new = vcat(-aff[1], aff[3], aff[2])
    JuMP.@constraint(model, aff_new in MOI.ExponentialCone())
    return
end

#=
Power12SSF(p) for p ∈ (1, 2]
(x > 0, y > 0, z > 0) : x > y * (z / y)^p = y^(1-p) * z^p
↔ x^(1/p) * y^(1-1/p) > |z|, z > 0
↔ (x, y, z) ∈ MOI.PowerCone(1/p), z ∈ ℝ₊
=#
function extend_atom_jump(
    h::Cones.Power12SSF,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert length(aff) == 3
    JuMP.@constraint(model, aff[3] >= 0)
    JuMP.@constraint(model, aff in MOI.PowerCone(inv(h.p)))
    return
end
