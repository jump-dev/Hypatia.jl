#=
JuMP helpers for constructing extended formulations for vector and matrix
spectral/eigenvalue cones
=#

abstract type SpectralExtender end

abstract type VecSpecExt <: SpectralExtender end

struct VecNegGeom <: VecSpecExt end
struct VecNegGeomEFExp <: VecSpecExt end
struct VecNegGeomEFPow <: VecSpecExt end

const VecNegGeomAll = Union{VecNegGeom, VecNegGeomEFExp, VecNegGeomEFPow}

get_val(x::Vector{<:Real}, ::VecNegGeomAll) = exp(sum(log, x) / length(x))

struct VecInv <: VecSpecExt end
struct VecInvEF <: VecSpecExt end
struct VecNegLog <: VecSpecExt end
struct VecNegLogEF <: VecSpecExt end
struct VecNegEntropy <: VecSpecExt end
struct VecNegEntropyEF <: VecSpecExt end
struct VecPower12 <: VecSpecExt p::Real end
struct VecPower12EF <: VecSpecExt p::Real end

const VecSepSpec = Union{VecInv, VecNegLog, VecNegEntropy, VecPower12}
const VecSepSpecEF = Union{VecInvEF, VecNegLogEF, VecNegEntropyEF, VecPower12EF}

get_val(x::Vector{<:Real}, ext::VecSpecExt) = Cones.h_val(x, get_ssf(ext))

abstract type MatSpecExt <: SpectralExtender end

struct MatNegGeom <: MatSpecExt end
struct MatNegGeomEFExp <: MatSpecExt end
struct MatNegGeomEFPow <: MatSpecExt end

const MatNegGeomEF = Union{MatNegGeomEFExp, MatNegGeomEFPow}

get_vec_ef(::MatNegGeomEFExp) = VecNegGeomEFExp()
get_vec_ef(::MatNegGeomEFPow) = VecNegGeomEFPow()

struct MatInv <: MatSpecExt end
struct MatInvEigOrd <: MatSpecExt end
struct MatInvDirect <: MatSpecExt end
struct MatNegLog <: MatSpecExt end
struct MatNegLogEigOrd <: MatSpecExt end
struct MatNegLogDirect <: MatSpecExt end
struct MatNegEntropy <: MatSpecExt end
struct MatNegEntropyEigOrd <: MatSpecExt end
struct MatPower12 <: MatSpecExt p::Real end
struct MatPower12EigOrd <: MatSpecExt p::Real end

const MatSepSpec = Union{MatInv, MatNegLog, MatNegEntropy, MatPower12}
const MatSepSpecEigOrd = Union{MatInvEigOrd, MatNegLogEigOrd,
    MatNegEntropyEigOrd, MatPower12EigOrd}

const SepSpecAll = Union{VecSepSpec, VecSepSpecEF, MatSepSpec, MatSepSpecEigOrd,
    MatInvDirect, MatNegLogDirect}

get_vec_ef(::MatInvEigOrd) = VecInvEF()
get_vec_ef(::MatNegLogEigOrd) = VecNegLogEF()
get_vec_ef(::MatNegEntropyEigOrd) = VecNegEntropyEF()
get_vec_ef(ext::MatPower12EigOrd) = VecPower12EF(ext.p)

get_ssf(::Union{VecInv, VecInvEF, MatInv}) = Cones.InvSSF()
get_ssf(::Union{VecNegLog, VecNegLogEF, MatNegLog}) = Cones.NegLogSSF()
get_ssf(::Union{VecNegEntropy, VecNegEntropyEF, MatNegEntropy}) =
    Cones.NegEntropySSF()
get_ssf(ext::Union{VecPower12, VecPower12EF, MatPower12}) =
    Cones.Power12SSF(ext.p)

#=
homogenizes separable spectral vector/matrix constraints
=#
function add_homog_spectral(
    ext::SepSpecAll,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    aff_new = vcat(aff[1], 1.0, aff[2:end])
    add_spectral(ext, d, aff_new, model)
    return
end

#=
VecNegGeom
(u, w > 0) : u > -(Πᵢ wᵢ)^(1/d)
=#
function add_homog_spectral(
    ::VecNegGeom,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert 1 + d == length(aff)
    aff_new = vcat(-aff[1], aff[2:end])
    JuMP.@constraint(model, aff_new in Hypatia.HypoGeoMeanCone{Float64}(1 + d))
    return
end

#=
VecNegGeomEFExp
negative geometric mean -> exponential cone EF
(u, w > 0) : u > -(Πᵢ wᵢ)^(1/d)
↔ ∃ y > 0, x ∈ ℝᵈ : e'x < 0, (-xᵢ, y - u, wᵢ) ∈ MOI.ExponentialCone, ∀i
=#
function add_homog_spectral(
    ::VecNegGeomEFExp,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert 1 + d == length(aff)

    y = JuMP.@variable(model)
    JuMP.@constraint(model, y >= 0)
    x = JuMP.@variable(model, [1:d])
    JuMP.@constraint(model, sum(x) <= 0)

    for i in 1:d
        aff_new = vcat(-x[i], y - aff[1], aff[1 + i])
        JuMP.@constraint(model, aff_new in MOI.ExponentialCone())
    end

    return
end

#=
VecNegGeomEFPow
negative geometric mean -> power cone EF
(u, w > 0) : u > -(Πᵢ wᵢ)^(1/d)
↔ ∃ y > 0, x ∈ ℝᵈ⁻² : (w₁, w₂, x₁) ∈ MOI.PowerCone(1/2),
(w[i], x[i-2], x[i-1]) ∈ MOI.PowerCone(1/i), ∀i = 3..d-1
(w[d], x[d-2], y - u) ∈ MOI.PowerCone(1/d)
=#
function add_homog_spectral(
    ::VecNegGeomEFPow,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert d > 2 # EF does not work for d = 2
    @assert 1 + d == length(aff)
    w = aff[2:end]

    y = JuMP.@variable(model)
    JuMP.@constraint(model, y >= 0)
    x = JuMP.@variable(model, [1:(d - 2)])

    JuMP.@constraint(model, vcat(w[1], w[2], x[1]) in MOI.PowerCone(1.0 / 2))
    for i in 3:(d - 1)
        aff_new_i = vcat(w[i], x[i - 2], x[i - 1])
        JuMP.@constraint(model, aff_new_i in MOI.PowerCone(1.0 / i))
    end
    aff_new = vcat(w[end], x[end], y - aff[1])
    JuMP.@constraint(model, aff_new in MOI.PowerCone(1.0 / d))

    return
end

#=
VecSepSpec
=#
function add_spectral(
    ext::VecSepSpec,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert 2 + d == length(aff)
    JuMP.@constraint(model, aff in Hypatia.EpiPerSepSpectralCone{Float64}(
        get_ssf(ext), Cones.VectorCSqr{Float64}, d))
    return
end

#=
VecSepSpecEF
=#
function add_spectral(
    ext::VecSepSpecEF,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert 2 + d == length(aff)

    # perspective nonnegative TODO could be redundant
    JuMP.@constraint(model, aff[2] >= 0)

    # linear constraint
    x = JuMP.@variable(model, [1:d])
    JuMP.@constraint(model, aff[1] >= sum(x))

    # 3-dim constraints
    h = get_ssf(ext)
    for i in 1:d
        extend_atom(h, vcat(x[i], aff[2], aff[2 + i]), model)
    end

    return
end

#=
MatNegGeom
=#
function add_homog_spectral(
    ::MatNegGeom,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert 1 + Cones.svec_length(d) == length(aff)
    aff_new = vcat(-aff[1], aff[2:end])
    JuMP.@constraint(model, aff_new in
        Hypatia.HypoRootdetTriCone{Float64, Float64}(length(aff)))
    return
end

#=
MatNegGeomEF
(u, W ≻ 0) : u > -rootdet(W)
↔ ∃ upper triangular U : [W U'; U Diag(δ)] ≻ 0, δ = diag(U),
(u, δ) ∈ NegGeomean
=#
function add_homog_spectral(
    ext::MatNegGeomEF,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    δ = extend_det(d, aff[2:end], model) # checks dimension
    add_homog_spectral(get_vec_ef(ext), d, vcat(aff[1], δ), model)
    return
end

#=
MatSepSpec
=#
function add_spectral(
    ext::MatSepSpec,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert 2 + Cones.svec_length(d) == length(aff)
    JuMP.@constraint(model, aff in Hypatia.EpiPerSepSpectralCone{Float64}(
        get_ssf(ext), Cones.MatrixCSqr{Float64, Float64}, d))
    return
end

#=
MatSepSpecEigOrd
uses eigenvalue ordering constraints from Ben-Tal & Nemirovski
=#
function add_spectral(
    ext::MatSepSpecEigOrd,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    W = get_smat_U(d, aff[3:end]) # checks dimension

    # eigenvalue ordering
    λ = JuMP.@variable(model, [1:d])
    JuMP.@constraint(model, [i in 1:(d - 1)], λ[i] >= λ[i + 1])
    JuMP.@constraint(model, tr(W) == sum(λ))

    for i in 1:(d - 1)
        Z_i = JuMP.@variable(model, [1:d, 1:d], PSD)
        s_i = JuMP.@variable(model)
        JuMP.@constraint(model, sum(λ[1:i]) - i * s_i - tr(Z_i) >= 0)
        mat_i = Symmetric(Z_i - W + s_i * Matrix(I, d, d), :U)
        JuMP.@SDconstraint(model, mat_i >= 0)
    end

    # vector separable spectral constraint
    vec_aff = vcat(aff[1], aff[2], λ)
    add_spectral(get_vec_ef(ext), d, vec_aff, model)

    return
end

#=
MatInvDirect
(u, v > 0, W ≻ 0) : u > v tr(inv(W / v))
↔ ∃ Z : [Z vI; vI W] ≻ 0, u > tr(Z), v > 0 (use Schur complement)
=#
function add_spectral(
    ::MatInvDirect,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    W = get_smat_U(d, aff[3:end]) # checks dimension

    Z = JuMP.@variable(model, [1:d, 1:d], Symmetric)
    JuMP.@constraint(model, aff[1] >= tr(Z))
    JuMP.@constraint(model, aff[2] >= 0)

    vI = aff[2] * Matrix(I, d, d)
    mat = Symmetric(hvcat((2, 2), Z, vI, vI, W), :U)
    JuMP.@SDconstraint(model, mat >= 0)

    return
end

#=
MatNegLogDirect
(u, v > 0, W ≻ 0) : u > -v logdet(W / v)
↔ ∃ upper triangular U : [W U'; U Diag(δ)] ≻ 0, δ = diag(U),
(u, v, δ) ∈ NegLogVector
=#
function add_spectral(
    ::MatNegLogDirect,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    δ = extend_det(d, aff[3:end], model) # checks dimension
    vec_aff = vcat(aff[1], aff[2], δ)
    add_spectral(VecNegLogEF(), d, vec_aff, model)
    return
end

#=
helpers
=#

# check dimension and get symmetric matrix W (upper triangle) of vectorized w
function get_smat_U(d::Int, w::Vector{JuMP.AffExpr})
    @assert Cones.svec_length(d) == length(w)
    W = zeros(JuMP.AffExpr, d, d)
    Cones.svec_to_smat!(W, w, sqrt(2))
    return W
end

# construct the matrix part of the det EF, return δ like eigenvalues
function extend_det(
    d::Int,
    w::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert Cones.svec_length(d) == length(w)

    u = JuMP.@variable(model, [1:length(w)])
    U = get_smat_U(d, 1.0 * u)
    @assert istriu(U)
    δ = diag(U)

    W = get_smat_U(d, w)
    mat = Symmetric(hvcat((2, 2), W, U', U, Diagonal(δ)), :U)
    JuMP.@SDconstraint(model, mat >= 0)

    return δ
end

#=
InvSSF:
(x > 0, y > 0, z > 0) : x > y * inv(z / y) = y^2 / z
↔ x * z > y^2
↔ (x, z, sqrt(2) * y) ∈ JuMP.RotatedSecondOrderCone
=#
function extend_atom(
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
function extend_atom(
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
function extend_atom(
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
function extend_atom(
    h::Cones.Power12SSF,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
    )
    @assert length(aff) == 3
    JuMP.@constraint(model, aff[3] >= 0)
    JuMP.@constraint(model, aff in MOI.PowerCone(inv(h.p)))
    return
end
