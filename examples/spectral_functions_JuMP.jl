#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
JuMP helpers for constructing extended formulations for vector and matrix
spectral/eigenvalue cones
=#

abstract type SpectralExtender end

# vector spectral formulations
abstract type VecSpecExt <: SpectralExtender end

# vector negative geomean formulations
struct VecNegRtdet <: VecSpecExt end
struct VecNegRtdetEFExp <: VecSpecExt end
struct VecNegRtdetEFPow <: VecSpecExt end

const VecNegRtdetAll = Union{VecNegRtdet, VecNegRtdetEFExp, VecNegRtdetEFPow}

# vector separable spectral primal formulations
struct VecLogCone <: VecSpecExt end
struct VecNegLog <: VecSpecExt end
struct VecNegLogEF <: VecSpecExt end
struct VecNegEntropy <: VecSpecExt end
struct VecNegEntropyEF <: VecSpecExt end
struct VecNegSqrt <: VecSpecExt end
struct VecNegSqrtEF <: VecSpecExt end
struct VecNegPower01 <: VecSpecExt
    p::Real
end
struct VecNegPower01EF <: VecSpecExt
    p::Real
end
struct VecPower12 <: VecSpecExt
    p::Real
end
struct VecPower12EF <: VecSpecExt
    p::Real
end

const VecSepSpecPrim =
    Union{VecLogCone, VecNegLog, VecNegEntropy, VecNegSqrt, VecNegPower01, VecPower12}
const VecSepSpecPrimEF =
    Union{VecNegLogEF, VecNegEntropyEF, VecNegSqrtEF, VecNegPower01EF, VecPower12EF}
const VecSepSpecPrimAll = Union{VecSepSpecPrim, VecSepSpecPrimEF}

# vector separable spectral dual formulations
struct VecNegEntropyConj <: VecSpecExt end
struct VecNegEntropyConjEF <: VecSpecExt end
struct VecNegSqrtConj <: VecSpecExt end
struct VecNegSqrtConjEF <: VecSpecExt end
struct VecNegPower01Conj <: VecSpecExt
    p::Real
end
struct VecNegPower01ConjEF <: VecSpecExt
    p::Real
end
struct VecPower12Conj <: VecSpecExt
    p::Real
end
struct VecPower12ConjEF <: VecSpecExt
    p::Real
end

const VecSepSpecDual =
    Union{VecNegEntropyConj, VecNegSqrtConj, VecNegPower01Conj, VecPower12Conj}
const VecSepSpecDualEF =
    Union{VecNegEntropyConjEF, VecNegSqrtConjEF, VecNegPower01ConjEF, VecPower12ConjEF}
const VecSepSpecDualAll = Union{VecSepSpecDual, VecSepSpecDualEF}

const VecSepSpec = Union{VecSepSpecPrim, VecSepSpecDual}
const VecSepSpecEF = Union{VecSepSpecPrimEF, VecSepSpecDualEF}

# matrix spectral formulations
abstract type MatSpecExt <: SpectralExtender end

# matrix negative geomean formulations
struct MatNegRtdet <: MatSpecExt end
struct MatNegRtdetEFExp <: MatSpecExt end
struct MatNegRtdetEFPow <: MatSpecExt end

const MatNegRtdetAll = Union{MatNegRtdet, MatNegRtdetEFExp, MatNegRtdetEFPow}

# matrix separable spectral primal formulations
struct MatLogdetCone <: MatSpecExt end
struct MatNegLog <: MatSpecExt end
struct MatNegLogEigOrd <: MatSpecExt end
struct MatNegLogDirect <: MatSpecExt end
struct MatNegEntropy <: MatSpecExt end
struct MatNegEntropyEigOrd <: MatSpecExt end
struct MatNegSqrt <: MatSpecExt end
struct MatNegSqrtEigOrd <: MatSpecExt end
struct MatNegPower01 <: MatSpecExt
    p::Real
end
struct MatNegPower01EigOrd <: MatSpecExt
    p::Real
end
struct MatPower12 <: MatSpecExt
    p::Real
end
struct MatPower12EigOrd <: MatSpecExt
    p::Real
end

const MatSepSpecPrim =
    Union{MatLogdetCone, MatNegLog, MatNegEntropy, MatNegSqrt, MatNegPower01, MatPower12}
const MatSepSpecPrimEF = Union{
    MatNegLogEigOrd,
    MatNegLogDirect,
    MatNegEntropyEigOrd,
    MatNegSqrtEigOrd,
    MatNegPower01EigOrd,
    MatPower12EigOrd,
}
const MatSepSpecPrimAll = Union{MatSepSpecPrim, MatSepSpecPrimEF}

# matrix separable spectral dual formulations
struct MatNegEntropyConj <: MatSpecExt end
struct MatNegEntropyConjEigOrd <: MatSpecExt end
struct MatNegSqrtConj <: MatSpecExt end
struct MatNegSqrtConjEigOrd <: MatSpecExt end
struct MatNegSqrtConjDirect <: MatSpecExt end
struct MatNegPower01Conj <: MatSpecExt
    p::Real
end
struct MatNegPower01ConjEigOrd <: MatSpecExt
    p::Real
end
struct MatPower12Conj <: MatSpecExt
    p::Real
end
struct MatPower12ConjEigOrd <: MatSpecExt
    p::Real
end

const MatSepSpecDual =
    Union{MatNegEntropyConj, MatNegSqrtConj, MatNegPower01Conj, MatPower12Conj}
const MatSepSpecDualEF = Union{
    MatNegEntropyConjEigOrd,
    MatNegSqrtConjEigOrd,
    MatNegSqrtConjDirect,
    MatNegPower01ConjEigOrd,
    MatPower12ConjEigOrd,
}
const MatSepSpecDualAll = Union{MatSepSpecDual, MatSepSpecDualEF}

const MatSepSpec = Union{MatSepSpecPrim, MatSepSpecDual}
const MatSepSpecEigOrd = Union{
    MatNegLogEigOrd,
    MatNegEntropyEigOrd,
    MatNegSqrtEigOrd,
    MatNegPower01EigOrd,
    MatPower12EigOrd,
    MatNegEntropyConjEigOrd,
    MatNegSqrtConjEigOrd,
    MatNegPower01ConjEigOrd,
    MatPower12ConjEigOrd,
}

const NegRtdetAll = Union{VecNegRtdetAll, MatNegRtdetAll}
const SepSpecPrimAll = Union{VecSepSpecPrimAll, MatSepSpecPrimAll}
const SepSpecDualAll = Union{VecSepSpecDualAll, MatSepSpecDualAll}
const SepSpecAll = Union{SepSpecPrimAll, SepSpecDualAll}

# helpers
get_vec_ef(::MatNegRtdetEFExp) = VecNegRtdetEFExp()
get_vec_ef(::MatNegRtdetEFPow) = VecNegRtdetEFPow()
get_vec_ef(::MatNegLogEigOrd) = VecNegLogEF()
get_vec_ef(::MatNegEntropyEigOrd) = VecNegEntropyEF()
get_vec_ef(::MatNegSqrtEigOrd) = VecNegSqrtEF()
get_vec_ef(ext::MatNegPower01EigOrd) = VecNegPower01EF(ext.p)
get_vec_ef(ext::MatPower12EigOrd) = VecPower12EF(ext.p)
get_vec_ef(::MatNegEntropyConjEigOrd) = VecNegEntropyConjEF()
get_vec_ef(::MatNegSqrtConjEigOrd) = VecNegSqrtConjEF()
get_vec_ef(ext::MatNegPower01ConjEigOrd) = VecNegPower01ConjEF(ext.p)
get_vec_ef(ext::MatPower12ConjEigOrd) = VecPower12ConjEF(ext.p)

function get_ssf(
    ::Union{
        VecLogCone,
        VecNegLog,
        VecNegLogEF,
        MatLogdetCone,
        MatNegLog,
        MatNegLogEigOrd,
        MatNegLogDirect,
    },
)
    return Cones.NegLogSSF()
end
function get_ssf(
    ::Union{
        VecNegEntropy,
        VecNegEntropyEF,
        VecNegEntropyConj,
        VecNegEntropyConjEF,
        MatNegEntropy,
        MatNegEntropyEigOrd,
        MatNegEntropyConj,
        MatNegEntropyConjEigOrd,
    },
)
    return Cones.NegEntropySSF()
end
function get_ssf(
    ::Union{
        VecNegSqrt,
        VecNegSqrtEF,
        VecNegSqrtConj,
        VecNegSqrtConjEF,
        MatNegSqrt,
        MatNegSqrtEigOrd,
        MatNegSqrtConj,
        MatNegSqrtConjEigOrd,
        MatNegSqrtConjDirect,
    },
)
    return Cones.NegSqrtSSF()
end
function get_ssf(
    ext::Union{
        VecNegPower01,
        VecNegPower01EF,
        VecNegPower01Conj,
        VecNegPower01ConjEF,
        MatNegPower01,
        MatNegPower01EigOrd,
        MatNegPower01Conj,
        MatNegPower01ConjEigOrd,
    },
)
    return Cones.NegPower01SSF(ext.p)
end
function get_ssf(
    ext::Union{
        VecPower12,
        VecPower12EF,
        VecPower12Conj,
        VecPower12ConjEF,
        MatPower12,
        MatPower12EigOrd,
        MatPower12Conj,
        MatPower12ConjEigOrd,
    },
)
    return Cones.Power12SSF(ext.p)
end

nat_name(::NegRtdetAll) = "NegRtdet"
get_name_ssf(ext::SepSpecAll) = string(nameof(typeof(get_ssf(ext))))
nat_name(ext::SepSpecPrimAll) = get_name_ssf(ext)
nat_name(ext::SepSpecDualAll) = get_name_ssf(ext) * "Conj"

is_domain_pos(::SpectralExtender) = true
is_domain_pos(ext::SepSpecDualAll) = Cones.h_conj_dom_pos(get_ssf(ext))

get_val(x::Vector, ::NegRtdetAll) = -exp(sum(log, x) / length(x))
get_val(x::Vector, ext::SepSpecPrimAll) = Cones.h_val(x, get_ssf(ext))
get_val(x::Vector, ext::SepSpecDualAll) = Cones.h_conj(x, get_ssf(ext))

function pos_only(x::Vector{T}, minval::T = eps(T)) where {T <: Real}
    return [(x_i < minval ? minval : x_i) for x_i in x]
end

#=
homogenizes separable spectral vector/matrix constraints
=#
function add_homog_spectral(
    ext::SepSpecAll,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
)
    # epigraph and perspective variables are swapped if dual cone is used
    aff_new = vcat(swap_if_dual(aff[1], 1.0, ext)..., aff[2:end])
    add_spectral(ext, d, aff_new, model)
    return
end

#=
VecNegRtdet
(u, w > 0) : u > -(Πᵢ wᵢ)^(1/d)
=#
function add_homog_spectral(
    ::VecNegRtdet,
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
VecNegRtdetEFExp
negative geometric mean -> exponential cone EF
(u, w > 0) : u > -(Πᵢ wᵢ)^(1/d)
↔ ∃ y > 0, x ∈ ℝᵈ : e'x < 0, (-xᵢ, y - u, wᵢ) ∈ MOI.ExponentialCone, ∀i
=#
function add_homog_spectral(
    ::VecNegRtdetEFExp,
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
VecNegRtdetEFPow
negative geometric mean -> power cone EF
(u, w > 0) : u > -(Πᵢ wᵢ)^(1/d)
↔ ∃ y > 0, x ∈ ℝᵈ⁻² : (w₁, w₂, x₁) ∈ MOI.PowerCone(1/2),
(w[i], x[i-2], x[i-1]) ∈ MOI.PowerCone(1/i), ∀i = 3..d-1
(w[d], x[d-2], y - u) ∈ MOI.PowerCone(1/d)
=#
function add_homog_spectral(
    ::VecNegRtdetEFPow,
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
VecLogCone
=#
function add_spectral(::VecLogCone, d::Int, aff::Vector{JuMP.AffExpr}, model::JuMP.Model)
    @assert 2 + d == length(aff)
    aff_new = vcat(-aff[1], aff[2], aff[3:end])
    JuMP.@constraint(model, aff_new in Hypatia.HypoPerLogCone{Float64}(length(aff)))
    return
end

#=
VecSepSpec
=#
function add_spectral(ext::VecSepSpec, d::Int, aff::Vector{JuMP.AffExpr}, model::JuMP.Model)
    @assert 2 + d == length(aff)
    is_dual = (ext isa VecSepSpecDualAll)
    cone = Hypatia.EpiPerSepSpectralCone{Float64}(
        get_ssf(ext),
        Cones.VectorCSqr{Float64},
        d,
        is_dual,
    )
    JuMP.@constraint(model, aff in cone)
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
    # epigraph and perspective variables are swapped if dual cone is used
    (epi, per) = swap_if_dual(aff[1], aff[2], ext)

    # linear constraint
    x = JuMP.@variable(model, [1:d])
    JuMP.@constraint(model, epi >= sum(x))

    # 3-dim constraints
    for i in 1:d
        extend_atom(ext, (1.0 * x[i], per, aff[2 + i]), model)
    end

    # perspective nonnegative NOTE could be redundant with atom constraints
    JuMP.@constraint(model, per >= 0)

    return
end

#=
MatNegRtdet
=#
function add_homog_spectral(
    ::MatNegRtdet,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
)
    aff_new = vcat(-aff[1], aff[2:end])
    R = get_R(d, length(aff) - 1)
    JuMP.@constraint(model, aff_new in Hypatia.HypoRootdetTriCone{Float64, R}(length(aff)))
    return
end

#=
MatNegRtdetEF
(u, W ≻ 0) : u > -rootdet(W)
↔ ∃ upper triangular U : [W U'; U Diag(δ)] ≻ 0, δ = diag(U),
(u, δ) ∈ NegRtdetean
=#
function add_homog_spectral(
    ext::Union{MatNegRtdetEFExp, MatNegRtdetEFPow},
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
)
    R = get_R(d, length(aff) - 1)
    δ = extend_det(R, d, aff[2:end], model)
    add_homog_spectral(get_vec_ef(ext), d, vcat(aff[1], δ), model)
    return
end

#=
MatLogdetCone
=#
function add_spectral(::MatLogdetCone, d::Int, aff::Vector{JuMP.AffExpr}, model::JuMP.Model)
    aff_new = vcat(-aff[1], aff[2], aff[3:end])
    R = get_R(d, length(aff) - 2)
    JuMP.@constraint(
        model,
        aff_new in Hypatia.HypoPerLogdetTriCone{Float64, R}(length(aff))
    )
    return
end

#=
MatSepSpec
=#
function add_spectral(ext::MatSepSpec, d::Int, aff::Vector{JuMP.AffExpr}, model::JuMP.Model)
    R = get_R(d, length(aff) - 2)
    is_dual = (ext isa MatSepSpecDualAll)
    cone = Hypatia.EpiPerSepSpectralCone{Float64}(
        get_ssf(ext),
        Cones.MatrixCSqr{Float64, R},
        d,
        is_dual,
    )
    JuMP.@constraint(model, aff in cone)
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
    λ = JuMP.@variable(model, [1:d])
    JuMP.@constraint(model, [i in 1:(d - 1)], λ[i] >= λ[i + 1])

    vec_aff = vcat(aff[1], aff[2], λ)
    add_spectral(get_vec_ef(ext), d, vec_aff, model)

    w = aff[3:end]
    R = get_R(d, length(w))

    if R <: Real
        W = get_smat(d, w)
        trW = tr(W)
        for i in 1:(d - 1)
            Z = JuMP.@variable(model, [1:d, 1:d], PSD)
            s = JuMP.@variable(model)
            JuMP.@constraint(model, sum(λ[1:i]) - i * s - tr(Z) >= 0)
            mat = Symmetric(Z - W + s * Matrix(I, d, d), :U)
            JuMP.@constraint(model, mat in JuMP.PSDCone())
        end
    else
        (Wr, Wi) = get_smat_complex(d, w)
        trW = tr(Wr)
        for i in 1:(d - 1)
            Zr = JuMP.@variable(model, [1:d, 1:d], Symmetric)
            Zia = JuMP.@variable(model, [1:(d - 1), 1:(d - 1)], Symmetric)
            Zi = make_skewsymm(d, 1.0 * Zia)
            Z2 = Symmetric(hvcat((2, 2), Zr, Zi', Zi, Zr), :U)
            JuMP.@constraint(model, Z2 in JuMP.PSDCone())

            s = JuMP.@variable(model)
            JuMP.@constraint(model, sum(λ[1:i]) - i * s - tr(Zr) >= 0)

            Mr = Symmetric(Zr - Wr + s * Matrix(I, d, d), :U)
            Mi = Zi - Wi
            mat = Symmetric(hvcat((2, 2), Mr, Mi', Mi, Mr), :U)
            JuMP.@constraint(model, mat in JuMP.PSDCone())
        end
    end
    JuMP.@constraint(model, trW == sum(λ))

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
    R = get_R(d, length(aff) - 2)
    δ = extend_det(R, d, aff[3:end], model)
    vec_aff = vcat(aff[1], aff[2], δ)
    add_spectral(VecNegLogEF(), d, vec_aff, model)
    return
end

#=
MatNegSqrtConjDirect
note this is a dual cone EF, so u and v are swapped
(u > 0, v > 0, W ≻ 0) : v > 1/4 * u tr(inv(W / u))
↔ ∃ Z : [Z uI; uI W] ≻ 0, 4 * v > tr(Z), u > 0 (use Schur complement)
=#
function add_spectral(
    ::MatNegSqrtConjDirect,
    d::Int,
    aff::Vector{JuMP.AffExpr},
    model::JuMP.Model,
)
    w = aff[3:end]
    R = get_R(d, length(w))

    JuMP.@constraint(model, aff[1] >= 0)
    Z = JuMP.@variable(model, [1:d, 1:d], Symmetric)
    JuMP.@constraint(model, 4 * aff[2] >= tr(Z))

    uI = aff[1] * Matrix(I, d, d)
    if R <: Real
        W = get_smat(d, w)
        mat = Symmetric(hvcat((2, 2), Z, uI, uI, W), :U)
    else
        (Wr, Wi) = get_smat_complex(d, w)
        Zia = JuMP.@variable(model, [1:(d - 1), 1:(d - 1)], Symmetric)
        Zi = make_skewsymm(d, 1.0 * Zia)

        Mr = Symmetric(hvcat((2, 2), Z, uI, uI, Wr), :U)
        Mi = hvcat((2, 2), Zi, zeros(d, d), zeros(d, d), Wi)
        mat = Symmetric(hvcat((2, 2), Mr, Mi', Mi, Mr), :U)
    end
    JuMP.@constraint(model, mat in JuMP.PSDCone())

    return
end

#=
helpers
=#

# get epigraph and perspective variables, swapping if using dual cone
swap_if_dual(x, y, ext::SepSpecDualAll) = (y, x)
swap_if_dual(x, y, ext::SepSpecAll) = (x, y)

# get real or complex type from length of affine expression
function get_R(d::Int, w_len::Int)
    if w_len == d^2
        return ComplexF64
    else
        @assert w_len == Cones.svec_length(d)
        return Float64
    end
end

# get upper triangle of symmetric matrix W from vectorized w
function get_smat(d::Int, w::Vector{JuMP.AffExpr})
    @assert Cones.svec_length(d) == length(w)
    W = zeros(JuMP.AffExpr, d, d)
    Cones.svec_to_smat!(W, w, sqrt(2))
    return W
end

# get real and imaginary matrix parts from vectorized w
function get_smat_complex(d::Int, w::Vector)
    @assert d^2 == length(w)
    rt2 = sqrt(2)
    Wr = zeros(eltype(w), d, d)
    Wi = zero(Wr)
    idx = 1
    for j in 1:d
        for i in 1:(j - 1)
            Wr[i, j] = Wr[j, i] = w[idx] / rt2
            Wi[i, j] = w[idx + 1] / rt2
            Wi[j, i] = -Wi[i, j]
            idx += 2
        end
        Wr[j, j] = w[idx]
        idx += 1
    end
    return (Wr, Wi)
end

# make a skew-symmetric matrix
function make_skewsymm(d::Int, U::Symmetric)
    Z = zeros(eltype(U), d, d)
    Z[1:(d - 1), 2:d] = UpperTriangular(U)
    @assert istriu(Z)
    Z .-= Z'
    @assert all(iszero, diag(Z))
    return Z
end

# construct the real matrix part of the det EF, return δ like eigenvalues
function extend_det(R::Type{Float64}, d::Int, w::Vector{JuMP.AffExpr}, model::JuMP.Model)
    u = JuMP.@variable(model, [1:length(w)])
    U = get_smat(d, 1.0 * u)
    δ = diag(U)

    W = get_smat(d, w)
    mat = Symmetric(hvcat((2, 2), W, U', U, Diagonal(δ)), :U)
    JuMP.@constraint(model, mat in JuMP.PSDCone())

    return δ
end

# construct the complex matrix part of the det EF, return δ like eigenvalues
function extend_det(R::Type{ComplexF64}, d::Int, w::Vector{JuMP.AffExpr}, model::JuMP.Model)
    (Wr, Wi) = get_smat_complex(d, w)

    Ur = UpperTriangular(JuMP.@variable(model, [1:d, 1:d], Symmetric))
    δ = diag(Ur)
    Uia = JuMP.@variable(model, [1:(d - 1), 1:(d - 1)], Symmetric)
    Ui = zero(Wi)
    Ui[1:(d - 1), 2:d] = UpperTriangular(Uia)

    Mr = Symmetric(hvcat((2, 2), Wr, Ur', Ur, Diagonal(δ)), :U)
    Mi = hvcat((2, 2), Wi, -Ui', Ui, zeros(d, d))
    mat = Symmetric(hvcat((2, 2), Mr, Mi', Mi, Mr), :U)
    JuMP.@constraint(model, mat in JuMP.PSDCone())

    return δ
end

#=
NegLogSSF:
(x, y > 0, z > 0) : x > y * -log(z / y)
↔ y * exp(-x / y) < z
↔ (-x, y, z) ∈ MOI.ExponentialCone
conjugate does not provide additional modeling power
=#
function extend_atom(::VecNegLogEF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    aff_new = vcat(-aff[1], aff[2], aff[3])
    JuMP.@constraint(model, aff_new in MOI.ExponentialCone())
    return
end

#=
NegEntropySSF:
(x, y > 0, z > 0) : x > z * log(z / y) = -z * log(y / z)
↔ z * exp(-x / z) < y
↔ (-x, z, y) ∈ MOI.ExponentialCone
or for the conjugate: (z can be negative)
(x > 0, y > 0, z) : x > y * exp(-z / y - 1) = y * exp((-z - y) / y)
↔ (-z - y, y, x) ∈ MOI.ExponentialCone
=#
function extend_atom(::VecNegEntropyEF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    aff_new = vcat(-aff[1], aff[3], aff[2])
    JuMP.@constraint(model, aff_new in MOI.ExponentialCone())
    return
end

function extend_atom(::VecNegEntropyConjEF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    aff_new = vcat(-aff[3] - aff[2], aff[2], aff[1])
    JuMP.@constraint(model, aff_new in MOI.ExponentialCone())
    return
end

#=
NegSqrtSSF:
(x, y > 0, z > 0) : x > -y * sqrt(z / y) = -sqrt(y * z)
↔ ∃ θ > 0 : (x - θ)^2 < y * z
↔ (y, z, sqrt(2) * (x - θ)) ∈ JuMP.RotatedSecondOrderCone, θ ∈ ℝ₊
or for the conjugate:
(x > 0, y > 0, z > 0) : x > 1/4 * y * inv(z / y) = 1/4 * y^2 / z
↔ 2 * x * z > y^2 / 2
↔ (x, z, y / sqrt(2)) ∈ JuMP.RotatedSecondOrderCone
=#
function extend_atom(::VecNegSqrtEF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    θ = JuMP.@variable(model)
    JuMP.@constraint(model, θ >= 0)
    aff_new = vcat(aff[2], aff[3], sqrt(2) * (aff[1] - θ))
    JuMP.@constraint(model, aff_new in JuMP.RotatedSecondOrderCone())
    return
end

function extend_atom(::VecNegSqrtConjEF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    aff_new = vcat(aff[1], aff[3], aff[2] / sqrt(2))
    JuMP.@constraint(model, aff_new in JuMP.RotatedSecondOrderCone())
    return
end

#=
NegPower01SSF(p) for p ∈ (0, 1)
(x, y > 0, z > 0) : x > -y * (z / y)^p = -y^(1-p) * z^p
↔ ∃ θ > 0 : z^p * y^(1-p) > |x - θ|
↔ (z, y, x - θ) ∈ MOI.PowerCone(p), θ ∈ ℝ₊
or for the conjugate:
let q = p / (p - 1) < 0, so 1 - p = 1 / (1 - q)
let c = (1 - p) * p^-q > 0
(x > 0, y > 0, z > 0) : x > c * y * (z / y)^q = c * z^q * y^(1-q)
↔ x^(1/(1-q)) > c^(1/(1-q)) * z^(q/(1-q)) * y
↔ z^(q/(q-1)) * x^(1/(1-q)) > c^(1/(1-q)) * y
↔ z^p * x^(1-p) > |b * y|, y > 0
where b = c^(1/(1-q)) = p^p * (1 - p)^(1-p) > 0
↔ (z, x, b * y) ∈ MOI.PowerCone(p), y ∈ ℝ₊
=#
function extend_atom(ext::VecNegPower01EF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    p = ext.p
    @assert 0 < p < 1
    θ = JuMP.@variable(model)
    JuMP.@constraint(model, θ >= 0)
    aff_new = vcat(aff[3], aff[2], aff[1] - θ)
    JuMP.@constraint(model, aff_new in MOI.PowerCone(p))
    return
end

function extend_atom(
    ext::VecNegPower01ConjEF,
    aff::NTuple{3, JuMP.AffExpr},
    model::JuMP.Model,
)
    p = ext.p
    @assert 0 < p < 1
    q = p / (p - 1)
    b = p^p * (1 - p)^(1 - p)
    aff_new = vcat(aff[3], aff[1], b * aff[2])
    JuMP.@constraint(model, aff_new in MOI.PowerCone(p))
    return
end

#=
Power12SSF(p) for p ∈ (1, 2]
(x > 0, y > 0, z > 0) : x > y * (z / y)^p = y^(1-p) * z^p
↔ x^(1/p) * y^(1-1/p) > |z|, z > 0
↔ (x, y, z) ∈ MOI.PowerCone(1/p), z ∈ ℝ₊
or for the conjugate: (z can be negative)
let z₋ = {0 if z ≥ 0, or -z if z < 0}, q = p / (p - 1) > 2
(x > 0, y > 0, z) : x > (p - 1) * y * (z₋ / y / p)^q
= (p - 1) / p^q * z₋^q * y^(1-q)
↔ x^(1/q) > c * z₋ * y^(1/q-1)
↔ x^(1/q) * y^(1-1/q) > c * z₋
↔ ∃ θ > 0 : x^(1/q) * y^(1-1/q) > |c * (z - θ)|
where c = (p - 1)^(1/q) / p > 0
↔ (x, y, c * (z - θ)) ∈ MOI.PowerCone(1/q), θ ∈ ℝ₊
=#
function extend_atom(ext::VecPower12EF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    p = ext.p
    @assert 1 < p <= 2
    JuMP.@constraint(model, aff[3] >= 0)
    JuMP.@constraint(model, collect(aff) in MOI.PowerCone(inv(p)))
    return
end

function extend_atom(ext::VecPower12ConjEF, aff::NTuple{3, JuMP.AffExpr}, model::JuMP.Model)
    p = ext.p
    @assert 1 < p <= 2
    q = p / (p - 1)
    c = (p - 1)^inv(q) / p
    θ = JuMP.@variable(model)
    JuMP.@constraint(model, θ >= 0)
    aff_new = vcat(aff[1], aff[2], c * (aff[3] - θ))
    JuMP.@constraint(model, aff_new in MOI.PowerCone(inv(q)))
    return
end
