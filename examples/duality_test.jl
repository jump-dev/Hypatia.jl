import Dualization

function run_duality_test(conetype::Type{<:Hypatia.HypatiaCones{T}}) where {T}
    model = JuMP.GenericModel{T}(Hypatia.Optimizer{T})
    JuMP.@variable(model, t)
    conepars, x = initialize_cone(conetype)
    JuMP.@constraint(model, generate_constraint(conetype, t, x) in conetype(conepars...))
    JuMP.@objective(model, Min, t)
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    primal_objective = JuMP.objective_value(model)
    JuMP.set_optimizer(
        model,
        Dualization.dual_optimizer(Hypatia.Optimizer{T}; coefficient_type = T),
    )
    JuMP.optimize!(model)
    dual_objective = JuMP.objective_value(model)
    @test primal_objective ≈ dual_objective rtol = cbrt(eps(T))
    return nothing
end

initialize_cone(::Type{Hypatia.NonnegativeCone{T}}) where {T} = (2, randn(T, 2))
generate_constraint(::Type{Hypatia.NonnegativeCone{T}}, t, x) where {T} = t .- x

function initialize_cone(::Type{Hypatia.PosSemidefTriCone{T, R}}) where {T, R}
    dim = Cones.svec_length(R, 2)
    x = randn(T, dim)
    return dim, x
end
function generate_constraint(::Type{Hypatia.PosSemidefTriCone{T, R}}, t, x) where {T, R}
    return t * id_svec(R, length(x)) - x
end

function initialize_cone(::Type{Hypatia.DoublyNonnegativeTriCone{T}}) where {T}
    dim = Cones.svec_length(T, 2)
    x = rand(T, dim)
    return dim, x
end
function generate_constraint(::Type{Hypatia.DoublyNonnegativeTriCone{T}}, t, x) where {T}
    return t * id_svec(T, length(x)) + x
end

function initialize_cone(::Type{Hypatia.PosSemidefTriSparseCone{I, T, R}}) where {I, T, R}
    side = 3
    row_idxs = [1, 3, 2, 3]
    col_idxs = [1, 1, 2, 3]
    dim = R <: Real ? 4 : 5
    x = randn(T, dim)
    return (side, row_idxs, col_idxs), x
end
function generate_constraint(
    ::Type{Hypatia.PosSemidefTriSparseCone{I, T, R}},
    t,
    x,
) where {I, T, R}
    id = R <: Real ? [1, 0, 1, 1] : [1, 0, 0, 1, 1]
    return t * id - x
end

function initialize_cone(::Type{Hypatia.LinMatrixIneqCone{T}}) where {T}
    As = Symmetric.([Matrix(T(1) * I(2)), randn(T, 2, 2), randn(T, 2, 2)])
    x = randn(T, 3)
    return (As,), x
end
function generate_constraint(::Type{Hypatia.LinMatrixIneqCone{T}}, t, x) where {T}
    return t * [1, 0, 0] - x
end

const EpiCones{T} = Union{
    Hypatia.EpiNormInfCone{T, T},
    Hypatia.EpiNormInfCone{T, Complex{T}},
    Hypatia.EpiNormEuclCone{T},
    Hypatia.EpiRelEntropyCone{T},
}
function initialize_cone(::Type{<:EpiCones{T}}) where {T}
    dim = 5
    x = rand(T, dim)
    return dim, x
end
generate_constraint(::Type{<:EpiCones{T}}, t, x) where {T} = t * e0(T, length(x)) + x

function initialize_cone(::Type{Hypatia.EpiPerSquareCone{T}}) where {T}
    dim = 4
    x = randn(T, dim)
    x[2] = -1
    return dim, x
end
function generate_constraint(::Type{<:Hypatia.EpiPerSquareCone{T}}, t, x) where {T}
    return t * e0(T, length(x)) - x
end

function initialize_cone(::Type{Hypatia.EpiNormSpectralTriCone{T, R}}) where {T, R}
    dim = 1 + Cones.svec_length(R, 2)
    x = randn(T, dim)
    return dim, x
end
function generate_constraint(
    ::Type{Hypatia.EpiNormSpectralTriCone{T, R}},
    t,
    x,
) where {T, R}
    return t * e0(T, length(x)) - x
end

function initialize_cone(::Type{Hypatia.EpiNormSpectralCone{T, R}}) where {T, R}
    d1 = 2
    d2 = 3
    dim = 1 + Cones.vec_length(R, d1 * d2)
    x = randn(T, dim)
    return (d1, d2), x
end
function generate_constraint(::Type{Hypatia.EpiNormSpectralCone{T, R}}, t, x) where {T, R}
    return t * e0(T, length(x)) - x
end

function initialize_cone(::Type{Hypatia.MatrixEpiPerSquareCone{T, R}}) where {T, R}
    d1 = 2
    d2 = 3
    dim = Cones.svec_length(R, d1) + 1 + Cones.vec_length(R, d1 * d2)
    x = randn(T, dim)
    x[Cones.svec_length(R, d1) + 1] = -1
    return (d1, d2), x
end
function generate_constraint(
    ::Type{Hypatia.MatrixEpiPerSquareCone{T, R}},
    t,
    x,
) where {T, R}
    return t * [id_svec(R, Cones.svec_length(R, 2)); zeros(T, 1 + Cones.vec_length(R, 6))] -
           x
end

function initialize_cone(::Type{Hypatia.GeneralizedPowerCone{T}}) where {T}
    α = rand(T, 2)
    α[2] = 1 - α[1]
    x = rand(T, 5)
    return (α, 3), x
end
function generate_constraint(::Type{Hypatia.GeneralizedPowerCone{T}}, t, x) where {T}
    return t * [1, 1, 0, 0, 0] + x
end

function initialize_cone(::Type{Hypatia.HypoPowerMeanCone{T}}) where {T}
    α = rand(T, 2)
    α[2] = 1 - α[1]
    x = rand(T, 3)
    return (α,), x
end
function generate_constraint(::Type{Hypatia.HypoPowerMeanCone{T}}, t, x) where {T}
    return -t * [1, 0, 0] + x
end

function initialize_cone(::Type{Hypatia.HypoGeoMeanCone{T}}) where {T}
    dim = 3
    x = rand(T, dim)
    return dim, x
end
generate_constraint(::Type{Hypatia.HypoGeoMeanCone{T}}, t, x) where {T} = -t * [1, 0, 0] + x

function initialize_cone(::Type{Hypatia.HypoRootdetTriCone{T, R}}) where {T, R}
    dim = 1 + Cones.svec_length(R, 2)
    x = randn(T, dim)
    return dim, x
end
function generate_constraint(::Type{Hypatia.HypoRootdetTriCone{T, R}}, t, x) where {T, R}
    return t * [-1; id_svec(R, length(x) - 1)] + x
end

initialize_cone(::Type{Hypatia.HypoPerLogCone{T}}) where {T} = (3, [0, 1, rand(T)])
generate_constraint(::Type{Hypatia.HypoPerLogCone{T}}, t, x) where {T} = -t * [1, 0, 0] + x

function initialize_cone(::Type{Hypatia.HypoPerLogdetTriCone{T, R}}) where {T, R}
    dim = 2 + Cones.svec_length(R, 2)
    x = randn(T, dim)
    x[2] = 1
    return dim, x
end
function generate_constraint(::Type{Hypatia.HypoPerLogdetTriCone{T, R}}, t, x) where {T, R}
    return t * [-1; 0; id_svec(R, length(x) - 2)] + x
end

function initialize_cone(::Type{Hypatia.EpiPerSepSpectralCone{T}}) where {T}
    h = Cones.NegSqrtSSF()
    Q = Cones.MatrixCSqr{T, Complex{T}}
    d = 3
    x = randn(T, 2 + Cones.svec_length(Complex{T}, d))
    x[2] = 1
    return (h, Q, d), x
end
function generate_constraint(::Type{Hypatia.EpiPerSepSpectralCone{T}}, t, x) where {T}
    return t * [1; 0; id_svec(Complex{T}, length(x) - 2)] + x
end

function initialize_cone(::Type{Hypatia.EpiTrRelEntropyTriCone{T, R}}) where {T, R}
    dim = 1 + 2Cones.svec_length(R, 2)
    x = randn(T, dim)
    return dim, x
end
function generate_constraint(
    ::Type{Hypatia.EpiTrRelEntropyTriCone{T, R}},
    t,
    x,
) where {T, R}
    id = id_svec(R, div(length(x) - 1, 2))
    return t * [1; id; id] + x
end

function id_svec(R, dim)
    v = zeros(real(R), dim)
    if R <: Real
        for i in 1:isqrt(2 * dim)
            v[div(i * (i + 1), 2)] = 1
        end
    else
        for i in 1:isqrt(dim)
            v[i^2] = 1
        end
    end
    return v
end

function e0(T, dim)
    v = zeros(T, dim)
    v[1] = 1
    return v
end

function cone_types(T::Type{<:Real})
    cones_T = [
        Hypatia.NonnegativeCone{T},
        Hypatia.PosSemidefTriCone{T, T},
        Hypatia.PosSemidefTriCone{T, Complex{T}},
        Hypatia.DoublyNonnegativeTriCone{T},
        Hypatia.PosSemidefTriSparseCone{Cones.PSDSparseDense, T, T},
        Hypatia.PosSemidefTriSparseCone{Cones.PSDSparseDense, T, Complex{T}},
        Hypatia.LinMatrixIneqCone{T},
        Hypatia.EpiNormInfCone{T, T},
        Hypatia.EpiNormInfCone{T, Complex{T}},
        Hypatia.EpiNormEuclCone{T},
        Hypatia.EpiPerSquareCone{T},
        Hypatia.EpiNormSpectralTriCone{T, T},
        Hypatia.EpiNormSpectralTriCone{T, Complex{T}},
        Hypatia.EpiNormSpectralCone{T, T},
        Hypatia.EpiNormSpectralCone{T, Complex{T}},
        Hypatia.MatrixEpiPerSquareCone{T, T},
        Hypatia.MatrixEpiPerSquareCone{T, Complex{T}},
        Hypatia.GeneralizedPowerCone{T},
        Hypatia.HypoPowerMeanCone{T},
        Hypatia.HypoGeoMeanCone{T},
        Hypatia.HypoRootdetTriCone{T, T},
        Hypatia.HypoRootdetTriCone{T, Complex{T}},
        Hypatia.HypoPerLogCone{T},
        Hypatia.HypoPerLogdetTriCone{T, T},
        Hypatia.HypoPerLogdetTriCone{T, Complex{T}},
        Hypatia.EpiPerSepSpectralCone{T},
        Hypatia.EpiRelEntropyCone{T},
        Hypatia.EpiTrRelEntropyTriCone{T, T},
        Hypatia.EpiTrRelEntropyTriCone{T, Complex{T}},
    ]

    if T <: LinearAlgebra.BlasReal
        append!(
            cones_T,
            [
                Hypatia.PosSemidefTriSparseCone{Cones.PSDSparseCholmod, T, T},
                Hypatia.PosSemidefTriSparseCone{Cones.PSDSparseCholmod, T, Complex{T}},
            ],
        )
    end

    return cones_T
end
