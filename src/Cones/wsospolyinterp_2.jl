#=
Copyright 2018, Chris Coey and contributors

-real or hermitian interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by
-- interpolation points
-- domain polynomials g_i (evaluated at interpolation points)
-- basis polynomials P_i (evaluated at interpolation points)

TODO decide whether to keep this version with the gi specified or the old version without gi
does it work if gi are negative for some points? some confusion here
should the cone be parametrized by real vs complex?
check if gradient and hessian are correct for complex case
=#

RealOrComplexF64 = Union{Float64, ComplexF64}

mutable struct WSOSPolyInterp_2{T <: RealOrComplexF64} <: Cone
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{T}}
    gs::Vector{Vector{Float64}}

    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    F

    function WSOSPolyInterp_2(dim::Int, Ps::Vector{Matrix{T}}, gs::Vector{Vector{Float64}}, is_dual::Bool) where {T <: RealOrComplexF64}
        for i in eachindex(Ps)
            @assert size(Ps[i], 1) == length(gs[i]) == dim
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.Ps = Ps
        cone.gs = gs
        cone.g = Vector{Float64}(undef, dim)
        cone.H = similar(cone.g, dim, dim)
        cone.H2 = similar(cone.H)
        cone.Hi = similar(cone.H)
        return cone
    end
end

WSOSPolyInterp_2(dim::Int, Ps::Vector{Matrix{T}}, gs::Vector{Vector{Float64}}) where {T <: RealOrComplexF64} = WSOSPolyInterp_2{T}(dim, Ps, gs, false)

get_nu(cone::WSOSPolyInterp_2) = sum(size(cone.Ps[i], 2) for i in eachindex(cone.Ps))

set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterp_2) = (@. arr = 1.0; arr)

function check_in_cone(cone::WSOSPolyInterp_2{T}) where T
    U = cone.dim
    Ps = cone.Ps
    gs = cone.gs
    x = cone.point
    ΛFs = Vector{CholeskyPivoted{T, Matrix{T}}}(undef, length(Ps)) # TODO prealloc

    for i in eachindex(Ps)
        Pi = Ps[i]
        gi = gs[i]
        Λi = Hermitian(Pi' * Diagonal(gi .* x) * Pi) # TODO prealloc

        ΛFs[i] = cholesky!(Λi, Val(true), check = false)
        if !isposdef(ΛFs[i])
            return false
        end
    end

    @. cone.g = 0.0
    @. cone.H = 0.0

    for i in eachindex(Ps)
        Pi = Ps[i]
        gi = gs[i]
        ΛFi = ΛFs[i]

        # TODO prealloc below
        # PΛinvPti = Hermitian(Pi * Hermitian(inv(ΛFs[i])) * Pi')
        PΛinvPthi = LowerTriangular(ΛFi.L) \ Matrix((Pi')[ΛFi.p, :])
        PΛinvPti = Hermitian(PΛinvPthi' * PΛinvPthi) # TODO syrk

        cone.g -= gi .* diag(PΛinvPti)
        cone.H += Symmetric(gi * gi') .* abs2.(PΛinvPti) # TODO simplify math here?
    end

    return factorize_hess(cone)
end
