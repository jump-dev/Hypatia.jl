#=
Copyright 2018, Chris Coey and contributors

real WSOS or hermitian WSOS interpolation-based cone

TODO decide whether to keep this version with the gi specified or the old version without gi
does it work if gi are negative for some points? some confusion here
should the cone be parametrized by real vs complex?
check if gradient and hessian are correct for complex case
=#

mutable struct WSOSPolyInterp_Complex <: Cone
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{ComplexF64}}
    gs::Vector{Vector{Float64}}

    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    F

    function WSOSPolyInterp_Complex(dim::Int, Ps::Vector{Matrix{ComplexF64}}, gs::Vector{Vector{Float64}}, is_dual::Bool)
        for i in eachindex(Ps)
            @assert size(Ps[i], 1) == length(gs[i]) == dim
        end
        cone = new()
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

WSOSPolyInterp_Complex(dim::Int, Ps::Vector{Matrix{ComplexF64}}, gs::Vector{Vector{Float64}}) = WSOSPolyInterp_Complex(dim, Ps, gs, false)

get_nu(cone::WSOSPolyInterp_Complex) = sum(size(cone.Ps[i], 2) for i in eachindex(cone.Ps))

set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterp_Complex) = (@. arr = 1.0; arr)

function check_in_cone(cone::WSOSPolyInterp_Complex)
    U = cone.dim
    Ps = cone.Ps
    gs = cone.gs
    x = cone.point
    ΛFs = Vector{CholeskyPivoted{ComplexF64, Matrix{ComplexF64}}}(undef, length(Ps)) # TODO prealloc

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
