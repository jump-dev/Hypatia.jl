#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by
- interpolation points
- domain polynomials g_i (evaluated at interpolation points)
- basis polynomials P_i (evaluated at interpolation points)

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterp_2 <: Cone
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{Float64}}
    gs::Vector{Vector{Float64}}

    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    F

    function WSOSPolyInterp_2(dim::Int, Ps::Vector{Matrix{Float64}}, gs::Vector{Vector{Float64}}, is_dual::Bool)
        for i in eachindex(Ps)
            @assert size(Ps[i], 1) == length(gs[i]) == dim
        end
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.Ps = Ps
        cone.gs = gs
        cone.g = similar(Ps[1], dim)
        cone.H = similar(Ps[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.Hi = similar(cone.H)
        return cone
    end
end

WSOSPolyInterp_2(dim::Int, Ps::Vector{Matrix{Float64}}, gs::Vector{Vector{Float64}}) = WSOSPolyInterp_2(dim, Ps, gs, false)

get_nu(cone::WSOSPolyInterp_2) = sum(size(cone.Ps[i], 2) for i in eachindex(cone.Ps))

set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterp_2) = (@. arr = 1.0; arr)

function check_in_cone(cone::WSOSPolyInterp_2)
    U = cone.dim
    Ps = cone.Ps
    gs = cone.gs
    x = cone.point

    @. cone.g = 0.0
    @. cone.H = 0.0

    for i in eachindex(Ps)
        Pi = Ps[i]
        gi = gs[i]
        Λi = Symmetric(Pi' * Diagonal(gi .* x) * Pi)

        F = cholesky!(Λi, Val(true), check = false)
        if !isposdef(F)
            return false
        end

        # PΛinvPt = Symmetric(Pi * inv(F) * Pi')
        PΛinvPt_half = LowerTriangular(F.L) \ Matrix((Pi')[F.p, :])
        PΛinvPt = Symmetric(PΛinvPt_half' * PΛinvPt_half)

        cone.g -= gi .* diag(PΛinvPt)
        cone.H += Symmetric(gi * gi') .* abs2.(PΛinvPt) # TODO simplify math here?
    end

    return factorize_hess(cone)
end
