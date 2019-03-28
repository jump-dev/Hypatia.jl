#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points ipwt

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO can perform loop for calculating g and H in parallel
TODO scale the interior direction
=#

mutable struct WSOSPolyInterp_2 <: Cone
    use_dual::Bool
    dim::Int
    P::Matrix{Float64}
    Ls::Vector{Int}
    gs::Vector{Vector{Float64}}

    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    F # TODO prealloc
    # tmp1::Vector{Matrix{Float64}}
    # tmp2::Vector{Matrix{Float64}}
    # tmp3::Matrix{Float64}

    function WSOSPolyInterp_2(dim::Int, P::Matrix{Float64}, Ls::Vector{Int}, gs::Vector{Vector{Float64}}, is_dual::Bool)
        @assert length(Ls) == length(gs)
        @assert size(P, 1) == length(gs[1])
        @assert size(P, 2) == Ls[1]

        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.P = P
        cone.Ls = Ls
        cone.gs = gs

        cone.g = similar(P, dim)
        cone.H = similar(P, dim, dim)
        cone.H2 = similar(cone.H)
        cone.Hi = similar(cone.H)
        # cone.tmp1 = [similar(P[1], L, L) for L in Ls]
        # cone.tmp2 = [similar(P[1], L, dim) for L in Ls]
        # cone.tmp3 = similar(P[1], dim, dim)
        return cone
    end
end

WSOSPolyInterp_2(dim::Int, P::Matrix{Float64}, Ls::Vector{Int}, gs::Vector{Vector{Float64}}) = WSOSPolyInterp_2(dim, P, Ls, gs, false)

get_nu(cone::WSOSPolyInterp_2) = sum(cone.Ls)

set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterp_2) = (@. arr = 1.0; arr)

function check_in_cone(cone::WSOSPolyInterp_2)
    U = cone.dim
    P = cone.P
    Ls = cone.Ls
    gs = cone.gs
    x = cone.point

    @. cone.g = 0.0
    @. cone.H = 0.0
    # tmp3 = cone.tmp3

    for i in eachindex(Ls)
        # ipwtj = cone.ipwt[j]
        # tmp1j = cone.tmp1[j]
        # tmp2j = cone.tmp2[j]
        Li = Ls[i]
        Pi = view(P, :, 1:Li)
        gi = gs[i]
        Λi = Symmetric(Pi' * Diagonal(gi .* x) * Pi)

        F = cholesky!(Λi, Val(true), check = false)
        if !isposdef(F)
            return false
        end

        PΛiPi_a = Symmetric(Pi * inv(F) * Pi')
        # half
        Pitp = Matrix((Pi')[F.p, :])
        PΛiPi_half = LowerTriangular(F.L) \ Pitp
        PΛiPi_b = Symmetric(PΛiPi_half' * PΛiPi_half)
        # check consistency
        # @show norm(PΛiPi_a - PΛiPi_b)
        # @show PΛiPi_a ./ PΛiPi_b
        # @show F.p

        cone.g -= gi .* diag(PΛiPi_b)
        cone.H += Symmetric(gi * gi') .* abs2.(PΛiPi_b)
    end

    return factorize_hess(cone)
end
