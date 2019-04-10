#=
Copyright 2018, Chris Coey and contributors
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

    # println() # TODO delete
    # @show x # TODO delete

    @. cone.g = 0.0
    @. cone.H = 0.0

    ΛFs = Vector{CholeskyPivoted{ComplexF64, Matrix{ComplexF64}}}(undef, length(Ps))

    for i in eachindex(Ps)
        Pi = Ps[i]
        gi = gs[i]
        if !ishermitian(Pi' * Diagonal(gi .* x) * Pi) # TODO delete
            tmp = Pi' * Diagonal(gi .* x) * Pi
            diff = norm(tmp - tmp')
            if diff > 1e-10
                @show diff
            end
        end
        Λi = Hermitian(Pi' * Diagonal(gi .* x) * Pi)
        # @show Λi # TODO delete

        ΛFs[i] = cholesky!(Λi, Val(true), check = false)
        if !isposdef(ΛFs[i])
            # @show eigvals(Λi) # TODO delete
            return false
        end
    end

    for i in eachindex(Ps)
        Pi = Ps[i]
        gi = gs[i]
        ΛFi = ΛFs[i]

        PΛinvPti = Hermitian(Pi * Hermitian(inv(ΛFs[i])) * Pi')
        # PΛinvPthi = LowerTriangular(ΛFi.L) \ Matrix((Pi')[ΛFi.p, :])
        # PΛinvPti = Hermitian(PΛinvPthi' * PΛinvPthi)

        cone.g -= gi .* diag(PΛinvPti)
        cone.H += Symmetric(gi * gi') .* abs2.(PΛinvPti) # TODO simplify math here?
    end

    # @show cone.H # TODO delete

    return factorize_hess(cone)
end
