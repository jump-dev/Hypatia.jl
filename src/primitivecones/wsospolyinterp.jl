#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points ipwt

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO can perform loop for calculating g and H in parallel
TODO maybe can avoid final factorization?
TODO scale the interior direction
=#

mutable struct WSOSPolyInterp <: PrimitiveCone
    usedual::Bool
    dim::Int
    P0::Matrix{Float64}
    weight_vecs::Vector{Vector{Float64}}
    lower_dims::Vector{Int}
    pnt::AbstractVector{Float64}
    scalpnt::Vector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F # TODO prealloc
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}
    scal::Float64
    # d::Vector{POSVXData}

    function WSOSPolyInterp(
        dim::Int,
        P0::Matrix{Float64},
        weight_vecs::Vector{Vector{Float64}},
        lower_dims::Vector{Int},
        isdual::Bool
        )

        @assert size(P0, 1) == dim
        for wv in weight_vecs
            @assert length(wv) == dim
        end
        prmtv = new()
        prmtv.usedual = !isdual # using dual barrier
        prmtv.dim = dim
        prmtv.P0 = P0
        prmtv.weight_vecs = weight_vecs
        prmtv.lower_dims = lower_dims
        prmtv.scalpnt = similar(P0, dim)
        prmtv.g = similar(P0, dim)
        prmtv.H = similar(P0, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.tmp1 = [similar(P0, l, l) for l in lower_dims]
        prmtv.tmp2 = [similar(P0, l, dim) for l in lower_dims]
        prmtv.tmp3 = similar(P0, dim, dim)
        return prmtv
    end
end

WSOSPolyInterp(dim::Int, P0::Matrix{Float64}, weight_vecs::Vector{Vector{Float64}}, lower_dims::Vector{Int}) = WSOSPolyInterp(dim, P0, weight_vecs, lower_dims, false)

dimension(prmtv::WSOSPolyInterp) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterp) = sum(prmtv.lower_dims) # TODO exclude P0
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. arr = 1.0; arr)
loadpnt_prmtv!(prmtv::WSOSPolyInterp, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterp, scal::Float64)
    prmtv.scal = scal
    @. prmtv.scalpnt = prmtv.pnt / prmtv.scal

    @. prmtv.g = 0.0
    @. prmtv.H = 0.0
    tmp3 = prmtv.tmp3

    for j in eachindex(prmtv.weight_vecs) # TODO can be done in parallel, but need multiple tmp3s
        lower_dimsj = prmtv.lower_dims[j]
        P0j = prmtv.P0[:, 1:lower_dimsj] # TODO compare with views
        weight_vecsj = prmtv.weight_vecs[j]
        tmp1j = prmtv.tmp1[j]
        tmp2j = prmtv.tmp2[j]

        # tmp1j = ipwtj'*Diagonal(pnt)*ipwtj
        # mul!(tmp2j, ipwtj', Diagonal(prmtv.scalpnt)) # TODO dispatches to an extremely inefficient method
        @timeit to "incone multiplications" begin
        @. tmp2j = P0j' * (prmtv.scalpnt .* weight_vecsj)'
        mul!(tmp1j, tmp2j, P0j)
        end

        # pivoted cholesky and triangular solve method
        F = cholesky!(Symmetric(tmp1j, :L), Val(true), check=false)
        if !isposdef(F)
            return false
        end

        tmp2j .= view(P0j', F.p, :)
        ldiv!(F.L, tmp2j) # TODO make sure calls best triangular solve
        # mul!(tmp3, tmp2j', tmp2j)
        BLAS.syrk!('U', 'T', 1.0, tmp2j, 0.0, tmp3)

        # posvx solve method
        # tmp2j .= ipwtj' # TODO eliminate by transposing in construction
        # PDPiP = similar(tmp2j)
        # success = hypatia_posvx!(PDPiP, tmp1j, tmp2j, prmtv.d[j])
        # if !success
        #     return false
        # end
        # mul!(tmp3, ipwtj, PDPiP)
        # TODO ask @chriscoey if we need to keep ^^

        @inbounds for i in eachindex(prmtv.g)
            prmtv.g[i] -= tmp3[i, i] * weight_vecsj[i]
            @inbounds for k in 1:i
                prmtv.H[k, i] += abs2(tmp3[k, i]) * weight_vecsj[k] * weight_vecsj[i]
            end
        end
    end

    return factH(prmtv)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. g = prmtv.g/prmtv.scal; g)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = (ldiv!(prod, prmtv.F, arr); @. prod = prod * prmtv.scal * prmtv.scal; prod)
