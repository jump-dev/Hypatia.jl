
using LinearAlgebra

function symH(GHG, GHh, F)
    # ldiv!(F.U, GHh)
    lmul!(F.U, GHh)
    BLAS.syrk!('U', 'T', true, GHh, false, GHG)
    return GHG
end

function symH(GHG, GHh, F, G)
    # ldiv!(F.U, GHh)
    mul!(GHh, F.U, G)
    BLAS.syrk!('U', 'T', true, GHh, false, GHG)
    return GHG
end

function nonsymH(GHG, HG, G, H)
    mul!(HG, H, G)
    mul!(GHG, G', HG)
    return GHG
end

for q in [10, 50, 100, 500, 1000, 5000, 10000]
    H = randn(q, q)
    H = Symmetric(H * H', :U)
    G = randn(q, q)
    GHGsym = zeros(q, q)
    GHGnonsym = zeros(q, q)
    HG = zeros(q, q)
    GHh = zeros(q, q)

    println(q)

    @time nonsymH(GHGnonsym, HG, G, H)
    println()

    GHh = copy(G)
    @time F = cholesky!(H)
    @time symH(GHGsym, GHh, F)
    @assert Symmetric(GHGsym, :U) ≈ Symmetric(GHGnonsym, :U)
    @time symH(GHGsym, GHh, F, G)
    @assert Symmetric(GHGsym, :U) ≈ Symmetric(GHGnonsym, :U)
    println()
    println()
end
