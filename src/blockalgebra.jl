# left hand side always a kronecker with the identity
function _block_lowertrisolve(fact, ipwt, blocknum, R, L, U)
    Lmat = fact.L
    ei = zeros(R)
    ei[blocknum] = 1.0
    rhs = kron(ei, ipwt')
    return Lmat \ view(rhs, fact.p, :)
    # resvec = zeros(R * L, U)
    # resi(i) = resvec[_blockrange(i, L), :]
    # Lmatij(i, j) = Lmat[_blockrange(i, L), _blockrange(j, L)]
    # tmp = zeros(L, U)
    # for r in 1:R
    #     if r == blocknum
    #         resvec[_blockrange(r, L), :] = LowerTriangular(Lmatij(r, r)) \ ipwt'
    #     elseif r > blocknum
    #         tmp .= 0.0
    #         for s in blocknum:(r - 1)
    #             tmp -= Lmatij(r, s) * resi(s)
    #         end
    #         resvec[_blockrange(r, L), :] = LowerTriangular(Lmatij(r, r)) \ tmp
    #     end
    # end
    # return resvec
end
function _block_lowertrisolve(fact, ipwt, R, L, U)
    resmat = zeros(R * L, R * U)
    for r in 1:R
        resmat[:, _blockrange(r, U)] = _block_lowertrisolve(fact, ipwt, r, R, L, U)
    end
    return resmat
end

# assume blockwise cholesky factorization
function _block_uppertrisolve(uppertri, Fvec, ipwt, blocknum, R, L, U)
    Lmat = LowerTriangular(Symmetric(uppertri, :U))
    resvec = zeros(R * L, U)
    resi(i) = resvec[_blockrange(i, L), :]
    Lmatij(i, j) = Lmat[_blockrange(i, L), _blockrange(j, L)]
    tmp = zeros(L, U)
    for r in 1:R
        if r == blocknum
            resvec[_blockrange(r, L), :] = Fvec[r].L \ view(ipwt', Fvec[r].p, :)
        elseif r > blocknum
            tmp .= 0.0
            for s in blocknum:(r - 1)
                tmp -= Lmatij(r, s) * resi(s)
            end
            resvec[_blockrange(r, L), :] = Fvec[r].L \ view(tmp, Fvec[r].p, :)
        end
    end
    return resvec
end
function _block_uppertrisolve(uppertri, Fvec, ipwt, R, L, U)
    resmat = zeros(R * L, R * U)
    for r in 1:R
        resmat[:, _blockrange(r, U)] = _block_uppertrisolve(uppertri, Fvec, ipwt, r, R, L, U)
    end
    return resmat
end

# because Ux only has blocks on the lower triangle, don't need to multiply all combinations for blocks
# This is the same thing as BLAS.syrk!('U', 'T', 1.0, Ux, 0.0, res)
# a lot could improve. for one the elements on the diagonal are symmetric so don't need to compute entire block.
function mulblocks(Ux, R, L, U)
    res = Matrix{Float64}(undef, R * U, R * U)
    BLAS.syrk!('U', 'T', 1.0, Ux, 0.0, res)
    # tmp = zeros(U, U)
    # for i in 1:R
    #     rinds = _blockrange(i, U)
    #     for j in i:R
    #         cinds = _blockrange(j, U)
    #         tmp .= 0.0
    #         # since Ux is block lower triangular rows only from max(i,j) start making a nonzero contribution to the product
    #         for k = j:R
    #             # actually each block is symmetric so could make this better
    #             tmp += Ux[_blockrange(k, L), _blockrange(i, U)]' * Ux[_blockrange(k, L), _blockrange(j, U)]
    #         end
    #         res[rinds, cinds] = tmp
    #     end
    # end
    return Symmetric(res)
end

function PLmabdaP(fact, ipwtj, R, L, U)
    ux = _block_lowertrisolve(fact, ipwtj, R, L, U)
    res = mulblocks(ux, R, L, U)
    return res
end

# function blockcholesky(A, R, L)
#     res = zeros(R * L, R * L)
#     tmp = zeros(L, L)
#     for i in 1:R
#         @show i
#         for j in 1:i
#             tmp .= 0.0
#             @show j
#             if i == j
#                 for k in 1:(j - 1)
#                     @show k
#                     tmp += res[_blockrange(i, L), _blockrange(k, L)] * res[_blockrange(i, L), _blockrange(k, L)]'
#                 end
#                 res[_blockrange(i, L), _blockrange(i, L)] = cholesky(A[_blockrange(i, L), _blockrange(i, L)] - tmp, Val(false)).L
#             else
#                 for k in 1:(j - 1)
#                     @show k
#                     tmp += res[_blockrange(i, L), _blockrange(k, L)] * res[_blockrange(j, L), _blockrange(k, L)]'
#                 end
#                 res[_blockrange(i, L), _blockrange(j, L)] = LowerTriangular(res[_blockrange(j, L), _blockrange(j, L)]) \ (A[_blockrange(i, L), _blockrange(j, L)] - tmp)
#             end
#         end
#     end
#     return res
# end

# warning: when pivoting used, result not really a cholesky or a pivoted cholesky. pivoting needed when using the diagonal blocks but not other blocks.
function blockcholesky(A, R, L)
    res = zeros(R * L, R * L)
    tmp = zeros(L, L)
    facts = Vector{CholeskyPivoted{Float64, Matrix{Float64}}}(undef, R)
    for i in 1:R
        for j in i:R
            tmp .= 0.0
            if i == j
                for k in 1:(i - 1)
                    tmp += res[_blockrange(k, L), _blockrange(i, L)]' * res[_blockrange(k, L), _blockrange(i, L)]
                end
                F = cholesky(A[_blockrange(i, L), _blockrange(i, L)] - tmp, Val(true))
                if !(isposdef(F))
                    return (res, facts, false)
                end
                facts[i] = F
                res[_blockrange(i, L), _blockrange(i, L)] = F.U
            else
                for k in 1:(i - 1)
                    tmp += res[_blockrange(k, L), _blockrange(i, L)]' * res[_blockrange(k, L), _blockrange(j, L)]
                end
                # res[_blockrange(i, L), _blockrange(j, L)] = facts[i].L \ (A[_blockrange(i, L), _blockrange(j, L)] - tmp)
                rhs = A[_blockrange(i, L), _blockrange(j, L)] - tmp
                res[_blockrange(i, L), _blockrange(j, L)] = facts[i].L \ view(rhs, facts[i].p, :)
            end
        end
    end
    return (res, facts, true)
end

_blockrange(inner::Int, outer::Int) = (outer * (inner - 1) + 1):(outer * inner)

using Test
using LinearAlgebra
R = 3; U = 5; L = 4;
ipwt = rand(U, L)
kron_ipwt = kron(Matrix(I, R, R), ipwt)
blocklambda = rand(R * L, R * L)
blocklambda = blocklambda * blocklambda'
F = cholesky(blocklambda, Val(true))
Ux = _block_lowertrisolve(F, ipwt, R, L, U)
@test Ux ≈ F.L \ view(kron_ipwt', F.p, :)
@test Ux' * Ux ≈ (cholesky(blocklambda).L \ kron_ipwt')' * (cholesky(blocklambda).L \ kron_ipwt')
@test (F.L \ view(kron_ipwt', F.p, :))' * (F.L \ view(kron_ipwt', F.p, :)) ≈ (cholesky(blocklambda).L \ kron_ipwt')' * (cholesky(blocklambda).L \ kron_ipwt')
# x = _block_uppertrisolve(F.U, Ux, R, L, U) # this is not being used, not needed
# @test x ≈ F \ kron_ipwt' # this is not being used, not needed
# # @test kron_ipwt * inv(F) * kron_ipwt' ≈ mul_ipwtkron(ipwt, x, R, L, U) # this is not being used, not needed

res = zeros(U * R, U * R)
BLAS.syrk!('U', 'T', 1.0, Ux, 0.0, res)
@test Symmetric(res) ≈ Symmetric(PLmabdaP(F, ipwt, R, L, U))

(uppertri, Fvec, _) = blockcholesky(blocklambda, R, L)
ux_pivoted = _block_uppertrisolve(uppertri, Fvec, ipwt, R, L, U)
