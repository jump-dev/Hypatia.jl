using LinearAlgebra, SparseArrays
import SuiteSparse
import SuiteSparse: decrement, libumfpack
import SuiteSparse:UMFPACK
import SuiteSparse.UMFPACK: UmfpackLU, umfpack_numeric!, umfpack_symbolic!, umfpack_report_symbolic, umf_ctrl, umf_info, UMFPACK_WARNING_singular_matrix, umferror

const UmfpackIndexTypes = (:Int32, :Int64)
const UMFITypes = Union{Int32, Int64}

macro isok(A)
    :(umferror($(esc(A))))
end

# example data
n = 4
S = sparse(randn(n, n))
b = randn(n, 2)
x = copy(b)
# make a UmfpackLU object for S
zerobased = S.colptr[1] == 0
U = UmfpackLU(C_NULL, C_NULL, S.m, S.n,
                zerobased ? copy(S.colptr) : decrement(S.colptr),
                zerobased ? copy(S.rowval) : decrement(S.rowval),
                copy(S.nzval), 0)

# calls to get umfpack_numeric! and umfpack_symbolic!
umf_nm(nm,Tv,Ti) = "umfpack_" * (Tv == :Float64 ? "d" : "z") * (Ti == :Int64 ? "l_" : "i_") * nm
itype = Int64
sym_r = umf_nm("symbolic", :Float64, itype)
num_r = umf_nm("numeric", :Float64, itype)
@eval begin
    function umfpack_symbolic!(U::UmfpackLU{Float64,$itype})
        if U.symbolic != C_NULL return U end
        tmp = Vector{Ptr{Cvoid}}(undef, 1)
        @isok ccall(($sym_r, :libumfpack), $itype,
                    ($itype, $itype, Ptr{$itype}, Ptr{$itype}, Ptr{Float64}, Ptr{Cvoid},
                     Ptr{Float64}, Ptr{Float64}),
                    U.m, U.n, U.colptr, U.rowval, U.nzval, tmp,
                    umf_ctrl, umf_info)
        U.symbolic = tmp[1]
        return U
    end
    function umfpack_numeric!(U::UmfpackLU{Float64,$itype})
        if U.numeric != C_NULL return U end
        if U.symbolic == C_NULL umfpack_symbolic!(U) end
        tmp = Vector{Ptr{Cvoid}}(undef, 1)
        @isok status = ccall(($num_r, :libumfpack), $itype,
                       (Ptr{$itype}, Ptr{$itype}, Ptr{Float64}, Ptr{Cvoid}, Ptr{Cvoid},
                        Ptr{Float64}, Ptr{Float64}),
                       U.colptr, U.rowval, U.nzval, U.symbolic, tmp,
                       umf_ctrl, umf_info)
        println(status)
        if status != UMFPACK_WARNING_singular_matrix
            umferror(status)
        end
        U.numeric = tmp[1]
        U.status = status
        return U
    end
end

# try them
umfpack_symbolic!(U) # ok
umfpack_numeric!(U) # errors


norm(b - S * (U \ b))





@which F \ b
