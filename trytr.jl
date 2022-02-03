using BenchmarkTools
using SparseArrays
import LinearAlgebra

function tr2(A::AbstractMatrix{T}) where T
    n = LinearAlgebra.checksquare(A)
    t = zero(T)
    for i=1:n
        t += A[i,i]
    end
    t
end

function tr3(A::AbstractMatrix{T}) where T
    n = LinearAlgebra.checksquare(A)
    sum(view(A, 1:size(A, 1)+1:length(A)))
end


function tr4(A::AbstractMatrix{T}) where T
    n = LinearAlgebra.checksquare(A)
    t = zero(T)
    for i in diagind(A)
        t += A[i]
    end
    t
    # sum(view(A, 1:size(A, 1)+1:length(A)))
end


# btw `sum(view(A, 1:size(A, 1)+1:length(A)))` is a bit slower on SparseArrays
# with a large number of nonzeros @simon @andreas
using LinearAlgebra, SparseArrays
n = 50_000; A = sprand(n, n, 0.01);
@time tr(A);
@time sum(view(A, 1:size(A, 1)+1:length(A)));
@time sum(view(A, diagind(A)));

# @show tr(A)
# @show tr4(A)
# @show sum(diag(A))
# @show sum(view(A, diagind(A)))
# @time tr(A)
@time tr4(A)
# @time sum(view(A, 1:size(A, 1)+1:length(A)));
@time sum(view(A, diagind(A)));
@time sum(diag(A))

# B = sprand(n, n, 0.1)
# @time tr2(B)
# @time tr3(B)
#
#
# C = sprand(n, n, 0.05)
# @time tr2(C)
# @time tr3(C)
