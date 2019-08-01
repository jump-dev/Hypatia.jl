import LinearAlgebra.mul!
import Base.adjoint
import Base.eltype
import Base.size
import Base.*
import Base.-
using Hypatia

struct Arrow
    point::Float64
    edge::Vector{Float64}
    diag::Vector{Float64}
end

eltype(::Arrow) = Float64
size(A::Arrow) = length(a.diag) + 1
size(A::Arrow, d) = length(a.diag) + 1
adjoint(A::Arrow) = A

function *(A::Arrow, x::AbstractVector)
    y = similar(x)
    y[1] = dot(A.edge, x[2:end]) + A.point * x[1]
    y[2:end] .= A.edge * x[1] .+ A.diag .* x[2:end]
    return y
end

(n, p, q) = (400, 1, 1202)
(q1, q2, q3) = (400, 401, 401)

arrow_mat_1 = A[(n + p + q1 + 1):(n + p + q1 + q2), (n + p + q1 + 1):(n + p + q1 + q2)]
arrow_block_1 = Arrow(arrow_mat_1[1, 1], arrow_mat_1[1, 2:end], diag(arrow_mat_1)[2:end])

arrow_mat_2 = A[(n + p + q1 + q2 + 1):(n + p + q1 + q2 + q3), (n + p + q1 + q2 + 1):(n + p + q1 + q2 + q3)]
arrow_block_2 = Arrow(arrow_mat_2[1, 1], arrow_mat_2[1, 2:end], diag(arrow_mat_2)[2:end])

diag_block = Diagonal(A[(n + p + 1):(n + p + q1), (n + p + 1):(n + p + q1)])

Acon = A[(n + 1):(n + p), 1:n]
Gcon = A[(n + p + 1):(n + p + q), 1:n]

A_block = Hypatia.HypBlockMatrix{Float64}(
    n + p + q,
    n + p + q,
    [Acon, Acon', Gcon, Gcon', diag_block, arrow_block_1, arrow_block_2],
    [(n + 1):(n + p), 1:n, (n + p + 1):(n + p + q), 1:n, (n + p + 1):(n + p + q1), (n + p + q1 + 1):(n + p + q1 + q2), (n + p + q1 + q2 + 1):(n + p + q)],
    [1:n, (n + 1):(n + p), 1:n, (n + p + 1):(n + p + q), (n + p + 1):(n + p + q1), (n + p + q1 + 1):(n + p + q1 + q2), (n + p + q1 + q2 + 1):(n + p + q)]
)


IterativeSolvers.TimerOutputs.reset_timer!()
(x, hist) = IterativeSolvers.minres!(prevsol, A_block, b, log = true, rtol = 1e-8, atol = 1e-8, verbose = false)
IterativeSolvers.TimerOutputs.print_timer()


;
