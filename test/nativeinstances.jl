#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
native test instances

TODO
- improve efficiency of many of the tests by doing in-place linear algebra etc
- maybe pass in tol?
=#

using Test
import Random
using LinearAlgebra
using SparseArrays
import GenericLinearAlgebra.svdvals
import GenericLinearAlgebra.eigvals
import DynamicPolynomials
import Hypatia
import Hypatia.Solvers.build_solve_check
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

function dimension1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0]
    A = zeros(T, 0, 2)
    b = T[]
    G = T[1 0]
    h = T[1]
    cones = CO.Cone{T}[CO.Nonnegative{T}(1)]

    for use_sparse in (false, true)
        if use_sparse
            A = sparse(A)
            G = sparse(G)
        end
        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -1 atol=tol rtol=tol
        @test r.x ≈ [1, 0] atol=tol rtol=tol
        @test isempty(r.y)

        @test_throws ErrorException options.linear_model{T}(T[-1, -1], A, b, G, h, cones)
    end
end

function consistent1(T; options...)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    c = zeros(T, n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(10I, q, n)
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[11:15, :] = rnd1 * A[1:5, :] - rnd2 * A[6:10, :]
    b = vec(sum(A, dims = 2))
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[:, 11:15] = rnd1 * A[:, 1:5] - rnd2 * A[:, 6:10]
    G[:, 11:15] = rnd1 * G[:, 1:5] - rnd2 * G[:, 6:10]
    c[11:15] = rnd1 * c[1:5] - rnd2 * c[6:10]
    h = zeros(T, q)
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; options...)
    @test r.status == :Optimal
end

function inconsistent1(T; options...)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(-I, q, n)
    b = rand(T, p)
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[11:15, :] = rnd1 * A[1:5, :] - rnd2 * A[6:10, :]
    b[11:15] = 2 * (rnd1 * b[1:5] - rnd2 * b[6:10])
    h = zeros(T, q)

    @test_throws ErrorException options.linear_model{T}(c, A, b, G, h, CO.Cone{T}[CO.Nonnegative{T}(q)])
end

function inconsistent2(T; options...)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(-I, q, n)
    b = rand(T, p)
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[:,11:15] = rnd1 * A[:,1:5] - rnd2 * A[:,6:10]
    G[:,11:15] = rnd1 * G[:,1:5] - rnd2 * G[:,6:10]
    c[11:15] = 2 * (rnd1 * c[1:5] - rnd2 * c[6:10])
    h = zeros(T, q)

    @test_throws ErrorException options.linear_model{T}(c, A, b, G, h, CO.Cone{T}[CO.Nonnegative{T}(q)])
end

function nonnegative1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = SparseMatrixCSC(-one(T) * I, q, n)
    h = zeros(T, q)
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = one(T), options...)
    @test r.status == :Optimal
end

function nonnegative2(T; options...)
    tol = 2 * sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = rand(T, q, n) - Matrix(T(2) * I, q, n)
    h = vec(sum(G, dims = 2))
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
end

function nonnegative3(T; options...)
    tol = 2 * sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = Diagonal(-one(T) * I, n)
    h = zeros(T, q)
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
end

function epinorminf1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, -1, -1]
    A = T[1 0 0; 0 1 0]
    b = [one(T), Tirt2]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 - Tirt2 atol=tol rtol=tol
    @test r.x ≈ [1, Tirt2, 1] atol=tol rtol=tol
    @test r.y ≈ [1, 1] atol=tol rtol=tol
end

function epinorminf2(T; options...)
    tol = 10 * sqrt(sqrt(eps(T)))
    l = 3
    L = 2l + 1
    c = collect(T, -l:l)
    A = spzeros(T, 2, L)
    A[1, 1] = A[1, L] = A[2, 1] = 1; A[2, L] = -1
    b = T[0, 0]
    G = [spzeros(T, 1, L); sparse(one(T) * I, L, L); spzeros(T, 1, L); sparse(T(2) * I, L, L)]
    h = zeros(T, 2L + 2); h[1] = 1; h[L + 2] = 1
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(L + 1, true), CO.EpiNormInf{T, T}(L + 1, false)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = one(T), options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -l + 2 atol=tol rtol=tol
    @test r.x[2] ≈ 0.5 atol=tol rtol=tol
    @test r.x[end - 1] ≈ -0.5 atol=tol rtol=tol
    @test sum(abs, r.x) ≈ 1 atol=tol rtol=tol
end

function epinorminf3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0, 0, 0, 0, 0]
    A = zeros(T, 0, 6)
    b = zeros(T, 0)
    G = Diagonal(-one(T) * I, 6)
    h = zeros(T, 6)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormInf{T, T}(6, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinorminf4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, -1]
    A = T[1 0 0; 0 1 0]
    b = T[1, -0.4]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(3, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.x ≈ [1, -0.4, 0.6] atol=tol rtol=tol
    @test r.y ≈ [1, 0] atol=tol rtol=tol
end

function epinorminf5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    c = T[1, 0, 0, 0, 0, 0]
    A = rand(T(-9):T(9), 3, 6)
    b = vec(sum(A, dims = 2))
    G = rand(T, 6, 6)
    h = vec(sum(G, dims = 2))
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(6, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=tol rtol=tol
end

function epinorminf6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, -1, -1, -1]
    A = T[1 0 0 0 0; 0 1 0 0 0; 0 0 0 1 0; 0 0 0 0 1]
    b = T[2, 0, 1, 0]
    G = SparseMatrixCSC(-one(T) * I, 5, 5)
    h = zeros(T, 5)
    cones = CO.Cone{T}[CO.EpiNormInf{T, Complex{T}}(5)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -3 atol=tol rtol=tol
    @test r.x ≈ [2, 0, 2, 1, 0] atol=tol rtol=tol
end

function epinorminf7(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0, 0, 0, 0, 0, 0]
    A = zeros(T, 0, 7)
    b = zeros(T, 0)
    G = Diagonal(-one(T) * I, 7)
    h = zeros(T, 7)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormInf{T, Complex{T}}(7, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinorminf8(T; options...)
    tol = eps(T) ^ 0.2
    c = T[1, -1, 1, 1]
    A = T[1 0 0 0 ; 0 1 0 0; 0 0 1 0]
    b = T[-0.4, 0.3, -0.3]
    G = vcat(zeros(T, 1, 4), Diagonal(T[-1, -1, -1, -1]))
    h = T[1, 0, 0, 0, 0]
    cones = CO.Cone{T}[CO.EpiNormInf{T, Complex{T}}(5, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1.4 atol=tol rtol=tol
    @test r.x ≈ [-0.4, 0.3, -0.3, -0.4] atol=tol rtol=tol
    @test r.y ≈ [0, 0.25, -0.25] atol=tol rtol=tol
    @test r.z ≈ [1.25, 1, -0.75, 0.75, 1] atol=tol rtol=tol
end

function epinormeucl1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    Tirt2 = inv(Trt2)
    c = T[0, -1, -1]
    A = T[10 0 0; 0 10 0]
    b = T[10, 10Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Trt2 atol=tol rtol=tol
    @test r.x ≈ [1, Tirt2, Tirt2] atol=tol rtol=tol
    @test r.y ≈ [Trt2 / 10, 0] atol=tol rtol=tol
end

function epinormeucl2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, -1]
    A = T[1 0 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function epinormeucl3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0, 0]
    A = T[0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=tol rtol=tol
    @test r.x ≈ [1, 1, 0] atol=tol rtol=tol
end

function epipersquare1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, -1, -1]
    A = T[1 0 0 0; 0 1 0 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 4, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -sqrt(T(2)) atol=tol rtol=tol
    @test r.x[3:4] ≈ [1, 1] / sqrt(T(2)) atol=tol rtol=tol
end

function epipersquare2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, 0, -1]
    A = T[1 0 0; 0 1 0]
    b = T[Tirt2 / 2, Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = -one(T), options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Tirt2 - 1 atol=tol rtol=tol
    @test r.x[2] ≈ Tirt2 atol=tol rtol=tol
end

function epipersquare3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, -1, -1]
    A = T[1 0 0 0]
    b = T[0]
    G = SparseMatrixCSC(-one(T) * I, 4, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = zero(T), options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function epiperexp1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texph = exp(T(0.5))
    c = T[1, 1, 1]
    A = T[0 1 0; 1 0 0]
    b = T[2, 1]
    G = sparse([3, 2, 1], [1, 2, 3], -ones(T, 3))
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerExp{T}()]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 * Texph + 3 atol=tol rtol=tol
    @test r.x ≈ [1, 2, 2 * Texph] atol=tol rtol=tol
    @test r.y ≈ -[1 + Texph / 2, 1 + Texph] atol=tol rtol=tol
    @test r.z ≈ -G * (c + A' * r.y) atol=tol rtol=tol
end

function epiperexp2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, -1]
    A = T[0 1 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerExp{T}()]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
end

function epiperexp3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([3, 2, 1, 4], [1, 2, 3, 1], -ones(T, 4))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerExp{T}(), CO.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function epiperexp4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texp2 = exp(T(-2))
    c = T[1, 0, 0]
    A = T[0 1 0; 0 0 1]
    b = T[1, -1]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerExp{T}(true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Texp2 atol=tol rtol=tol
    @test r.x ≈ [Texp2, 1, -1] atol=tol rtol=tol
end

# TODO add use_dual = true tests
function episumperentropy1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for w_dim in [1, 2, 3]
        dim = 1 + 2 * w_dim
        c = T[1]
        A = zeros(T, 0, 1)
        b = zeros(T, 0)
        G = zeros(T, dim, 1)
        G[1, 1] = -1
        h = zeros(T, dim)
        h[2:(w_dim + 1)] .= 1
        w = rand(T, w_dim) .+ 1
        h[(w_dim + 2):end] .= w
        cones = CO.Cone{T}[CO.EpiSumPerEntropy{T}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ sum(wi * log(wi) for wi in w) atol=tol rtol=tol
    end
end

function episumperentropy2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    for w_dim in [1, 2, 4]
        dim = 1 + 2 * w_dim
        c = fill(-one(T), w_dim)
        A = zeros(T, 0, w_dim)
        b = zeros(T, 0)
        G = vcat(zeros(T, 1 + w_dim, w_dim), Matrix{T}(-I, w_dim, w_dim))
        h = vcat(zero(T), ones(T, w_dim), zeros(T, w_dim))
        cones = CO.Cone{T}[CO.EpiSumPerEntropy{T}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -w_dim atol=tol rtol=tol
        @test r.x ≈ fill(1, w_dim) atol=tol rtol=tol
    end
end

function episumperentropy3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    for w_dim in [2, 4]
        dim = 1 + 2 * w_dim
        c = fill(-one(T), w_dim)
        A = ones(T, 1, w_dim)
        b = T[dim]
        G = vcat(zeros(T, 1, w_dim), Matrix{T}(-I, w_dim, w_dim), zeros(T, w_dim, w_dim))
        h = vcat(zeros(T, 1 + w_dim), ones(T, w_dim))
        cones = CO.Cone{T}[CO.EpiSumPerEntropy{T}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -dim atol=tol rtol=tol
        @test r.x ≈ fill(dim / w_dim, w_dim) atol=tol rtol=tol
    end
end

function episumperentropy4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1]
    A = zeros(T, 0, 1)
    b = zeros(T, 0)
    G = Matrix{T}(-I, 5, 1)
    h = T[0, 1, 5, 2, 3]
    cones = CO.Cone{T}[CO.EpiSumPerEntropy{T}(5)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    entr = 2 * log(T(2)) + 3 * log(3 / T(5))
    @test r.primal_obj ≈ entr atol=tol rtol=tol
    @test r.s ≈ [entr, 1, 5, 2, 3] atol=tol rtol=tol
    @test r.z ≈ [1, 2, 3 / T(5), log(inv(T(2))) - 1, log(5 / T(3)) - 1] atol=tol rtol=tol
end

function hypoperlog1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texph = exp(T(0.5))
    c = T[1, 1, 1]
    A = T[0 1 0; 1 0 0]
    b = T[2, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 * Texph + 3 atol=tol rtol=tol
    @test r.x ≈ [1, 2, 2 * Texph] atol=tol rtol=tol
    @test r.y ≈ -[1 + Texph / 2, 1 + Texph] atol=tol rtol=tol
    @test r.z ≈ c + A' * r.y atol=tol rtol=tol
end

function hypoperlog2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = T[0 1 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
end

function hypoperlog3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(T, 4))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3), CO.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function hypoperlog4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texp2 = exp(T(-2))
    c = T[0, 0, 1]
    A = T[0 1 0; 1 0 0]
    b = T[1, -1]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Texp2 atol=tol rtol=tol
    @test r.x ≈ [-1, 1, Texp2] atol=tol rtol=tol
end

function hypoperlog5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tlogq = log(T(0.25))
    c = T[-1, 0, 0]
    A = T[0 1 1]
    b = T[1]
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = T[0, 1, 0, 0]
    cones = CO.Cone{T}[CO.HypoPerLog{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Tlogq atol=tol rtol=tol
    @test r.x ≈ [Tlogq, 0.5, 0.5] atol=tol rtol=tol
    @test r.y ≈ [2] atol=tol rtol=tol
end

function hypoperlog6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test r.x[1] ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function hypogeomean1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = T[0 0 1; 0 1 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(ones(T, 2) / 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 0 : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[2:3] ≈ [1, 0.5] atol=tol rtol=tol
    end
end

function hypogeomean2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = vcat(zero(T), ones(T, l))
    A = T[one(T) zeros(T, 1, l)]
    G = SparseMatrixCSC(-one(T) * I, l + 1, l + 1)
    h = zeros(T, l + 1)

    for is_dual in (true, false)
        b = is_dual ? [-one(T)] : [one(T)]
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(l)), l), is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 1 : l) atol=tol rtol=tol
        @test r.x[2:end] ≈ (is_dual ? fill(inv(T(l)), l) : ones(l)) atol=tol rtol=tol
    end
end

function hypogeomean3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = ones(T, l)
    A = zeros(T, 0, l)
    b = zeros(T, 0)
    G = [zeros(T, 1, l); Matrix{T}(-I, l, l)]
    h = zeros(T, l + 1)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(l)), l), is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function power1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, 1]
    A = T[1 0 0; 0 1 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for is_dual in (false, true)
        cones = CO.Cone{T}[CO.Power{T}(ones(T, 2) / 2, 1, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? -sqrt(T(2)) : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[3] ≈ (is_dual ? -sqrt(T(2)) : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[1:2] ≈ [0.5, 1] atol=tol rtol=tol
    end
end

function power2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, -1, -1]
    A = T[0 1 0 0; 1 0 0 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 4, 4)
    h = zeros(T, 4)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.Power{T}(ones(T, 2) / 2, 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? -T(2) : -1) atol=tol rtol=tol
        @test norm(r.x[3:4]) ≈ (is_dual ? sqrt(T(2)) : inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[3:4] ≈ (is_dual ? ones(T, 2) : fill(inv(T(2)), 2)) atol=tol rtol=tol
        @test r.x[1:2] ≈ [1, 0.5] atol=tol rtol=tol
    end
end

function power3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = vcat(fill(T(10), l), zeros(T, 2))
    A = T[zeros(T, 1, l) one(T) zero(T); zeros(T, 1, l) zero(T) one(T)]
    G = SparseMatrixCSC(-T(10) * I, l + 2, l + 2)
    h = zeros(T, l + 2)

    for is_dual in (true, false)
        b = [one(T), zero(T)]
        cones = CO.Cone{T}[CO.Power{T}(fill(inv(T(l)), l), 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 10 : 10 * T(l)) atol=tol rtol=tol
        @test r.x[1:l] ≈ (is_dual ? fill(inv(T(l)), l) : ones(l)) atol=tol rtol=tol
    end
end

function power4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = ones(T, l)
    A = zeros(T, 0, l)
    b = zeros(T, 0)
    G = [zeros(T, 3, l); Matrix{T}(-I, l, l)]
    h = zeros(T, l + 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.Power{T}(fill(inv(T(l)), l), 3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinormspectral1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    for is_complex in (false, true)
        dim = Xn * Xm
        if is_complex
            dim *= 2
        end
        c = vcat(one(T), zeros(T, dim))
        A = hcat(zeros(T, dim, 1), Matrix{T}(I, dim, dim))
        b = rand(T, dim)
        G = Matrix{T}(-I, dim + 1, dim + 1)
        h = vcat(zero(T), rand(T, dim))

        for is_dual in (true, false)
            R = (is_complex ? Complex{T} : T)
            cones = CO.Cone{T}[CO.EpiNormSpectral{T, R}(Xn, Xm, is_dual)]

            r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
            @test r.status == :Optimal

            S = zeros(R, Xn, Xm)
            @views CO.vec_copy_to!(S, r.s[2:end])
            prim_svdvals = svdvals(S)
            Z = similar(S)
            @views CO.vec_copy_to!(Z, r.z[2:end])
            dual_svdvals = svdvals(Z)
            if is_dual
                @test sum(prim_svdvals) ≈ r.s[1] atol=tol rtol=tol
                @test dual_svdvals[1] ≈ r.z[1] atol=tol rtol=tol
            else
                @test prim_svdvals[1] ≈ r.s[1] atol=tol rtol=tol
                @test sum(dual_svdvals) ≈ r.z[1] atol=tol rtol=tol
            end
        end
    end
end

function epinormspectral2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    for is_complex in (false, true)
        R = (is_complex ? Complex{T} : T)
        dim = Xn * Xm
        if is_complex
            dim *= 2
        end
        mat = rand(R, Xn, Xm)
        c = zeros(T, dim)
        CO.vec_copy_to!(c, mat)
        c .*= -1
        A = zeros(T, 0, dim)
        b = T[]
        G = vcat(zeros(T, 1, dim), Matrix{T}(-I, dim, dim))
        h = vcat(one(T), zeros(T, dim))

        for is_dual in (true, false)
            cones = CO.Cone{T}[CO.EpiNormSpectral{T, R}(Xn, Xm, is_dual)]
            r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
            @test r.status == :Optimal
            if is_dual
                @test r.primal_obj ≈ -svdvals(mat)[1] atol=tol rtol=tol
            else
                @test r.primal_obj ≈ -sum(svdvals(mat)) atol=tol rtol=tol
            end
        end
    end
end

function epinormspectral3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    for is_complex in (false, true), (Xn, Xm) in ((1, 1), (1, 3), (2, 2), (3, 4))
        dim = Xn * Xm
        if is_complex
            dim *= 2
        end
        c = fill(-one(T), dim)
        A = zeros(T, 0, dim)
        b = T[]
        G = vcat(zeros(T, 1, dim), Matrix{T}(-I, dim, dim))
        h = zeros(T, dim + 1)

        for is_dual in (true, false)
            R = (is_complex ? Complex{T} : T)
            cones = CO.Cone{T}[CO.EpiNormSpectral{T, R}(Xn, Xm, is_dual)]
            r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
            @test r.status == :Optimal
            @test r.primal_obj ≈ 0 atol=tol rtol=tol
            @test norm(r.x) ≈ 0 atol=tol rtol=tol
        end
    end
end

function epinormspectral4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    G = zeros(T, 7, 1)
    G[1, 1] = -1
    h = T[0, 1, 1, 1, -1, 0, 1]

    rt2 = sqrt(T(2))
    rt3 = sqrt(T(3))
    invrt2 = inv(rt2)
    invrt3 = inv(rt3)
    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormSpectral{T, T}(2, 3, is_dual)]
        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        if is_dual
            @test r.primal_obj ≈ rt2 + rt3 atol=tol rtol=tol
            @test r.s ≈ T[rt2 + rt3, 1, 1, 1, -1, 0, 1] atol=tol rtol=tol
            @test r.z ≈ T[1, -invrt2, -invrt3, -invrt2, invrt3, 0, -invrt3] atol=tol rtol=tol
        else
            @test r.primal_obj ≈ rt3 atol=tol rtol=tol
            @test r.s ≈ T[rt3, 1, 1, 1, -1, 0, 1] atol=tol rtol=tol
            @test r.z ≈ T[1, 0, -invrt3, 0, invrt3, 0, -invrt3] atol=tol rtol=tol
        end
    end
end

function matrixepipersquare1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for is_complex in (false, true), (Xn, Xm) in [(1, 1), (1, 3), (2, 2), (2, 3)]
        R = (is_complex ? Complex{T} : T)
        per_idx = (is_complex ? Xn ^ 2 + 1 : CO.svec_length(Xn) + 1)
        dim = per_idx + (is_complex ? 2 : 1) * Xn * Xm
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, dim, 1)
        G[per_idx] = -1
        h = zeros(T, dim)
        @views CO.smat_to_svec!(h[1:(per_idx - 1)], Matrix{R}(I, Xn, Xn), sqrt(T(2)))
        W = rand(R, Xn, Xm)
        @views CO.vec_copy_to!(h[(per_idx + 1):end], W)
        WWt = Hermitian(W * W')
        dual_epi = tr(WWt) / 2
        primal_epi = svdvals(WWt)[1] / 2

        for is_dual in (false, true)
            cones = CO.Cone{T}[CO.MatrixEpiPerSquare{T, R}(Xn, Xm, is_dual)]
            r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
            @test r.status == :Optimal
            @test r.primal_obj >= 0
            if is_dual
                @test r.primal_obj ≈ dual_epi atol=tol rtol=tol
                @test r.s[per_idx] ≈ dual_epi atol=tol rtol=tol
                @test r.z[per_idx] ≈ 1 atol=tol rtol=tol
            else
                @test r.primal_obj ≈ primal_epi atol=tol rtol=tol
                @test r.s[per_idx] ≈ primal_epi atol=tol rtol=tol
                @test r.z[per_idx] ≈ 1 atol=tol rtol=tol
            end
        end
    end
end

function matrixepipersquare2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    for is_complex in (false, true)
        R = (is_complex ? Complex{T} : T)
        per_idx = (is_complex ? Xn ^ 2 + 1 : CO.svec_length(Xn) + 1)
        dim = per_idx + (is_complex ? 2 : 1) * Xn * Xm
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, dim, 1)
        G[per_idx] = -1
        h = zeros(T, dim)
        U_half = rand(R, Xn, Xn)
        U = Hermitian(U_half * U_half')
        @views CO.smat_to_svec!(h[1:(per_idx - 1)], U.data, sqrt(T(2)))
        W = rand(R, Xn, Xm)
        @views CO.vec_copy_to!(h[(per_idx + 1):end], W)

        for is_dual in (false, true)
            cones = CO.Cone{T}[CO.MatrixEpiPerSquare{T, R}(Xn, Xm, is_dual)]
            r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
            @test r.status == :Optimal
            @test r.primal_obj >= 0
            if is_dual
                @test 2 * r.s[per_idx] ≈ tr(W' * (U \ W)) atol=tol rtol=tol
            else
                primal_viol = Hermitian(2 * r.s[per_idx] * U - W * W')
                @test minimum(eigvals(primal_viol)) ≈ 0 atol=tol rtol=tol
            end
        end
    end
end

function matrixepipersquare3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (2, 4)
    for is_complex in (false, true)
        R = (is_complex ? Complex{T} : T)
        per_idx = (is_complex ? Xn ^ 2 + 1 : CO.svec_length(Xn) + 1)
        W_dim = (is_complex ? 2 : 1) * Xn * Xm
        dim = per_idx + W_dim
        c = ones(T, W_dim)
        A = zeros(T, 0, W_dim)
        b = T[]
        G = vcat(zeros(T, per_idx, W_dim), Matrix{T}(-10I, W_dim, W_dim))
        h = zeros(T, dim)
        U_half = rand(R, Xn, Xn)
        U = Hermitian(U_half * U_half')
        @views CO.smat_to_svec!(h[1:(per_idx - 1)], U.data, sqrt(T(2)))

        for is_dual in (false, true)
            cones = CO.Cone{T}[CO.MatrixEpiPerSquare{T, R}(Xn, Xm, is_dual)]
            r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
            @test r.status == :Optimal
            @test norm(r.x) ≈ 0 atol=2tol rtol=2tol
        end
    end
end

function linmatrixineq1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for side in [2, 3, 5], R in [T, Complex{T}]
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, 2, 1)
        G[1, 1] = -1
        h = T[0, 2]
        A_1_half = rand(R, side, side)
        A_1 = Hermitian(A_1_half * A_1_half' + 2I)
        F = eigen(A_1)
        val_1 = F.values[end]
        vec_1 = F.vectors[:, end]
        As = [A_1, Hermitian(-vec_1 * vec_1')]
        cones = CO.Cone{T}[CO.LinMatrixIneq{T}(As)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 2 / val_1 atol=tol rtol=tol
        @test r.s ≈ [2 / val_1, 2] atol=tol rtol=tol
    end
end

function linmatrixineq2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for Rs in [[T, T], [Complex{T}, Complex{T}], [T, Complex{T}, T], [Complex{T}, T, T]]
        dim = length(Rs)
        c = ones(T, dim - 1)
        A = zeros(T, 0, dim - 1)
        b = T[]
        G = vcat(spzeros(T, 1, dim - 1), sparse(-one(T) * I, dim - 1, dim - 1))
        h = zeros(T, dim)
        h[1] = 1
        As = Hermitian[]
        for R in Rs
            A_half = rand(R, 3, 3)
            push!(As, Hermitian(A_half * A_half'))
        end
        As[1] += I
        cones = CO.Cone{T}[CO.LinMatrixIneq{T}(As)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj < 0
    end
end

function linmatrixineq3(T; options...)
    dense1 = [1 0; 0 1]
    dense2 = [1 0; 0 -1]
    sparse1 = sparse(dense1)
    sparse2 = sparse(dense2)
    diag1 = Diagonal([1, 1])
    diag2 = Diagonal([1, -1])
    # NOTE not all combinations work due to missing methods in LinearAlgebra
    As_list = [
        [dense1, dense2],
        # [dense1, sparse2],
        [dense1, diag2],
        # [sparse1, dense2],
        [sparse1, sparse2],
        # [sparse1, diag2],
        [diag1, dense2],
        # [diag1, sparse2],
        [diag1, diag2],
        [I, dense2],
        # [I, sparse2],
        [I, diag2],
        ]
    for As in As_list
        tol = sqrt(sqrt(eps(T)))
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, 2, 1)
        G[1, 1] = -1
        h = T[0, -1]
        cones = CO.Cone{T}[CO.LinMatrixIneq{T}(As)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 1 atol=tol rtol=tol
        @test r.s ≈ [1, -1] atol=tol rtol=tol
    end
end

function possemideftri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 0; 0 0 1]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -one(T) atol=tol rtol=tol
    @test r.x[2] ≈ one(T) atol=tol rtol=tol
end

function possemideftri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 1]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

# maximum eigenvalue problem
function possemideftri3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    rand_mat = Hermitian(rand(T, 2, 2), :U)
    G = reshape(T[-1, 0, -1], 3, 1)
    h = -CO.smat_to_svec!(zeros(T, 3), rand_mat, rt2)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ eig_max atol=tol rtol=tol
    @test r.x[1] ≈ eig_max atol=tol rtol=tol
end

# dual formulation to the above
function possemideftri4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    s = 3
    rand_mat = Hermitian(rand(T, s, s), :U)
    dim = sum(1:s)
    c = -CO.smat_to_svec!(zeros(T, dim), rand_mat, rt2)
    A = reshape(CO.smat_to_svec!(zeros(T, dim), Matrix{T}(I, s, s), rt2), 1, dim)
    b = T[1]
    G = Diagonal(-one(T) * I, dim)
    h = zeros(T, dim)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(dim)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ -eig_max atol=tol rtol=tol
end

function possemideftri5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    Trt2i = inv(Trt2)
    c = T[1, 0, 0, 1]
    A = T[0 0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Trt2 atol=tol rtol=tol
    @test r.x ≈ [Trt2i, 0, 1, Trt2i] atol=tol rtol=tol
end

# maximum eigenvalue problem
function possemideftri6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    rand_mat = Hermitian(rand(Complex{T}, 2, 2), :U)
    G = reshape(T[-1, 0, 0, -1], 4, 1)
    h = -CO.smat_to_svec!(zeros(T, 4), rand_mat, rt2)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ eig_max atol=tol rtol=tol
    @test r.x[1] ≈ eig_max atol=tol rtol=tol
end

# dual formulation to the above
function possemideftri7(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    s = 3
    rand_mat = Hermitian(rand(Complex{T}, s, s), :U)
    dim = abs2(s)
    c = -CO.smat_to_svec!(zeros(T, dim), rand_mat, rt2)
    A = reshape(CO.smat_to_svec!(zeros(T, dim), Matrix{Complex{T}}(I, s, s), rt2), 1, dim)
    b = T[1]
    G = Diagonal(-one(T) * I, dim)
    h = zeros(T, dim)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(dim)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ -eig_max atol=tol rtol=tol
end

function possemideftri8(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    G = zeros(T, 15, 1)
    G[[1, 3, 6, 10, 15]] .= -1
    h = zeros(T, 15)
    @. h[[7, 8, 9, 11, 12, 13]] = rt2 * [1, 1, 0, 1, -1, 1]
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(15)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    rt3 = sqrt(T(3))
    @test r.primal_obj ≈ rt3 atol=tol rtol=tol
    @test r.s ≈ [rt3, 0, rt3, 0, 0, rt3, rt2, rt2, 0, rt3, rt2, -rt2, rt2, 0, rt3] atol=tol rtol=tol
    inv6 = inv(T(6))
    rt2inv6 = rt2 / 6
    invrt6 = inv(rt2 * rt3)
    @test r.z ≈ [inv6, -rt2inv6, inv6, rt2inv6, -rt2inv6, inv6, 0, 0, 0, 0, -invrt6, invrt6, -invrt6, 0, inv(T(2))] atol=tol rtol=tol
end

function possemideftri9(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    inv2 = inv(T(2))
    c = vcat(one(T), zeros(T, 9))
    A = zeros(T, 0, 10)
    b = T[]
    G = zeros(T, 16, 10)
    G[1, 2] = G[1, 4] = G[1, 7] = G[1, 8] = G[1, 10] = inv2
    G[1, 1] = G[2, 2] = G[4, 4] = G[7, 7] = G[11, 8] = G[16, 10] = -1
    G[3, 3] = G[5, 5] = G[6, 6] = G[15, 9] = -rt2
    h = zeros(T, 16)
    @. h[[8, 9, 10, 12, 13, 14]] = rt2 * [1, 1, 0, 1, -1, 1]
    cones = CO.Cone{T}[CO.Nonnegative{T}(1), CO.PosSemidefTri{T, T}(15)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    rt3 = sqrt(T(3))
    @test r.primal_obj ≈ rt2 + rt3 atol=tol rtol=tol
    invrt2 = inv(rt2)
    invrt3 = inv(rt3)
    @test r.s ≈ [0, invrt2 + invrt3, 1 - rt2 / rt3, invrt2 + invrt3, rt2 * invrt3, -rt2 * invrt3, invrt3, rt2, rt2, 0, rt2, rt2, -rt2, rt2, 0, rt3] atol=tol rtol=tol
    invrt6 = invrt2 * invrt3
    @test r.z ≈ [1, inv2, 0, inv2, 0, 0, inv2, -inv2, -inv2, 0, inv2, -invrt6, invrt6, -invrt6, 0, inv2] atol=tol rtol=tol
end





function possemideftrisparse1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 0; 0 0 1]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    row_idxs = [1, 2, 2]
    col_idxs = [1, 1, 2]
    cones = CO.Cone{T}[CO.PosSemidefTriSparse{T, T}(2, row_idxs, col_idxs)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -one(T) atol=tol rtol=tol
    @test r.x[2] ≈ one(T) atol=tol rtol=tol
end

function possemideftrisparse2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    G = zeros(T, 10, 1)
    G[[1, 2, 3, 6, 10]] .= -1
    h = zeros(T, 10)
    @. h[[4, 5, 7, 8, 9]] = rt2 * [1, 1, 1, -1, 1]
    row_idxs = [1, 2, 3, 4, 4, 4, 5, 5, 5, 5]
    col_idxs = [1, 2, 3, 1, 2, 4, 1, 2, 3, 5]
    cones = CO.Cone{T}[CO.PosSemidefTriSparse{T, T}(5, row_idxs, col_idxs)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    rt3 = sqrt(T(3))
    @test r.primal_obj ≈ rt3 atol=tol rtol=tol
    @test r.s ≈ [rt3, rt3, rt3, rt2, rt2, rt3, rt2, -rt2, rt2, rt3] atol=tol rtol=tol
    inv6 = inv(T(6))
    rt2inv6 = rt2 / 6
    invrt6 = inv(rt2 * rt3)
    @test r.z ≈ [inv6, inv6, inv6, 0, 0, 0, -invrt6, invrt6, -invrt6, inv(T(2))] atol=tol rtol=tol
end





function hypoperlogdettri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 4
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + CO.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1, 0]
        A = T[0 1]
        b = T[1]
        G = Matrix{T}(-I, dim, 2)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        CO.smat_to_svec!(view(h, 3:dim), mat, rt2)
        cones = CO.Cone{T}[CO.HypoPerLogdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
        @test r.x[2] ≈ 1 atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        CO.svec_to_smat!(sol_mat, r.s[3:end] / r.s[2], rt2)
        @test r.s[2] * logdet(cholesky!(Hermitian(sol_mat, :U))) ≈ r.s[1] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, -r.z[3:end] / r.z[1], rt2)
        @test r.z[1] * (logdet(cholesky!(Hermitian(sol_mat, :U))) + T(side)) ≈ r.z[2] atol=tol rtol=tol
    end
end

function hypoperlogdettri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + CO.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[0, 1]
        A = T[1 0]
        b = T[-1]
        G = Matrix{T}(-I, dim, 2)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        CO.smat_to_svec!(view(h, 3:dim), mat, rt2)
        cones = CO.Cone{T}[CO.HypoPerLogdetTri{T, R}(dim, true)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.x[2] ≈ r.primal_obj atol=tol rtol=tol
        @test r.x[1] ≈ -1 atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        CO.svec_to_smat!(sol_mat, -r.s[3:end] / r.s[1], rt2)
        @test r.s[1] * (logdet(cholesky!(Hermitian(sol_mat, :U))) + T(side)) ≈ r.s[2] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, r.z[3:end] / r.z[2], rt2)
        @test r.z[2] * logdet(cholesky!(Hermitian(sol_mat, :U))) ≈ r.z[1] atol=tol rtol=tol
    end
end

function hypoperlogdettri3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + CO.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1, 0]
        A = T[0 1]
        b = T[0]
        G = SparseMatrixCSC(-one(T) * I, dim, 2)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        CO.smat_to_svec!(view(h, 3:dim), mat, rt2)
        cones = CO.Cone{T}[CO.HypoPerLogdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function hyporootdettri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + CO.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1]
        A = zeros(T, 0, 1)
        b = T[]
        G = Matrix{T}(-I, dim, 1)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        CO.smat_to_svec!(view(h, 2:dim), mat, rt2)
        cones = CO.Cone{T}[CO.HypoRootdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        CO.svec_to_smat!(sol_mat, r.s[2:end], rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ r.s[1] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, r.z[2:end] .* T(side), rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ -r.z[1] atol=tol rtol=tol
    end
end

function hyporootdettri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 4
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + CO.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = Matrix{T}(-I, dim, 1)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        CO.smat_to_svec!(view(h, 2:dim), mat, rt2)
        cones = CO.Cone{T}[CO.HypoRootdetTri{T, R}(dim, true)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ r.primal_obj atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        CO.svec_to_smat!(sol_mat, r.s[2:end] .* T(side), rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ -r.s[1] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, r.z[2:end], rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ r.z[1] atol=tol rtol=tol
    end
end

function hyporootdettri3(T; options...)
    # max u: u <= rootdet(W) where W is not full rank
    tol = eps(T) ^ 0.15
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + CO.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1]
        A = zeros(T, 0, 1)
        b = T[]
        G = SparseMatrixCSC(-one(T) * I, dim, 1)
        mat_half = T(0.2) * rand(R, side, side - 1)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        CO.smat_to_svec!(view(h, 2:dim), mat, rt2)
        cones = CO.Cone{T}[CO.HypoRootdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test r.x[1] ≈ zero(T) atol=tol rtol=tol
    end
end

function wsosinterpnonnegative1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps, _) = MU.interpolate(MU.Box{T}(-ones(T, 2), ones(T, 2)), 2, sample = false)
    DynamicPolynomials.@polyvar x y
    fn = x ^ 4 + x ^ 2 * y ^ 2 + 4 * y ^ 2 + 4

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[fn(pts[j, :]...) for j in 1:U]
    cones = CO.Cone{T}[CO.WSOSInterpNonnegative{T, T}(U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -T(4) atol=tol rtol=tol
    @test r.x[1] ≈ T(4) atol=tol rtol=tol
end

function wsosinterpnonnegative2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps, _) = MU.interpolate(MU.Box{T}(zeros(T, 2), fill(T(3), 2)), 2, sample = false)
    DynamicPolynomials.@polyvar x y
    fn = (x - 2) ^ 2 + (x * y - 3) ^ 2

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[fn(pts[j, :]...) for j in 1:U]
    cones = CO.Cone{T}[CO.WSOSInterpNonnegative{T, T}(U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ zero(T) atol=tol rtol=tol
    @test r.x[1] ≈ zero(T) atol=tol rtol=tol
end

function wsosinterpnonnegative3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps, _) = MU.interpolate(MU.Box{T}(zeros(T, 2), fill(T(3), 2)), 2, sample = false)
    DynamicPolynomials.@polyvar x y
    fn = (x - 2) ^ 2 + (x * y - 3) ^ 2

    c = T[fn(pts[j, :]...) for j in 1:U]
    A = ones(T, 1, U)
    b = T[1]
    G = Diagonal(-one(T) * I, U)
    h = zeros(T, U)
    cones = CO.Cone{T}[CO.WSOSInterpNonnegative{T, T}(U, Ps, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ zero(T) atol=tol rtol=tol
end

function wsosinterppossemideftri1(T; options...)
    # convexity parameter for (x + 1) ^ 2 * (x - 1) ^ 2
    tol = sqrt(sqrt(eps(T)))
    DynamicPolynomials.@polyvar x
    fn = (x + 1) ^ 2 * (x - 1) ^ 2
    (U, pts, Ps, _) = MU.interpolate(MU.Box{T}([-one(T)], [one(T)]), 1, sample = false)
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[H(pts[u, :]...) for u in 1:U]
    cones = CO.Cone{T}[CO.WSOSInterpPosSemidefTri{T}(1, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(4) atol=tol rtol=tol
    @test r.x[1] ≈ -T(4) atol=tol rtol=tol
end

function wsosinterppossemideftri2(T; options...)
    # convexity parameter for x[1] ^ 4 - 3 * x[2] ^ 2
    tol = sqrt(sqrt(eps(T)))
    n = 2
    DynamicPolynomials.@polyvar x[1:n]
    fn = x[1] ^ 4 - 3 * x[2] ^ 2
    (U, pts, Ps, _) = MU.interpolate(MU.FreeDomain{T}(n), 1, sample = false)
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = vcat(ones(T, U, 1), zeros(T, U, 1), ones(T, U, 1))
    h = T[H[i, j](pts[u, :]...) for i in 1:n for j in 1:i for u in 1:U]
    MU.vec_to_svec!(h, incr = U)
    cones = CO.Cone{T}[CO.WSOSInterpPosSemidefTri{T}(n, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(6) atol=tol rtol=tol
    @test r.x[1] ≈ -T(6) atol=tol rtol=tol
end

function wsosinterppossemideftri3(T; options...)
    # convexity parameter for sum(x .^ 6) - sum(x .^ 2)
    tol = sqrt(sqrt(eps(T)))
    n = 3
    DynamicPolynomials.@polyvar x[1:n]
    fn = sum(x .^ 4) - sum(x .^ 2)
    (U, pts, Ps, _) = MU.interpolate(MU.FreeDomain{T}(n), 2, sample = false)
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = vcat(ones(T, U, 1), zeros(T, U, 1), ones(T, U, 1), zeros(T, U, 1), zeros(T, U, 1), ones(T, U, 1))
    h = T[H[i, j](pts[u, :]...) for i in 1:n for j in 1:i for u in 1:U]
    MU.vec_to_svec!(h, incr = U)
    cones = CO.Cone{T}[CO.WSOSInterpPosSemidefTri{T}(n, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(2) atol=tol rtol=tol
    @test r.x[1] ≈ -T(2) atol=tol rtol=tol
end

function wsosinterpepinormeucl1(T; options...)
    # mint t(x) : t(x) ^ 2 >= x ^ 4 on [-1, 1] where t(x) a constant
    tol = sqrt(sqrt(eps(T)))
    DynamicPolynomials.@polyvar x
    fn = x ^ 2
    (U, pts, Ps, _) = MU.interpolate(MU.Box{T}([-one(T)], [one(T)]), 1, sample = false)
    @assert U == 3

    c = ones(T, U)
    A = T[1 -1 0; 1 0 -1]
    b = zeros(T, 2)
    G = vcat(-Matrix{T}(I, U, U), zeros(T, U, U))
    h = vcat(zeros(T, U), T[fn(pts[u, :]...) for u in 1:U])
    cones = CO.Cone{T}[CO.WSOSInterpEpiNormEucl{T}(2, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(U) atol=tol rtol=tol
    @test r.x ≈ ones(T, U) atol=tol rtol=tol
end

function wsosinterpepinormeucl2(T; options...)
    # min t(x) : t(x) ^ 2 >= x ^ 4 + (x - 1) ^ 2 on [-1, 1]^2 where t(x) a constant
    tol = sqrt(sqrt(eps(T)))
    DynamicPolynomials.@polyvar x
    fn1 = x ^ 2
    fn2 = (x - 1)
    (U, pts, Ps, _) = MU.interpolate(MU.Box{T}([-one(T)], [one(T)]), 1, sample = false)

    c = ones(T, U)
    A = T[1 -1 0; 1 0 -1]
    b = zeros(T, 2)
    G = vcat(-Matrix{T}(I, U, U), zeros(T, U, U), zeros(T, U, U))
    h = vcat(zeros(T, U), T[fn1(pts[u, :]...) for u in 1:U], T[fn2(pts[u, :]...) for u in 1:U])
    cones = CO.Cone{T}[CO.WSOSInterpEpiNormEucl{T}(3, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ sqrt(T(5)) * U atol=tol rtol=tol
    @test r.x ≈ fill(sqrt(T(5)), U) atol=tol rtol=tol
end

function wsosinterpepinormeucl3(T; options...)
    # max: w'f: 25x^4 >= f(x)^4 + 9x^4 on [-1, 1], soln is +/- 4x^2
    tol = sqrt(sqrt(eps(T)))
    DynamicPolynomials.@polyvar x
    # TODO when calc_w = true works with BigFloat, enable it
    (U, pts, Ps, _) = MU.interpolate(MU.Box{T}([-one(T)], [one(T)]), 1, sample = false, calc_w = false)
    w = fill(T(1 / 3), 3)
    fn1 = 5x^2
    fn2 = 3x^2

    c = -w
    A = zeros(T, 0, U)
    b = T[]
    G = vcat(spzeros(T, U, U), Diagonal(-one(T) * I, U), spzeros(T, U, U))
    h = vcat(T[fn1(pts[u, :]...) for u in 1:U], zeros(T, U), T[fn2(pts[u, :]...) for u in 1:U])
    cones = CO.Cone{T}[CO.WSOSInterpEpiNormEucl{T}(3, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -8 / 3 atol=tol rtol=tol
    fn_sol = 4x^2
    @test abs2.(r.x) ≈ abs2.([fn_sol(pts[u, :]...) for u in 1:U]) atol=tol rtol=tol
end

function primalinfeas1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0]
    A = T[1 1]
    b = [-T(2)]
    G = SparseMatrixCSC(-one(T) * I, 2, 2)
    h = zeros(T, 2)
    cones = CO.Cone{T}[CO.Nonnegative{T}(2)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function primalinfeas2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = T[]
    G = vcat(SparseMatrixCSC(-one(T) * I, 3, 3), Diagonal([one(T), one(T), -one(T)]))
    h = vcat(zeros(T, 3), one(T), one(T), -T(2))
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3), CO.Nonnegative{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function primalinfeas3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = zeros(T, 3)
    A = SparseMatrixCSC(-one(T) * I, 3, 3)
    b = [one(T), one(T), T(3)]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function dualinfeas1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, -1, 0]
    A = zeros(T, 0, 3)
    b = T[]
    G = repeat(SparseMatrixCSC(-one(T) * I, 3, 3), 2, 1)
    h = zeros(T, 6)
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(3), CO.EpiNormInf{T, T}(3, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :DualInfeasible
end

function dualinfeas2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0]
    A = zeros(T, 0, 2)
    b = T[]
    G = T[-1 0; 0 0; 0 -1]
    h = T[0, 1, 0]
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :DualInfeasible
end

function dualinfeas3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, 1, 0]
    A = zeros(T, 0, 4)
    b = T[]
    G = SparseMatrixCSC(-one(T) * I, 4, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :DualInfeasible
end
