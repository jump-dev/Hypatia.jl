#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
import Random
using LinearAlgebra
using SparseArrays
import DynamicPolynomials
import GenericSVD.svdvals
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
    cones = CO.Cone{T}[CO.Nonnegative{T}(1, false)]

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
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(I, q, n)
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
    cones = CO.Cone{T}[CO.Nonpositive{T}(q)]

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

function orthant1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    h = zeros(T, q)

    # nonnegative cone
    G = SparseMatrixCSC(-one(T) * I, q, n)
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]
    rnn = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = one(T), options...)
    @test rnn.status == :Optimal

    # nonpositive cone
    G = SparseMatrixCSC(one(T) * I, q, n)
    cones = CO.Cone{T}[CO.Nonpositive{T}(q)]
    rnp = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = one(T), options...)
    @test rnp.status == :Optimal

    @test rnp.primal_obj ≈ rnn.primal_obj atol=tol rtol=tol
end

function orthant2(T; options...)
    tol = 2 * sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = rand(T, q, n) - Matrix(T(2) * I, q, n)
    h = G * ones(T, n)

    # use dual barrier
    cones = CO.Cone{T}[CO.Nonnegative{T}(q, true)]
    r1 = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r1.status == :Optimal

    # use primal barrier
    cones = CO.Cone{T}[CO.Nonnegative{T}(q, false)]
    r2 = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=tol rtol=tol
end

function orthant3(T; options...)
    tol = 2 * sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = Diagonal(one(T) * I, n)
    h = zeros(T, q)

    # use dual barrier
    cones = CO.Cone{T}[CO.Nonpositive{T}(q, true)]
    r1 = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r1.status == :Optimal

    # use primal barrier
    cones = CO.Cone{T}[CO.Nonpositive{T}(q, false)]
    r2 = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=tol rtol=tol
end

function orthant4(T; options...)
    tol = 4 * sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = rand(T, q, n) - Matrix(T(2) * I, q, n)
    h = vec(sum(G, dims = 2))

    # mixture of nonnegative and nonpositive cones
    cones = CO.Cone{T}[CO.Nonnegative{T}(4, false), CO.Nonnegative{T}(6, true)]
    r1 = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r1.status == :Optimal

    # only nonnegative cone
    cones = CO.Cone{T}[CO.Nonnegative{T}(10, false)]
    r2 = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=tol rtol=tol
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
    tol = sqrt(sqrt(eps(T)))
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
    A = T[1 0 0; 0 1 0]
    b = T[1, Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormEucl{T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -Trt2 atol=tol rtol=tol
        @test r.x ≈ [1, Tirt2, Tirt2] atol=tol rtol=tol
        @test r.y ≈ [Trt2, 0] atol=tol rtol=tol
    end
end

function epinormeucl2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, -1]
    A = T[1 0 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormEucl{T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epipersquare1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, -1, -1]
    A = T[1 0 0 0; 0 1 0 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 4, 4)
    h = zeros(T, 4)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiPerSquare{T}(4, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -sqrt(T(2)) atol=tol rtol=tol
        @test r.x[3:4] ≈ [1, 1] / sqrt(T(2)) atol=tol rtol=tol
    end
end

function epipersquare2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, 0, -1]
    A = T[1 0 0; 0 1 0]
    b = T[Tirt2 / 2, Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiPerSquare{T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = -one(T), options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -Tirt2 - 1 atol=tol rtol=tol
        @test r.x[2] ≈ Tirt2 atol=tol rtol=tol
    end
end

function epipersquare3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, -1, -1]
    A = T[1 0 0 0]
    b = T[0]
    G = SparseMatrixCSC(-one(T) * I, 4, 4)
    h = zeros(T, 4)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiPerSquare{T}(4, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = zero(T), options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function possemideftri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    Trt2i = inv(Trt2)
    c = T[0, -1, 0]
    A = T[1 0 0; 0 0 1]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        if is_dual
            @test r.primal_obj ≈ -Trt2 atol=tol rtol=tol
            @test r.x[2] ≈ Trt2 atol=tol rtol=tol
        else
            @test r.primal_obj ≈ -Trt2i atol=tol rtol=tol
            @test r.x[2] ≈ Trt2i atol=tol rtol=tol
        end
    end
end

function possemideftri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 1]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function possemideftricomplex1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    c = T[1, 0, 0, 1]
    A = T[0 0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 atol=tol rtol=tol
    @test r.x ≈ [1, 0, 1, 1] atol=tol rtol=tol
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
        @test r.x[3] ≈ (is_dual ? -sqrt(T(2)) : -inv(sqrt(T(2))))
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
        @test norm(r.x[3:4]) ≈ (is_dual ? sqrt(T(2)) : inv(sqrt(T(2))))
        @test r.x[3:4] ≈ (is_dual ? ones(T, 2) : fill(inv(T(2)), 2))
        @test r.x[1:2] ≈ [1, 0.5] atol=tol rtol=tol
    end
end

function power3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = vcat(ones(T, l), zeros(T, 2))
    A = T[zeros(T, 1, l) one(T) zero(T); zeros(T, 1, l) zero(T) one(T)]
    G = SparseMatrixCSC(-one(T) * I, l + 2, l + 2)
    h = zeros(T, l + 2)

    for is_dual in (true, false)
        b = [one(T), zero(T)]
        cones = CO.Cone{T}[CO.Power{T}(fill(inv(T(l)), l), 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 1 : T(l)) atol=tol rtol=tol
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
            @views CO.vec_copy_to!(S[:], r.s[2:end])
            prim_svdvals = svdvals(S)
            Z = similar(S)
            @views CO.vec_copy_to!(Z[:], r.z[2:end])
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
        CO.vec_copy_to!(c, -mat[:])
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
    for is_complex in (false, true), (Xn, Xm) in ((1, 1), (1, 3), (2, 2))
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

function hypoperlogdettri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 4
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + div(side * (side + 1), 2))
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
        CO.svec_to_smat!(sol_mat, r.s[3:end], rt2)
        @test r.s[2] * logdet(Hermitian(sol_mat, :U) / r.s[2]) ≈ r.s[1] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, -r.z[3:end], rt2)
        @test r.z[1] * (logdet(Hermitian(sol_mat, :U) / r.z[1]) + T(side)) ≈ r.z[2] atol=tol rtol=tol
    end
end

function hypoperlogdettri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + div(side * (side + 1), 2))
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
        CO.svec_to_smat!(sol_mat, -r.s[3:end], rt2)
        @test r.s[1] * (logdet(Hermitian(sol_mat, :U) / r.s[1]) + T(side)) ≈ r.s[2] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, r.z[3:end], rt2)
        @test r.z[2] * logdet(Hermitian(sol_mat, :U) / r.z[2]) ≈ r.z[1] atol=tol rtol=tol
    end
end

function hypoperlogdettri3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + div(side * (side + 1), 2))
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
        dim = (is_complex ? 1 + side^2 : 1 + div(side * (side + 1), 2))
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
        @test det(Hermitian(sol_mat, :U)) ^ inv(T(side)) ≈ r.s[1] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, r.z[2:end] .* T(side), rt2)
        @test det(Hermitian(sol_mat, :U)) ^ inv(T(side)) ≈ -r.z[1] atol=tol rtol=tol
    end
end

function hyporootdettri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 4
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + div(side * (side + 1), 2))
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
        @test det(Hermitian(sol_mat, :U)) ^ inv(T(side)) ≈ -r.s[1] atol=tol rtol=tol
        CO.svec_to_smat!(sol_mat, r.z[2:end], rt2)
        @test det(Hermitian(sol_mat, :U)) ^ inv(T(side)) ≈ r.z[1] atol=tol rtol=tol
    end
end

function hyporootdettri3(T; options...)
    # max u: u <= rootdet(W) where W is not full rank
    tol = 5 * sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + div(side * (side + 1), 2))
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

function epiperexp1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 5
    c = vcat(zero(T), -ones(T, l))
    A = hcat(one(T), zeros(T, 1, l))
    b = T[1]
    G = [-one(T) zeros(T, 1, l); zeros(T, 1, l + 1); zeros(T, l, 1) sparse(-one(T) * I, l, l)]
    h = zeros(T, l + 2)
    cones = CO.Cone{T}[CO.EpiPerExp{T}(l + 2)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=tol rtol=tol
    @test r.s[2] ≈ 0 atol=tol rtol=tol
    @test r.s[1] ≈ 1 atol=tol rtol=tol
end

function epiperexp2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 5
    c = vcat(zero(T), -ones(T, l))
    A = hcat(one(T), zeros(T, 1, l))
    b = T[1]
    G = [-one(T) spzeros(T, 1, l); spzeros(T, 1, l + 1); spzeros(T, l, 1) sparse(-one(T) * I, l, l)]
    h = zeros(T, l + 2); h[2] = 1
    cones = CO.Cone{T}[CO.EpiPerExp{T}(l + 2)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=tol rtol=tol
    @test r.s[2] ≈ 1 atol=tol rtol=tol
    @test r.s[2] * sum(exp, r.s[3:end] / r.s[2]) ≈ r.s[1] atol=tol rtol=tol
end

function wsospolyinterp1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, P0, PWts, _) = MU.interpolate(MU.Box{T}(-ones(T, 2), ones(T, 2)), 2, sample = false)
    DynamicPolynomials.@polyvar x y
    fn = x ^ 4 + x ^ 2 * y ^ 2 + 4 * y ^ 2 + 4

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[fn(pts[j, :]...) for j in 1:U]
    cones = CO.Cone{T}[CO.WSOSPolyInterp{T, T}(U, [P0, PWts...])]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -T(4) atol=tol rtol=tol
    @test r.x[1] ≈ T(4) atol=tol rtol=tol
end

function wsospolyinterp2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, P0, PWts, _) = MU.interpolate(MU.Box{T}(zeros(T, 2), fill(T(3), 2)), 2, sample = false)
    DynamicPolynomials.@polyvar x y
    fn = (x - 2) ^ 2 + (x * y - 3) ^ 2

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[fn(pts[j, :]...) for j in 1:U]
    cones = CO.Cone{T}[CO.WSOSPolyInterp{T, T}(U, [P0, PWts...])]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ zero(T) atol=tol rtol=tol
    @test r.x[1] ≈ zero(T) atol=tol rtol=tol
end

function wsospolyinterp3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, P0, PWts, _) = MU.interpolate(MU.Box{T}(zeros(T, 2), fill(T(3), 2)), 2, sample = false)
    DynamicPolynomials.@polyvar x y
    fn = (x - 2) ^ 2 + (x * y - 3) ^ 2

    c = T[fn(pts[j, :]...) for j in 1:U]
    A = ones(T, 1, U)
    b = T[1]
    G = Diagonal(-one(T) * I, U)
    h = zeros(T, U)
    cones = CO.Cone{T}[CO.WSOSPolyInterp{T, T}(U, [P0, PWts...], true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ zero(T) atol=tol rtol=tol
end

function wsospolyinterpmat1(T; options...)
    # convexity parameter for (x + 1) ^ 2 * (x - 1) ^ 2
    tol = sqrt(sqrt(eps(T)))
    DynamicPolynomials.@polyvar x
    fn = (x + 1) ^ 2 * (x - 1) ^ 2
    # the half-degree is div(4 - 2, 2) = 1
    (U, pts, P0, PWts, _) = MU.interpolate(MU.Box{T}([-one(T)], [one(T)]), 1, sample = false)
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    # the "1" polynomial
    G = ones(T, U, 1)
    # dimension of the Hessian is 1x1
    h = T[H(pts[u, :]...) for u in 1:U]
    cones = CO.Cone{T}[CO.WSOSPolyInterpMat{T}(1, U, [P0, PWts...])]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(4) atol=tol rtol=tol
    @test r.x[1] ≈ -T(4) atol=tol rtol=tol
end

function wsospolyinterpmat2(T; options...)
    # convexity parameter for x[1] ^ 4 - 3 * x[2] ^ 2
    tol = sqrt(sqrt(eps(T)))
    n = 2
    DynamicPolynomials.@polyvar x[1:n]
    fn = x[1] ^ 4 - 3 * x[2] ^ 2
    # the half-degree is div(4 - 2, 2) = 1
    (U, pts, P0, _, _) = MU.interpolate(MU.FreeDomain{T}(n), 1, sample = false)
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    # the "1" polynomial on the diagonal
    G = vcat(ones(T, U, 1), zeros(T, U, 1), ones(T, U, 1))
    h = T[H[i, j](pts[u, :]...) * (i == j ? 1 : sqrt(T(2))) for i in 1:n for j in 1:i for u in 1:U]
    cones = CO.Cone{T}[CO.WSOSPolyInterpMat{T}(n, U, [P0])]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(6) atol=tol rtol=tol
    @test r.x[1] ≈ -T(6) atol=tol rtol=tol
end

function wsospolyinterpmat3(T; options...)
    # convexity parameter for sum(x .^ 6) - sum(x .^ 2)
    tol = sqrt(sqrt(eps(T)))
    n = 3
    DynamicPolynomials.@polyvar x[1:n]
    fn = sum(x .^ 4) - sum(x .^ 2)
    # half-degree is div(6 - 2, 2) = 2
    (U, pts, P0, _, _) = MU.interpolate(MU.FreeDomain{T}(n), 2, sample = false)
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    # the "1" polynomial on the diagonal
    G = vcat(ones(T, U, 1), zeros(T, U, 1), ones(T, U, 1), zeros(T, U, 1), zeros(T, U, 1), ones(T, U, 1))
    h = T[H[i, j](pts[u, :]...) * (i == j ? 1 : sqrt(T(2))) for i in 1:n for j in 1:i for u in 1:U]
    cones = CO.Cone{T}[CO.WSOSPolyInterpMat{T}(n, U, [P0])]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(2) atol=tol rtol=tol
    @test r.x[1] ≈ -T(2) atol=tol rtol=tol
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
    cones = CO.Cone{T}[CO.EpiNormInf{T}(3), CO.EpiNormInf{T}(3, true)]

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
