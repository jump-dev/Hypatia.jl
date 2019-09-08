using BenchmarkTools, SparseArrays, LinearAlgebra

num_threads = 8
println("using $num_threads threads")
ENV["OMP_NUM_THREADS"] = num_threads
BLAS.set_num_threads(num_threads)


using Pardiso


# for num_threads in 8:8


    function testsolve()

        for n in 5000:5000:100000, p in [0.001, 0.01, 0.05, 0.1]
            println()
            println("n = $n, p = $p")

            A = sprandn(n, n, p)
            A = A + eps() * I
            B = A * rand(n, 2)

            X = zeros(n, 2)
            ps = PardisoSolver()
            # Pardiso.set_matrixtype!(ps, -2)
            # don't ignore the parameters we set
            # Pardiso.set_iparm!(ps, 1, 1)
            # use transpose
            # Pardiso.set_iparm!(ps, 12, 1)
            println("pardiso")
            t = @timed pardiso(ps, X, A, B) # for Pardiso, need to transpose, for solve!, don't
            println("time: ", t[2], "mem: ", t[3])
            flush(stdout)

            println("suitesparse")
            t = @timed ldiv!(lu(A), B)
            println("time: ", t[2], "mem: ", t[3])
            flush(stdout)

            println()
            println("norm:")
            println(norm(X - B))

        end
    end

    testsolve()

# end
