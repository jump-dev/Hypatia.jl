using BenchmarkTools, SparseArrays, LinearAlgebra, Printf

for num_cones in [5000], cone_dim in [10]
    println("num_cones = $num_cones and cone_dim = $cone_dim")
    num_cones == 5000 && cone_dim >= 5000 && continue
    dist = rand()
    grad = randn(cone_dim)
    J = Diagonal(vcat(-1, ones(cone_dim - 1)))
    hess = J / dist + grad * grad'
    hess_expanded = sparse([
        J / dist  grad;
        grad'     -1;
        ])
    id = Matrix(I, num_cones, num_cones)
    dense_lhs = sparse(kron(id, hess))
    sparse_lhs = sparse(kron(id, hess_expanded))

    # @time ldlt(dense_lhs)
    # @time ldlt(sparse_lhs)

    rhs = randn(cone_dim)
    dense_rhs = repeat(rhs, num_cones)
    sparse_rhs = repeat(vcat(rhs, 0), num_cones)

    @time sol_dense = dense_lhs \ dense_rhs
    @time sol_sparse = sparse_lhs \ sparse_rhs

    @assert isapprox(sol_dense[1:cone_dim], sol_sparse[1:cone_dim])
end
