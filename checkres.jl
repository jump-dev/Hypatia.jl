using IterativeSolvers, CSV
using DelimitedFiles
using LinearAlgebra
using Random


LHS = CSV.read("lhs.csv", header = false);
LHS = convert(Matrix, LHS);
i = 3
b = CSV.read("rhs.csv", header = false);
b = convert(Matrix, b);
b = b[:, i];
prevsol = CSV.read("prevsol.csv", header = false);
prevsol = convert(Matrix, prevsol);
prevsol = prevsol[:, i];
A = Symmetric(LHS, :L);

# open("res_init.csv", "a") do io
#     println("")
#     println(io, "tol,gmresres,minresres,gmresiters,minresites,gmrescon,minrescon")
#     for tol in -1:-1:-12
#         x0 = copy(prevsol)
#         x_g, log_g = gmres!(x0, A, b, restart = size(A, 2), tol = 10.0^tol, log = true)
#         x0 = copy(prevsol)
#         x_m, log_m = minres!(x0, A, b, tol = 10.0^tol, log = true, maxiter = 500)
#         println(io, "$(tol),$(norm(b - A * x_g)),$(norm(b - A * x_m)),$(log_g.iters),$(log_m.iters),$(log_g.isconverged),$(log_m.isconverged)")
#     end
# end

Random.seed!(1)
sol = A \ b
noise_dir = randn(size(A, 1))
noise_dir ./= norm(noise_dir)
open("pert.csv", "a") do io
    println(io, "pert,gmresres,minresres,gmresiters,minresites,gmrescon,minrescon")
    for pert in -16:0
        init = sol .+ 10.0^pert * noise_dir
        x0 = copy(init)
        x_g, log_g = gmres!(init, A, b, restart = size(A, 2), tol = 1e-8, log = true)
        x_m, log_m = minres!(x0, A, b, rtol = 1e-8, atol = 1e-8, log = true, maxiter = 500)
        println(io, "$(pert),$(norm(b - A * x_g)),$(norm(b - A * x_m)),$(log_g.iters),$(log_m.iters),$(log_g.isconverged),$(log_m.isconverged)")
    end
end
