#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

real_types = [Float64, Float32, BigFloat]
tf = [true, false]
seeds = 1:2

function solve_formulation(d)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5, time_limit = 600)
    t = @timed SO.solve(solver)
    r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
    dimx = size(d.G, 2)
    dimy = size(d.A, 1)
    dimz = size(d.G, 1)
    return (solver, r, dimx, dimy, dimz, t)
end

# densityest

# compile run
for T in real_types, use_wsos in tf
    d = densityest(T, 10, 4, 2, use_wsos = use_wsos)
end
# run
n_range = [1, 2]
deg_range = [4, 6]
io = open("densityest.csv", "w")
println(io, "usewsos,real,seed,n,deg,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
for use_wsos in tf n in n_range, deg in d_range, T in real_types, seed in seeds
    Random.seed!(seed)
    d = densityest(T, 200, n, deg, use_wsos = use_wsos)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
    println(io, "$use_wsos,$T,$seed,$n,$deg,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
        "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
        "$(solver.y_feas),$(solver.z_feas)"
        )
    end
end
close(io)

# portfolio

# # compile run
# for use_ball in tf
#     d = portfolio(4, risk_measures = [:l1, :linf], use_l1ball = use_ball, use_linfball = use_ball)
#     (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
# end
# # run
# n_range = [100, 200, 400, 600]
# io = open("portfoliol1.csv", "w")
# println(io, "usel1ball,seed,n,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
# for n in n_range, seed in seeds, use_ball in tf
#     Random.seed!(seed)
#     d = portfolio(T, n, risk_measures = [:l1, :linf], use_l1ball = use_ball, use_linfball = use_ball)
#     (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
#     println(io, "$use_ball,$seed,$n,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
#         "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
#         "$(solver.y_feas),$(solver.z_feas)"
#         )
# end
# close(io)

# expdesign

# compile run
nmax = 5
(p, q, n) = (3, 2, nmax)
for T in real_types, use_logdet in tf, use_sumlog in tf
    d = expdesign(T, q, p, n, nmax, use_logdet = use_logdet, use_sumlog = use_sumlog)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
end
# run
io = open("expdesign.csv", "w")
println(io, "uselogdet,use_sumlog,real,seed,q,p,n,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
for q in q_range, T in real_types, seed in seeds, use_logdet in [false], use_sumlog in tf
    p = 2 * q
    n = 2 * q
    Random.seed!(seed)
    d = expdesign(T, q, p, n, nmax, use_logdet = use_logdet, use_sumlog = use_sumlog)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
    println(io, "$use_logdet,$use_sumlog,$T,$seed,$q,$p,$n,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
        "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
        "$(solver.y_feas),$(solver.z_feas)"
        )
end
close(io)

# matrix completion
n_range = [10]

# compile run
for T in real_types, use_geomean in tf
    d = matrixcompletion(5, 5, use_geomean = use_geomean, T = T)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
end
# run
io = open("matrixcopletion.csv", "w")
println(io, "use3dim,real,seed,m,n,unknown,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
for n in n_range, T in real_types, seed in seeds, use_geomean in tf
    m = n + 10
    num_known = round(Int, 0.1 * m * n)
    Random.seed!(seed)
    d = matrixcompletion(T, m, n, num_known = num_known, use_geomean = use_geomean)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
    println(io, "$use_geomean,$T,$seed,$m,$n,$num_known,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
        "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
        "$(solver.y_feas),$(solver.z_feas)"
        )
end
close(io)

# sparse PCA
p_range = [50, 100]
k_range = [5, 10, 20]
# compile run
for T in real_types, use_l1ball in tf
    d = sparsepca(T, 3, 3, 3, use_l1ball = use_l1ball)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
end
# run
io = open("sparsepca.csv", "w")
println(io, "usel1ball,real,seed,p,k,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
for p in p_range, k in k_range, T in real_types, seed in seeds, use_l1ball in tf
    Random.seed!(seed)
    d = sparsepca(T, p, p, k, use_l1ball = use_l1ball)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
    println(io, "$use_l1ball,$T,$seed,$p,$k,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
        "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
        "$(solver.y_feas),$(solver.z_feas)"
        )
end
close(io)

# polymin
n_range = [2, 3, 4]
halfdeg_range = [3, 4, 5]
# compile run
for use_wsos in tf
    d = polyminreal(:random, 2, use_wsos = use_wsos, n = 2)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
end
# compile
io = open("polyminreal.csv", "w")
println(io, "usewsos,seed,n,halfdeg,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
for n in n_range, halfdeg in halfdeg_range, use_wsos in tf, seed in seeds
    Random.seed!(seed)
    d = polyminreal(:random, halfdeg, use_wsos = use_wsos, n = n)
    (solver, r, dimx, dimy, dimz, t) = solve_formulation(d)
    println(io, "$use_wsos,$seed,$n,$halfdeg,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
        "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
        "$(solver.y_feas),$(solver.z_feas)"
        )
end
close(io)
