function JuMP_polysoc_envelope(; use_soc = false)
    Random.seed!(1)
    n = 1
    dom = MU.FreeDomain(n)
    DP.@polyvar x[1:n]
    d = 2
    (U, pts, P0, _, w) = MU.interpolate(dom, d, sample = false, calc_w = true)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2d)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 400, tol_feas = 1e-12, tol_abs_opt = 1e-12, tol_rel_opt = 1e-12))
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))

    vec_length = 5
    npoly = vec_length - 1
    LDegs = size(P0, 2)
    polys = P0[:, 1:LDegs] * rand(-9:9, LDegs, npoly)

    if !use_soc
        matrix_condition = GenericAffExpr{Float64,VariableRef}[]
        push!(matrix_condition, f...)
        for i in 2:vec_length
            push!(matrix_condition, rt2 * polys[:, i - 1]...)
            for j in 2:(i - 1)
                push!(matrix_condition, zeros(U)...)
            end
            push!(matrix_condition, f...)
        end
        wsos_cone_mat = HYP.WSOSPolyInterpMatCone(vec_length, U, [P0])
        sqrconstr = JuMP.@constraint(model, matrix_condition in wsos_cone_mat)
    else
        cone = HYP.WSOSPolyInterpSOCCone(vec_length, U, [P0])
        sqrconstr = JuMP.@constraint(model, vcat(f, [polys[u, i] for i in 1:npoly for u in 1:U]...) in cone)
    end
    JuMP.optimize!(model)



    dummy_cone = HYP.Cones.WSOSPolyInterpSOC(vec_length, U, [P0])
    dual_soln = JuMP.dual(sqrconstr)
    dummy_point = zeros(vec_length * U)
    ind_mat = 0
    ind_vec = 0
    for i in 1:vec_length, j in 1:i, u in 1:U
        ind_mat += 1
        if j == 1
            ind_vec += 1
            fact = (i == 1 ? 1.0 : inv(sqrt(2)))
            dummy_point[ind_vec] = dual_soln[ind_mat] * fact
        end
    end
    dummy_cone.point = dummy_point
    incone = HYP.Cones.check_in_cone(dummy_cone)
    @show incone

    lambda1 = P0' * Diagonal(dual_soln[1:U]) * P0
    @show isposdef(Symmetric(lambda1))
    @show eigen(Symmetric(lambda1)).values

    dummy_cone_mat = HYP.Cones.WSOSPolyInterpMat(vec_length, U, [P0])
    dummy_cone_mat.point = dual_soln
    incone_mat = HYP.Cones.check_in_cone(dummy_cone_mat)
    @show incone_mat






    # now impose opposite condition and check
    # sol = JuMP.value.(f)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 400, tol_feas = 1e-12, tol_abs_opt = 1e-12, tol_rel_opt = 1e-12))
    # dualsol = JuMP.dual(sqrconstr)
    # lambda1 = P0' * Diagonal(dualsol[1:U]) * P0
    # @assert isposdef(Symmetric(lambda1))
    # if !use_soc
    #     cone = HYP.WSOSPolyInterpSOCCone(vec_length, U, [P0])
    #     JuMP.@constraint(model, vcat(sol, [polys[:, i] for i in 1:npoly]...) in cone)
    # else
    #     matrix_condition = GenericAffExpr{Float64,VariableRef}[]
    #     push!(matrix_condition, sol...)
    #     for i in 2:vec_length
    #         push!(matrix_condition, rt2 * polys[:, i - 1]...)
    #         for j in 2:(i - 1)
    #             push!(matrix_condition, zeros(U)...)
    #         end
    #         push!(matrix_condition, sol...)
    #     end
    #     wsos_cone_mat = HYP.WSOSPolyInterpMatCone(vec_length, U, [P0])
    #     JuMP.@constraint(model, matrix_condition in wsos_cone_mat)
    # end
    # JuMP.optimize!(model)





    return model
end


function jordan_mul(x, y)
    res = [dot(x, y); x[end] .* y[1:(end - 1)] + y[end] .* x[1:(end - 1)]]
    return res
end
