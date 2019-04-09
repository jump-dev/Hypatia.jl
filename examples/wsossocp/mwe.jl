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
    rand_polys = [dot(polys[:, i], lagrange_polys) for i in 1:npoly]

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
        JuMP.@constraint(model, matrix_condition in wsos_cone_mat)
    else
        cone = HYP.WSOSPolyInterpSOCCone(vec_length, U, [P0])
        JuMP.@constraint(model, vcat(f, [polys[:, i] for i in 1:npoly]...) in cone)
    end
    JuMP.optimize!(model)





    # now impose opposite condition and check
    sol = JuMP.value.(f)
    if !use_soc
        cone = HYP.WSOSPolyInterpSOCCone(vec_length, U, [P0])
        JuMP.@constraint(model, vcat(sol, [polys[:, i] for i in 1:npoly]...) in cone)
    else
        matrix_condition = GenericAffExpr{Float64,VariableRef}[]
        push!(matrix_condition, sol...)
        for i in 2:vec_length
            push!(matrix_condition, rt2 * polys[:, i - 1]...)
            for j in 2:(i - 1)
                push!(matrix_condition, zeros(U)...)
            end
            push!(matrix_condition, sol...)
        end
        wsos_cone_mat = HYP.WSOSPolyInterpMatCone(vec_length, U, [P0])
        JuMP.@constraint(model, matrix_condition in wsos_cone_mat)
    end
    JuMP.optimize!(model)





    return model
end
