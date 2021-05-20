#=
experiment design optimizes a function of the information matrix

adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.5

minimize    f(V × Diagonal(x) × V')
subject to  x ≥ 0
            Σ x = k
            A x = b
where variable x ∈ ℝᵖ is the frequency of each experiment, k is the number of
experiments to run (let k = p), the columns of V ∈ ℝ^(q × p) correspond to each
experiment (let q ≈ p/2), and f is a convex spectral function
=#

struct ExperimentDesignJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    p::Int
    ssf::Cones.SepSpectralFun
end

function build(inst::ExperimentDesignJuMP{T}) where {T <: Float64}
    p = inst.p
    @assert p >= 2
    q = div(p, 2)
    V = randn(T, q, p)
    A = randn(T, round(Int, sqrt(p - 1)), p)
    b = sum(A, dims = 2)

    model = JuMP.Model()
    JuMP.@variable(model, x[1:p] >= 0)
    JuMP.@constraint(model, sum(x) == p)
    JuMP.@constraint(model, A * x .== b)

    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)

    vec_dim = Cones.svec_length(q)
    Q = V * diagm(x) * V' # information matrix
    Q_vec = zeros(JuMP.AffExpr, vec_dim)
    Cones.smat_to_svec!(Q_vec, Q, sqrt(T(2)))
    f_cone = Hypatia.EpiPerSepSpectralCone{T}(inst.ssf, Cones.MatrixCSqr{T, T}, q)
    JuMP.@constraint(model, vcat(epi, 1, Q_vec) in f_cone)

    return model
end
