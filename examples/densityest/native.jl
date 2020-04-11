#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each X_i in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    max prod_i f(X_i)
    ∫f = 1
    f ≥ 0
=#

include(joinpath(@__DIR__, "../common_native.jl"))
import DelimitedFiles

struct DensityEstNative{T <: Real} <: ExampleInstanceNative{T}
    dataset_name::Symbol
    num_obs::Int
    n::Int
    X::Matrix{T}
    deg::Int
    use_wsos::Bool # use WSOS cone formulation, else PSD formulation
    hypogeomean_obj::Bool # use geomean objective, else sum of logs objective
    use_hypogeomean::Bool # use hypogeomean cone if applicable, else hypoperlog formulation
end
function DensityEstNative{T}(
    dataset_name::Symbol,
    deg::Int,
    use_wsos::Bool,
    hypogeomean_obj::Bool,
    use_hypogeomean::Bool,
    ) where {T <: Real}
    X = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$dataset_name.txt"))
    X = convert(Matrix{T}, X)
    (num_obs, n) = size(X)
    return DensityEstNative{T}(dataset_name, num_obs, n, X, deg, use_wsos, hypogeomean_obj, use_hypogeomean)
end
function DensityEstNative{T}(
    num_obs::Int,
    n::Int,
    args...) where {T <: Real}
    X = randn(T, num_obs, n)
    return DensityEstNative{T}(:Random, num_obs, n, X, args...)
end

example_tests(::Type{<:DensityEstNative{<:BlasReal}}, ::MinimalInstances) = [
    ((5, 1, 2, true, true, true),),
    ((5, 1, 2, false, true, true),),
    ((5, 1, 2, true, false, true),),
    ((5, 1, 2, true, true, false),),
    ((:iris, 2, true, true, true),),
    ]
example_tests(::Type{DensityEstNative{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((50, 2, 2, true, true, true), options),
    ((50, 2, 2, false, true, true), options),
    ((50, 2, 2, true, false, true), options),
    ((50, 2, 2, true, true, false), options),
    ((100, 8, 2, true, true, true), options),
    ((100, 8, 2, false, true, true), options),
    ((100, 8, 2, true, false, true), options),
    ((100, 8, 2, true, true, false), options),
    ((250, 4, 6, true, true, true), options),
    ((250, 4, 6, false, true, true), options),
    ((250, 4, 6, true, false, true), options),
    ((250, 4, 6, true, true, false), options),
    ((:iris, 4, true, true, true), options),
    ((:iris, 5, true, true, true), options),
    ((:iris, 6, true, true, true), options),
    ((:iris, 4, false, true, true), options),
    ((:iris, 4, true, false, true), options),
    ((:iris, 4, true, true, false), options),
    ((:cancer, 4, true, true, true), options),
    ((:cancer, 4, false, true, true), options),
    ((:cancer, 4, true, false, true), options),
    ((:cancer, 4, true, true, false), options),
    ]
end
example_tests(::Type{DensityEstNative{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((:cancer, 6, true, true, true), options),
    ((:cancer, 6, false, true, true), options),
    ((:cancer, 6, true, false, true), options),
    ((:cancer, 6, true, true, false), options),
    ((400, 5, 6, true, true, true), options),
    ((400, 5, 6, false, true, true), options),
    ((400, 5, 6, true, false, true), options),
    ((400, 5, 6, true, true, false), options),
    ]
end

function build(inst::DensityEstNative{T}) where {T <: Real}
    (X, num_obs) = (inst.X, inst.num_obs)

    domain = ModelUtilities.Box{T}(-ones(T, inst.n), ones(T, inst.n))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(inst.deg + 1, 2)
    (U, pts, Ps, w) = ModelUtilities.interpolate(domain, halfdeg, calc_w = true)
    lagrange_polys = ModelUtilities.recover_lagrange_polys(pts, 2 * halfdeg)
    basis_evals = Matrix{T}(undef, num_obs, U)
    for i in 1:num_obs, j in 1:U
        basis_evals[i, j] = lagrange_polys[j](X[i, :])
    end

    cones = Cones.Cone{T}[]

    num_psd_vars = 0
    if inst.use_wsos
        # U variables
        h_poly = zeros(T, U)
        b_poly = T[]
        push!(cones, Cones.WSOSInterpNonnegative{T, T}(U, Ps))
    else
        # U polynomial coefficient variables plus PSD variables
        # there are length(Ps) new PSD variables, we will store them scaled, lower triangle, row-wise
        psd_var_list = Matrix{T}[]
        for i in eachindex(Ps)
            L = size(Ps[i], 2)
            dim = Cones.svec_length(L)
            num_psd_vars += dim
            push!(psd_var_list, zeros(T, U, dim))
            idx = 1
            # we will work with PSD vars with scaled off-diagonals
            # relevant columns (not rows) in A need to be scaled by sqrt(2) also
            for k in 1:L
                for l in 1:(k - 1)
                    psd_var_list[i][:, idx] = Ps[i][:, k] .* Ps[i][:, l] * sqrt(T(2))
                    idx += 1
                end
                psd_var_list[i][:, idx] = Ps[i][:, k] .* Ps[i][:, k]
                idx += 1
            end
            push!(cones, Cones.PosSemidefTri{T, T}(dim))
        end
        A_psd = hcat(psd_var_list...)
        b_poly = zeros(T, U)
        h_poly = zeros(T, num_psd_vars)
    end

    if inst.hypogeomean_obj
        num_hypo_vars = 1
        if inst.use_hypogeomean
            G_likl = [
                -one(T) zeros(T, 1, U)
                zeros(T, num_obs) -basis_evals
                ]
            h_likl = zeros(T, 1 + num_obs)
            push!(cones, Cones.HypoGeomean{T}(fill(inv(T(num_obs)), num_obs)))
            A_ext = zeros(T, 0, num_obs)
        else
            num_ext_geom_vars = 1 + num_obs
            h_likl = zeros(T,  3 * num_obs + 2)
            # order of variables is: hypograph vars, f(obs), psd_vars, geomean ext vars (y, z)
            G_likl = zeros(T, 3 * num_obs + 2, 2 + U + num_psd_vars + num_obs)
            # u - y <= 0
            G_likl[1, :] = vcat(one(T), zeros(T, U + num_psd_vars), -one(T), zeros(T, num_obs))
            push!(cones, Cones.Nonnegative{T}(1))
            # e'z >= 0
            G_likl[2, :] = vcat(zeros(T, 2 + U + num_psd_vars), -ones(T, num_obs))
            push!(cones, Cones.Nonnegative{T}(1))
            # f(x) <= y * log(z / y)
            row_offset = 3
            # number of columns before extended variables start
            ext_offset = 2 + U + num_psd_vars
            for i in 1:num_obs
                G_likl[row_offset, ext_offset + i] = -1
                G_likl[row_offset + 1, ext_offset] = -1
                G_likl[row_offset + 2, 2:(1 + U)] = -basis_evals[i, :]
                row_offset += 3
                push!(cones, Cones.HypoPerLog{T}(3))
            end
        end
    else
        num_hypo_vars = num_obs
        h_likl = zeros(T, 3 * num_obs)
        G_likl = zeros(T, 3 * num_obs, num_obs + U)
        offset = 1
        for i in 1:num_obs
            G_likl[offset + 2, (num_obs + 1):(num_obs + U)] = -basis_evals[i, :]
            h_likl[offset + 1] = 1
            G_likl[offset, i] = -1
            offset += 3
            push!(cones, Cones.HypoPerLog{T}(3))
        end
    end

    # extended formulation variables for hypogeomean cone are added after psd ones, so psd vars were already accounted for in hypogeomean_obj && !use_hypogeomean path
    if !inst.hypogeomean_obj || inst.use_hypogeomean
        G_likl = hcat(G_likl, zeros(T, size(G_likl, 1), num_psd_vars))
        num_ext_geom_vars = 0
    end
    c = zeros(T, num_hypo_vars + U + num_psd_vars + num_ext_geom_vars)
    @views c[1:num_hypo_vars] .= -1
    h = vcat(h_poly, h_likl)
    b = vcat(b_poly, one(T))

    if inst.use_wsos
        A = zeros(T, 1, num_hypo_vars + U + num_ext_geom_vars)
        A[1, num_hypo_vars .+ (1:U)] = w
        G = zeros(T, U + size(G_likl, 1), size(G_likl, 2))
        G[1:U, num_hypo_vars .+ (1:U)] = Diagonal(-I, U)
        G[(U + 1):end, :] = G_likl
    else
        A = zeros(T, U + 1, num_hypo_vars + U + num_psd_vars + num_ext_geom_vars)
        A[1:U, num_hypo_vars .+ (1:U)] = Diagonal(-I, U)
        A[1:U, (num_hypo_vars + U) .+ (1:num_psd_vars)] = A_psd
        A[U + 1, num_hypo_vars .+ (1:U)] = w
        G = zeros(T, num_psd_vars + size(G_likl, 1), size(G_likl, 2))
        G[1:num_psd_vars, (num_hypo_vars + U) .+ (1:num_psd_vars)] = Diagonal(-I, num_psd_vars)
        G[(num_psd_vars + 1):end, :] = G_likl
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

return DensityEstNative
