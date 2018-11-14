#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
const conemap_mpb_to_hypatia = Dict(
    :NonPos => Hypatia.Nonpositive,
    :NonNeg =>  Hypatia.Nonnegative,
    :SOC => Hypatia.EpiNormEucl,
    :SOCRotated => Hypatia.EpiPerSquare,
    :ExpPrimal => Hypatia.HypoPerLog,
    # :ExpDual => TODO
    :SDP => Hypatia.PosSemidef,
    :Power => Hypatia.HypoGeomean
)

const DimCones = Union{Type{Hypatia.Nonpositive}, Type{Hypatia.Nonnegative}, Type{Hypatia.EpiNormEucl}, Type{Hypatia.EpiPerSquare}, Type{Hypatia.PosSemidef}}
const ParametricCones =  Union{Type{Hypatia.HypoGeomean}}

function get_hypatia_cone(t::T, dim::Int) where T <: DimCones
    t(dim, false)
end
function get_hypatia_cone(t::Type{T}, ::Int) where T <: Hypatia.PrimitiveCone
    t()
end
function get_hypatia_cone(t::T, alphas::Vector{Float64}) where T <: ParametricCones
    t(alphas ./ sum(alphas), false)
end

function add_hypatia_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, idxs::UnitRange{Int})
    conetype = conemap_mpb_to_hypatia[conesym]
    conedim = length(idxs)
    push!(hypatia_cone.prmtvs, get_hypatia_cone(conetype, conedim))
    push!(hypatia_cone.idxs, idxs)
    hypatia_cone
end

function add_parametric_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, alphas::Vector{Float64}, idxs::UnitRange{Int})
    conetype = conemap_mpb_to_hypatia[conesym]
    conedim = length(idxs)
    push!(hypatia_cone.prmtvs, get_hypatia_cone(conetype, alphas))
    push!(hypatia_cone.idxs, idxs)
    hypatia_cone
end

function mpbcones_to_hypatiacones!(hypatia_cone::Hypatia.Cone, mpb_cones::Vector{Tuple{Symbol,Vector{Int}}}, parametric_refs::Vector{Int}, parameters::Vector{Vector{Float64}},offset::Int=0)
    power_cones_count = 0
    for (i, c) in enumerate(mpb_cones)
        c[1] in (:Zero, :Free) && continue
        smallest_ind = minimum(c[2])
        start_ind = offset + 1
        end_ind = offset + maximum(c[2]) - minimum(c[2]) + 1
        output_idxs = UnitRange{Int}(start_ind, end_ind)
        # output_idxs = (offset- smallest_ind + 1) .+ c[2]
        offset += length(c[2])
        if c[1] == :Power
            power_cones_count += 1
            alphas = parameters[parametric_refs[power_cones_count]]
            add_parametric_cone!(hypatia_cone, c[1], alphas, output_idxs)
        else
            add_hypatia_cone!(hypatia_cone, c[1], output_idxs)
        end
    end
    hypatia_cone
end

function mpbtohypatia(c_in::Vector{Float64},
    A_in::AbstractMatrix,
    b_in::Vector{Float64},
    con_cones::Vector{Tuple{Symbol,Vector{Int}}},
    var_cones::Vector{Tuple{Symbol,Vector{Int}}},
    sense::Symbol,
    con_power_refs::Vector{Int},
    var_power_refs::Vector{Int},
    power_alphas::Vector{Vector{Float64}},
    objoffset::Float64;
    dense::Bool=true
    )

    # dimension of x
    n = length(c_in)

    # count the number of "zero" constraints
    zero_constrs = 0
    cone_constrs = 0
    for (cone_type, inds) in con_cones
        if cone_type == :Zero
            zero_constrs += length(inds)
        else
            cone_constrs += length(inds)
        end
    end

    # count the number of cone variables
    cone_vars = 0
    zero_vars = 0
    cone_var_inds = Int[]
    zero_var_inds = Int[]
    zero_var_cones = Int[]
    cone_count = 0
    for (cone_type, inds) in var_cones
        cone_count += 1
        # TODO treat fixed variables better
        if cone_type == :Zero
            push!(zero_var_inds, inds...)
            push!(zero_var_cones, cone_count)
            zero_vars += length(inds)
        elseif cone_type != :Free
            cone_vars += length(inds)
            push!(cone_var_inds, inds...)
        end
    end
    @assert length(cone_var_inds) == cone_vars
    # variables that are fixed at zero count as constraints
    zero_constrs += zero_vars

    h = zeros(cone_constrs + cone_vars)
    b = zeros(zero_constrs)
    if dense
        A = zeros(zero_constrs, n)
        G = zeros(cone_constrs + cone_vars, n)
    else
        A = spzeros(zero_constrs, n)
        G = spzeros(cone_constrs + cone_vars, n)
    end

    # keep index of constraints in A and G
    i = 0
    j = 0
    # constraints are split among A and G
    for (cone_type, inds) in con_cones
        if cone_type == :Zero
            nexti = i + length(inds)
            out_inds = i+1:nexti
            A[out_inds, :] .= A_in[inds, :]
            b[out_inds] = b_in[inds]
            i = nexti
        else
            if cone_type == :Power
                inds .= [inds[end]; inds[1:end-1]]
            end
            nextj = j + length(inds)
            out_inds = j+1:nextj
            G[out_inds, :] .= A_in[inds, :]
            h[out_inds] .= b_in[inds]
            j = nextj
        end
    end
    # corner case, add variables fixed at zero as constraints TODO treat fixed variables better
    if zero_vars > 0
        fixed_var_ref = zero_constrs-zero_vars+1:zero_constrs
        @assert all(b[fixed_var_ref] .≈ 0.0)
        @assert all(A[fixed_var_ref, zero_var_inds] .≈ 0.0)
        @assert length(zero_var_inds) == zero_vars
        for ind in zero_var_inds
            i += 1
            A[i, ind] = 1.0
        end
    end

    # append G
    G[cone_constrs+1:end, :] .= Matrix(-1I, n, n)[cone_var_inds, :]
    # reorder any variables in the power cone
    for (cone_type, inds) in var_cones
        if cone_type == :Power
            out_inds = [inds[end]; inds[1:end-1]]
            A[:, inds] .= A[:, out_inds]
            G[:, inds] .= G[:, out_inds]
            c_in[inds] .= c_in[out_inds]
        end
    end

    # prepare cones
    hypatia_cone = Hypatia.Cone()
    mpbcones_to_hypatiacones!(hypatia_cone, con_cones, con_power_refs, power_alphas)
    mpbcones_to_hypatiacones!(hypatia_cone, var_cones, var_power_refs, power_alphas, cone_constrs)

    return (c_in, A, b, G, h, hypatia_cone)
end

function cbftohypatia(dat::CBFData; remove_ints::Bool=false, dense::Bool=true)
    if !isempty(dat.intlist)
        @warn "ignoring integrality constraints"
    end
    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true, roundints=true)
    if dat.sense == :Max
        c .*= -1.0
    end
    if remove_ints
        (c, A, b, con_cones, var_cones, vartypes) = remove_ints_in_nonlinear_cones(c, A, b, con_cones, var_cones, vartypes)
    end
    (c, A, b, G, h, hypatia_cone) = mpbtohypatia(c, A, b, con_cones, var_cones, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset, dense = dense)
    (c, A, b, G, h, hypatia_cone, dat.objoffset)
end
