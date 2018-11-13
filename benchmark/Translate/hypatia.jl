#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
const conemap_mpb_to_hypatia = Dict(
    :NonPos => Hypatia.Nonpositive,
    :NonNeg =>  Hypatia.Nonnegative,
    :SOC => Hypatia.EpiNormEucl,
    :SOCRotated => Hypatia.EpiPerSquare,
    :ExpPrimal => Hypatia.HypoPerLog,
    # :ExpDual => "EXP*"
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
# function add_hypatia_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, idxs::UnitRange{Int})
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

function mbgtohypatia(c_in::Vector{Float64},
    A_in::AbstractMatrix,
    b_in::Vector{Float64},
    con_cones::Vector{Tuple{Symbol,Vector{Int}}},
    var_cones::Vector{Tuple{Symbol,Vector{Int}}},
    vartypes::Vector{Symbol},
    sense::Symbol,
    con_power_refs::Vector{Int},
    var_power_refs::Vector{Int},
    power_alphas::Vector{Vector{Float64}},
    objoffset::Float64;
    dense::Bool=true
    )

    # cannot do integer variables yet
    for v in vartypes
        if v != :Cont
            error("We cannot handle binary or integer variables yet.")
        end
    end

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
        elseif cone_type == :ExpPrimal
            # take out if this ever happens
            error("We didn't know CBF allows variables in exponential cones.")
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
    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true, roundints=true)
    (dat.sense == :Max) && (c .*= -1.0)
    if remove_ints
        (c, A, b, con_cones, var_cones, vartypes) = remove_ints_in_nonlinear_cones(c, A, b, con_cones, var_cones, vartypes)
    end
    (c, A, b, G, h, hypatia_cone) = mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset, dense = dense)
    (c, A, b, G, h, hypatia_cone, dat.objoffset)
end



# const conemap_mpb_to_hypatia = Dict(
#     "L+" => Hypatia.NonpositiveCone,
#     "L-" =>  Hypatia.NonnegativeCone,
#     "Q" => Hypatia.SecondOrderCone,
#     "QR" => Hypatia.RotatedSecondOrderCone,
#     "EXP" => Hypatia.ExponentialCone,
#     # :ExpDual => "EXP*"
#     "SDP" => Hypatia.PositiveSemidefiniteCone
# )
#
# function add_hypatia_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, idxs::UnitRange{Int})
#     conetype = conemap_mpb_to_hypatia[conesym]
#     conedim = length(idxs)
#     push!(hypatia_cone.prmtvs, get_hypatia_cone(conetype, conedim))
#     push!(hypatia_cone.idxs, output_idxs)
#     push!(hypatia_cone.useduals, false)
#     nothing
# end
#
# function cbfcones_to_hypatiacones!(hypatia_cone::Hypatia.Cone, c::Vector{Tuple{String,Int}}, offset::Int=0)
#     for (cname, count) in c
#         cname in ("L=", "F") && continue
#         cname == "EXP" && @assert count == 3
#         cname == "EXP*" && error("We cannot handle the exponential dual cone yet.")
#         output_idxs = UnitRange{Int}(offset+1, offset+count)
#         offset += count
#         add_hypatia_cone!(hypatia_cone, cname, output_idxs)
#     end
#     nothing
# end
#
# psdconstartidx = Int[]
# for i in 1:length(dat.psdcon)
#     if i == 1
#         push!(psdconstartidx,dat.nconstr+1)
#     else
#         push!(psdconstartidx,psdconstartidx[i-1] + psd_len(dat.psdcon[i-1]))
#     end
#     push!(con_cones,(:SDP,psdconstartidx[i]:psdconstartidx[i]+psd_len(dat.psdcon[i])-1))
# end
# nconstr = (length(dat.psdcon) > 0) ? psdconstartidx[end] + psd_len(dat.psdcon[end]) - 1 : dat.nconstr
#
# function add_psd_cones!(hypatia_cone::Hypatia.Cone, psd_dims::Vector{Int}, offset::Int)
#     total_len = 0
#     for d in psd_dims
#         len = psd_len(d)
#         total_len += len
#         output_idxs = UnitRange{Int}(offset+1, offset+len)
#         add_hypatia_cone!(hypatia_cone, "SDP", output_idxs)
#         offset += len
#     end
#     total_len
# end
#
# """
#     cbftohypatia(dat::CBFData; roundints::Bool=true, dense::Bool=false)
# """
# function cbftohypatia(dat::CBFData; roundints::Bool=true, dense::Bool=false)
#     @assert dat.nvar == (isempty(dat.var) ? 0 : sum(c -> c[2], dat.var))
#     @assert dat.nconstr == (isempty(dat.con) ? 0 : sum(c -> c[2], dat.con))
#
#     if !isempty(intlist)
#         @warn "We cannot handle integer variables yet. Variables will be treated as continuous."
#     end
#
#     # x variables are ordered as vectors followed by all PSD variables
#     # A matrix constraints ordered as L= constraints, then fixed variables
#     # G matrix ordered as cone constraints (PSD last), then cone variables (PSD last)
#     # cones order must match order of constraints in G
#
#     # cone
#     hypatia_cone = Hypatia.Cone()
#     cbfcones_to_hypatiacones!(hypatia_cone, dat.con)
#     len_psd_con = add_psd_cones!(hypatia_cone, dat.psdvar, dat.nconstr)
#     cbfcones_to_hypatiacones!(hypatia_cone, dat.var, dat.nconstr + len_psd_con)
#     len_psd_var = add_psd_cones!(hypatia_cone, dat.psdcon, dat.nconstr + len_psd_con + dat.nvar)
#     n = dat.nvar + len_psd_var
#
#     # c
#     c = zeros(dat.nvar + len_psd_var)
#     for (i, v) in dat.objacoord
#         c[i] = v
#     end
#     nvarcones = 0
#     for v in dat.var
#         !(v[1] in ("L=", "F")) && (nvarcones += 1)
#     end
#     for (matidx, i, j, v) in dat.objfcoord
#         # variable index of first element in PSD cone + column major offset
#         ix = (hypatia_cone.idxs[nvarcones + matidx + 1]).start + idx_to_offset(dat.psdvar[matidx], i, j, true)
#         @assert c[ix] == 0.0
#         scale = (i == j) ? 1.0 : sqrt(2)
#         c[ix] = scale * v
#     end
#     if dat.sense == :Max
#         c .= -c
#     end
#
#     # A, b include all vars in zero cone, constraints in zero cone
#     # zero-cone constraints
#     m1 = sum(cone[2] for cone in dat.con if cone[1] == "L=")
#     # number of fixed variables
#     m2 = sum(cone[2] for cone in dat.var if cone[1] == "L=")
#     m = m1 + m2
#     if dense
#         A = zeros(m)
#     else
#         A = spzeros(m)
#     end
#     b = zeros(m)
#
#     I_A, J_A, V_A = unzip(dat.acoord)
#     # also include constraints involving PSD variables
#     for (conidx, matidx, i, j, v) in dat.fcoord
#         ix = (hypatia_cone.idxs[nvarcones + matidx + 1]).start + idx_to_offset(dat.psdvar[matidx], i, j, true)
#         push!(I_A, conidx)
#         push!(J_A, ix)
#         scale = (i == j) ? 1.0 : sqrt(2)
#         push!(V_A, scale * v)
#     end
#
#     # for L= constraints just copy over data
#     Ainds = [dat.con[I_A[i]][1] == "L=" for i in 1:length(I_A)]
#     A[1:m1, :] .= sparse(I_A[Ainds], J_A[Ainds], -V_A[Ainds], m1, dat.nvar + len_psd_var)
#     # identity for vars in L=
#     fixed_var_idxs = Int[]
#     idx = 0
#     for var in dat.var
#         if var[1] == "L="
#             push!(fixed_var_idxs, collect(idx+1:idx+var[2])...)
#             idx += var[2]
#         end
#     end
#     A[m1+1:end, fixed_var_idxs] .= Matrix(I, m2, m2)
#
#     for (i, v) in dat.bcoord
#         if Ainds[i]
#             b[i] = v
#         end
#     end
#
#     # G, h
#     # constraints in cones
#     m3 = dat.nconstr + len_psd_con - m1
#     # variables in cones
#     m4 = 0
#     for v in dat.var
#         (v[1] in ("F", "L=")) || (m4 += 1)
#     end
#     m = m3 + m4 + len_psd_var
#     if dense
#         G = zeros(m)
#     else
#         G = spzeros(m)
#     end
#     h = zeros(m)
#
#     # lookup psd constraint starts
#     psdconstartidx = Int[]
#     for i in 1:length(dat.psdcon)
#         if i == 1
#             push!(psdconstartidx, dat.nconstr + 1)
#         else
#             push!(psdconstartidx,psdconstartidx[i-1] + psd_len(dat.psdcon[i-1]))
#         end
#     end
#
#     # TODO not copy
#     I_G, J_G, V_G = copy(I_A[!Ainds]), copy(J_A[!Ainds]), copy(V_A[!Ainds])
#     # add PSD constraints
#     for (conidx, varidx, i, j, v) in dat.hcoord
#         ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx], i, j, true)
#         push!(I_G, ix)
#         push!(J_G, varidx)
#         scale = (i == j) ? 1.0 : sqrt(2)
#         push!(V_G, scale * v)
#     end
#     Ginds = [i for i in 1:length(I_A) if dat.con[I_A[i]][1] != "L="]
#
#     for (conidx,i,j,v) in dat.dcoord
#         ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx],i,j,col_major)
#         @assert b[ix] == 0.0
#         scale = (i == j) ? 1.0 : sqrt(2)
#         b[ix] = scale*v
#     end
#     # for cone constraints just copy over data
#
#     G[1:dat.nconstr, :] .= sparse(I_A[!Ainds], J_A[!Ainds], -V_A[!Ainds], dat.nconstr, dat.nvar + len_psd_var)
#     # positive semidefinite cone constraints
#
#
#     # exponential cone constraints are actually in reverse order
#     exp_con_indices = Vector{Int}[]
#     offset = 0
#     for (cone_type, count) in dat.con
#         if cone_type == :ExpPrimal
#             push!(exp_con_indices, collect(offset+1:offset+3))
#         end
#         offset += count
#     end
#
#     # TODO when going CBF -> hypatia directly this will definitely not need to happen
#     for ind_collection in exp_con_indices
#         A_in[ind_collection, :] .= A_in[reverse(ind_collection), :]
#         b_in[ind_collection, :] .= b_in[reverse(ind_collection), :]
#     end
#
#
#
#
#     #
#     # I_A, J_A, V_A = unzip(dat.acoord)
#     for (i, j, v) in dat.acoord
#         offset = zero_count + cone_count + 1
#         if dat.con[i][1] == "L="
#             zero_count += dat.con[i][2]
#             A[zero_count, j] = v
#         else
#             if dat.con[i][1] == "EXP"
#
#             cone_count += 1
#             G[cone_count, j] = v
#         end
#     end
#
#     A = sparse(I_A,J_A,-V_A,nconstr,nvar)
#     b = zeros(dat.nconstr)
#     for (i, v) in dat.bcoord
#         b[i] = v
#     end
#
#     # psdvarstartidx = Int[]
#     # for i in 1:length(dat.psdvar)
#     #     if i == 1
#     #         push!(psdvarstartidx, dat.nvar + 1)
#     #     else
#     #         push!(psdvarstartidx, psdvarstartidx[i-1] + psd_len(dat.psdvar[i-1]))
#     #     end
#     #     push!(var_cones,(:SDP, psdvarstartidx[i]:psdvarstartidx[i] + psd_len(dat.psdvar[i])-1))
#     # end
#     # nvar = (length(dat.psdvar) > 0) ? psdvarstartidx[end] + psd_len(dat.psdvar[end]) - 1 : dat.nvar
#
#     # psdconstartidx = Int[]
#     # for i in 1:length(dat.psdcon)
#     #     if i == 1
#     #         push!(psdconstartidx,dat.nconstr+1)
#     #     else
#     #         push!(psdconstartidx,psdconstartidx[i-1] + psd_len(dat.psdcon[i-1]))
#     #     end
#     #     push!(con_cones,(:SDP,psdconstartidx[i]:psdconstartidx[i]+psd_len(dat.psdcon[i])-1))
#     # end
#     # nconstr = (length(dat.psdcon) > 0) ? psdconstartidx[end] + psd_len(dat.psdcon[end]) - 1 : dat.nconstr
#
#
#
#     for (conidx,matidx,i,j,v) in dat.fcoord
#         ix = psdvarstartidx[matidx] + idx_to_offset(dat.psdvar[matidx],i,j,col_major)
#         push!(I_A,conidx)
#         push!(J_A,ix)
#         scale = (i == j) ? 1.0 : sqrt(2)
#         push!(V_A,scale*v)
#     end
#
#     for (conidx,varidx,i,j,v) in dat.hcoord
#         ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx],i,j,col_major)
#         push!(I_A,ix)
#         push!(J_A,varidx)
#         scale = (i == j) ? 1.0 : sqrt(2)
#         push!(V_A,scale*v)
#     end
#
#     b = [b;zeros(nconstr-dat.nconstr)]
#     for (conidx,i,j,v) in dat.dcoord
#         ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx],i,j,col_major)
#         @assert b[ix] == 0.0
#         scale = (i == j) ? 1.0 : sqrt(2)
#         b[ix] = scale*v
#     end
#
#     A = sparse(I_A,J_A,-V_A,nconstr,nvar)
#
#     vartypes = fill(:Cont, nvar)
#     if !roundints
#         vartypes[dat.intlist] .= :Int
#     end
#
#     return c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset
# end
