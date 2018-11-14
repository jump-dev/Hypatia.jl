# Contains code modified from https://github.com/JuliaOpt/ConicBenchmarkUtilities.jl (Miles Lubin and contributors)

const conemap = Dict(
    "L=" => :Zero,
    "F" => :Free,
    "L-" => :NonPos,
    "L+" => :NonNeg,
    "Q" => :SOC,
    "QR" => :SOCRotated,
    "EXP" => :ExpPrimal,
    "EXP*" => :ExpDual,
    "POWER" => :Power,
)

const conemap_rev = Dict(
    :Zero => "L=",
    :Free => "F",
    :NonPos => "L-",
    :NonNeg => "L+",
    :SOC => "Q",
    :SOCRotated => "QR",
    :ExpPrimal => "EXP",
    :ExpDual => "EXP*"
)

function cbfcones_to_mpbcones(c::Vector{Tuple{String,Int}},total)
    i = 1
    mpb_cones = Tuple{Symbol,Vector{Int}}[]

    for (cname,count) in c
        conesymbol = conemap[cname]
        if conesymbol == :ExpPrimal
            @assert count == 3
            indices = i+2:-1:i
        else
            indices = i:(i+count-1)
        end
        push!(mpb_cones, (conesymbol, collect(indices)))
        i += count
    end
    @assert i == total + 1
    return mpb_cones
end

# https://github.com/JuliaLang/julia/issues/13942#issuecomment-217324812
function unzip(A::Array{T}) where T<:Tuple
    res = map(x -> x[], T.parameters)
    res_len = length(res)
    for t in A
        for i in 1:res_len
            push!(res[i], t[i])
        end
    end
    return res
end

psd_len(n) = div(n*(n+1),2)
# returns offset from starting index for (i,j) term in n x n matrix
function idx_to_offset(n,i,j,col_major::Bool=false)
    @assert 1 <= i <= n
    @assert 1 <= j <= n
    # upper triangle
    if i > j
        i,j = j,i
    end
    if col_major
        return psd_len(j) + i - j - 1
    else
        # think row major
        return psd_len(n) - psd_len(n-i+1) + (j-i)
    end
end

function cbftompb(dat::CBFData; roundints::Bool=true, col_major::Bool=false)
    @assert dat.nvar == (isempty(dat.var) ? 0 : sum(c->c[2],dat.var))
    @assert dat.nconstr == (isempty(dat.con) ? 0 : sum(c->c[2],dat.con))

    c = zeros(dat.nvar)
    for (i,v) in dat.objacoord
        c[i] = v
    end

    var_cones = cbfcones_to_mpbcones(dat.var, dat.nvar)
    con_cones = cbfcones_to_mpbcones(dat.con, dat.nconstr)

    I_A, J_A, V_A = unzip(dat.acoord)
    b = zeros(dat.nconstr)
    for (i,v) in dat.bcoord
        b[i] = v
    end

    psdvarstartidx = Int[]
    for i in 1:length(dat.psdvar)
        if i == 1
            push!(psdvarstartidx,dat.nvar+1)
        else
            push!(psdvarstartidx,psdvarstartidx[i-1] + psd_len(dat.psdvar[i-1]))
        end
        push!(var_cones,(:SDP,psdvarstartidx[i]:psdvarstartidx[i]+psd_len(dat.psdvar[i])-1))
    end
    nvar = (length(dat.psdvar) > 0) ? psdvarstartidx[end] + psd_len(dat.psdvar[end]) - 1 : dat.nvar

    psdconstartidx = Int[]
    for i in 1:length(dat.psdcon)
        if i == 1
            push!(psdconstartidx,dat.nconstr+1)
        else
            push!(psdconstartidx,psdconstartidx[i-1] + psd_len(dat.psdcon[i-1]))
        end
        push!(con_cones,(:SDP,psdconstartidx[i]:psdconstartidx[i]+psd_len(dat.psdcon[i])-1))
    end
    nconstr = (length(dat.psdcon) > 0) ? psdconstartidx[end] + psd_len(dat.psdcon[end]) - 1 : dat.nconstr

    c = [c;zeros(nvar-dat.nvar)]
    for (matidx,i,j,v) in dat.objfcoord
        ix = psdvarstartidx[matidx] + idx_to_offset(dat.psdvar[matidx],i,j,col_major)
        @assert c[ix] == 0.0
        scale = (i == j) ? 1.0 : sqrt(2)
        c[ix] = scale*v
    end

    for (conidx,matidx,i,j,v) in dat.fcoord
        ix = psdvarstartidx[matidx] + idx_to_offset(dat.psdvar[matidx],i,j,col_major)
        push!(I_A,conidx)
        push!(J_A,ix)
        scale = (i == j) ? 1.0 : sqrt(2)
        push!(V_A,scale*v)
    end

    for (conidx,varidx,i,j,v) in dat.hcoord
        ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx],i,j,col_major)
        push!(I_A,ix)
        push!(J_A,varidx)
        scale = (i == j) ? 1.0 : sqrt(2)
        push!(V_A,scale*v)
    end

    b = [b;zeros(nconstr-dat.nconstr)]
    for (conidx,i,j,v) in dat.dcoord
        ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx],i,j,col_major)
        @assert b[ix] == 0.0
        scale = (i == j) ? 1.0 : sqrt(2)
        b[ix] = scale*v
    end

    A = sparse(I_A,J_A,-V_A,nconstr,nvar)

    vartypes = fill(:Cont, nvar)
    if !roundints
        vartypes[dat.intlist] .= :Int
    end

    return c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset
end

function mpbtocbf(name, c, A, b, con_cones, var_cones, vartypes, sense=:Min)
    num_scalar_var = 0
    for (cone,idx) in var_cones
        if cone != :SDP
            num_scalar_var += length(idx)
        end
    end
    num_scalar_con = 0
    for (cone,idx) in con_cones
        if cone != :SDP
            num_scalar_con += length(idx)
        end
    end

    # need to shuffle rows and columns to put them in order
    var_idx_old_to_new = zeros(Int,length(c))
    con_idx_old_to_new = zeros(Int,length(b))
    var_idx_new_to_old = zeros(Int,num_scalar_var)
    con_idx_new_to_old = zeros(Int,num_scalar_con)

    # CBF fields
    var = Tuple{String,Int}[]
    con = Tuple{String,Int}[]

    i = 1
    for (cone,idx) in var_cones
        if cone == :ExpPrimal
            @assert all(var_idx_old_to_new[idx] .== 0)
            @assert length(idx) == 3
            # MPB: (x,y,z) : y*exp(x/y) <= z
            # CBF: (z,y,x) : y*exp(x/y) <= z
            var_idx_old_to_new[idx] = i+2:-1:i
            var_idx_new_to_old[i+2:-1:i] = idx
            i += 3
            push!(var, (conemap_rev[cone],length(idx)))
        elseif cone != :SDP
            for k in idx
                var_idx_old_to_new[k] = i
                var_idx_new_to_old[i] = k
                i += 1
            end
            push!(var, (conemap_rev[cone],length(idx)))
        end
    end
    @assert i - 1 == num_scalar_var

    i = 1
    for (cone,idx) in con_cones
        if cone == :ExpPrimal
            @assert all(con_idx_old_to_new[idx] .== 0)
            @assert length(idx) == 3
            con_idx_old_to_new[idx] = i+2:-1:i
            con_idx_new_to_old[i+2:-1:i] = idx
            i += 3
            push!(con, (conemap_rev[cone],length(idx)))
        elseif cone != :SDP
            for k in idx
                @assert con_idx_old_to_new[k] == 0
                con_idx_old_to_new[k] = i
                con_idx_new_to_old[i] = k
                i += 1
            end
            push!(con, (conemap_rev[cone],length(idx)))
        end
    end
    @assert i - 1 == num_scalar_con

    objacoord = collect(zip(findnz(sparse(c[var_idx_new_to_old]))...))
    bcoord = collect(zip(findnz(sparse(b[con_idx_new_to_old]))...))
    # MPB is b - Ax ∈ K, CBF is b + Ax ∈ K
    Acbf = -A[con_idx_new_to_old,var_idx_new_to_old]

    acoord = collect(zip(findnz(Acbf)...))::Vector{Tuple{Int,Int,Float64}}

    intlist = Int[]
    for i in 1:length(vartypes)
        if var_idx_old_to_new[i] == 0 && vartypes[i] != :Cont
            error("CBF format does not support integer restrictions on PSD variables")
        end
        if vartypes[i] == :Cont
        elseif vartypes[i] == :Int
            push!(intlist,var_idx_old_to_new[i])
        elseif vartypes[i] == :Bin
            # TODO: Check if we need to add variable bounds also
            push!(intlist,var_idx_old_to_new[i])
        else
            error("Unrecognized variable category $(vartypes[i])")
        end
    end

    psdvar = Int[]
    psdcon = Int[]

    # Map from MPB linear variable index to (psdvar,i,j)
    psdvar_idx_old_to_new = fill((0,0,0), length(c))
    # Map from MPB linear constraint index to (psdcon,i,j)
    psdcon_idx_old_to_new = fill((0,0,0),length(b))

    for (cone,idx) in var_cones
        if cone == :SDP
            y = length(idx)
            conedim = round(Int, sqrt(0.25 + 2y) - 0.5)
            push!(psdvar, conedim)
            k = 1
            for i in 1:conedim, j in i:conedim
                psdvar_idx_old_to_new[idx[k]] = (length(psdvar), i, j)
                k += 1
            end
            @assert length(idx) == k - 1
        end
    end

    for (cone,idx) in con_cones
        if cone == :SDP
            y = length(idx)
            conedim = round(Int, sqrt(0.25 + 2y) - 0.5)
            push!(psdcon, conedim)
            k = 1
            for i in 1:conedim, j in i:conedim
                psdcon_idx_old_to_new[idx[k]] = (length(psdcon), i, j)
                k += 1
            end
            @assert length(idx) == k - 1
        end
    end

    objfcoord = Tuple{Int,Int,Int,Float64}[]
    for i in 1:length(c)
        if c[i] != 0.0 && psdvar_idx_old_to_new[i] != (0,0,0)
            varidx, vari, varj = psdvar_idx_old_to_new[i]
            scale = (vari == varj) ? 1.0 : sqrt(2)
            push!(objfcoord, (varidx, vari, varj, c[i]/scale))
        end
    end
    dcoord = Tuple{Int,Int,Int,Float64}[]
    for i in 1:length(b)
        if b[i] != 0.0 && psdcon_idx_old_to_new[i] != (0,0,0)
            conidx, coni, conj = psdcon_idx_old_to_new[i]
            scale = (coni == conj) ? 1.0 : sqrt(2)
            push!(dcoord, (conidx, coni, conj, b[i]/scale))
        end
    end

    A_I, A_J, A_V = findnz(A)
    fcoord = Tuple{Int,Int,Int,Int,Float64}[]
    hcoord = Tuple{Int,Int,Int,Int,Float64}[]

    for (i,j,v) in zip(A_I,A_J,A_V)
        if psdvar_idx_old_to_new[j] != (0,0,0)
            if psdcon_idx_old_to_new[i] != (0,0,0)
                error("CBF format does not allow PSD variables to appear in affine expressions defining PSD constraints")
            end
            newrow = con_idx_old_to_new[i]
            @assert newrow != 0
            varidx, vari, varj = psdvar_idx_old_to_new[j]
            scale = (vari == varj) ? 1.0 : sqrt(2)
            push!(fcoord, (newrow, varidx, vari, varj, -v/scale))
        elseif psdcon_idx_old_to_new[i] != (0,0,0)
            newcol = var_idx_old_to_new[j]
            conidx, coni, conj = psdcon_idx_old_to_new[i]
            scale = (coni == conj) ? 1.0 : sqrt(2)
            push!(hcoord, (conidx, newcol, coni, conj, -v/scale))
        end
    end

    return CBFData(name,sense,var,psdvar,con,psdcon,[],[],[],objacoord,objfcoord,0.0,fcoord,acoord,bcoord,hcoord,dcoord,intlist,num_scalar_var,num_scalar_con)
end

# converts an MPB solution to CBF solution
# no transformation needed unless PSD vars present
function mpb_sol_to_cbf(dat::CBFData,x::Vector)
    scalar_solution = x[1:dat.nvar]

    psdvar_solutions = Matrix{Float64}[]
    startidx = dat.nvar+1
    for i in 1:length(dat.psdvar)
        endidx = startidx + psd_len(dat.psdvar[i]) - 1
        svec_solution = x[startidx:endidx]
        push!(psdvar_solutions, make_smat!(Matrix{Float64}(undef, dat.psdvar[i],dat.psdvar[i]), svec_solution))
        startidx = endidx + 1
    end

    return scalar_solution, psdvar_solutions
end

# Copied from Pajarito.jl
function make_smat!(smat::Matrix{Float64}, svec::Vector{Float64})
    dim = size(smat, 1)
    kSD = 1
    for jSD in 1:dim, iSD in jSD:dim
        if jSD == iSD
            smat[iSD, jSD] = svec[kSD]
        else
            smat[iSD, jSD] = smat[jSD, iSD] = (1/sqrt(2)) * svec[kSD]
        end
        kSD += 1
    end
    return smat
end
