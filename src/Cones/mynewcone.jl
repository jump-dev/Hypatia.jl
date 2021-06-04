mutable struct MyNewCone{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    # fields specific to this cone
    A
    side
    fact
    Aw_inv
    rt2

    function MyNewCone{T}(
        A;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        # these are some standard fields
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.hess_fact_cache = hess_fact_cache

        cone.A = A
        # infer cone dimension from size of A
        cone.dim = size(A, 2)
        # cache the side dimesnion of smat(A * w)
        cone.side = svec_side(T, cone.dim)
        # cache √2 for convencience
        cone.rt2 = sqrt(T(2))

        return cone
    end
end

get_nu(cone::MyNewCone) = cone.side

function set_initial_point!(
    arr::AbstractVector{T},
    cone::MyNewCone{T},
    ) where {T <: Real}
    side = cone.side
    A = cone.A
    temp = zeros(T, size(A, 1))
    I_vec = smat_to_svec!(temp, Matrix{T}(I, side, side), cone.rt2)
    arr .= A \ I_vec
    return arr
end

function update_feas(cone::MyNewCone{T}) where {T <: Real}
    side = cone.side
    A = cone.A
    w = cone.point
    temp = zeros(T, side, side)

    Aw_vec = A * w
    Aw_mat = svec_to_smat!(temp, Aw_vec, cone.rt2)
    @show Aw_mat
    @show inv(Symmetric(Aw_mat))
    fact = cholesky(Symmetric(Aw_mat, :U), check = false)
    cone.fact = fact
    cone.is_feas = isposdef(fact)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::MyNewCone{T}) where {T <: Real}
    A = cone.A
    Aw_inv = cone.Aw_inv = inv(cone.fact)
    temp = zeros(T, size(A, 1))
    smat_to_svec!(temp, -Aw_inv, cone.rt2)
    cone.grad .= A' * temp
    # smat_to_svec!(cone.grad, -Aw_inv, cone.rt2)

    # function bar(w)
    #     T = eltype(w)
    #     Aw_mat = svec_to_smat!(zeros(T, cone.side, cone.side), cone.A * w, sqrt(T(2)))
    #     Aw_mat = (Aw_mat + Aw_mat') / 2
    #     return -logdet(Symmetric(Aw_mat))
    # end
    # fd = ForwardDiff.gradient(bar, cone.point)
    # @show cone.grad
    # @show fd
    @show cone.grad

    cone.grad_updated = true
    return cone.grad
end

using ForwardDiff

function update_hess(cone::MyNewCone{T}) where {T <: Real}
    side = cone.side
    A = cone.A
    Aw_inv = cone.Aw_inv
    rt2 = cone.rt2
    temp1 = zeros(T, side, side)
    temp2 = zeros(T, size(A, 1))
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data

    k = 1
    for j in 1:cone.side, i in 1:j
        Aij = A[:, k]
        Aij_mat = svec_to_smat!(temp1, Aij, rt2)
        H[:, k] = A' * smat_to_svec!(temp2, Aw_inv * Aij_mat * Aw_inv, rt2)
        k += 1
    end

    function bar(w)
        E = eltype(w)
        Aw_mat = svec_to_smat!(zeros(E, cone.side, cone.side), cone.A * w, sqrt(E(2)))
        return -logdet(Symmetric(Aw_mat))
    end
    fd = ForwardDiff.hessian(bar, cone.point)
    @show H
    @show fd

    cone.hess_updated = true
    return cone.hess
end
