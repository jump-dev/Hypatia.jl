mutable struct MySpectrahedron{T <: Real} <: Cone{T}
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
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    # fields specific to this cone
    A
    side
    fact
    Aw_inv
    rt2

    function MySpectrahedron{T}(
        A;
        use_dual::Bool = false,
        ) where {T <: Real}
        cone = new{T}()
        cone.use_dual_barrier = use_dual

        cone.A = A
        (d1, d2) = size(A)
        # infer cone dimension from size of A
        cone.dim = d2
        # cache the side dimesnion of smat(A * w)
        cone.side = svec_side(T, d1)
        # cache √2 for convencience
        cone.rt2 = sqrt(T(2))

        return cone
    end
end

get_nu(cone::MySpectrahedron) = cone.side

function set_initial_point!(
    arr::AbstractVector{T},
    cone::MySpectrahedron{T},
    ) where {T <: Real}
    side = cone.side
    A = cone.A
    temp = zeros(T, size(A, 1))
    I_vec = smat_to_svec!(temp, Matrix{T}(I, side, side), cone.rt2)
    arr .= A \ I_vec
    return arr
end

function update_feas(cone::MySpectrahedron{T}) where {T <: Real}
    side = cone.side
    A = cone.A
    w = cone.point
    temp = zeros(T, side, side)

    Aw_vec = A * w
    Aw_mat = svec_to_smat!(temp, Aw_vec, cone.rt2)
    fact = cholesky(Symmetric(Aw_mat, :U), check = false)
    cone.fact = fact
    cone.is_feas = isposdef(fact)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::MySpectrahedron{T}) where {T <: Real}
    A = cone.A
    Aw_inv = cone.Aw_inv = inv(cone.fact)
    temp = zeros(T, size(A, 1))
    smat_to_svec!(temp, -Aw_inv, cone.rt2)
    cone.grad = A' * temp

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::MySpectrahedron{T}) where {T <: Real}
    isdefined(cone, :hess) || alloc_hess!(cone)
    side = cone.side
    A = cone.A
    Aw_inv = cone.Aw_inv
    rt2 = cone.rt2
    temp1 = zeros(T, side, side)
    temp2 = zeros(T, size(A, 1))
    H = cone.hess.data

    for k in 1:cone.dim
        Ak = A[:, k]
        Ak_mat = Symmetric(svec_to_smat!(temp1, Ak, rt2), :U)
        H[:, k] = A' * smat_to_svec!(temp2, Aw_inv * Ak_mat * Aw_inv, rt2)
    end

    cone.hess_updated = true
    return cone.hess
end

use_dder3(::MySpectrahedron)::Bool = false
