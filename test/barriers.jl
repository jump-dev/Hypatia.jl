#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import Random

function pass_through_cone(cone::CO.Cone{T}; num_checks::Int = 2) where {T <: HypReal}
    CO.setup_data(cone)
    for _ in 1:num_checks
        point = Vector{T}(undef, CO.dimension(cone))
        CO.set_initial_point(point, cone)
        CO.load_point(cone, point)
        @test CO.check_in_cone(cone)
        tol = max(1e-12, sqrt(eps(T)))
        @test -dot(cone.point, CO.grad(cone)) ≈ CO.get_nu(cone) atol=tol rtol=tol
        @test CO.hess(cone) * cone.point ≈ -CO.grad(cone) atol=tol rtol=tol
    end
    return
end

function test_epinormeucl_barrier(T::Type{<:HypReal})
    for dim in [2, 3, 5]
        cone = CO.EpiNormEucl{T}(dim)
        pass_through_cone(cone)
    end
    return
end

function test_epinorinf_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.EpiNormInf{T}(dim)
        pass_through_cone(cone)
    end
    return
end

function test_epinormspectral_barrier(T::Type{<:HypReal})
    for (n, m) in [(1, 3), (2, 4)]
        cone = CO.EpiNormSpectral{T}(n, m)
        pass_through_cone(cone)
    end
    return
end

function test_epiperpower_barrier(T::Type{<:HypReal})
    for alpha in [1.5, 2.5]
        cone = CO.EpiPerPower{T}(alpha)
        pass_through_cone(cone)
    end
    return
end

function test_epipersquare_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.EpiPerSquare{T}(dim)
        pass_through_cone(cone)
    end
    return
end

function test_epipersumexp_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.EpiPerSumExp{T}(dim)
        pass_through_cone(cone)
    end
    return
end

function test_hypogeomean_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for dim in [3, 5, 8]
        alpha = rand(T, dim - 1)
        alpha ./= sum(alpha)
        cone = CO.HypoGeomean{T}(alpha)
        pass_through_cone(cone)
    end
    return
end

function test_hypoperlog_barrier(T::Type{<:HypReal})
    cone = CO.HypoPerLog{T}()
    pass_through_cone(cone)
    return
end

function test_hypoperlogdet_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.HypoPerLogdet{T}(dim)
        pass_through_cone(cone)
    end
    return
end

function test_semidefinite_barrier(T::Type{<:HypReal})
    for dim in [1, 3, 6]
        cone = CO.PosSemidef{T, T}(dim) # real
        pass_through_cone(cone)
    end
    for dim in [1, 4, 9]
        cone = CO.PosSemidef{T, Complex{T}}(dim) # complex
        pass_through_cone(cone)
    end
    return
end

function test_wsospolyinterp_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        cone = CO.WSOSPolyInterp{T, T}(U, [P0], true)
        pass_through_cone(cone)
    end
    # TODO also test complex case CO.WSOSPolyInterp{T, Complex{T}} - need complex MU interp functions first
    return
end

function test_wsospolyinterpmat_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:3, R in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        cone = CO.WSOSPolyInterpMat{T}(R, U, [P0], true)
        pass_through_cone(cone)
    end
    return
end

function test_wsospolyinterpsoc_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for n in 1:2, halfdeg in 1:2, R in 3:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        cone = CO.WSOSPolyInterpSOC{T}(R, U, [P0], true)
        pass_through_cone(cone)
    end
    return
end
