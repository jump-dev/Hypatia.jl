#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

utilities for Hypatia domains and SemialgebraicSets.jl
=#

# construct domain inequalities for SumOfSquares models from Hypatia domains

function get_domain_inequalities(dom::Box, x)
    bss = SAS.BasicSemialgebraicSet{Float64, DynamicPolynomials.Polynomial{true, Float64}}()
    for i in 1:dimension(dom)
        SAS.addinequality!(bss, (-x[i] + dom.u[i]) * (x[i] - dom.l[i]))
    end
    return bss
end

get_domain_inequalities(dom::Ball, x) = SAS.@set(sum((x - dom.c) .^ 2) <= dom.r^2)

get_domain_inequalities(dom::Ellipsoid, x) = SAS.@set((x - dom.c)' * dom.Q * (x - dom.c) <= 1.0)

get_domain_inequalities(dom::SemiFreeDomain, x) = get_domain_inequalities(dom.sampling_region, x)
