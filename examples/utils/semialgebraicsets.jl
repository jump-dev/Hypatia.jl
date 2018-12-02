#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import DynamicPolynomials
import SemialgebraicSets

# construct domain inequalities for SumOfSquares models from Hypatia domains

function get_domain_inequalities(dom::Hypatia.Box, x)
    bss = SemialgebraicSets.BasicSemialgebraicSet{Float64, DynamicPolynomials.Polynomial{true, Float64}}()
    for i in 1:Hypatia.dimension(dom)
        SemialgebraicSets.addinequality!(bss, (-x[i] + dom.u[i]) * (x[i] - dom.l[i]))
    end
    return bss
end

function get_domain_inequalities(dom::Hypatia.Ball, x)
    return SemialgebraicSets.@set(sum((x - dom.c).^2) <= dom.r^2)
end

function get_domain_inequalities(dom::Hypatia.Ellipsoid, x)
    return SemialgebraicSets.@set((x - dom.c)' * dom.Q * (x - dom.c) <= 1.0)
end

function get_domain_inequalities(dom::Hypatia.SemiFreeDomain, x)
    return get_domain_inequalities(dom.sampling_region, x)
end
