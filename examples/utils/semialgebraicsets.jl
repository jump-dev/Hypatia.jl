#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import DynamicPolynomials
import SemialgebraicSets

function get_bss(dom::Hypatia.Box, x)
    bss = SemialgebraicSets.BasicSemialgebraicSet{Float64, DynamicPolynomials.Polynomial{true, Float64}}()
    for i in 1:Hypatia.dimension(dom)
        SemialgebraicSets.addinequality!(bss, (-x[i] + dom.u[i]) * (x[i] - dom.l[i]))
    end
    return bss
end
get_bss(dom::Hypatia.Ball, x) = SemialgebraicSets.@set(sum((x - dom.c).^2) <= dom.r^2)
get_bss(dom::Hypatia.Ellipsoid, x) = SemialgebraicSets.@set((x - dom.c)' * dom.Q * (x - dom.c) <= 1.0)
get_bss(dom::Hypatia.SemiFreeDomain, x) = get_bss(dom.sampling_region, x)
