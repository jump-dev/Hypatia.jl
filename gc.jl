using DynamicPolynomials
using LinearAlgebra
using PolyJuMP
using JuMP

# function weirdgc()
#     n = 600
#     @polyvar x[1:n]
#     w = rand(n)
#     monos = monomials(x, 0:1)
#     U = length(monos)
#     c = rand(U)
#     p = dot(monos, c)
#     @show typeof(p)
#     p(w)
# end

function weirdgc()
    n = 200
    @polyvar x[1:n]
    w = rand(n)
    monos = monomials(x, 0:1)
    U = length(monos)
    model = Model()
    # @variable(model, p, Poly(monos))
    @variable(model, p[1:n])
    @show typeof(p)
    r = zeros(JuMP.AffExpr, n)
    for i in 1:n
        r[i] = dot(p, w)
        # r[i] = p(w)
    end
end

@time weirdgc()
