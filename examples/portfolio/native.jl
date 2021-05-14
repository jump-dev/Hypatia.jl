#=
maximize expected returns subject to risk constraints

TODO
- add entropic ball constraint using entropy cone
- describe formulation and options
=#

using SparseArrays

struct PortfolioNative{T <: Real} <: ExampleInstanceNative{T}
    num_stocks::Int
    epinormeucl_constr::Bool # add L2 ball constraints
    epinorminf_constrs::Bool # add Linfty ball constraints
    use_epinorminf::Bool # use epinorminf cone, else nonnegative cones
end

function build(inst::PortfolioNative{T}) where {T <: Real}
    @assert xor(inst.epinormeucl_constr, inst.epinorminf_constrs)
    num_stocks = inst.num_stocks

    returns = rand(T, num_stocks)
    sigma_half = T.(randn(num_stocks, num_stocks))
    x = T.(randn(num_stocks))
    x ./= norm(x)
    gamma = sum(abs, sigma_half * x) / sqrt(T(num_stocks))

    c = -returns
    # investments add to one, nonnegativity
    A = ones(T, 1, num_stocks)
    G = sparse(-one(T) * I, num_stocks, num_stocks)
    b = T[1]
    h = zeros(T, num_stocks)
    cones = Cones.Cone{T}[Cones.Nonnegative{T}(num_stocks)]
    cone_offset = num_stocks

    function add_ball_constr(cone, gamma_new)
        G = vcat(G, spzeros(T, 1, num_stocks), -sigma_half)
        h_risk = vcat(gamma_new, zeros(T, num_stocks))
        h = vcat(h, h_risk)
        push!(cones, cone)
        cone_offset += num_stocks + 1
    end

    if inst.epinormeucl_constr
        add_ball_constr(Cones.EpiNormEucl{T}(num_stocks + 1), gamma)
    end

    if inst.epinorminf_constrs
        if inst.use_epinorminf
            add_ball_constr(Cones.EpiNormInf{T, T}(num_stocks + 1,
                use_dual = true), gamma * sqrt(T(num_stocks)))
            add_ball_constr(Cones.EpiNormInf{T, T}(num_stocks + 1), gamma)
        else
            c = vcat(c, zeros(T, 2 * num_stocks))
            A = [
                A    spzeros(T, 1, 2 * num_stocks);
                sigma_half    -I    I;
                ]
            padding = spzeros(T, num_stocks, 2 * num_stocks)
            G = [
                G    spzeros(T, size(G, 1), 2 * num_stocks);
                spzeros(T, 2 * num_stocks, num_stocks)    -I;
                spzeros(T, 1, num_stocks)    ones(T, 1, 2 * num_stocks);
                sigma_half    padding;
                -sigma_half    padding;
                ]
            b = vcat(b, zeros(T, num_stocks))
            h = vcat(h, zeros(T, 2 * num_stocks), gamma * sqrt(T(num_stocks)),
                gamma * ones(T, 2 * num_stocks))
            push!(cones, Cones.Nonnegative{T}(4 * num_stocks + 1))
            cone_offset += 4 * num_stocks + 1
        end
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end
