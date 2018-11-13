#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using DelimitedFiles

function sample_families()
    problem_stats = readdlm(joinpath(@__DIR__(), "data/cblib_problem_stats.csv"), ',', header = true)
    sampled_problems = fill(false, size(problem_stats[1], 1))
    families = Vector{Int}[]
    prev = 0
    # group problems in the same family together
    for i in 1:size(problem_stats[1], 1)
        if problem_stats[1][i, end] != prev
            push!(families, Int[])
            prev = problem_stats[1][i, end]
        end
    end
    for (i, f) in enumerate(problem_stats[1][:, end])
        push!(families[f], i)
    end
    # sample at most five problems from each family
    for fam in families
        nproblems = min(5, length(families))
        ids = rand(fam, nproblems)
        sampled_problems[ids] .= true
    end
    return sampled_problems
end
