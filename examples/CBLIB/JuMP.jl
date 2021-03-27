#=
run a CBLIB instance
cblib_dir (defined in JuMP_test.jl) is the directory of CBLIB files
=#

struct CBLIBJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    name::String # filename of CBLIB instance
end

function build(inst::CBLIBJuMP{T}) where {T <: Float64}
    # read in model from file
    model = JuMP.read_from_file(joinpath(cblib_dir, inst.name * ".cbf.gz"))

    # delete integer constraints
    JuMP.delete.(model, JuMP.all_constraints(model, JuMP.VariableRef, MOI.Integer))

    return model
end
