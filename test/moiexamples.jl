
# TODO 
# @testset "envelope example" begin
#     opt = build_envelope(2, 5, 1, 5, use_data=true, native=false)
#     MOI.optimize!(opt)
#     @test MOI.get(opt, MOI.TerminationStatus()) == MOI.Success
#     @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -25.502777 atol=1e-4
#     @test MOI.get(opt, MOI.ObjectiveBound()) ≈ -25.502777 atol=1e-4
# end
#
# @testset "lp example" begin
#     opt = build_lp(500, 1000, use_data=true, native=false)
#     MOI.optimize!(opt)
#     @test MOI.get(opt, MOI.TerminationStatus()) == MOI.Success
#     @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 2055.807 atol=1e-4
#     @test MOI.get(opt, MOI.ObjectiveBound()) ≈ 2055.807 atol=1e-4
# end
#
# @testset "namedpoly examples" begin
#     @testset "Goldstein-Price" begin
#         opt = build_namedpoly(:goldsteinprice, 7, native=false)
#         MOI.optimize!(opt)
#         @test MOI.get(opt, MOI.TerminationStatus()) == MOI.Success
#         @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 3 atol=1e-4
#         @test MOI.get(opt, MOI.ObjectiveBound()) ≈ 3 atol=1e-4
#     end
#
#     @testset "Robinson" begin
#         opt = build_namedpoly(:robinson, 8, native=false)
#         MOI.optimize!(opt)
#         @test MOI.get(opt, MOI.TerminationStatus()) == MOI.Success
#         @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 0.814814 atol=1e-4
#         @test MOI.get(opt, MOI.ObjectiveBound()) ≈ 0.814814 atol=1e-4
#     end
#
#     @testset "Lotka-Volterra" begin
#         opt = build_namedpoly(:lotkavolterra, 3, native=false)
#         MOI.optimize!(opt)
#         @test MOI.get(opt, MOI.TerminationStatus()) == MOI.Success
#         @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -20.8 atol=1e-4
#         @test MOI.get(opt, MOI.ObjectiveBound()) ≈ -20.8 atol=1e-4
#     end
# end
