using NormalizingFlows
using Test

@testset "MADE" begin
    include("MADE_flows_test.jl")
end

@testset "Planar Flow" begin
    include("Planar_test.jl")
end

# @testset "Linear Flow" begin
#     include("GLOW_test.jl")
# end