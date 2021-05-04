using NormalizingFlows, Random
using Base.Test

@testset "MADE" begin
    include("MADE_flows_test.jl")
end

@testset "Planar" begin
    include("Planar_test.jl")
end

