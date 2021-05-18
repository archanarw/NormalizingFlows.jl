using NormalizingFlows, Flux
using Test, Random
rng = MersenneTwister(0)

@testset "MADE" begin
    include("MADE_flows_test.jl")
end

@testset "Planar Flow" begin
    model = PlanarFlow(rng, Float32, 28^2)
    @test size.(NormalizingFlows.params(model)) == [(784,),(784,),(784,)]
    @test all(eltype.([model.v, model.w, model.b]) .== Float32)
end

@testset "Linear Flow" begin
    model = GLOW(rng, Float32, 1, 6)
    @test size(params(model)[6]) == (3,) #Checking if the coupling layer is half the size of the input
end

@testset "Radial Flow" begin
    model = RadialFlow(rng, Float32, 28^2)
    @test size.(NormalizingFlows.params(model)) == [(784,),(1,),(1,)]
    @test all(eltype.([model.z₀, model.α, model.β]) .== Float32)
end