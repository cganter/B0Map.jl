using Test

@testset "GREMultiEchoWFFW.jl" begin
    include("test_GREMultiEchoWFFW.jl")
end;

@testset "GREMultiEchoWFRW.jl" begin
    include("test_GREMultiEchoWFRW.jl")
end;

@testset "GREMultiEchoWF.jl" begin
    include("test_GREMultiEchoWF.jl")
end;

