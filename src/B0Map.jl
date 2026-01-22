module B0Map
# tissue VARPRO models for multi-echo GRE sequences
include("GREMultiEcho.jl")
include("GREMultiEchoWFFW.jl")
include("GREMultiEchoWFRW.jl")
include("GREMultiEchoWF.jl")
# local fit
include("FitTools.jl")
include("LocalFit.jl")
# PHASER
#include("phaser.jl")
#include("BFourierLin.jl")
end
