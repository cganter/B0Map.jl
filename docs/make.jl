using B0Map
using Documenter

DocMeta.setdocmeta!(B0Map, :DocTestSetup, :(using B0Map); recursive=true)

makedocs(;
    authors="Carl Ganter <cganter@tum.de>",
    sitename="B0Map.jl",
    format=Documenter.HTML(;
        canonical="https://cganter.github.io/B0Map.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quick Start" => Any[
            "Set up parameters" => "man/fittools.md",
            "Local fit" => "man/localfit.md",
            "Regularized fit" => "man/phaser.md",
            "Get results" => "man/results.md",
        ],
        "Signal models" => Any[
            "Multi-echo GRE" => "man/GREgeneral.md",
            "Unconstrained water-fat mixture" => "man/WFFW.md",
            "Real-valued water fat mixture" => "man/WFRW.md",
            "Constrained water-fat mixture" => "man/WF.md",
        ],
        "API" => Any[
            "man/api.md",
        ],
    ],
)

#deploydocs(;
#    repo="github.com/cganter/B0Map.jl",
#    devbranch="main",
#)
