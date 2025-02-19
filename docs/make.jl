using B0Map
using Documenter

DocMeta.setdocmeta!(B0Map, :DocTestSetup, :(using B0Map); recursive=true)

makedocs(;
    #modules=[B0Map],
    authors="Carl Ganter <cganter@tum.de>",
    sitename="B0Map.jl",
    format=Documenter.HTML(;
        canonical="https://cganter.github.io/B0Map.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Multi-echo GRE signal models" => Any[
            "Unconstrained water-fat mixture" => "man/WFFW.md",
            "Real-valued water fat mixture" => "man/WFRW.md",
            "Constrained water-fat mixture" => "man/WF.md",
        ],
        "Data fitting" => Any[
            "General steps" => "man/fittools.md",
            "Local fitting" => "man/localfit.md",
            "PHASER" => "man/phaser.md",
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
