using B0Map
using Documenter

DocMeta.setdocmeta!(B0Map, :DocTestSetup, :(using B0Map); recursive=true)

makedocs(;
    modules=[B0Map],
    authors="Carl Ganter <cganter@tum.de>",
    sitename="B0Map.jl",
    format=Documenter.HTML(;
        canonical="https://cganter.github.io/B0Map.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cganter/B0Map.jl",
    devbranch="main",
)
