using SaddleSearch
using Base.Test

include("testsets.jl")

V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.002, a_rot=0.002, len=1e-3, maxnit=100)
x, v, log = run!(dimer, E, dE, x0, v0)
