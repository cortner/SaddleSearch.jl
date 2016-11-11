using SaddleSearch
using Base.Test

include("testsets.jl")

@testset "StaticDimerMethod" begin

println("Test with the muller potential")
println("------------------------------")
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.002, a_rot=0.002, len=1e-3, maxnit=100)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot


println("Test with the standard double-well")
println("-----------------------------------")

V = DoubleWell()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.5, a_rot=0.2, len=1e-3, maxnit=100)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

end
