using SaddleSearch
using Base.Test

include("testsets.jl")

verbose=1

@testset "StaticDimerMethod" begin

println("Test with the muller potential")
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.002, a_rot=0.002, len=1e-3, maxnit=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot


println("Test with the standard double-well")
V = DoubleWell()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.66, a_rot=0.4, len=1e-3, maxnit=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot


println("Test with ill-conditioned double-well")
V = DoubleWell(diagm([1.0, 10.0]))
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.1, a_rot=0.1, len=1e-3, maxnit=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

println("same test, but now with preconditioner")
dimer = StaticDimerMethod(a_trans=0.66, a_rot=0.4, len=1e-3, maxnit=100,
                           verbose=verbose, precon=V.A, precon_rot=true)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

end
