

@testset "StaticDimerMethod vs BBDimer" begin

println("Test with the muller potential")
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.002, a_rot=0.002, len=1e-3, maxnit=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.002, a0_rot=0.002, maxnumdE=100, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test bblog.res_trans[end] <= dimer.tol_trans
@test bblog.res_rot[end] <= dimer.tol_rot
@test vecnorm(xbb - x, Inf) < 1e-4


dimer = StaticDimerMethod( a_trans=0.25, a_rot=0.5, len=1e-3, maxnit=100,
      verbose=2, precon=eye(2), precon_rot=true,
      precon_prep! = (P,x) -> hessprecond(V, x) )
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot


dimer = BBDimer( a0_trans=0.25, a0_rot=0.5, len=1e-3, maxnumdE=100,
      verbose=verbose, precon=eye(2), precon_rot=true,
      precon_prep! = (P,x) -> hessprecond(V, x) )
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot


##### TODO: this test still fails - WHY?!?!?
# dimer = BBDimer( a0_trans=0.1, a0_rot=0.1, len=1e-3, maxnumdE=100,
#       verbose=2, precon=eye(2), precon_rot=true,
#       precon_prep! = (P,x) -> hessprecond(V, x),
#       ls = Backtracking() )
# x, v, log = run!(dimer, E, dE, x0, v0)
# @test log.res_trans[end] <= dimer.tol_trans
# @test log.res_rot[end] <= dimer.tol_rot




println("Test with the standard double-well")
V = DoubleWell()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.66, a_rot=0.4, len=1e-3, maxnit=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.5, a0_rot=0.5, maxnumdE=100, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test bblog.res_trans[end] <= dimer.tol_trans
@test bblog.res_rot[end] <= dimer.tol_rot
@test vecnorm(xbb - x, Inf) < 1e-4


println("Test with ill-conditioned double-well")
V = DoubleWell(diagm([1.0, 10.0]))
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.1, a_rot=0.1, len=1e-3, maxnit=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.1, a0_rot=0.1, maxnumdE=100, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test bblog.res_trans[end] <= dimer.tol_trans
@test bblog.res_rot[end] <= dimer.tol_rot
@test vecnorm(xbb - x, Inf) < 1e-4

println("same test, but now with preconditioner")
dimer = StaticDimerMethod(a_trans=0.66, a_rot=0.4, len=1e-3, maxnit=100,
                           verbose=verbose, precon=V.A, precon_rot=true)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.66, a0_rot=0.4, maxnumdE=100,
            precon=V.A, precon_rot=true, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test bblog.res_trans[end] <= dimer.tol_trans
@test bblog.res_rot[end] <= dimer.tol_rot
@test vecnorm(xbb - x, Inf) < 1e-4

end
