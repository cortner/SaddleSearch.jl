
using SaddleSearch, LinearAlgebra, Test
using SaddleSearch.TestSets

using SaddleSearch: res_trans, res_rot

verbose = 1

##

@info("TEST: StaticDimer, BBDimer with and without Preconditioning")

@info("Muller potential")

V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimer(a_trans=0.002, a_rot=0.002, len=1e-3, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.002, a0_rot=0.002, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test res_trans(bblog)[end] <= dimer.tol_trans
@test res_rot(bblog)[end] <= dimer.tol_rot
@test norm(xbb - x, Inf) < 1e-4

##

@info("Muller potential and preconditioning")

dimer = StaticDimer( a_trans =0.25, a_rot=0.5, len=1e-3, verbose=verbose,
                     precon=[1.0 0.0; 0.0 1.0], precon_rot=true, 
                     precon_prep! = (P,x) -> hessprecond(V, x) )
x, v, log = run!(dimer, E, dE, x0, v0)
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot


dimer = BBDimer( a0_trans=0.25, a0_rot=0.5, len=1e-3, verbose=verbose,
                 precon=[1.0 0.0; 0.0 1.0], precon_rot=true, 
                 precon_prep! = (P,x) -> hessprecond(V, x) )
x, v, log = run!(dimer, E, dE, x0, v0)
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot



@info("Standard double-well")

V = DoubleWell()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimer(a_trans=0.66, a_rot=0.4, len=1e-3, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.5, a0_rot=0.5, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test res_trans(bblog)[end] <= dimer.tol_trans
@test res_rot(bblog)[end] <= dimer.tol_rot
@test norm(xbb - x, Inf) < 1e-4


@info("Ill-conditioned double-well")

V = DoubleWell(diagm([1.0, 10.0]))
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimer(a_trans=0.1, a_rot=0.1, len=1e-3, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.1, a0_rot=0.1, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test res_trans(bblog)[end] <= dimer.tol_trans
@test res_rot(bblog)[end] <= dimer.tol_rot
@test norm(xbb - x, Inf) < 1e-4

@info("Ill-conditioned double-well, but now with preconditioner")

dimer = StaticDimer(a_trans=0.66, a_rot=0.4, len=1e-3, verbose=verbose,
      precon=V.A, precon_rot=true)
x, v, log = run!(dimer, E, dE, x0, v0)
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.66, a0_rot=0.4, precon=V.A,
      precon_rot=true, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test res_trans(bblog)[end] <= dimer.tol_trans
@test res_rot(bblog)[end] <= dimer.tol_rot
@test norm(xbb - x, Inf) < 1e-4

