

@testset "Dimer with Linesearch" begin

heading1("TEST: Dimer type methods with linesearch")

heading2("Muller potential")
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimer(a_trans=0.002, a_rot=0.002, len=1e-3, maxnumdE=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, copy(x0), copy(v0))
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.002, a0_rot=0.002, maxnumdE=100, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, copy(x0), copy(v0))
@test res_trans(bblog)[end] <= dimer.tol_trans
@test res_rot(bblog)[end] <= dimer.tol_rot
@test vecnorm(xbb - x, Inf) < 1e-4

lsbbdimer = BBDimer(a0_trans=0.002, a0_rot=0.002, maxnumdE=100, verbose=verbose,
                     ls = Backtracking() )
xls, vls, lslog = run!(lsbbdimer, E, dE, copy(x0), copy(v0))
@test res_trans(lslog)[end] <= lsbbdimer.tol_trans
@test res_rot(lslog)[end] <= lsbbdimer.tol_rot
@test vecnorm(xls - x, Inf) < 1e-4

supdimer = SuperlinearDimer(maximum_translation=0.02, max_num_rot=1, len=1e-3,
                            maxnumdE=1000, verbose=verbose)
xsup, vsup, logsup = run!(supdimer, E, dE, copy(x0), copy(v0))
@test res_trans(logsup)[end] <= supdimer.tol_trans
@test vecnorm(xsup - x, Inf) < 1e-4

odedimer = ODEDimer(verbose=verbose)
xode, vode, logode = run!(odedimer, E, dE, copy(x0), copy(v0))
# @test res_trans(logode)[end] <= odedimer.tol_trans
@test maxres(logode)[end] <= odedimer.tol_trans
@test vecnorm(xode - x, Inf) < 1e-4


heading2("Standard double-well potential")

V = DoubleWell()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimer(a_trans=0.66, a_rot=0.4, len=1e-3, maxnumdE=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test res_trans(log)[end] <= dimer.tol_trans
@test res_rot(log)[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.66, a0_rot=0.4, maxnumdE=100, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test res_trans(bblog)[end] <= dimer.tol_trans
@test res_rot(bblog)[end] <= dimer.tol_rot
@test vecnorm(xbb - x, Inf) < 1e-4

lsbbdimer = BBDimer(a0_trans=0.66, a0_rot=0.4, maxnumdE=100, verbose=verbose,
                     ls = Backtracking() )
xls, vls, lslog = run!(lsbbdimer, E, dE, x0, v0)
@test res_trans(lslog)[end] <= lsbbdimer.tol_trans
@test res_rot(lslog)[end] <= lsbbdimer.tol_rot
@test vecnorm(xls - x, Inf) < 1e-4

supdimer = SuperlinearDimer(maximum_translation=1.0, max_num_rot=1, len=1e-3,
                            maxnumdE=1000, verbose=verbose)
xsup, vsup, logsup = run!(supdimer, E, dE, x0, v0)
@test res_trans(logsup)[end] <= supdimer.tol_trans
@test vecnorm(xsup - x, Inf) < 1e-4

odedimer = ODEDimer(verbose=verbose)
xode, vode, logode = run!(odedimer, E, dE, copy(x0), copy(v0))
# @test res_trans(logode)[end] <= odedimer.tol_trans
@test maxres(logode)[end] <= odedimer.tol_trans
@test vecnorm(xode - x, Inf) < 1e-4

end
