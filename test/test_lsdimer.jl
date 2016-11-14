

@testset "Dimer with Linesearch" begin

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

lsbbdimer = BBDimer(a0_trans=0.002, a0_rot=0.002, maxnumdE=100, verbose=verbose,
                     ls = Backtracking() )
xls, vls, lslog = run!(lsbbdimer, E, dE, x0, v0)
@test lslog.res_trans[end] <= lsbbdimer.tol_trans
@test lslog.res_rot[end] <= lsbbdimer.tol_rot
@test vecnorm(xls - x, Inf) < 1e-4

lsbbdimer = BBDimer(a0_trans=0.0005, a0_rot=0.0005, maxnumdE=100, verbose=2,
                     ls = Backtracking() )
xls, vls, lslog = run!(lsbbdimer, E, dE, x0, v0)
@test lslog.res_trans[end] <= lsbbdimer.tol_trans
@test lslog.res_rot[end] <= lsbbdimer.tol_rot
@test vecnorm(xls - x, Inf) < 1e-4

rotdimer = RotOptimDimer(a_trans=0.002, len=1e-3, maxnit=100, verbose=2)
xrot, vrot, logrot = run!(rotdimer, E, dE, x0, v0)
@test logrot.res_trans[end] <= rotdimer.tol_trans
@test vecnorm(xrot - x, Inf) < 1e-4


println("Test with the standard double-well potential")
V = DoubleWell()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.66, a_rot=0.4, len=1e-3, maxnit=100, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

bbdimer = BBDimer(a0_trans=0.66, a0_rot=0.4, maxnumdE=100, verbose=verbose)
xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
@test bblog.res_trans[end] <= dimer.tol_trans
@test bblog.res_rot[end] <= dimer.tol_rot
@test vecnorm(xbb - x, Inf) < 1e-4

lsbbdimer = BBDimer(a0_trans=0.66, a0_rot=0.4, maxnumdE=100, verbose=2,
                     ls = Backtracking() )
xls, vls, lslog = run!(lsbbdimer, E, dE, x0, v0)
@test lslog.res_trans[end] <= lsbbdimer.tol_trans
@test lslog.res_rot[end] <= lsbbdimer.tol_rot
@test vecnorm(xls - x, Inf) < 1e-4

lsbbdimer = BBDimer(a0_trans=0.1, a0_rot=0.1, maxnumdE=100, verbose=2,
                     ls = Backtracking() )
xls, vls, lslog = run!(lsbbdimer, E, dE, x0, v0)
@test lslog.res_trans[end] <= lsbbdimer.tol_trans
@test lslog.res_rot[end] <= lsbbdimer.tol_rot
@test vecnorm(xls - x, Inf) < 1e-4

rotdimer = RotOptimDimer(a_trans=0.66, len=1e-3, maxnit=100, verbose=2)
xrot, vrot, logrot = run!(rotdimer, E, dE, x0, v0)
@test logrot.res_trans[end] <= rotdimer.tol_trans
@test vecnorm(xrot - x, Inf) < 1e-4

end
