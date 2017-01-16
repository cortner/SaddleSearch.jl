
using SaddleSearch
using SaddleSearch.TestSets
using SaddleSearch.TestSets: hessprecond, precond, hessian

# @testset "LJ Vacancy Tests" begin

println("Test with the LJVacancy2D potential")
V = LJVacancy2D(R = 5.1, bc = :free)
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

dimer = StaticDimerMethod(a_trans=0.002, a_rot=0.002, len=1e-3, maxnit=1000, verbose=verbose)
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot

# bbdimer = BBDimer(a0_trans=0.001, a0_rot=0.001, maxnumdE=1000, verbose=verbose)
# xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
# @test bblog.res_trans[end] <= dimer.tol_trans
# @test bblog.res_rot[end] <= dimer.tol_rot
# @test vecnorm(xbb - x, Inf) < 1e-4


dimer = StaticDimerMethod( a_trans=0.5, a_rot=0.3, len=1e-3, maxnit=500,
      verbose=verbose, precon=eye(2), precon_rot=true,
      precon_prep! = (P,x) -> precond(V, x) )
x, v, log = run!(dimer, E, dE, x0, v0)
@test log.res_trans[end] <= dimer.tol_trans
@test log.res_rot[end] <= dimer.tol_rot


# dimer = BBDimer( a0_trans=0.25, a0_rot=0.5, len=1e-3, maxnumdE=100,
#       verbose=verbose, precon=eye(2), precon_rot=false,
#       precon_prep! = (P,x) -> exp_precond(V, x),
#       ls = StaticLineSearch() )
# x, v, log = run!(dimer, E, dE, x0, v0)
# @test log.res_trans[end] <= dimer.tol_trans
# @test log.res_rot[end] <= dimer.tol_rot

# end





# H = hessian(V, x0)
# P = hessprecond(V, x0; stab=0.01)
# eP = eigvals(full(H), full(P))
# @show maximum(eP) / minimum(eP)
# eI = eigvals(full(H))
# @show maximum(eI) / minimum(eI)
#
# println( full(H[1:10,1:10]) )
# println( full(H[1:2:20, 1:2:20]) )
#
# P = exp_precond(V, x0)
# println( full(P[1:10,1:10]) )
# println( full(P[1:2:20, 1:2:20]) )
