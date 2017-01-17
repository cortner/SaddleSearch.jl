# TODO: tests fail with the FF-type preconditioner
#       so for the moment we just use 25 * I. This still needs to be fixed?!

using SaddleSearch
using SaddleSearch.TestSets
using SaddleSearch.TestSets: hessprecond, precond, hessian

@testset "Molecule2D Tests" begin

   println("Molecule2D Test and Benchmark: ")
   locverb = 0

   V = Molecule2D()
   x0, v0 = ic_dimer(V, :near)
   E, dE = objective(V)

   # println("R = $R : Nat = $(size(V.Xref, 2))")

   dimer = StaticDimerMethod(a_trans=0.05, a_rot=0.04, len=1e-3,
                              maxnit=3000, verbose=locverb, rescale_v = true)
   x, v, log = run!(dimer, E, dE, x0, v0)
   @test log.res_trans[end] <= dimer.tol_trans
   @test log.res_rot[end] <= dimer.tol_rot
   println("      Dimer(I): $(length(log.res_trans)) iterations")

   bbdimer = BBDimer(a0_trans=0.01, a0_rot=0.01, maxnumdE=1000, verbose=locverb,
                     ls = StaticLineSearch(), rescale_v = false)
   xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
   @test bblog.res_trans[end] <= bbdimer.tol_trans
   @test bblog.res_rot[end] <= bbdimer.tol_rot
   @test vecnorm(xbb - x, Inf) < 1e-4
   println("   BB-Dimer(I): $(length(bblog.res_trans)) iterations")

   dimer = StaticDimerMethod( a_trans=0.9, a_rot=0.5, len=1e-3, maxnit=100,
         verbose=locverb, precon=25.0*eye(3), precon_rot=true,
         precon_prep! = (P,x) -> P, rescale_v = true )   # precond(V, x)
   x, v, log = run!(dimer, E, dE, x0, v0)
   @test log.res_trans[end] <= dimer.tol_trans
   @test log.res_rot[end] <= dimer.tol_rot
   println("      Dimer(P): $(length(log.res_trans)) iterations")

   bbdimer = BBDimer( a0_trans=0.25, a0_rot=0.5, len=1e-3, maxnumdE=100,
         verbose=locverb, precon=25.0*eye(3), precon_rot=true,
         rescale_v = true,
         precon_prep! = (P,x) -> P,
         ls = StaticLineSearch() )
   x, v, log = run!(bbdimer, E, dE, x0, v0)
   @test log.res_trans[end] <= bbdimer.tol_trans
   @test log.res_rot[end] <= bbdimer.tol_rot
   println("   BB-Dimer(P): $(length(log.res_trans)) iterations")

end





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







# println("cond(min)")
# xm = TestSets.mol2dpath(2*π/3)
# H = hessian(V, xm)
# @show eigvals(H)
# P = precond(V, xm)
# @show cond(H)
# Q = pinv(sqrtm(P))
# @show cond(Q * H * Q)
#
# dimer = StaticDimerMethod(a_trans=0.05, a_rot=0.04, len=1e-3,
#                            maxnit=3000, verbose=false)
# xs, v, log = run!(dimer, E, dE, x0, v0)
#
# println("cond(saddle)")
# # xs = x0
# H = hessian(V, xs)
# P = precond(V, xs)
# Q = pinv(sqrtm(P))
# @show eigvals(H)
# @show eigvals(P)
# @show eigvals(H, P)
# @show cond(H)
# @show cond(Q * H * Q)
#
# println("cond(saddle-improved-P)")
# D, V = eig(H, P)
# @show D
# Imin = find(D .== minimum(D))[1]
# v = V[:, Imin]
# P2 = (eye(3) - v * v') * P * (eye(3) - v * v') + D[Imin] * (v * v')
# Q2 = pinv(sqrtm(P2))
# @show eigvals(H, P2)
# @show cond(Q2 * H * Q2)
#
# println("cond(saddle-improved-P)")
# D, V = eig(H)
# Imin = find(D .== minimum(D))[1]
# v = V[:, Imin]
# P2 = (eye(3) - v * v') * P * (eye(3) - v * v') + D[Imin] * (v * v')
# Q2 = pinv(sqrtm(P2))
# @show eigvals(H, P2)
# @show cond(Q2 * H * Q2)
#
# println("cond(saddle-hessprecond)")
# P3 = V * diagm(abs(D)) * V'
# Q3 = pinv(sqrtm(P3))
# @show eigvals(H, P3)
# @show cond(Q3 * H * Q3)
