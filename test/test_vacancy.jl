
using SaddleSearch
using SaddleSearch.TestSets
using SaddleSearch.TestSets: hessprecond, precond, hessian

@testset "LJ Vacancy Tests" begin

   println("LJVacancy2D Test and Benchmark: ")
   locverb = 0

   for R in (3.1, 4.1, 5.1, 7.1)

      V = LJVacancy2D(R = R, bc = :free)
      x0, v0 = ic_dimer(V, :near)
      E, dE = objective(V)

      println("R = $R : Nat = $(size(V.Xref, 2))")

      dimer = StaticDimerMethod(a_trans=0.002, a_rot=0.002, len=1e-3,
                                 maxnit=3000, verbose=locverb)
      x, v, log = run!(dimer, E, dE, x0, v0)
      @test res_trans(log)[end] <= dimer.tol_trans
      @test res_rot(log)[end] <= dimer.tol_rot
      println("      Dimer(I): $(length(res_trans(log))) iterations")

      bbdimer = BBDimer(a0_trans=0.001, a0_rot=0.001, maxnumdE=1000, verbose=locverb,
                        ls = StaticLineSearch())
      xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
      @test res_trans(bblog)[end] <= bbdimer.tol_trans
      @test res_rot(bblog)[end] <= bbdimer.tol_rot
      # @test vecnorm(xbb - x, Inf) < 1e-4
      println("   BB-Dimer(I): $(length(res_trans(bblog))) iterations")

      dimer = StaticDimerMethod( a_trans=1.0, a_rot=0.3, len=1e-3, maxnit=500,
            verbose=locverb, precon=precond(V, x0), precon_rot=true,  rescale_v=true,
            precon_prep! = (P,x) -> precond(V, x) )
      x, v, log = run!(dimer, E, dE, x0, v0)
      @test res_trans(log)[end] <= dimer.tol_trans
      @test res_rot(log)[end] <= dimer.tol_rot
      println("      Dimer(P): $(length(res_trans(log))) iterations")

      bbdimer = BBDimer( a0_trans=0.25, a0_rot=0.5, len=1e-3, maxnumdE=100,
            verbose=locverb, precon=precond(V, x0), precon_rot=true,
            precon_prep! = (P,x) -> precond(V, x), rescale_v=true,
            ls = StaticLineSearch() )
      x, v, log = run!(bbdimer, E, dE, x0, v0)
      @test res_trans(log)[end] <= bbdimer.tol_trans
      @test res_rot(log)[end] <= bbdimer.tol_rot
      println("   BB-Dimer(P): $(length(res_trans(log))) iterations")

   end

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
