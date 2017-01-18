# TODO: tests fail with the FF-type preconditioner
#       so for the moment we just use 25 * I. This still needs to be fixed?!

using SaddleSearch
using SaddleSearch.TestSets
using SaddleSearch.TestSets: hessprecond, precond, hessian

@testset "Molecule2D Tests" begin

   println("Molecule2D Test and Benchmark: ")
   locverb = 1

   # NOTE: this test does not seem terribly robust; many of the methods
   #       fail for many values of μ. We should probably include the
   #       superlinear dimer method in this test.

   # for μ in (3.0, 5.0, 10.0)
   for μ in (5.0,)

      println("μ = $μ")
      V = Molecule2D()  # kbb=μ, kab=μ^2)
      x0, v0 = ic_dimer(V, :near)
      E, dE = objective(V)

      P0 = μ^2 * eye(3)

      dimer = StaticDimerMethod(a_trans=0.5/μ^2, a_rot=0.5/μ^2, len=1e-3,
                                 maxnit=3000, verbose=locverb, rescale_v = true)
      x, v, log = run!(dimer, E, dE, x0, v0)
      @test log.res_trans[end] <= dimer.tol_trans
      @test log.res_rot[end] <= dimer.tol_rot
      println("      Dimer(I): $(length(log.res_trans)) iterations")

      bbdimer = BBDimer(a0_trans=0.5/μ^2, a0_rot=0.5/μ^2, maxnumdE=1000, verbose=locverb,
                        ls = StaticLineSearch(), rescale_v = false)
      xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
      @test bblog.res_trans[end] <= bbdimer.tol_trans
      @test bblog.res_rot[end] <= bbdimer.tol_rot
      @test vecnorm(xbb - x, Inf) < 1e-4
      println("   BB-Dimer(I): $(length(bblog.res_trans)) iterations")

      dimer = StaticDimerMethod( a_trans=0.9, a_rot=0.5, len=1e-3, maxnit=100,
            verbose=locverb, precon=P0, precon_rot=true,
            precon_prep! = (P,x) -> P, rescale_v = true )
      x, v, log = run!(dimer, E, dE, x0, v0)
      @test log.res_trans[end] <= dimer.tol_trans
      @test log.res_rot[end] <= dimer.tol_rot
      println("      Dimer(P): $(length(log.res_trans)) iterations")

      bbdimer = BBDimer( a0_trans=0.25, a0_rot=0.5, len=1e-3, maxnumdE=100,
            verbose=locverb, precon=P0, precon_rot=true,
            rescale_v = true, precon_prep! = (P,x) -> P,
            ls = StaticLineSearch() )
      x, v, log = run!(bbdimer, E, dE, x0, v0)
      @test log.res_trans[end] <= bbdimer.tol_trans
      @test log.res_rot[end] <= bbdimer.tol_rot
      println("   BB-Dimer(P): $(length(log.res_trans)) iterations")
   end
end
