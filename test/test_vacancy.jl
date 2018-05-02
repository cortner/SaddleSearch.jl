
@testset "LJ Vacancy Tests" begin

   heading1("LJVacancy2D Test and Benchmark: ")

   locverb = 0

   for R in (3.1, 4.1, 5.1)    # add 7.1 ??

      V = LJVacancy2D(R = R, bc = :free)
      x0, v0 = ic_dimer(V, :near)
      E, dE = objective(V)

      heading2("Domain with R = $R, Nat = $(size(V.Xref, 2))")

      dimer = StaticDimerMethod(a_trans=0.002, a_rot=0.002, len=1e-3,
                                 maxnumdE=3000, verbose=locverb)
      x, v, log = run!(dimer, E, dE, x0, v0)
      @test res_trans(log)[end] <= dimer.tol_trans
      @test res_rot(log)[end] <= dimer.tol_rot
      println("      Dimer(I): $(numdE(log)[end]) ∇E evaluations")

      bbdimer = BBDimer(a0_trans=0.001, a0_rot=0.001, maxnumdE=1000, verbose=locverb,
                        ls = StaticLineSearch())
      xbb, vbb, bblog = run!(bbdimer, E, dE, x0, v0)
      @test res_trans(bblog)[end] <= bbdimer.tol_trans
      @test res_rot(bblog)[end] <= bbdimer.tol_rot
      # @test vecnorm(xbb - x, Inf) < 1e-4
      println("   BB-Dimer(I): $(numdE(bblog)[end]) ∇E evaluations")

      supdimer = SuperlinearDimer(maximum_translation=0.1, max_num_rot=3,
                                  len=1e-3, maxnumdE=1000, verbose=locverb)
      xsup, vsup, logsup = run!(supdimer, E, dE, x0, v0)
      @test res_trans(logsup)[end] <= supdimer.tol_trans
      @test res_rot(logsup)[end] <= supdimer.tol_rot
      println("Superlinear(I): $(numdE(logsup)[end]) ∇E evaluations")


      dimer = StaticDimerMethod( a_trans=1.0, a_rot=0.3, len=1e-3, maxnumdE=500,
            verbose=locverb, precon=precond(V, x0), precon_rot=true,  rescale_v=true,
            precon_prep! = (P,x) -> precond(V, x) )
      x, v, log = run!(dimer, E, dE, x0, v0)
      @test res_trans(log)[end] <= dimer.tol_trans
      @test res_rot(log)[end] <= dimer.tol_rot
      println("      Dimer(P): $(numdE(log)[end]) ∇E evaluations")

      bbdimer = BBDimer( a0_trans=0.25, a0_rot=0.5, len=1e-3, maxnumdE=100,
            verbose=locverb, precon=precond(V, x0), precon_rot=true,
            precon_prep! = (P,x) -> precond(V, x), rescale_v=true,
            ls = StaticLineSearch() )
      x, v, log = run!(bbdimer, E, dE, x0, v0)
      @test res_trans(log)[end] <= bbdimer.tol_trans
      @test res_rot(log)[end] <= bbdimer.tol_rot
      println("   BB-Dimer(P): $(numdE(log)[end]) ∇E evaluations")

   end

end
