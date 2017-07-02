using SaddleSearch, SaddleSearch.TestSets
using Base.Test
using SaddleSearch: numE, numdE, res_trans, res_rot

using SaddleSearch.TestSets: hessprecond, precond
using CTKSolvers
using SaddleSearch: ODE12r, odesolve, IterationLog

verbose = 0
translation_method = "CG"

println("Superlinear Dimer, Müller")
# MullerPotential with good IC
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=500, verbose=verbose,
            translation_method = translation_method)
x1, v, res = run!(dimer, E, dE, x0, v0)
println("   num_dE = ", numdE(res)[end])

println("Superlinear Dimer, LJcluster")
# Lennard-Jones Cluster
V = LJcluster()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=500, verbose=verbose,
            translation_method = translation_method)
x, v, res = run!(dimer, E, dE, x0, v0)
println("   num_dE = ", numdE(res)[end])

println("Superlinear Dimer, Vacancy")
V = LJVacancy2D()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=500, verbose=verbose,
            translation_method = translation_method)
x, v, res = run!(dimer, E, dE, x0, v0)
println("   num_dE = ", numdE(res)[end])


println("NSOLI, Müller")
# MullerPotential with good IC
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
sol, it_hist, ierr, x_hist = nsoli(x0, dE)
println("   num_dE = ", it_hist[:, 2][end])


println("NSOLI, LJcluster")
V = LJcluster()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
sol, it_hist, ierr, x_hist = nsoli(x0, dE)
println("   num_dE = ", it_hist[:, 2][end])

println("NSOLI, LJcluster, P(EXP)")
sol, it_hist, ierr, x_hist = nsoli(x0, x-> precond(V,x) \ dE(x))
println("  num_dE = ", it_hist[:, 2][end])

println("NSOLI, LJVacancy2D")
V = LJVacancy2D()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
sol, it_hist, ierr, x_hist = nsoli(x0, dE)
println("   num_dE = ", it_hist[:, 2][end])

println("NSOLI, LJVacancy2D, P(EXP)")
sol, it_hist, ierr, x_hist = nsoli(x0, x->precond(V,x) \ dE(x))
println("   num_dE = ", it_hist[:, 2][end])
