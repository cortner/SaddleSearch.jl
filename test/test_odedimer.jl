using SaddleSearch.TestSets: hessprecond
using Isaac
using SaddleSearch: ODE12r, odesolve, IterationLog

verbose = 1

# MullerPotential with good IC
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=100, verbose=verbose)
x1, v, res = run!(dimer, E, dE, x0, v0)
@show numE(res)[end], numdE(res)[end]

# # MullerPotential with good IC
# V = MullerPotential()
# x0, v0 = ic_dimer(V, :near)
# E, dE = objective(V)
# dimer = DimerMethod(verbose=verbose, a0_trans = 0.001, a0_rot = 0.001)
# x2, v, res = run!(dimer, E, dE, x0, v0)
# @show numE(res)[end], numdE(res)[end]

# @show x1, x2


# Lennard-Jones Cluster
V = LJcluster()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=500, verbose=verbose)
x, v, res = run!(dimer, E, dE, x0, v0)
@show numE(res)[end], numdE(res)[end]


V = LJVacancy2D()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=500, verbose=verbose)
x, v, res = run!(dimer, E, dE, x0, v0)
@show numE(res)[end], numdE(res)[end]


# # MullerPotential with good IC
# V = MullerPotential()
# x0, v0 = ic_dimer(V, :near)
# E, dE = objective(V)
#
# function Fdimer(t, xv)
#    n = length(xv) ÷ 2
#    x, v = xv[1:n], xv[n+1:end]
#    v /= norm(v)
#    fx = - (eye(n) - 2 * v * v') * dE(x)
#    fv = - (eye(n) - v * v') * ((dE(x + 1e-3 * v) - dE(x))/1e-3)
#    return [fx; fv]
# end
#
# ode = ODE12r(atol = 1e-3, rtol = 1e-3)
# odesolve(ode, Fdimer, [x0;v0], 2,
#          IterationLog(:numE => Int, :numdE => Int, :res => Float64),
#          verbose = 3)


# MullerPotential with good IC
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
sol, it_hist, ierr, x_hist = nsoli(x0, dE)
@show length(x_hist)
@show it_hist[:, 2][end]


V = LJcluster()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
sol, it_hist, ierr, x_hist = nsoli(x0, dE)
@show length(x_hist)
@show it_hist[:, 2][end]


V = LJVacancy2D()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
sol, it_hist, ierr, x_hist = nsoli(x0, dE)
@show length(x_hist)
@show it_hist[:, 2][end]
