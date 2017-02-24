using SaddleSearch.TestSets: hessprecond

# # MullerPotential with good IC
# V = MullerPotential()
# x0, v0 = ic_dimer(V, :near)
# E, dE = objective(V)
# dimer = SuperlinearDimer(maximum_translation=0.1, max_num_rot=1, maxnumdE=100, verbose=2)
# run!(dimer, E, dE, x0, v0)


# MullerPotential with good IC
V = LJcluster()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
dimer = SuperlinearDimer(maximum_translation=0.1, max_num_rot=1, maxnumdE=500, verbose=verbose)
x, v, res = run!(dimer, E, dE, x0, v0)
@show numE(res)[end], numdE(res)[end]

# MullerPotential with good IC
bb = BBDimer(verbose=2, a0_trans=0.01, a0_rot = 0.01) # , ls = Backtracking())
x, v, resbb = run!(bb, E, dE, x0, v0)
@show numE(resbb)[end], numdE(resbb)[end]



# precon_prep! = (P,x) -> hessprecond(V, x)+0.1*eye(length(x))
