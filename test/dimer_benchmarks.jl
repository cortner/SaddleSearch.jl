
# TODO: add preconditioner to the Problem description??

struct WalkerProblem
   V
   E
   dE
   x0
   v0
   id
end


testsets = []
methods = []

maxnumdE = 1_000

# ============ Generate the Benchmark Tests ===========

# TODO: maybe we can assign a "good" initial step too all the methods?

# MullerPotential with good IC
V = MullerPotential()
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)
push!(testsets, WalkerProblem(V, E, dE, x0, v0, "Muller:near"))

# MüllerPotential with bad IC
V = MullerPotential()
x0, v0 = ic_dimer(V, :far)
E, dE = objective(V)
push!(testsets, WalkerProblem(V, E, dE, x0, v0, "Muller:far"))



# ============ List of Methods to Benchmark ===========

push!(methods,  BBDimer(a0_trans=0.002, a0_rot=0.002, maxnumdE=maxnumdE,
                        verbose=verbose, ls = StaticLineSearch(), id="BBDimer") )

push!(methods,  BBDimer(a0_trans=0.002, a0_rot=0.002, maxnumdE=maxnumdE,
                        verbose=verbose, ls = Backtracking(), id="BBDimer+LS") )

push!(methods,  RotOptimDimer(a_trans=0.001, maxnumdE=maxnumdE, verbose=verbose) )

push!(methods,  SuperlinearDimer(maximum_translation=0.1, max_num_rot=1, maxnumdE=maxnumdE, verbose=verbose) )


# ============= run the all the benchmarks ================

using DataFrames
results = DataFrame()
results[:Methods] = [m.id for m in methods]
for t in testsets
   r = Int[]
   for method in methods
      try
         x, v, log = run!(method, t.E, t.dE, t.x0, t.v0)
         push!(r, maximum(log.numdE))
         @show x, v
      catch e
         println("($(t.id), $(method.id)) : failed")
         @show e
         push!(r, -1)
      end
      # TODO: test the result is correct
   end
   results[Symbol(t.id)] = r
end

display(results)
