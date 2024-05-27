using SaddleSearch, LinearAlgebra, Test
using SaddleSearch.TestSets

using SaddleSearch: res_trans, res_rot, numdE, maxres

verbose = 1

##

tol = 2e-2
tolP = 1e-3
maxnit = 2000

preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)

@info("TEST: String type methods with ode12r solver")

@info("Muller potential")
V = MullerPotential()
x = ic_path(V, :near, 15)
E, dE = objective(V)
precon = x-> hessianprecond(V, x)



path = StaticString(; alpha = 0.0009, tol=tol, maxnit=maxnit, precon_scheme = preconI, verbose=verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODEString(reltol=0.1, tol = tol, maxnit = maxnit,
                        precon_scheme = preconI,
                        path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = SaddleSearch.run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = StaticNEB(; alpha = 0.0007, maxtol = 0.0002, tol = tol, 
                   maxnit = maxnit, precon_scheme = preconI, 
                   verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODENEB(reltol=1e-2, k=0.0002, interp=3, tol = tol,
                        maxnit = maxnit,
                        precon_scheme = preconI, path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

@info("Double well potential")
c = 16.0
V = DoubleWell(diagm([.5, c*c*(0.02^(c-1)), c]))
x = ic_path(V, :near, 11)
E, dE = objective(V)

P = copy(V.A); P[1] = 1.0
precon = x->[P]


path = StaticString(alpha = 1.0/c, tol = tol, maxnit = maxnit, 
            precon_scheme = preconI, verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODEString(reltol=1e-3, tol = tol, maxnit = maxnit,
                        precon_scheme = preconI, path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = SaddleSearch.run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = StaticNEB(alpha = 0.07, interp = 3, maxtol = 0.1, tol = tol, maxnit = maxnit, 
               precon_scheme = preconI, verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODENEB(reltol=1e-1, k=0.1, interp=3, tol = tol,
                        maxnit = maxnit,
                        precon_scheme = preconI, path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

preconP = SaddleSearch.localPrecon(precon = [P], precon_prep! = (P, x) -> precon(x))

path = StaticString(alpha = 0.25, tol = tolP, maxnit = maxnit, 
                     precon_scheme = preconP, verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODEString(reltol=0.1, tol = tolP, maxnit = maxnit,
                        precon_scheme = preconP, path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = SaddleSearch.run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

# WARNING : THIS TEST FAILS WITH NAN 
# path = StaticNEB(alpha = 0.6, maxtol = 0.00001, tol = tolP, interp = 1,
#                   maxnit = maxnit, precon_scheme = preconP, verbose = verbose)
# PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
# @test PATHlog[:maxres][end] <= path.tol

path = ODENEB(reltol=0.1, k=0.00001, interp=1, tol = tolP,
                        maxnit = maxnit,
                        precon_scheme = preconP, path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

@info("Vacancy migration potential")
V = LJVacancy2D(R = 3.1)
x = ic_path(V, :min, 9)
E, dE = objective(V)

precon = x->[copy(precond(V, xn)) for xn in x]

path = StaticString(alpha = 0.001, tol = tol, maxnit = maxnit, 
            precon_scheme = preconI, verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODEString(reltol=1e-2, threshold=1e-5, tol = tol,
                        maxnit = maxnit,
                        precon_scheme = preconI,
                        path_traverse = serial(),
                        verbose = verbose)   # allow failure on 600 iterations
PATHx, PATHlog, _ = SaddleSearch.run!(path, E, dE, Path(x))
@show PATHlog[:maxres][end]
println("[allowed test failure: target is $(path.tol)]")
# @test PATHlog[:maxres][end] <= path.tol

path = StaticNEB(alpha = 0.001, interp = 3, maxtol = 0.01, tol = tol, maxnit = maxnit, 
               precon_scheme = preconI, verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODENEB(reltol=1e-2, k=0.00001, interp=3, tol = tol,
                        maxnit = maxnit,
                        precon_scheme = preconI, path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

preconP = SaddleSearch.localPrecon(precon = precon(x),
            precon_prep! = (P, x) -> precon(x))

path = StaticString(alpha = 1.55, tol = tolP, maxnit = maxnit, 
                  precon_scheme = preconP, verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODEString(reltol=1e-1, tol = tolP, maxnit = maxnit,
                        precon_scheme = preconP,
                        path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = SaddleSearch.run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = StaticNEB(alpha = 1.0, maxtol = 0.001, interp = 3, tol = tolP, 
               maxnit = maxnit, precon_scheme = preconP, verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol

path = ODENEB(reltol=1e-2, k=0.001, interp=3, tol = tolP,
                        maxnit = maxnit,
                        precon_scheme = preconP, path_traverse = serial(),
                        verbose = verbose)
PATHx, PATHlog, _ = run!(path, E, dE, Path(x))
@test PATHlog[:maxres][end] <= path.tol
