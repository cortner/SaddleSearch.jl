tol = 2e-2
tolP = 1e-3
maxnit = 2000

# @testset "String with ODE" begin

preconI = SaddleSearch.localPrecon(precon = [I], precon_prep! = (P, x) -> P)

heading1("TEST: String type methods with ode12r solver")

# heading2("Muller potential")
# V = MullerPotential()
# x = ic_path(V, :near, 15)
# E, dE = objective(V)
# precon = x-> hessianprecond(V, x)
#
# path = StaticString(0.0009, tol, maxnit, preconI, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODEString(reltol=0.1, tol = tol, maxnit = maxnit,
#                         precon_scheme = preconI, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = SaddleSearch.run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = StaticNEB(0.0007, 0.0002, 1, tol, maxnit, preconI, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODENEB(reltol=1e-2, k=0.0002, interp=1, tol = tol, maxnit = maxnit,
#                         precon_scheme = preconI, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# heading2("Double well potential")
# c = 16.0
# V = DoubleWell(diagm([.5, c*c*(0.02^(c-1)), c]))
# x = ic_path(V, :near, 11)
# E, dE = objective(V)
#
# P = copy(V.A); P[1] = 1.0
# precon = x->[P]
#
# path = StaticString(1./c, tol, maxnit, preconI, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODEString(reltol=1e-3, tol = tol, maxnit = maxnit,
#                         precon_scheme = preconI, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = SaddleSearch.run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = StaticNEB(0.07, 0.1, 3, tol, maxnit, preconI, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODENEB(reltol=1e-1, k=0.1, interp=1, tol = tol, maxnit = maxnit,
#                         precon_scheme = preconI, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# preconP = SaddleSearch.localPrecon(precon = [P], precon_prep! = (P, x) -> precon(x))
#
# path = StaticString(0.25, tolP, maxnit, preconP, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODEString(reltol=0.1, tol = tolP, maxnit = maxnit,
#                         precon_scheme = preconP, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = SaddleSearch.run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = StaticNEB(0.6, 0.00001, 1, tolP, maxnit, preconP, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODENEB(reltol=0.1, k=0.00001, interp=1, tol = tolP, maxnit = maxnit,
#                         precon_scheme = preconP, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol

heading2("Vacancy migration potential")
V = LJVacancy2D(R = 3.1)
x = ic_path(V, :min, 9)
E, dE = objective(V)

precon = x->[copy(precond(V, xn)) for xn in x]

# path = StaticString(0.001, tol, maxnit, preconI, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODEString(reltol=1e-2, threshold=1e-5, tol = tol, maxnit = maxnit,
#                         precon_scheme = preconI, path_traverse = serial(),
#                         verbose = 1)   # allow failure on 600 iterations
# PATHx, PATHlog = SaddleSearch.run!(path, E, dE, x)
# @show PATHlog[:maxres][end]
# println("[allowed test failure: target is $(path.tol)]")
# # @test PATHlog[:maxres][end] <= path.tol
#
# path = StaticNEB(0.001, 0.01, 3, tol, maxnit, preconI, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODENEB(reltol=1e-2, k=0.00001, interp=1, tol = tol, maxnit = maxnit,
#                         precon_scheme = preconI, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# preconP = SaddleSearch.localPrecon(precon = precon(x),
# precon_prep! = (P, x) -> precon(x))
#
# path = StaticString(1.55, tolP, maxnit, preconP, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODEString(reltol=1e-1, tol = tolP, maxnit = maxnit,
#                         precon_scheme = preconP, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = SaddleSearch.run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = StaticNEB(1.0, 0.001, 3, tolP, maxnit, preconP, serial(), 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol
#
# path = ODENEB(reltol=1e-2, k=0.001, interp=1, tol = tolP, maxnit = maxnit,
#                         precon_scheme = preconP, path_traverse = serial(),
#                         verbose = 1)
# PATHx, PATHlog = run!(path, E, dE, x)
# @test PATHlog[:maxres][end] <= path.tol

path = SaddleSearch.LBFGSNEB(hmax = 0.03, k=0.01, interp=3, tol = tol, maxnit = 70,
                        precon_scheme = preconI, path_traverse = serial(),
                        verbose = 2)
PATHx, PATHlog = run!(path, E, dE, x)
@test PATHlog[:maxres][end] <= path.tol

# end
