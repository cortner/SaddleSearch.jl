using SaddleSearch, SaddleSearch.TestSets
using Base.Test
using SaddleSearch: numE, numdE, res_trans, res_rot

using SaddleSearch.TestSets: hessprecond, precond
using Isaac
using SaddleSearch: ODE12r, odesolve, IterationLog


@testset "newtonkrylov" begin

@testset "blocklanczos" begin
println("Testing `blocklanczos`")

# create A = -Δ , B = diag(V)   with V positive
d = 30
x = LinRange(-1, 1, d)
A = d^2 * SymTridiagonal(2*ones(d), -ones(d-1))
B = Diagonal(exp(- x.^2) + 0.1)

# construct an index-1 system
μ = 30.0
H = A - μ * B
σ = sort(eigvals(full(H)))
@assert σ[1] < 0 && σ[2] > 0

b = ones(d) + 0.1 * sin.(x)

# define a linear function
fl = x -> H * x

# then try the same with a quadratic nonlinearity mixed in
#   (hessian remains the same!)
q = x -> [ [x[i]*x[i+1] for i = 1:d-1]; x[end] * x[1] ]
fq = x -> H * x + q(x)

# testing unpreconditioned lanczos
for (f, V0, msg) in [ (fl, reshape(b, d, 1), "Linear - Basic Lanczos"),
                      (fl, [b rand(d)], "Linear - Block Lanczos"),
                      (fq, reshape(b, d, 1), "Nonlinear - Basic Lanczos"),
                      (fq, [b rand(d)], "Nonlinear - Block Lanczos")
                     ]
   println("Testing: ", msg)
   x, λ, v, _ = blocklanczos(zeros(d), f, zeros(d), 1e-6, d;
                              b = b, V0 = V0, debug = false)
   @test norm(H * x - b) < 1e-6
   @test norm(H * v - λ * v, Inf) < 1e-7
   @test abs(λ - σ[1]) < 1e-7
end

# next we add some preconditioning
P = 0.9 * sparse(A) + μ/2 * speye(d)
λP = minimum(eigvals(full(H), full(P)))
srand(12345)
vrand = rand(d)

for (V0, msg, numfo) in [ (reshape(P\b, d, 1), "Preconditioned Lanczos", 9),
                           ([P\b P\vrand], "Preconditioned Block-Lanczos", 14) ]
   println("Testing: ", msg)
   x, λ, v, numf = blocklanczos(zeros(d), fq, zeros(d), 1e-6, d;
                                P = P, b = b, debug = false, V0=V0)
   @test numf == numfo
   @test norm(H * x - b) < 1e-6
   @test norm(H * v - λ * P * v, Inf) < 1e-6
   @test abs(λ - λP) < 1e-7
   if numf != numfo
      warn("numf was $numfo in original tests; now it is $numf")
   end
end

end  # @testset "blocklanczos"


@testset "NK Muller" begin

xe = []

for init in (:near, :far)
   println("NK Dimer, Muller, $(init)")
   V = MullerPotential()
   x0, v0 = ic_dimer(V, init)
   E, dE = objective(V)
   x, ndE = run!(NK(), E, dE, x0, v0)
   println("   num_dE = ", ndE)
   if init == :near
      xe = copy(x)
   else
      println("  |x - xe| = ", norm(x - xe, Inf))
      @test norm(x - xe, Inf) < 1e-8
   end

   println("Superlinear Dimer, Müller, $(init)")
   # MullerPotential with good IC
   V = MullerPotential()
   x0, v0 = ic_dimer(V, init)
   E, dE = objective(V)
   dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=500, verbose=verbose)
   x1, v, res = run!(dimer, E, dE, x0, v0)
   println("   num_dE = ", numdE(res)[end])
   println(" |x - xe| = $(norm(x1-xe, Inf))")
   if init == :near
      @test norm(x1 - xe, Inf) < 1e-10
   end
end

end # @testset "NK Muller"

@testset "NK Vacancy" begin

V = LJVacancy2D(bc = :clamped)
x0, v0 = ic_dimer(V, :near)
E, dE = objective(V)

println("Superlinear Dimer, Vacancy")
dimer = SuperlinearDimer(maximum_translation=0.2, max_num_rot=1, maxnumdE=500,
                        verbose=1)
x1, v, res = run!(dimer, E, dE, x0, v0)
# H1 = TestSets.hessian(V, x1)
println("   num_dE = ", numdE(res)[end])
# println("   |dE| = ", norm(dE(x1), Inf))
# println("   min-eig = ", sort(eigvals(H1))[1:2])
@test norm(dE(x1), Inf) < 1e-5

println("NK, Vacancy")
x2, ndE = run!(NK(), E, dE, x0, v0)
# H2 = TestSets.hessian(V, x2)
println("   num_dE = ", ndE)
@test norm(dE(x2), Inf) < 1e-5
@test norm(x1 - x2, Inf) < 1e-6
# println("   |dE| = ", norm(dE(x2), Inf))
# println("   |x_sl - x_nk| = ", norm(x1 - x2, Inf))
# println("   min-eig = ", sort(eigvals(H2))[1:2])

println("Superlinear Dimer, Vacancy, P(EXP)")
dimer = SuperlinearDimer( maximum_translation=0.2, max_num_rot=1, maxnumdE=500,
                          verbose=1,
                          precon = precond(V, x0),
                          precon_prep! = (P, x) -> precond(V, x) )
x3, v, res = run!(dimer, E, dE, x0, v0)
# H3 = TestSets.hessian(V, x3)
println("   num_dE = ", numdE(res)[end])
println("   |dE| = ", norm(dE(x3), Inf))
println("   |x_slP - x_sl| = ", norm(x1 - x3, Inf))
println("      >>> this converged to a minimum (cf commented-out hessian test")
# println("   min-eig = ", sort(eigvals(H3))[1:2])

println("NK, Vacancy, P(EXP)")
nk = NK(precon = precond(V, x0), precon_prep! = (P, x) -> precond(V, x) )
x4, ndE = run!(nk, E, dE, x0, v0)
println("   num_dE = ", ndE)
@test norm(dE(x4), Inf) < 1e-5
@test norm(x1 - x4, Inf) < 1e-6
# println("   |dE| = ", norm(dE(x2), Inf))
# println("   |x_sl - x_nkP| = ", norm(x1 - x4, Inf))

end  # @testset "NK Vacancy"



end  # @testset "newtonkrylov"
