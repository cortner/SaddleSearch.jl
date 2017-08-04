# This Example requires
#   * JuLIP.jl,
#   * atomistica
#   * CTKSolvers.jl
#

using JuLIP, JuLIP.Potentials, SaddleSearch, Optim
using CTKSolvers


using PyCall
@pyimport atomistica
kumagai() = JuLIP.ASE.ASECalculator(atomistica.KumagaiScr())
tersoff() = JuLIP.ASE.ASECalculator(atomistica.TersoffScr())
sw() = StillingerWeber()

function vacancy_saddle(L)
    at = bulk("Si", cubic=true, pbc = true) * L
    xv = positions(at)[1]
    deleteat!(at, 1)
    X = positions(at)
    xn = X[1]
    @assert norm(xv - xn) ≈ rnn("Si")
    X[1] = 0.5 * (xv + xn)
    set_positions!(at, X)
    v0 = zeros(length(at) * 3)
    v0[1:3] = xn - xv
    # v0 += 0.01 * (rand(v0) - 0.5)

    # set_constraint!(at, JuLIP.Constraints.FixedCell(at, clamp = [10]))
    set_constraint!(at, JuLIP.Constraints.FixedCell(at))
    set_calculator!(at, kumagai())
    return at, v0
end

L = 8
at, v0 = vacancy_saddle(L)
x0 = dofs(at)

E = xx -> energy(at, xx)
dE = xx -> gradient(at, xx)

Estab = xx -> energy(at, xx) + 0.5 * norm(xx[1:3] - x0[1:3])^2
dEstab = xx -> gradient(at, xx) + [xx[1:3] - x0[1:3]; zeros(length(x0)-3)]
dE0 = dE(x0)

function check_saddle(x0)
   x = copy(x0)
   H = zeros(length(x0), length(x0))
   h = 1e-7
   dE0 = dE(x)
   for n = 1:length(x0)
      x[n] += h
      H[:, n] = (dE(x) - dE0) / h
      x[n] -= h
   end
   σ = sort(eigvals(Symmetric(H)))
   return σ
end

# NewtonKrylov - I
nkI = NK(len = 1e-4, verbose=1, maxnumdE = 500, maxstep = 0.2,
         krylovinit = :resrot )
x1, gcalls = run!(nkI, Estab, dEstab, x0, v0)
println("  NK(I): gcalls =  ", gcalls, "     | res = $(norm(dE(x1)))")
L <= 4 && println("         σ[1:6] = ", check_saddle(x1)[1:6])

# Superlinear - I
sld = SuperlinearDimer( maximum_translation=0.2, max_num_rot=1, maxnumdE=500,
                          verbose=1 )
x3, v, res = run!(sld, E, dE, x0, v0)
println("  SLD(I): gcalls = ", SaddleSearch.numdE(res)[end], "    | res = $(norm(dE(x3), Inf))")
L <= 4 && println("          σ[1:6] = ", check_saddle(x3)[1:6])
println("          |x_nk - x_sld| = ", norm(x1 - x3, Inf))

# NewtonKrylov - P
nkP = NK(len = 1e-4, verbose=1, maxnumdE = 500, maxstep = 0.2,
         krylovinit = :res,
         precon = JuLIP.Preconditioners.Exp(at),
         precon_prep! = (P, x) -> JuLIP.update!(P, at, x).amg.A)
x1P, gcalls = run!(nkP, Estab, dEstab, x0, v0)
println("  NK(EXP): gcalls =  ", gcalls, "     | res = $(norm(dE(x1P)))")
L <= 4 && println("           σ[1:6] = ", check_saddle(x1P)[1:6])
println("           |x_nk - x_nkP| = ", norm(x1 - x1P, Inf))

# NewtonKrylov - FF
nkFF = NK(len = 1e-4, verbose=1, maxnumdE = 500, maxstep = 0.2,
         krylovinit = :resrot,
         precon = JuLIP.Preconditioners.FF(at, sw()),
         precon_prep! = (P, x) -> JuLIP.update!(P, at, x).amg.A)
x1FF, gcalls = run!(nkFF, Estab, dEstab, x0, v0)
println("  NK(FF): gcalls =  ", gcalls, "     | res = $(norm(dE(x1FF)))")
L <= 4 && println("          σ[1:6] = ", check_saddle(x1FF)[1:6])
println("          |x_nk - x_nkFF| = ", norm(x1 - x1FF, Inf))


# # Superlinear - P
# sldP = SuperlinearDimer( maximum_translation=0.2, max_num_rot=1, maxnumdE=500,
#                           verbose=1,
#                           precon = JuLIP.Preconditioners.Exp(at),
#                           precon_prep! = (P, x) -> JuLIP.update!(P, at, x) )
# x3P, v, res = run!(sldP, Estab, dEstab, x0, v0)
# println("  SLD: gcalls = ", SaddleSearch.numdE(res)[end], "    | res = $(norm(dE(x3P), Inf))")
# L <= 4 && println("       σ[1:6] = ", check_saddle(x3P)[1:6])
# println("       |x_nk - x_sldP| = ", norm(x1 - x3P, Inf))







# function runsaddle(init, LL; verbose = 1)
#     gcalls_nk = Int[]
#     gcalls_nk_exp = Int[]
#     gcalls_nk_ff = Int[]
#
#     for L in LL
#         at, v0 = init(L)
#         X0 = positions(at)
#         println("len(at) = ", length(at))
#
#         E = x -> energy(set_dofs!(at, x))
#         dE = x -> gradient(set_dofs!(at, x))
#
#
#         # NewtonKrylov  (ID)
#         set_positions!(at, X0)
#         nkI = NK(len = 1e-4, verbose=3, maxnumdE = 100, maxstep = 0.1,
#                   krylovinit = :res,
#                   precon = JuLIP.Preconditioners.Exp(at),
#                   precon_prep! = (P, x) -> JuLIP.update!(P, at, x).amg.A)
#         x0 = dofs(at)
#         x0 += 0.01 * rand(x0)
#         x, gcalls = run!(nkI, E, dE, dofs(at), v0)
#         println("    >> gcalls(NK-I) =  ", gcalls, "     | res = $(norm(dE(x)))")
#         push!(gcalls_nk, gcalls)
#     end
# end
#
#
# # LL = [2, 4, 8, 16]
#
# runsaddle(vacancy_saddle, [4])


# at, v0 = vacancy_saddle(4)
# println("len(at) = ", length(at))
# E = x -> energy(set_dofs!(at, x))
# T = x -> 0.5 * sum(sum(reshape(x, 3, length(x) ÷ 3), 2).^2)
# dT = x -> ForwardDiff.gradient(T, x)
# dE = x -> gradient(set_dofs!(at, x)) + dT(x)
#
# x = dofs(at)
# N = length(x)
# H = zeros(N, N)
# g0 = dE(x)
# h = 1e-6
# for n = 1:N
#    x[n] += h
#    H[:, n] = (dE(x) - g0) / h
#    x[n] -= h
# end
# H = 0.5 * (H + H')
#
# λ = eigvals(H)
# @show λ[1:11]




# TESTING HOW THE HESSIAN BEHAVES
# H = zeros(length(x0), length(x0))
# h = 1e-7
# x = copy(x0)
# for n = 1:length(x0)
#    x[n] += h
#    H[:, n] = (dE(x) - dE0) / h
#    x[n] -= h
# end
# σ = sort(eigvals(Symmetric(H)))
# @show σ[1:6]

# b = dE0
# V0 = [b v0]
# blocklanczos( dE0, dE, x0, 0.5 * norm(b), 100;
#       P = I, b = b, V0 = V0,
#       eigatol = 1e-1, eigrtol = 1e-1,
#       debug = true, h = 1e-7 )



# TESTING HOW THE HESSIAN BEHAVES

# b = dE0
# V0 = [b v0]
# blocklanczos( dE0, dE, x0, 0.5 * norm(b), 100;
#       P = I, b = b, V0 = V0,
#       eigatol = 1e-1, eigrtol = 1e-1,
#       debug = true, h = 1e-7 )
