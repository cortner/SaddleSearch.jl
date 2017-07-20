using JuLIP, JuLIP.Potentials, SaddleSearch, Optim
using MaterialsScienceTools
using CTKSolvers
SI = MaterialsScienceTools.Silicon

using PyCall
@pyimport atomistica
kumagai() = JuLIP.ASE.ASECalculator(atomistica.Kumagai())
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
    v0 += 0.01 * (rand(v0) - 0.5)

   #  set_constraint!(at, JuLIP.Constraints.FixedCell(at, clamp = [10]))
    set_constraint!(at, JuLIP.Constraints.FixedCell(at))
    set_calculator!(at, tersoff())
    return at, v0
end



function runsaddle(init, LL; verbose = 1)
    gcalls_nk = Int[]
    gcalls_nk_exp = Int[]
    gcalls_nk_ff = Int[]

    for L in LL
        at, v0 = init(L)
        X0 = positions(at)
        println("len(at) = ", length(at))

        E = x -> energy(set_dofs!(at, x))
        dE = x -> gradient(set_dofs!(at, x))


        # NewtonKrylov  (ID)
        set_positions!(at, X0)
        nkI = NK(len = 1e-4, verbose=3, maxnumdE = 100, maxstep = 0.1,
                  krylovinit = :res,
                  precon = JuLIP.Preconditioners.Exp(at),
                  precon_prep! = (P, x) -> JuLIP.update!(P, at, x).amg.A)
        x0 = dofs(at)
        x0 += 0.01 * rand(x0)
        x, gcalls = run!(nkI, E, dE, dofs(at), v0)
        println("    >> gcalls(NK-I) =  ", gcalls, "     | res = $(norm(dE(x)))")
        push!(gcalls_nk, gcalls)
    end
end


# LL = [2, 4, 8, 16]

runsaddle(vacancy_saddle, [4])


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
