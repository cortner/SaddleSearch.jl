
"""
Available Testsets:

* MullerPotential
* DoubleWell
* LJcluster
* LJVacancy2D

"""
module TestSets

using Parameters
import ForwardDiff

export objective,
   MullerPotential, DoubleWell, LJcluster, LJVacancy2D, Molecule2D,
   ic_dimer, ic_string



"""
return the objective and objective gradient functions
"""
objective(V) = x->energy(V, x), x->gradient(V, x)

# no need to compute gradients if ForwardDiff does it for us.
gradient(V, x) = ForwardDiff.gradient(y->energy(V,y), x)

hessian(V, x) = ForwardDiff.hessian(y->energy(V,y), x)

function hessprecond(V, x; stab=0.0)
   H = Symmetric(hessian(V, x))
   D, V = eig(H)
   D = abs.(D) + stab
   return V * diagm(D) * V'
end



# ============================================================================
# TEST SET: Müller Potential
#   TODO: add reference
# ============================================================================

@with_kw type MullerPotential
   B::Vector{Matrix{Float64}} = [ [ -1 0; 0 -10], [-1 0; 0 -10],
                                    [-6.5 5.5; 5.5 -6.5], [0.7 0.3; 0.3 0.7] ]
   A::Vector{Float64} = [-200, -100, -170, 15]
   R::Vector{Vector{Float64}} = [ [1,0], [0,0.5], [-0.5,1.5], [-1,1] ]
end

energy(V::MullerPotential, r) = sum( 0.5 * A * exp(dot(r-R, B*(r-R)))
                                     for (A, R, B) in zip(V.A, V.R, V.B) )

function ic_dimer(V::MullerPotential, case=:near)
   if case == :near
      return [-0.6, 0.5], [-1.0, 1.0]
   elseif case ==:far
      return [-0.75, 1.0], [-1.0, -1.0]
   end
   error("unknown initial condition")
end

function ic_path(V::MullerPotential, case=:near)
   if case == :near
      return [-0.5; 1.5],  [0.7; .0]
   elseif case == :far
      return [-1.0; 0.5], [0.7; .5]
   end
end

# ============================================================================
# TEST SET: DoubleWell
# ============================================================================

@with_kw type DoubleWell
   A::Matrix{Float64} = eye(2)
end

DoubleWell(c::Float64) = DoubleWell( diagm([1.0, c])  )

fdw(r) = [r[1]^2 - 1.0; r[2:end]]

energy(V::DoubleWell, r) = energy_sq(V, fdw(r))
energy_sq(V::DoubleWell, s) = 0.5 * dot(s, V.A * s)

function ic_dimer(V::DoubleWell, case=:near)
   d = size(V.A,1)-1  # extra dimensions
   dA = diag(V.A)[2:end]
   if case == :near
      x0 = [0.7; 0.2 * dA.^(-1/2)]
      return x0, - x0
   elseif case ==:far
      return [0.8; 0.1*dA.^(-1/2)], [-1.0, zeros(d)]
   end
   error("unknown initial condition")
end

function ic_path(V::DoubleWell, case=:nothing)
   d = size(V.A,1)-1

   if case == :near
      x0 = [1.0; 0.2/sqrt(d) * ones(d)]
      return x0, - x0
   elseif case ==:far
      x0 = [0.8; 0.4/sqrt(d) * ones(d)]
      return x0, - x0
   end
   error("unknown initial condition")
end
# ============================================================================
# TEST SET: Lennard-Jones Cluster
# ============================================================================

@with_kw type LJcluster
   ε::Float64 = 0.25
   σ::Float64 = 1.
   ρ_min::Float64 = 1.0
end

dists(r::Matrix) = [norm(r[:,i]-r[:,j])
                     for i = 1:size(r,2)-1 for j = i+1:size(r,2)]

dists(r::Vector) = dists(reshape(r, 2, length(r) ÷ 2))

LJpot(r) = r.^(-12) - 2 * r.^(-6)
LJenergy(r) = sum(LJpot(dists(r)))

energy(V::LJcluster, r) = 4.0 * V.ε * LJenergy(r / V.σ)

function lj_refconfig()
   ω = π / 3.0
   return [ [ 0.0, 0.0], [ 1.0, 0.0], [cos(ω), sin(ω)], [cos(2*ω), sin(2*ω)],
            [-1.0, 0.0], [cos(4*ω), sin(4*ω)], [cos(5*ω), sin(5*ω)] ]
end

function ic_dimer(V::LJcluster, case=:near)
   if case == :near
      x = [-0.516917, 0.46756, 1.06511, -0.0369461, 0.350324, 0.476477,
            -0.229551, 1.31733, -1.15186, -0.120603, -0.501187, -0.870874,
            0.502517, -0.870205]
      v = [-0.254188, 0.666259, -0.017954, 0.0379428, -0.226884, -0.232232,
           -0.0667094, 0.546372, 0.279263, 0.056547, 0.00602156, -0.0110506,
            0.00103608, 0.000875767]
      return x, v
   elseif case == :far
      R = lj_refconfig()
      S = copy(R)
      R[4] += 0.5 * R[3]     # push third atom outwards
      R[1] += 0.5 * S[4] + 0.25 * R[5] # push middle atom outwards towards third atom
      R[3] *= 0.5            # push second atom towards centre
      x = vcat(R...)
      v = [ R[4] [0,0] (- 0.5 * R[3]) R[4] [0,0] [0,0] [0,0] ][:]
      return x, v
   end
   error("unkown `case` in `icdimer(::LJCluster,...)`")
end


function ic_path(V::LJcluster)
   R = lj_refconfig()
   r1 = [R[5]; R[4]; R[3]; R[2]; R[7]; R[6]; R[1]]
   r2 = [R[5]; R[3]; R[1]; R[2]; R[7]; R[6]; R[4]]
   return V.ρ_min * r1, V.ρ_min * r2
end

precond(V::LJcluster, r) = LJaux.exp_precond(reshape(r, 2, length(r) ÷ 2))


# ============================================================================
# TEST SET: 2D LJ Vacancy
# ============================================================================

"""
This module just collects the basic Lennard-Jones implementation,
the functions below wrap this into a test type and implement the
boundary condition.
"""
module LJaux

function vacancy_refconfig(R, bc)
   A = [1.0 cos(π/3); 0.0 sin(π/3)]
   cR = ceil(Int, R / minimum(svd(A)[2]))
   t = collect(-cR:cR)
   x = ones(length(t)) * t'
   y = t * ones(length(t))'
   X = A * [x[:] y[:]]'
   r = sqrt(sumabs2(X, 1))
   Xref = X[:, find(0 .< r .<= R)]
   r = sqrt(sumabs2(Xref, 1))
   I0 = find(r .<= 1.1)[1]
   if I0 != 1
      Xref[:, [1,I0]] = Xref[:, [I0,1]]
      r[[1,I0]] = r[[I0,1]]
   end
   if bc == :free
      Ifree = 1:size(Xref, 2)
   elseif bc == :clamped
      Ifree = find(r .<= R - 2.1)   # freeze two layers of atoms
   else
      error("unknown parameter `bc = $bc`")
   end
   return Xref, Ifree
end

phi(r) = r^(-12) - 2 * r^(-6)
dphi(r) = -12 * r^(-13) + 12 * r^(-7)
ddphi(r) = 12*13 * r^(-14) - 7*12 * r^(-8)
pphi(r) = (12*13 - 7*12) * r^(-8)

function energy(X::Matrix)
    E = 0.0
    nX = size(X,2)
    for i = 1:nX-1, j = i+1:nX
        rij = norm(X[:,i] - X[:,j])
        E += phi(rij)
    end
    return E
end

function gradient(X::Matrix)
    dE = zeros(size(X))
    nX = size(X,2)
    for i = 1:nX-1, j = i+1:nX
        Rij = X[:,i] - X[:,j]
        rij = norm(Rij)
        a = dphi(rij) * Rij / rij
        dE[:,i] += a
        dE[:,j] -= a
    end
    return dE
end

function exp_precond(X::Matrix; rcut = 2.5, α=3.0)
   nX = size(X, 2)
   P = zeros(2*nX, 2*nX)
   I = zeros(Int, 2, nX)
   I[:] = 1:2*nX
   for i = 1:nX, j = i+1:nX
      Ii, Ij = I[:,i], I[:,j]
      Rij = X[:,i] - X[:,j]
      rij = norm(Rij)
      if 0 < rij < rcut
         a = pphi(rij) * eye(2)
         P[Ii, Ij] -= a
         P[Ij, Ii] -= a
         P[Ii, Ii] += a
         P[Ij, Ij] += a
      end
   end
   return sparse(P) + 0.001 * speye(2*nX)
end

end

"""
`LJVacancy2D(; R::Float64 = 5.1, bc=:free)`

`bc = :clamped` is also allowed
"""
type LJVacancy2D
   R::Float64
   Xref::Matrix{Float64}
   Ifree::Vector{Int}
end

LJVacancy2D(; R::Float64 = 5.1, bc=:free) =
   LJVacancy2D(R, LJaux.vacancy_refconfig(R, bc)...)

function dofs2pos{T}(V::LJVacancy2D, r::Vector{T})
   X = convert(Matrix{T}, V.Xref)
   X[:, V.Ifree] = reshape(r, 2, length(r) ÷ 2)
   return X
end

pos2dofs(V::LJVacancy2D, X::Matrix) = X[:, V.Ifree][:]

energy(V::LJVacancy2D, x::Vector) =
   LJaux.energy(dofs2pos(V, x))

gradient(V::LJVacancy2D, x::Vector) =
   pos2dofs(V, LJaux.gradient(dofs2pos(V, x)))

function ic_dimer(V::LJVacancy2D, case=:near)
   X = copy(V.Xref)
   if case == :near
      X[:, 1] *= 0.6
   elseif case == :far
      X[:, 1] *= 0.9
   elseif case == :min
      nothing
   else
      error("unkown `case` $(case) in `icdimer(::LJVacancy2D,...)`")
   end
   x0 = X[:, V.Ifree][:]
   v0 = zeros(length(x0))
   v0[1:2] = - x0[1:2] / norm(x0[1:2])
   return x0, v0
end

function ic_path(V::LJVacancy2D, case=:near)
   X0 = copy(V.Xref); X1 = copy(V.Xref)
   if case == :near
      X0[:,1] *= .1; X1[:,1] *= .9
   elseif case == :far
      X0[:,1] *= -.3; X1[:,1] *= 1.3
   elseif case == :min
      X0[:,1] *= .0; X1[:,1] *= 1.
   else
      error("unkown `case` $(case) in `icdimer(::LJVacancy2D,...)`")
   end
   return X0[:, V.Ifree][:], X1[:, V.Ifree][:]
end

function pos2dofs(V::LJVacancy2D, P::AbstractMatrix)
   free = [V.Ifree' * 2 - 1; V.Ifree' * 2][:]
   return P[free, free]
end

precond(V::LJVacancy2D, x::Vector; kwargs...) =
   pos2dofs(V, LJaux.exp_precond(dofs2pos(V, x), kwargs...))
   # LJaux.exp_precond(dofs2pos(V, x), kwargs...)




# ============================================================================
# TEST SET: Molecule2D
# ============================================================================
#
#  [B2]
#      \
#        [A] - [B1]

@with_kw type Molecule2D
   k::Float64 = 1.0
   kab::Float64 = 25.0
   kbb::Float64 = 5.0
   rbb::Float64 = √3
end

function bonds(V::Molecule2D, r)
   # A is always at [0,0]
   # B1 at [r1, 0]
   # B2 at [r2, r3]
   Rab1 = [r[1],0]
   Rab2 = [r[2], r[3]]
   Rbb = Rab1 - Rab2
   rab1 = norm(Rab1)
   rab2 = norm(Rab2)
   rbb = norm(Rab1 - Rab2)
   return rab1, rab2, rbb, Rab1/rab1, Rab2/rab2, Rbb/rbb
end


function energy(V::Molecule2D, r)
   rab1, rab2, rbb, Sab1, Sab2, Sbb = bonds(V, r)
   # AB bonds
   E = V.kab/2 * ((rab1 - 1)^2 + (rab2 - 1)^2)
   # AA bond
   E += V.kbb/2 * (rbb - V.rbb)^2
   # bond-angle
   E += V.k/2 * (dot(Sab1, Sab2) + 0.5)^2
   return E
end

"Θ = 2π/3 is the minimum, Θ = π near the saddle, Θ=4π/3 is the second minimum"
function mol2dpath(Θ)
   Rab1 = [1.0,0.0]
   Rab2 = [cos(Θ) -sin(Θ); sin(Θ) cos(Θ)] * Rab1
   return [Rab1[1], Rab2[1], Rab2[2]]
end

function ic_dimer(V::Molecule2D, case=:near)
   if case == :near
      r0 = mol2dpath(π-0.2)
   elseif case ==:far
      r0 = mol2dpath(2*π/3 + 0.2)
   end
   rs = mol2dpath(π)
   v0 = (rs-r0) / norm(rs-r0)
   return r0, v0
end

"an FF-type preconditioner for Molecule2D"
function precond(V::Molecule2D, r)
   # < Pu, u> = ∑_{i=1,2}  kab (uabi ⋅ Sabi)^2 + kbb (ubb ⋅ Sbb)^2
   #                + bond-angle terms
   #          = kab (u[1])^2 + kab (Sab2 ⋅ u[2:3])^2 +
   #             + kbb (Sbb ⋅ [u[2], u[3]-u[1]])^2
   #             + bond-angle terms
   #  [u[2], u[3]-u[1]] = [ 0 1 0; -1 0 1] u
   #
   # bond-angle terms: Φ = dot(Sab1, Sab2) then
   #      ~ k  ∑_{i,j} (∂_{Rabi}Φ ⋅ u_{abi}) (∂_{Rabj} ⋅ u_{abj})
   #      ~ k  ( ∑_i [(I - Sabi ⊗ Sabi) Sabj] ⋅ u_{abi} )^2

   rab1, rab2, rbb, Sab1, Sab2, Sbb = bonds(V, r)
   P = zeros(3,3)

   # 2-body bonds
   P[1,1] += V.kab                       # AB1
   P[2:3,2:3] += V.kab * Sab2 * Sab2'    # AB2
   Ubb = [0 1 0; -1 0 1]' * Sbb
   P += V.kbb * Ubb * Ubb'               # BB

   # bond-angle
   q1 = Sab2 - dot(Sab1, Sab2) * Sab1
   q2 = Sab1 - dot(Sab1, Sab2) * Sab2
   q = [q1[1]; q2]
   P += V.k * (0.9 * q * q' + 0.1 * eye(3))

   return P
end

end
