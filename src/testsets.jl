
module TestSets

using Parameters
import ForwardDiff

export objective,
   MullerPotential, DoubleWell, LJcluster, LJVacancy2D,
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


function ic_string(V::LJcluster)
   R = lj_refconfig()
   r1 = [R[5]; R[4]; R[3]; R[2]; R[7]; R[6]; R[1]]
   r2 = [R[5]; R[3]; R[1]; R[2]; R[7]; R[6]; R[4]]
   return V.ρ_min * r1, V.ρ_min * r2
end



# ============================================================================
# TEST SET: 2D LJ Vacancy
# ============================================================================

type LJVacancy2D
   R::Float64
   Xref::Matrix{Float64}
   Ifree::Vector{Int}
end

LJVacancy2D(;R::Float64 = 5.1) = LJVacancy2D(R, vacancy_refconfig(R)...)

energy(V::LJVacancy2D, r) = LJenergy(dofs2pos(V,r))

function vacancy_refconfig(R)
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
   Ifree = find(r .<= R - 2.1)   # freeze two layers of atoms
   return Xref, Ifree
end

function dofs2pos{T}(V::LJVacancy2D, r::Vector{T})
   X = convert(Matrix{T}, V.Xref)
   X[:, V.Ifree] = reshape(r, 2, length(r) ÷ 2)
   return X
end

function ic_dimer(V::LJVacancy2D, case=:near)
   X = copy(V.Xref)
   if case == :near
      X[:, 1] *= 0.6
   elseif case == :far
      X[:, 1] *= 0.9
   else
      error("unkown `case` $(case) in `icdimer(::LJVacancy2D,...)`")
   end
   x0 = X[:, V.Ifree][:]
   v0 = zeros(length(x0))
   v0[1:2] = - x0[1:2] / norm(x0[1:2])
   return x0, v0
end


function exp_precond(V::LJVacancy2D, r; rcut = 2.5, α=3.0, μ=70.0)
   X = dofs2pos(V, r)
   nX = size(X, 2)
   P = zeros(nX, nX)
   for i = 1:nX, j = 1:nX
      Rij = X[:,i] - X[:,j]
      rij = norm(Rij)
      if 0 < rij < rcut
         Ii = (i-1) * 2 + [1,2]
         Ij = (j-1) * 2 + [1,2]
         a = μ * (exp(-α * (rij - 1.0)) - exp(-α * (rcut - 1.0)))
         # A = a * eye(2)  # (Rij/rij) * (Rij/rij)'
         # P[Ii, Ii] += A
         # P[Ij, Ij] += A
         # P[Ii, Ij] -= A
         # P[Ij, Ii] -= A
         P[i,j] -= a
         P[j,i] -= a
         P[i,i] += a
         P[j,j] += a
      end
   end
   # free = [ (V.Ifree - 1) * 2 + 1; (V.Ifree - 1) * 2 + 2 ][:]
   P = sparse(P[V.Ifree, V.Ifree])
   return kron(P, speye(2))
end

end
