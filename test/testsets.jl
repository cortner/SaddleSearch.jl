
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

# Available Test Sets:
export MullerPotential,
       DoubleWell,
       LJcluster,
       LJVacancy2D,
       MorseIsland

# other exports
export objective, ic_dimer, ic_path, precond, hessprecond


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
   D = abs.(D) .+ stab
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
   r = sqrt.(sum(abs2, X, 1))
   Xref = X[:, find(0 .< r .<= R)]
   r = sqrt.(sum(abs2, Xref, 1))
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
# TEST SET: MorseIsland
# ============================================================================


module MorseAux

function island_refconfig(;n_bc1=6, n_bc2=14)
   data = joinpath(Pkg.dir("SaddleSearch"), "data") * "/"
   Xref = readdlm(data*"morse_island_min01a.dat")
   Ifix = find(u->u<=n_bc1, Xref[:,3])
   Ifree = setdiff(1:size(Xref,1), Ifix)
   Iisl = find(u->u>=n_bc2, Xref[:,3])
   Ibulk = setdiff(Ifree, Iisl)
   nfree = length(Ifree)
   return Xref, Ifix, Ifree, Iisl, Ibulk, nfree
end

end


type MorseIsland
   Xref::Matrix{Float64}
   Ifix::Vector{Int}
   Ifree::Vector{Int}
   Iisl::Vector{Int}
   Ibulk::Vector{Int}
   nfree::Int
end

MorseIsland(; n_bc1=6, n_bc2=14) =
   MorseIsland(MorseAux.island_refconfig(n_bc1=n_bc1, n_bc2=n_bc2)...)

function energy(V::MorseIsland, r)
   aa = 0.7102
   a = 1.6047
   r0 = 2.8970
   rc = 9.5
   c = 2.74412

   np = 343; nd = np*3

   e = exp(-a*(rc-r0))
   ecut = e*(e - 2)

   box = [7*c 4*c*sqrt(3)]

   En = 0

   # atoms below zfix are fixed
   zfix = 6.7212

   Rfree = reshape(r, (length(V.Ifree),3))
   Rfix = V.Xref[V.Ifix, :]

   R = zeros(np,3)
   R[V.Ifree, :] = Rfree
   R[V.Ifix,: ]= Rfix

   # R = reshape(r, (np,3))

   for i=1:np, j=i+1:np
      if (R[i,3]>zfix || R[j,3]>zfix)
         rel = [R[i,k]-R[j,k] for k=1:3]
         [rel[i] -= box[i] * round(rel[i]./box[i]) for i=1:2]
         r2sqrt = norm(rel)

         # if r2sqrt < rc
            e = exp( -a*(r2sqrt - r0) )
            # potential energy
            En += e*(e - 2) - ecut
        # end
      end
   end

   En *= aa
   return En
end

function gradient(V::MorseIsland, r)
   aa = 0.7102
   a = 1.6047
   r0 = 2.8970
   rc = 9.5
   c = 2.74412

   np = 343; nd = np*3

   e = exp(-a*(rc-r0))
   ecut = e*e - 2*e

   box = [7*c 4*c*sqrt(3)]

   f = [zeros(np) for i=1:3]
   F = zeros(nd,1);

   # atoms below zfix are fixed
   zfix = 6.7212

   Rfree = reshape(r, (length(V.Ifree),3))
   Rfix = V.Xref[V.Ifix, :]

   R = zeros(np,3)
   R[V.Ifree, :] = Rfree
   R[V.Ifix,: ]= Rfix
   # R = reshape(r, (np,3))

   I = zeros(Int, np, 3)
   I[:] = 1:nd

    for i=1:np, j=i+1:np
        if (R[i,3]>zfix || R[j,3]>zfix)
            Ii, Ij = I[i,:], I[j,:]
            rel = [R[i,k]-R[j,k] for k=1:3]
            [rel[i] -= box[i] * round(rel[i]/box[i]) for i=1:2]
            r2sqrt = norm(rel)

            # if r2sqrt < rc
                e = exp( -a*(r2sqrt - r0) )

                # force
                ff = (e*e - e) / r2sqrt
                df = ff .* rel
                if R[i,3] > zfix; F[Ii] += df; end
                if R[j,3] > zfix; F[Ij] -= df; end
                # [f[k][i] += df[k] for k=1:3]
                # [f[k][j] -= df[k] for k=1:3]
            # end
        end
    end

   # F = cat(1, f...)
   F *= - 2 * a * aa
   return F
end

function precond(V::MorseIsland, r)
    aa = 0.7102
    a = 1.6047
    r0 = 2.8970
    rc = 9.5
    c = 2.74412

    np = 343; nd = np*3

    e = exp(-a*(rc-r0))
    ecut = e*e - 2*e

    box = [7*c 4*c*sqrt(3)]

    # atoms below zfix are fixed
    zfix = 6.7212

    Rfree = reshape(r, (length(V.Ifree),3))
    Rfix = V.Xref[V.Ifix, :]

    R = zeros(np,3)
    R[V.Ifree, :] = Rfree
    R[V.Ifix,: ]= Rfix

    # R = reshape(r, (np,3))
    P = zeros(nd, nd)
    I = zeros(Int, np, 3)
    I[:] = 1:nd
    for i=1:np, j=i+1:np
        if (R[i,3]>zfix && R[j,3]>zfix)
            Ii, Ij = I[i,:], I[j,:]
            rel = [R[i,k]-R[j,k] for k=1:3]
            [rel[i] -= box[i] * round(rel[i]/box[i]) for i=1:2]
            r2sqrt = norm(rel)

            # if r2sqrt < rc
                e = exp( -a*(r2sqrt - r0) )
                ddf = 4 * aa * a * a * e * e * eye(3)

                P[Ii, Ij] -= ddf
                P[Ij, Ii] -= ddf
                P[Ii, Ii] += ddf
                P[Ij, Ij] += ddf

            # end
        end
    end

    Q = P[I[V.Ifree,:][:], I[V.Ifree,:][:]]
   return sparse(Q) + 0.001 * speye(length(I[:,V.Ifree]))
end

function ic_path(V::MorseIsland)
   data = joinpath(Pkg.dir("SaddleSearch"), "data") * "/"
   X0 = readdlm(data*"morse_island_min01a.dat")
   x0 = X0[V.Ifree, :][:]
   X1 = readdlm(data*"morse_island_min02a.dat")
   x1 = X1[V.Ifree, :][:]

   return X0, X1
end

end
