
module TestSets

using Parameters
import ForwardDiff

export objective,
   MullerPotential, DoubleWell, LJcluster,
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
# TEST CASE 1: Müller Potential
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
# TEST CASE 2: DoubleWell
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
# TEST CASE 3: Lennard-Jones Cluster
# ============================================================================

@with_kw type LJcluster
   ε::Float64 = 0.25
   σ::Float64 = 1.
   ρ_min::Float64 = 2.0^(1./6)
end

dists(r::Matrix) = [norm(r[:,i]-r[:,j]) for i = 1:6 for j = i+1:7]

dists(r::Vector) = dists(reshape(r, 2, length(r) ÷ 2))

dispForce(V::LJcluster, r) = (V.σ ./ dists(r)).^6

energy(V::LJcluster, r) = 4.0 * V.ε * sum( (dispForce(V,r) - 1.0) .* dispForce(V,r) )


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

end
