
using Parameters
import ForwardDiff


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

diffs(r) = broadcast(-,r,[r[i]' for i=1:7]')
dists(r) = broadcast(norm,diffs(r))
dispForce(V::LJcluster, r) = (V.σ ./ dists(r)).^6

energy(V::LJcluster, r) = 4V.ε * sum(triu( (dispForce(V,r) - 1)
                                                      .* dispForce(V,r), 1 ))

function ic_dimer(V::LJcluster, case=:near)
#  TODO add ICs for dimer method
end

function ic_string(V::LJcluster, case=:near)
   if case == :near
      omega=pi/3
      r_0=[ 0; 0]
      r_1=[ 1; 0]; r_2=[cos( omega ); sin( omega )]
      r_3=[cos(2*omega); sin(2*omega)]
      r_4=[-1; 0]; r_5=[cos(4*omega); sin(4*omega)]
      r_6=[cos(5*omega); sin(5*omega)]

      r1 = Array{Array{Float64}}(0)
      r2 = Array{Array{Float64}}(0)
      push!(r1, r_4, r_3, r_2, r_1, r_6, r_5, r_0)
      push!(r2, r_4, r_2, r_0, r_1, r_6, r_5, r_3)
      return V.ρ_min*r1, V.ρ_min*r2
  #  elseif case ==:far
  #     return
   end
   error("unknown initial condition")
end
