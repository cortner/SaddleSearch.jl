
using Parameters
import ForwardDiff


"""
return the objective and objective gradient functions
"""
objective(V) = x->energy(V, x), x->gradient(V, x)

# no need to compute gradients if ForwardDiff does it for us.
gradient(V, x) = ForwardDiff.gradient(y->energy(V,y), x)

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
      return [-0.8, 1.3], [-1.0, -1.0]
   end
   error("unknown initial condition")
end


# ============================================================================
# TEST CASE 2: DoubleWell
# ============================================================================

@with_kw type DoubleWell
   A::Matrix{Float64} = [1.0 0.0; 0.0 1.0]
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
      return [0.8; 0.1*dA^(-1/2)], [-1.0, zeros(d)]
   end
   error("unknown initial condition")
end
