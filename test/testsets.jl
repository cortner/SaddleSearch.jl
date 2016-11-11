
using Parameters

objective(V) = x->energy(V, x), x->gradient(V, x)

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

energy(V::MullerPotential, r) =
   sum(0.5 * A * exp(dot(r-R, B*(r-R))) for (A, R, B) in zip(V.A, V.R, V.B))

gradient(V::MullerPotential, r) =
   sum(A * exp(dot(r-R, B*(r-R))) * (B*(r-R)) for (A, R, B) in zip(V.A, V.R, V.B))


function ic_dimer(V::MullerPotential, case=:near)
   if case == :near
      return [-0.6, 0.5], [-1.0, 1.0]
   elseif case ==:far
      return [-0.8, 1.3], [-1.0, -1.0]
   end
   error("unknown initial condition")
end


# ============================================================================
