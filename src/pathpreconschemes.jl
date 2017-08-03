export coordTransform, forcePrecon

"""
the two preconditioning schemes implemented for Sting and NEB-type methods
"""

@with_kw type coordTransform
   precon = I
   precon_prep! = (P, x) -> P
   precon_cond::Bool = false
   tangent_norm = (P, t) -> norm(P, t)
   gradDescent⟂ = (P, ∇E, t) -> zeros(length(t)) # dummy variable
   force_eval = (P, ∇E, ∇E⟂, t) -> P \ ∇E - dot(∇E,t)*t
   maxres = (P, ∇E⟂, force) ->  maximum([norm(P(i)*force[i],Inf) for i = 1:length(force)])
end

@with_kw type forcePrecon
   precon = I
   precon_prep! = (P, x) -> P
   precon_cond::Bool = false
   tangent_norm = (P, t) -> norm(t)
   gradDescent⟂ = (P, ∇E, t) -> [∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)]
   force_eval = (P, ∇E, ∇E⟂, t) -> P \ ∇E⟂
   maxres = (P, ∇E⟂, force) -> vecnorm(cat(2, ∇E⟂...)', Inf)
end

# @with_kw type refForcePrecon
#    precon = I
#    precon_prep! = (P,x) -> P
#    precon_cond::Bool = false
#    tangent_norm = (P, t) -> norm(t)
#    gradDescent⟂ = (P, ∇E, t) -> [∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)]
#    force_eval = (P, ∇E, ∇E⟂, t) -> P \ ∇E⟂
#    maxres = (P, ∇E⟂, force) -> vecnorm(cat(2, ∇E⟂...)', Inf)
# end
