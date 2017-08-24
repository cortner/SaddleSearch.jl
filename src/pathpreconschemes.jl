export coordTransform, forcePrecon, refForcePrecon

"""
the three preconditioning schemes implemented for Sting and NEB-type methods

'ccordTransfor' : the preconditioner acts as a coordinate transformation of the state space
'forcePrecon' : the preconditioner is applied to the force directly, each point along the path is preconditioned independently.
'refForcePrecon' : same to force Precon but preconditioning of the whole path is allowed

### Parameters:
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `precon_cond` : true/false whether to precondition the minimisation step
* 'tangent_norm' : function evaluating the norm of tangents according to the scheme of choice
* 'gradDescent⟂' : function evaluating the perpendicular component of the gradient ONLY IF this is needed
* 'force_eval' : function evaluating the forces according to the scheme of choice
* 'maxres' : function evaluating the residual error of the path, relative to infinity norm
"""

@with_kw type coordTransform
   precon = I
   precon_prep! = (P, x) -> P
   precon_cond::Bool = false
   tangent_norm = (P, t) -> norm(P, t)
   gradDescent⟂ = (P, ∇E, t) -> zeros(length(t)) # dummy variable
   force_eval = (P, ∇E, ∇E⟂, t) -> [P(i) \ ∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)]
   maxres = (P, ∇E⟂, force) ->  maximum([norm(P(i)*force[i],Inf) for i = 1:length(force)])
end

@with_kw type forcePrecon
   precon = I
   precon_prep! = (P, x) -> P
   precon_cond::Bool = false
   tangent_norm = (P, t) -> norm(t)
   gradDescent⟂ = (P, ∇E, t) -> [∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)]
   force_eval = (P, ∇E, ∇E⟂, t) -> [P(i) \ ∇E⟂[i] for i=1:length(t)]
   maxres = (P, ∇E⟂, force) -> vecnorm(cat(2, ∇E⟂...)', Inf)
end

@with_kw type refForcePrecon
   precon = I
   precon_prep! = (P, x) -> P
   precon_cond::Bool = false
   tangent_norm = (P, t) -> norm(t)
   gradDescent⟂ = (P, ∇E, t) -> cat(1, [∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)]...)
   force_eval = (P, ∇E, ∇E⟂, t) -> [(P(1) \ ∇E⟂)[i:i+length(t)-1] for i=1:length(t):length(∇E)-length(t)+1]
   maxres = (P, ∇E⟂, force) -> vecnorm(∇E⟂, Inf)
end
