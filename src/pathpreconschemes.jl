export localPrecon, globalPrecon

"""
the three preconditioning schemes implemented for Sting and NEB-type methods

'ccordTransfor' : the preconditioner acts as a coordinate transformation of the
state space
'forcePrecon' : the preconditioner is applied to the force directly, each point
along the path is preconditioned independently.
'refForcePrecon' : same to force Precon but preconditioning of the whole path is
allowed

### Parameters:
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `precon_cond` : true/false whether to precondition the minimisation step
* 'tangent_norm' : function evaluating the norm of tangents according to the
                   scheme of choice
* 'gradDescent⟂' : function evaluating the perpendicular component of the
                   gradient ONLY IF this is needed
* 'force_eval' : function evaluating the forces according to the scheme of choice
* 'maxres' : function evaluating the residual error of the path, relative to
            infinity norm
"""

@with_kw type localPrecon
   precon = I
   precon_prep! = (P, x) -> P
   precon_cond::Bool = false
   dist = (P, x, i) ->  norm(0.5*(P(i)+P(i+1)), x[i+1]-x[i])
   tangent_norm = (P, t) -> [norm(P(i), t[i]) for i=1:length(t)]
   gradDescent⟂ = (P, ∇E, t) -> -[P(i) \ ∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)]
   force_eval = (P, ∇E⟂) -> ref(∇E⟂)
   maxres = (P, ∇E⟂) ->  maximum([norm(P(i)*∇E⟂[i],Inf) for i = 1:length(∇E⟂)])
end

# @with_kw type forcePrecon
#    precon = I
#    precon_prep! = (P, x) -> P
#    precon_cond::Bool = false
#    tangent_norm = (P, t) -> [norm(t[i]) i=1:length(t)]
#    gradDescent⟂ = (∇E, t) -> [∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)]
#    force_eval = (P, ∇E, ∇E⟂, t) -> -[P(i) \ ∇E⟂[i] for i=1:length(t)]
#    maxres = (P, ∇E⟂, force) -> vecnorm(cat(2, ∇E⟂...)', Inf)
# end

@with_kw type globalPrecon
   precon = I
   precon_prep! = (P, x) -> P
   precon_cond::Bool = false
   dist = (P, x, i) -> norm(x[i+1]-x[i])
   tangent_norm = (P, t) -> [norm(t[i]) for i=1:length(t)]
   gradDescent⟂ = (P, ∇E, t) -> ref(-[∇E[i] - dot(∇E[i],t[i])*t[i] for i=1:length(t)])
   force_eval = (P, ∇E⟂) -> ref(P) \ ∇E⟂
   # [(P(1) \ ∇E⟂)[i:i+length(t)-1] for i=1:length(t):length(∇E)-length(t)+1]
   maxres = (P, ∇E⟂) -> vecnorm(∇E⟂, Inf)
end

function ref{T}(x::Vector{T})
   return cat(1, x...)
end

function ref{T}(A::Array{Array{T,2},2})
    return cat(1,[cat(2,A[n,:]...) for n=1:size(A,1)]...)
end

function set_ref!{T}(x::Vector{T}, xref::Vector{Float64})
   Nimg = length(x); Nref = length(xref) ÷ Nimg
   X = reshape(xref, Nref, Nimg)
   x = [ X[:, n] for n = 1:Nimg ]
   return x
end
