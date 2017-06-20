
using Dierckx
export ODENudgedElasticbandMethod

"""
`ODENudgedElasticbandMethod`: neb variant utilising adaptive time step ode solvers

### Parameters:
* `alpha` : step length
* 'k' : spring constrant
* 'abstol' : absolute errors tolerance
* 'reltol' : relative errors tolerance
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_cond` : true/false whether to precondition the minimisation step
"""
@with_kw type ODENudgedElasticBandMethod
   solver = ode12(1e-6, 1e-3, true)
   k::Float64
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   precon = [I]
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_cond::Bool = false
end


function run!{T}(method::ODENudgedElasticBandMethod, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack solver, k, tol_res, maxnit,
            precon_prep!, verbose, precon_cond = method
   P=method.precon
   # initialise variables
   x = copy(x0)
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
      @printf(" nit |  sup|∇E|_∞   \n")
      @printf("-----|-----------------\n")
   end

   αout, xout, log = odesolve(solver, (α_,x_) -> forces(x, x_, k, dE, P, precon_prep!), ref(x), length(x), log, method; tol_res = tol_res, maxnit=maxnit )

   x = set_ref!(x, xout[end])
   return x, log, αout
end

function forces{T}(x::Vector{T}, xref::Vector{Float64},
                     k::Float64, dE, P, precon_prep!)
   x = set_ref!(x, xref)
   N = length(x)
   P = precon_prep!(P, x)
   Np = length(P)

   # central finite differences
   dxds = [(x[i+1]-x[i-1])/2 for i=2:N-1]
   dxds ./= [sqrt(dot(dxds[i], P[mod(i-Np+1,Np)+1], dxds[i])) for i=1:length(dxds)]
   dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
   Fk = k*[dot(x[i+1] - 2*x[i] + x[i-1], P[mod(i-Np+1,Np)+1], dxds[i]) * dxds[i] for i=2:N-1]
   Fk = [[zeros(x[1])]; Fk; [zeros(x[1])] ]

   dE0 = [dE(x[i]) for i=1:length(x)]

   dE0⟂ = [P[mod(i-Np+1,Np)+1] \ dE0[i] - dot(dE0[i], dxds[i])*dxds[i] for i = 1:length(x)]

   maxres = maximum([norm(P[mod(i-Np+1,Np)+1]*dE0⟂[i],Inf) for i = 1:length(x)])

   return ref(- dE0⟂), maxres

end
