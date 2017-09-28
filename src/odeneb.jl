
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
            precon, precon_prep!, verbose, precon_cond = method
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

   αout, xout, log = odesolve(solver, (α_,x_) -> forces(x, x_, k, dE, precon, precon_prep!), ref(x), length(x), log, method; tol_res = tol_res, maxnit=maxnit )

   x = set_ref!(x, xout[end])
   return x, log, αout
end

function forces{T}(x::Vector{T}, xref::Vector{Float64},
                     k::Float64, dE, precon, precon_prep!)
   x = set_ref!(x, xref)
   N = length(x)
   precon = precon_prep!(precon, x)
   Np = length(precon); P = i -> precon[mod(i-1,Np)+1]

   # central finite differences
   dxds = [(x[i+1]-x[i-1])/2 for i=2:N-1]
   dxds ./= [norm(P(i), dxds[i]) for i=1:length(dxds)]
   dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
   Fk = k*[dot(x[i+1] - 2*x[i] + x[i-1], P(i), dxds[i]) * dxds[i] for i=2:N-1]
   Fk = [[zeros(x[1])]; Fk; [zeros(x[1])] ]

   dE0 = [dE(x[i]) for i=1:length(x)]

   dE0⟂ = [P(i) \ dE0[i] - dot(dE0[i], dxds[i])*dxds[i] for i = 1:length(x)]

   res = maximum([norm(P(i)*dE0⟂[i],Inf) for i = 1:length(x)])

   return ref(- dE0⟂+Fk), res

end
