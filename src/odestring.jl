
using Dierckx
export ODEStringMethod

"""
`ODEStringMethod`: string variant utilising adaptive time step ode solvers.

### Parameters:
* `alpha` : step length
* 'abstol' : absolute errors tolerance
* 'reltol' : relative errors tolerance
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_cond` : true/false whether to precondition the minimisation step
"""
@with_kw type ODEStringMethod
   solver = ode12(1e-6, 1e-3, true)
   precon_scheme = coordTransform()
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   # precon = I
   # precon_prep! = (P, x) -> P
   verbose::Int = 2
   # precon_cond::Bool = false
end

function run!{T}(method::ODEStringMethod, E, dE, x0::Vector{T}, t0::Vector{T})
   # read all the parameters
   @unpack solver, precon_scheme, tol_res, maxnit, verbose = method
   # initialise variables
   x, t = copy(x0), copy(t0)
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
      @printf(" nit |  sup|∇E|_∞   \n")
      @printf("-----|-----------------\n")
   end

   αout, xout, log = odesolve(solver, (α_,x_) -> forces(precon_scheme, x, x_, dE), ref(x), length(x), log, method; g = x_ -> reparametrise(x, x_, precon_scheme), tol_res = tol_res, maxnit=maxnit )

   x = set_ref!(x, xout[end])
   return x, log, αout
end

function forces{T}(precon_scheme, x::Vector{T}, xref::Vector{Float64}, dE)
   @unpack precon, precon_prep!, precon_cond, tangent_norm, gradDescent⟂, force_eval, maxres = precon_scheme

   x = set_ref!(x, xref)

   precon = precon_prep!(precon, x)

   Np = length(precon); P = i -> precon[mod(i-Np+1,Np)+1]

   ds = [norm(P(i), x[i+1]-x[i]) for i=1:length(x)-1]

   param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param /= param[end]; param[end] = 1.
   S = [Spline1D(param, [x[j][i] for j=1:length(x)], w = ones(length(x)),
         k = 3, bc = "error") for i=1:length(x[1])]
   t= [[derivative(S[i], s) for i in 1:length(S)] for s in param]
   t ./= [tangent_norm(P(i), t[i]) for i=1:length(x)]

   t[1] =zeros(t[1]); t[end]=zeros(t[1])

   dE0 = [dE(x[i]) for i=1:length(x)]
   dE0⟂ = gradDescent⟂(P, dE0, t)
   F = [force_eval(P(i), dE0[i], dE0⟂[i], t[i]) for i=1:length(x)]

   res = maxres(P, dE0⟂, F)

   return ref(- F), res
end

function ref{T}(x::Vector{T})
   return cat(1, x...)
end

function set_ref!{T}(x::Vector{T}, xref::Vector{Float64})
   Nimg = length(x); Nref = length(xref) ÷ Nimg
   X = reshape(xref, Nref, Nimg)
   x = [ X[:, n] for n = 1:Nimg ]
   return x
end

function reparametrise{T}(x::Vector{T}, xref::Vector{Float64}, precon_scheme)
   @unpack precon_prep!, precon_cond = precon_scheme
   P=precon_scheme.precon

   P = precon_prep!(P, x)
   Np = length(P)

   x = set_ref!(x, xref)

   ds = [sqrt(dot(x[i+1]-x[i], (P[mod(i-Np+1,Np)+1]+P[mod(i-1-Np+1,Np)+1])/2, x[i+1]-x[i])) for i=1:length(x)-1]
   s = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   s /= s[end]; s[end] = 1.
   S = [Spline1D(s, [x[j][i] for j=1:length(s)], w = ones(length(x)),
        k = 3, bc = "error") for i=1:length(x[1])]
   x = [ [S[i](s) for i in 1:length(S)] for s in linspace(0., 1., length(x)) ]

   return ref(x)
end
