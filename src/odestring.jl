
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
   abstol::Float64 = 1e-6
   reltol::Float64 = 1e-3
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_cond::Bool = false
end


function run!{T}(method::ODEStringMethod, E, dE, x0::Vector{T}, t0::Vector{T})
   # read all the parameters
   @unpack abstol, reltol, tol_res, maxnit,
            precon_prep!, verbose, precon_cond = method
   P=method.precon
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

   αout, xout, log = bs23((α_,x_) -> forces(x, x_, dE, P, precon_prep!), ref(x), length(x), log, method; g = x_ -> reparametrise(x, x_, P, precon_prep!), atol = abstol, rtol = reltol, tol_res = tol_res, maxnit=maxnit )

   x = set_ref!(x, xout[end])
   return x, log, αout
end

function forces{T}(x::Vector{T}, xref::Vector{Float64}, dE, P, precon_prep!)
   x = set_ref!(x, xref)
   P = precon_prep!(P, x)
   Np = length(P)

   ds = [sqrt(dot(x[i+1]-x[i], P[mod(i-Np+1,Np)+1], x[i+1]-x[i])) for i=1:length(x)-1]
   param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param /= param[end]; param[end] = 1.
   S = [Spline1D(param, [x[j][i] for j=1:length(x)], w = ones(length(x)),
         k = 3, bc = "error") for i=1:length(x[1])]
   t= [[derivative(S[i], s) for i in 1:length(S)] for s in param]
   t ./= [sqrt(dot(t[i], P[mod(i-Np+1,Np)+1], t[i])) for i=1:length(x)]
   t[1] =zeros(t[1]); t[end]=zeros(t[1])

   dE0 = [dE(x[i]) for i=1:length(x)]
   PdE0 = [P[mod(i-Np+1,Np)+1] \ dE0[i] for i = 1:length(x)]
   dE0⟂ = [P[mod(i-Np+1,Np)+1] \ dE0[i] - dot(dE0[i],t[i])*t[i] for i = 1:length(x)]

   maxres = maximum([norm(P[mod(i-Np+1,Np)+1]*dE0⟂[i],Inf) for i = 1:length(x)])

   return ref(- PdE0), maxres

end

function ref{T}(x::Vector{T})
   Nimg = length(x)
   Ndim = length(x[1])
   X = zeros(Ndim, Nimg)
   [X[:,n] = x[n] for n=1:Nimg]
   return X[:]
end

function set_ref!{T}(x::Vector{T}, xref::Vector{Float64})
   Nimg = length(x); Nref = length(xref) ÷ Nimg
   X = reshape(xref, Nref, Nimg)
   x = [ X[:, n] for n = 1:Nimg ]
   return x
end

function reparametrise{T}(x::Vector{T}, xref::Vector{Float64}, P, precon_prep!)
   x = set_ref!(x, xref)
   P = precon_prep!(P, x)
   Np = length(P)

   ds = [sqrt(dot(x[i+1]-x[i], (P[mod(i-Np+1,Np)+1]+P[mod(i-1-Np+1,Np)+1])/2, x[i+1]-x[i])) for i=1:length(x)-1]
   s = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   s /= s[end]; s[end] = 1.
   S = [Spline1D(s, [x[j][i] for j=1:length(s)], w = ones(length(x)),
        k = 3, bc = "error") for i=1:length(x[1])]
   x = [ [S[i](s) for i in 1:length(S)] for s in linspace(0., 1., length(x)) ]

   return ref(x)
end
