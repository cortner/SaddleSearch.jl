import Optim, LineSearches

export RotOptimDimer

# TODO: change maxnit to maxn_dE

"""
`RotOptimDimer`: dimer variant with optimised rotation

### Parameters:
* `a_trans` : translation step
* `tol_trans` : translation residual
* `tol_rot` : rotation residual
* `maxnit` : maximum number of iterations
* `len` : dimer-length (i.e. distance of the two walkers)
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_rot` : true/false whether to precondition the rotation step
"""
@with_kw type RotOptimDimer
   a_trans::Float64
   # ------ shared parameters ------
   tol_trans::Float64 = 1e-5
   tol_rot::Float64 = 1e-2
   maxnumdE::Int = 1000
   len::Float64 = 1e-3
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_rot::Bool = false
   rmemory::Int = 100
   tmemory::Int = 100
   id::AbstractString = "RotOptDimer"
end


function run!{T}(method::RotOptimDimer, E, dE, x0::Vector{T}, v0::Vector{T})

   # read all the parameters
   @unpack a_trans, tol_trans, tol_rot, maxnumdE, len,
            precon_prep!, verbose, rmemory, tmemory, precon_rot = method
   P=method.precon
   # initialise variables
   x, v = copy(x0), copy(v0)
   nit = 0
   numdE, numE = 0, 0
   log = DimerLog()
   # LBFGS stuff for translation
   reset_hessian = true
   s = []
   y = []
   rho = []
   x0, v0 = x, v
   vec0 = x
   # and just start looping
   if verbose >= 2
      @printf(" nit |  |∇E|_∞    |∇R|_∞    λ   \n")
      @printf("-----|--------------------------\n")
   end
   P = precon_prep!(P, x)
   v /= sqrt(dot(v, P, v))
   dimerE(v) = ( E(x + len * v / sqrt(dot(v, P, v))) + E(x - len * v / sqrt(dot(v, P, v))) ) / len^2
   for nit = 0:2*maxnumdE
      # normalise v
      P = precon_prep!(P, x)
      # evaluate gradients, and more stuff
      # rotation step
#     res = Optim.optimize(dimerE,v,method=Optim.LBFGS(m = rmemory, linesearch! = LineSearches.morethuente!),g_tol = tol_rot,store_trace=true,show_trace=(verbose>2))
      res = Optim.optimize(dimerE,v,method=Optim.LBFGS(m = rmemory),g_tol = tol_rot,store_trace=true,show_trace=(verbose>2))
      v = Optim.minimizer(res)
      v /= sqrt(dot(v, P, v))
      # translation step
      dE0 = dE(x)
#     vec = - P \ dE0 + 2.0 * dot(v, dE0) * v
#     if !reset_hessian
#         s0 = x - x0
#         push!(s, s0)
#         y0 = vec0 - vec
#         push!(y, y0)
#         rho0 = 1.0 / dot(y0, s0)
#         push!(rho, rho0)
#         if length(s) > tmemory
#             deleteat!(s,1)
#             deleteat!(y,1)
#             deleteat!(rho,1)
#         end
#     end

#     reset_hessian = false

#     loopmax = length(s)
#     alpha = zeros(loopmax)

#     q = -vec

#     for i=loopmax:-1:1
#         alpha[i] = rho[i] * dot(s[i], q)
#         q -= alpha[i] * y[i]
#     end

#     if P == I
#         z = (I / 70.0) * q
#     else
#         z = P \ q
#     end

#     for i=1:loopmax
#         beta = rho[i] * dot(y[i], z)
#         z += s[i] * (alpha[i] - beta)
#     end

#     x0 = x
#     vec0 = vec

#     p_trans = -z

#     p_trans_norm = norm(p_trans)
#     if p_trans_norm > a_trans
#         p_trans = p_trans / p_trans_norm * sqrt(a_trans)
#     end

#     x += p_trans
      p_trans = - P \ dE0 + 2.0 * dot(v, dE0) * v
      x += a_trans * p_trans

      # translation residual, store history
      res_trans = vecnorm(dE0, Inf)
      res_rot = tol_rot
      numE += res.f_calls
      numdE += res.g_calls + 1
      push!(log, numE, numdE, res_trans, res_rot)
      if verbose >= 2
         @printf("%4d | %1.2e  %1.2e  %1.2e \n", nit, res_trans, res_rot, res.f_minimum)
      end
      if res_trans <= tol_trans
         if verbose >= 1
            println("RotOptimDimer terminates succesfully after $(nit) iterations")
         end
         return x, v, log
      end
      if numdE > maxnumdE
         if verbose >= 1
            println("BBDimer terminates unsuccesfully due to numdE >= $(maxnumdE)")
         end
         return x, v, log
      end

   end
   error("why am I here?")
   return x, v, log
end
