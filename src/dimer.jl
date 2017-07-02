

export StaticDimerMethod, DimerMethod

# TODO: change maxnit to maxn_dE

"""
`StaticDimerMethod`: the most basic dimer variant, simply taking alternating
steps with a fixed step-size.

### Parameters:
* `a_trans` : translation step
* `a_rot` : rotation step
* `tol_trans` : translation residual
* `tol_rot` : rotation residual
* `maxnit` : maximum number of iterations
* `len` : dimer-length (i.e. distance of the two walkers)
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_rot` : true/false whether to precondition the rotation step
"""
@with_kw type StaticDimerMethod
   a_trans::Float64
   a_rot::Float64
   # ------ shared parameters ------
   tol_trans::Float64 = 1e-5
   tol_rot::Float64 = 1e-2
   maxnit::Int = 1000
   len::Float64 = 1e-3
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_rot::Bool = false
   rescale_v::Bool = false
end


function run!{T}(method::StaticDimerMethod, E, dE, x0::Vector{T}, v0::Vector{T})

   # read all the parameters
   @unpack a_trans, a_rot, tol_trans, tol_rot, maxnit, len,
            precon_prep!, verbose, precon_rot, rescale_v = method
   P0=method.precon
   # initialise variables
   x, v = copy(x0), copy(v0)
   nit = 0
   numdE, numE = 0, 0
   log = DimerLog()
   # and just start looping
   if verbose >= 2
      @printf(" nit |  |∇E|_∞    |∇R|_∞     λ   \n")
      @printf("-----|-----------------------------\n")
   end
   for nit = 0:maxnit
      # normalise v
      P0 = precon_prep!(P0, x)
      v /= sqrt(dot(v, P0, v))
      # evaluate gradients, and more stuff
      dE0 = dE(x)
      dEv = dE(x + len * v)
      numdE += 2
      Hv = (dEv - dE0) / len

      # NEWTON TYPE RESCALING IN v DIRECTION
      if rescale_v
         P = PreconSMW(P0, v, abs(dot(Hv, v)) - 1.0)
         v /= sqrt(dot(v, P, v))
      else
         P = P0
      end

      # translation and rotation residual, store history
      res_trans = vecnorm(dE0, Inf)
      q_rot = - Hv + dot(v, Hv) * (P * v)
      res_rot = vecnorm(q_rot, Inf)
      push!(log, numE, numdE, res_trans, res_rot)
      if verbose >= 2
         @printf("%4d | %1.2e  %1.2e  %4.2f  \n", nit, res_trans, res_rot, dot(v, Hv))
      end
      if res_trans <= tol_trans && res_rot <= tol_rot
         if verbose >= 1
            println("StaticDimerMethod terminates succesfully after $(nit) iterations")
         end
         return x, v, log
      end
      # translation step
      p_trans = - (P \ dE0) + 2.0 * dot(v, dE0) * v
      x += a_trans * p_trans
      # rotation step
      if precon_rot
         p_rot = - (P \ Hv) + dot(Hv, v) * v
      else
         p_rot = q_rot
      end
      v += a_rot * p_rot
   end
   if verbose >= 1
      println("StaticDimerMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, v, log
end










@with_kw type DimerMethod
   a0_trans::Float64 = 0.01
   a0_rot::Float64 = 0.01
   atol::Float64 = 1e-1
   rtol::Float64 = 1e-1
   C1 = 1e-2
   C2 = 2.0
   hmin = 1e-8
   # ------ shared parameters ------
   tol_trans::Float64 = 1e-5
   tol_rot::Float64 = 1e-2
   maxnit::Int = 1000
   len::Float64 = 1e-3
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
end


function run!{T}(method::DimerMethod, E, dE, x0::Vector{T}, v0::Vector{T})
   @unpack a0_trans, a0_rot, atol, rtol, maxnit, len, precon, precon_prep!, verbose,
         tol_trans, tol_rot, C1, C2, hmin = method

   # function odesolve(solver::ODE12r, f, x0::Vector{Float64}, N::Int,
   #                   log::IterationLog, method;
   #                   g=x->x, tol_res=1e-4, maxnit=100 )

   threshold = atol/rtol
   t = 0.0
   x = copy(x0)
   v = copy(v0)

   numdE, numE = 0, 0

   # compute initial driving forces
   # ---------------------------------------------------
   P = precon_prep!(precon, x)
   v /= sqrt(dot(v, P, v))
   dE0 = dE(x)
   dEv = dE(x + len * v)
   numdE += 2
   Hv = (dEv - dE0) / len
   λ = dot(v, Hv)

   pE0 = P \ dE0

   fx = - pE0 + 2 * dot(v, dE0) * v  # - (I - 2 v ⊗ v) * ∇E
   fv = - Hv + dot(v, Hv) * (P*v)     # - (I - (Pv) ⊗ v) * Hv

   # initial residuals and error estimates
   res_trans = norm(dE0, Inf)
   res_rot = norm(fv, Inf)
   # ---------------------------------------------------
   log = DimerLog()
   push!(log, numE, numdE, res_trans, res_rot)
   if verbose >= 2
      @printf("%4d | %1.2e  %1.2e  %4.2f  \n", 0, res_trans, res_rot, λ)
   end
   if res_trans < tol_trans && res_rot < tol_rot
      println("$(typeof(method)) was initialised with an equilibrium configuration")
      return x, v, log
   end

   hx = a0_trans   # TODO: could do something more clever here > see ode12r
   hv = a0_rot

   for nit = 1:maxnit
      # make a trial step
      # hx = hv = min(hx, hv)
      xnew = x + hx * fx
      vnew = v + hv * fv

      # compute all driving forces
      # ---------------------------------------------------
      Pnew = precon_prep!(P, xnew)
      vnew /= sqrt(dot(vnew, P, vnew))
      dE0 = dE(xnew)
      dEv = dE(xnew + len * vnew)
      numdE += 2
      Hv = (dEv - dE0) / len
      λnew = dot(v, Hv)

      pE0 = P \ dE0

      fx_new = - pE0 + 2 * dot(v, dE0) * v  # - (I - 2 v ⊗ v) * ∇E
      fv_new = - Hv + λ * (P*v)             # - (I - (Pv) ⊗ v) * Hv

      # residuals
      res_trans_new = norm(dE0, Inf)
      res_rot_new = norm(fv, Inf)
      # ---------------------------------------------------

      # check for convergence
      if res_trans_new < tol_trans && res_rot_new < tol_rot
         push!(log, numE, numdE, res_trans_new, res_rot_new)
         if verbose >= 2
            @printf("%4d | %1.2e  %1.2e  %4.2f  \n", nit, res_trans, res_rot, λ)
         end
         println("$(typeof(method)) is terminating succesfully after $(maxnit) iterations.")
         return xnew, vnew, log
      end

      # error estimation
      ex = 0.5 * hx * (fx_new - fx)
      ev = 0.5 * hv * (fv_new - fv)
      err_x = norm(ex ./ max(1.0, threshold), Inf) + realmin(Float64)
      err_v = norm(ev ./ max(1.0, threshold), Inf) + realmin(Float64)

      accept_x = ( (res_trans_new <= res_trans * (1 - C1 * hx)) ||
                   ( (res_trans_new <= res_trans * C2) && (err_x <= rtol) ) )
      accept_v = ( (res_rot_new <= res_rot * (1 - C1 * hv)) ||
                   ( (res_rot_new <= res_rot * C2) && (err_v <= rtol) ) )
      accept = accept_x & accept_v

      # whether we accept or reject this step, we now need a good guess for
      # the next step-size, from a line-search-like construction
      yx, yv = fx - fx_new, fv - fv_new
      hx_ls = hx * dot(fx, yx) / (norm(yx)^2 + 1e-10)
      hv_ls = hv * dot(fv, yv) / (norm(yv)^2 + 1e-10)
      # or from the error estimate
      hx_err = hx * 0.5 * sqrt(rtol/err_x)
      hv_err = hv * 0.5 * sqrt(rtol/err_v)

      if accept
         x, fx, res_trans = xnew, fx_new, res_trans_new
         v, fv, res_rot = vnew, fv_new, res_rot_new
         push!(log, numE, numdE, res_trans_new, res_rot_new)
         if verbose >= 2
            @printf("%4d | %1.2e  %1.2e  %4.2f  \n", nit, res_trans, res_rot, λ)
         end

         # Compute a new step size.
         hx = max(0.25 * hx, min(4*hx, hx_err, hx_ls))
         hv = max(0.25 * hv, min(4*hv, hv_err, hv_ls))
         if verbose >= 3
            println("     accept: new h = $hx, |F| = $(res_trans), $(res_rot)")
            println("               hls = $(hx_ls), $(hv_ls)")
            println("              herr = $(hx_err), $(hv_err)")
         end
      else
         hx = max(0.1 * hx, min(hx_err, hx_ls))
         hv = max(0.1 * hv, min(hv_err, hv_ls))
         if verbose >= 3
            println("     reject: new h = $hx")
            println("              |Fnew| = $(res_trans_new)")
            println("              |Fold| = $(res_trans)")
         end
      end

      if hx <= hmin || hv <= hmin
         warn("Step size too small at nit = $nit: hx = $hx, hv = $hv")
         return xnew, vnew, log
      end
   end

   if verbose >= 1
      println("$(typeof(method)) terminated unsuccesfully after $(maxnit) iterations.")
   end

   # println("DEBUG")
   # Fold, _ = f(t, x)
   # println("  |Fold| = ", norm(Fold, Inf))
   # for h in (1e-1, 1e-2, 1e-3, 1e-4)
   #    Fnew, _ = f(t, x + h * Fold)
   #    println("   |Fnew(h)| = ", norm(Fnew, Inf))
   # end

   return xnew, vnew, log
end
