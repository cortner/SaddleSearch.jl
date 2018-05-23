
function rayleigh(v, x, len, E0, E, P)
   w = v / sqrt(dot(v, P, v))
   return 2.0 * (E(x+len/2*w) - 2.0 * E0 + E(x-len/2*v)) / len^2
end


localmerit(x, x0, v0, len, g0, λ0, E) = (
   0.5 * ( E(x+len/2*v0) + E(x-len/2*v0) )
   - 2.0 * dot(v0, g0) * dot(v0, x-x0)
   - λ0 * dot(v0, x-x0)^2  )




function run!{T}(method::StaticDimer, E, dE, x0::Vector{T}, v0::Vector{T})

   # read all the parameters
   @unpack a_trans, a_rot, tol_trans, tol_rot, maxnumdE, len,
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
   for nit = 0:maxnumdE  # there will be at most this many evaluations
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
            println("StaticDimer terminates succesfully after $(nit) iterations")
         end
         return x, v, log
      end
      if numdE > maxnumdE
         if verbose >= 1
            println("StaticDimer terminates unsuccesfully due to numdE >= $(maxnumdE)")
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

   error("why am I here?")
end






function run!{T}(method::BBDimer, E, dE, x0::Vector{T}, v0::Vector{T})

   # read all the parameters
   @unpack a0_trans, a0_rot, tol_trans, tol_rot, maxnumdE, len,
            precon_prep!, verbose, precon_rot, rescale_v, ls = method
   P0=method.precon
   # initialise variables
   x, v = copy(x0), copy(v0)
   β, γ = -1.0, -1.0   # this tells the loop that they haven't been initialised
   numdE, numE = 0, 0
   log = DimerLog()
   p_trans_old = zeros(length(x0))
   p_rot_old = zeros(length(x0))
   # and just start looping
   if verbose >= 2
      @printf(" nit |  |∇E|_∞    |∇R|_∞        λ         β         γ \n")
      @printf("-----|------------------------------------------------\n")
   end
   nit = 0
   while true
      nit += 1

      if any(isnan, v) || any(isnan, x)
         error("""BBDimer method has encountered NANs. This can happen with
                  BB type step size selection, which can be fast but is sometimes
                  unstable. To prevent this, use try different initial conditions
                  or use a different saddle-search method.""")
      end

      # normalise v
      P0 = precon_prep!(P0, x)
      v /= sqrt(dot(v, P0, v))
      # evaluate gradients, and more stuff
      dEm = dE(x - len/2 * v)
      dEp = dE(x + len/2 * v)
      numdE += 2
      Hv = (dEp - dEm) / len
      dE0 = 0.5 * (dEp + dEm)

      # NEWTON TYPE RESCALING IN v DIRECTION
      if rescale_v
         P = PreconSMW(P0, v, abs(dot(Hv, v)) - 1.0)
         v /= sqrt(dot(v, P, v))
      else
         P = P0
      end

      # translation and rotation residual, store history
      res_trans = vecnorm(dE0, Inf)
      λ = dot(v, Hv)
      q_rot = - Hv + λ * (P * v)
      res_rot = vecnorm(q_rot, Inf)
      push!(log, numE, numdE, res_trans, res_rot)
      if verbose >= 2
         @printf("%4d | %1.2e  %1.2e  %1.2e  %1.2e  %1.2e \n",
               nit, res_trans, res_rot, λ, β, γ)
      end

      # check whether to terminate
      if res_trans <= tol_trans && res_rot <= tol_rot
         if verbose >= 1
            println("BBDimer terminates succesfully after $(nit) iterations")
         end
         return x, v, log
      end
      if numdE > maxnumdE
         if verbose >= 1
            println("BBDimer terminates unsuccesfully due to numdE >= $(maxnumdE)")
         end
         return x, v, log
      end

      # compute the two "search" directions
      p_trans = - (P \ dE0) + 2.0 * dot(v, dE0) * v
      if precon_rot
         p_rot = - (P \ Hv - λ * v)
      else
         p_rot = q_rot
      end

      # initial step-sizes guess
      if nit == 1     # first iteration
         β = a0_trans
         γ = a0_rot
      else             # subsequent iterations: BB step
         # Δx, Δv has already been computed (see end of loop)
         Δg = p_trans - p_trans_old
         Δd = p_rot - p_rot_old
         β = abs( dot(Δx, P, Δg) / dot(Δg, P, Δg) )
         γ = abs( dot(Δv, P, Δd) / dot(Δd, P, Δd) )
      end

      # perform linesearch on the translation step
      F_trans = xx -> localmerit(xx, x, v, len, dE0, λ, E)
      β, numE_ls, _ = linesearch!(ls, F_trans, F_trans(x), - dot(p_trans, P, p_trans),
                                    x, p_trans, β)
      numE += numE_ls

      # perform line-search on rotation step
      E0 = E(x);  numE += 1
      F_rot = vv -> rayleigh(vv, x, len, E0, E, P)
      γ, numE_ls, _ = linesearch!(ls, F_rot, F_rot(v), -dot(p_rot, P, p_rot),
                                    v, p_rot, γ)
      numE += numE_ls

      if isnan(β) || isnan(γ)
         if verbose >= 1
            println("BBDimer terminates unsuccesfully due to unsuccesful linesearch")
         end
         return x, v, log
      end

      x_old, v_old = x, v
      # translation step
      x += β * p_trans
      # rotation step
      v += γ * p_rot

      # remember the change in x and v, also remember the old p_trans and p_rot
      Δx = β * p_trans
      Δv = γ * p_rot
      copy!(p_trans_old, p_trans)
      copy!(p_rot_old, p_rot)
   end
   error("why am I here?")
end




function dimer_ode(z, dE, P, precon_prep!, len)
   n = length(z) ÷ 2
   x, v = z[1:n], z[n+1:end]
   P = precon_prep!(P, x)
   # evaluate gradients, and more stuff
   dE0 = dE(x)
   dEv = dE(x + len * v)
   Hv = (dEv - dE0) / len

   p_trans = - (P \ dE0) + 2.0 * dot(v, dE0) * v
   p_rot = - (P \ Hv) + dot(Hv, v) * v
   F = [p_trans; p_rot]
   return F, norm(F, Inf)
end

function dimer_project(z, P, precon_prep!)
   n = length(z) ÷ 2
   x, v = z[1:n], z[n+1:end]
   P = precon_prep!(P, x)
   v /= sqrt(dot(v, P, v))
   znew = copy(z)
   znew[n+1:end] = v
   return znew
end


function run!(method::ODEDimer, E, dE, x0::Vector, v0::Vector)

   # read all the parameters
   @unpack abstol, reltol, order, damping, tol_trans, tol_rot, maxnumdE, len,
            precon_prep!, verbose, precon_rot, rescale_v = method
   P0=method.precon

   # initial condition
   n = length(x0)
   z0 = [x0; v0]
   # nonlinear system
   F = (t, z, nit) -> dimer_ode(z, dE, P0, precon_prep!, len)
   # projection (normalisation) step
   G = z -> dimer_project(z, P0, precon_prep!)
   # initialise a log
   log = PathLog()

   # run the ODE solver
   tout, zout, log = odesolve_co(ODE12r(atol=abstol, rtol=reltol),
                              F,
                              z0,   # initial condition
                              2,    # number of dE evaluations per F call
                              log,  # store iteration information in this log
                              method,
                              g = G,
                              maxnit = maxnumdE,
                              tol_res=min(tol_trans, tol_rot))
   z = zout[end]
   return z[1:n], z[n+1:end], log
end
