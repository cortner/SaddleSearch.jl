




function run!{T}(method::StaticDimerMethod, E, dE, x0::Vector{T}, v0::Vector{T})

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
            println("StaticDimerMethod terminates succesfully after $(nit) iterations")
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
