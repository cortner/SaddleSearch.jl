# using SaddleSearch: run!


function run!(method::SuperlinearDimer, E, dE, x0::Vector{T}, v0::Vector{T}) where {T}
   # read all the parameters
   @unpack maximum_translation, max_num_rot, trial_angle, trial_trans_step, extrapolate, translation_method,
           tol_trans, tol_rot, maxnumdE, len, precon_prep!, verbose = method
  # @show translation_method
  P=method.precon
   # initialise variables
   x, v = copy(x0), copy(v0)
   nit = 0
   numdE, numE = 0, 0
   log = DimerLog()
   if translation_method == "CG"
      cg_init = true
      direction_old = nothing
      cg_direction = nothing
   elseif translation_method == "LBFGS"
      reset_hessian = true
      s = []
      y = []
      rho = []
      x0, v0 = x, v
      vec0 = x
   else
      error("unknown translation_method")
   end

   # and just start looping
   if verbose >= 2
      @printf(" nit |  n∇E  |   |∇E|_∞    |∇R|_∞    λ        \n")
      @printf("-----|-------|--------------------------------\n")
   end
   P = precon_prep!(P, x)
   v /= sqrt(dotP(v, P, v))
   normalize(a) = a / norm(a)
   parallel_vector(a, b) = dot(a, b) * b    # assume b is normalised
   perpendicular_vector(a, b) = a - parallel_vector(a, b)
   rotate_vectors(a, b, phi) =    # do a, b need to be normalised??? CHECK!
         normalize(a * cos(phi) + b * sin(phi)) * norm(a),
         normalize(b * cos(phi) - a * sin(phi)) * norm(b)
   curv = 0.0
   for nit = 0:2*maxnumdE
      # normalise v
      P = precon_prep!(P, x)
      # evaluate gradients, and more stuff
      dE0 = dE(x)
      numdE += 1

      # rotation step

      stoprot = false
      nrot = 0
      dE1e = nothing      # extrapolated guess for dE1
      res_rot = 0.0
      while !stoprot     # ROTATION STEP

          nrot += 1

          if dE1e == nothing
             dE1 = dE(x + len * v)
             numdE += 1
          else
             dE1 = dE1e
          end
          dE2 = dE(x - len * v)
          numdE += 1

          dE1a = copy(dE1)     # dE1a store for later use (does it need to be a copy?)

          Hxv = (dE1 - dE2) / 2.0 / len
          curv = dot(Hxv, v)
          f_rot_A = perpendicular_vector(Hxv, v)   # rotation residual!
          res_rot = norm(f_rot_A, Inf)

          if res_rot <= tol_rot
             stoprot = true
          else                  # single rotation step
             n_A = v
             rot_unit_A = normalize(f_rot_A)

             # get the curvature and its derivative
             c0 = curv
             c0d = dot(dE1 - dE2, rot_unit_A) / len

             # trial rotation
             n_B, rot_unit_B = rotate_vectors(n_A, rot_unit_A, trial_angle)
             v = n_B

             # update gradients
             dE1 = dE(x + len * v)
             dE2 = dE(x - len * v)
             numdE += 2

             dE1b = copy(dE1)

             # get the curvature's derivative
             c1d = dot(dE1 - dE2, rot_unit_B) / len

             # calculate Fourier coefficients
             a1 = c0d * cos(2.0 * trial_angle) - c1d / (2.0 * sin(2.0 * trial_angle))
             b1 = 0.5 * c0d
             a0 = 2 * (c0 - a1)

             # estimate rotation angle
             rotangle = atan(b1 / a1) / 2.0

             # make sure its not the maximum
             cmin = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
             if c0 < cmin
                rotangle += pi / 2.0
             end

             # rotate into the estimated lowest eigenmode
             n_C, rot_unit_C = rotate_vectors(n_A, rot_unit_A, rotangle)
             v = n_C

             curv = cmin

             # force extrapolation to reduce number of calls
             if extrapolate
                dE1e = sin(trial_angle - rotangle) / sin(trial_angle) * dE1a + sin(rotangle) / sin(trial_angle) * dE1b + (1 - cos(rotangle) - sin(rotangle) * tan(trial_angle / 2.0)) * dE0
             else
                dE1e = nothing
             end

             if !stoprot
                if nrot >= max_num_rot
                   stoprot = true
                end
             end
          end
      end

      # TRANSLATION STEP

      # choose e-vec following or standard dimer direction
      if curv > 0
         f0p = -parallel_vector(-dE0, v)
      else
         f0p = -dE0 - 2.0 * parallel_vector(-dE0, v)
      end
      direction = f0p

      # mix up directions according to CG or LBFGS (or nothing)
      if translation_method == "CG"
         if cg_init
            cg_init = false
            direction_old = direction
            cg_direction = direction
         end
         old_norm = dot(direction_old, direction_old)
         # Polak-Ribiere Conjugate Gradient
         if old_norm != 0.0
            betaPR = dot(direction, (direction - direction_old)) / old_norm
         else
            betaPR = 0.0
         end
         if betaPR < 0.0
            betaPR = 0.0
         end
         cg_direction = direction + cg_direction * betaPR
         direction_old = direction

         direction = cg_direction
      elseif translation_method == "LBFGS"
         error("LBFGS not implemented")
      end


      direction = normalize(direction)

      if curv > 0.0
         xstep = direction * maximum_translation
      else
         dE0t = dE(x + direction * trial_trans_step)
         numdE += 1
         f0tp = -dE0t - 2.0 * parallel_vector(-dE0t, v)
         # f0p: current translation residual (not necessarily search direction!)
         # f0tp: translation residual at trial step
         # F: some kind of slope in search direction???
         # C: curvature in search direction?
         F = dot(f0tp + f0p, direction) / 2.0
         C = dot(f0tp - f0p, direction) / trial_trans_step
         xstep = ( -F / C + trial_trans_step / 2.0 ) * direction
         if norm(xstep) > maximum_translation
            xstep = direction * maximum_translation
         end
      end

      x += xstep

      # translation residual, store history
      res_trans = norm(dE0, Inf)
      push!(log, numE, numdE, res_trans, res_rot)
      if verbose >= 2
         @printf("%4d | %4d  |  %1.2e  %1.2e  %1.2e \n", nit, numdE, res_trans, res_rot, curv)
      end
      if res_trans <= tol_trans
         if verbose >= 1
            println("SuperlinearDimer terminates succesfully after $(nit) iterations")
         end
         return x, v, log
      end
      if numdE > maxnumdE
         if verbose >= 1
            println("SuperlinearDimer terminates unsuccesfully due to numdE >= $(maxnumdE)")
         end
         return x, v, log
      end

   end
   error("why am I here?")
   return x, v, log
end
