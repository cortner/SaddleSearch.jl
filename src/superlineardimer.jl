export SuperlinearDimer

# TODO: change maxnit to maxn_dE

"""
`SuperlinearDimer`: dimer variant based on Kastner's JCP 128, 014106 (2008) article & ASE implementation

### Parameters:
* `tol_trans` : translation residual
* `tol_rot` : rotation residual
* `maxnit` : maximum number of iterations
* `len` : dimer-length (i.e. distance of the two walkers)
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
"""
@with_kw type SuperlinearDimer
   maximum_translation::Float64 = 0.001
   max_num_rot::Int = 1
   trial_angle::Float64 = pi / 4.0
   trial_trans_step::Float64 = 0.0001
   use_central_forces::Bool = false
   extrapolate::Bool = true
   translation_method::AbstractString = "LCG"
   # ------ shared parameters ------
   tol_trans::Float64 = 1e-5
   tol_rot::Float64 = 1e-1
   maxnumdE::Int = 1000
   len::Float64 = 1e-3
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   id::AbstractString = "SuperlinearDimer"
end


function run!{T}(method::SuperlinearDimer, E, dE, x0::Vector{T}, v0::Vector{T})

   # read all the parameters
   @unpack maximum_translation, max_num_rot, trial_angle, trial_trans_step, use_central_forces, extrapolate, translation_method, 
           tol_trans, tol_rot, maxnumdE, len, precon_prep!, verbose = method 
   P=method.precon
   # initialise variables
   x, v = copy(x0), copy(v0)
   nit = 0
   numdE, numE = 0, 0
   log = IterationLog()
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
   end
   # and just start looping
   if verbose >= 2
      @printf(" nit |  |∇E|_∞    |∇R|_∞    λ    \n")
      @printf("-----|---------------------------\n")
   end
   P = precon_prep!(P, x)
   v /= sqrt(dot(v, P, v))
   normalize(a) = a / norm(a)
   parallel_vector(a, b) = dot(a, b) * b
   perpendicular_vector(a, b) = a - parallel_vector(a, b)
   rotate_vectors(a, b, phi) = normalize(a * cos(phi) + b * sin(phi)) * norm(a), normalize(b * cos(phi) - a * sin(phi)) * norm(b)
   curv = 0
   for nit = 0:2*maxnumdE
      # normalise v
      P = precon_prep!(P, x)
      # evaluate gradients, and more stuff
      dE0 = dE(x)
      numdE += 1

      # rotation step

      stoprot = false
      nrot = 0
      dE1e = nothing
      while !stoprot

          nrot += 1 

          if dE1e == nothing
             dE1 = dE(x + len * v)
             numdE += 1
          else
             dE1 = dE1e
          end
          if use_central_forces
             dE2 = dE1 - 2.0 * dE0
          else
             dE2 = dE(x - len * v)
             numdE += 1
          end

          dE1a = dE1

          curv = dot(dE1 - dE2, v) / 2.0 / len
          f_rot_A = perpendicular_vector(dE2 - dE1, v) / 2.0 / len

          if norm(f_rot_A) <= tol_rot
             stoprot = true
          else
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
             numdE += 1
             if use_central_forces
                dE2 = dE1 - 2.0 * dE0
             else
                dE2 = dE(x - len * v)
                numdE += 1
             end

             dE1b = dE1

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

      # translation step

      if curv > 0
         f0p = -parallel_vector(-dE0, v)
      else
         f0p = -dE0 - 2.0 * parallel_vector(-dE0, v)
      end
      direction = f0p 
      
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
      end

      direction = normalize(direction)

      if curv > 0.0
         xstep = direction * maximum_translation 
      else
         dE0t = dE(x + direction * trial_trans_step)
         numdE += 1
         f0tp = -dE0t - 2.0 * parallel_vector(-dE0t, v)
         F = dot(f0tp + f0p, direction) / 2.0
         C = dot(f0tp - f0p, direction) / trial_trans_step
         xstep = ( -F / C + trial_trans_step / 2.0 ) * direction
         if norm(xstep) > maximum_translation
            xstep = direction * maximum_translation
         end
      end

      x += xstep 

      # translation residual, store history
      res_trans = vecnorm(dE0, Inf)
#     res_rot = tol_rot
      res_rot = norm(v)
      push!(log, numE, numdE, res_trans, res_rot)
      if verbose >= 2
         @printf("%4d | %1.2e  %1.2e  %1.2e \n", nit, res_trans, res_rot, curv)
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
