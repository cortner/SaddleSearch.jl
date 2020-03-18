
export Dimer, StaticDimer, BBDimer, SuperlinearDimer, ODEDimer, AccelDimer

dimer_shared_docs =  """
### Shared Parameters
* `tol_trans` : translation residual tolerance
* `tol_rot` : rotation residual tolerance
* `maxnumdE` : maximum number of dE evalluations
* `len` : dimer-length (i.e. distance of the two walkers)
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration, 3:file log)
* `precon_rot` : true/false whether to precondition the rotation step
* `rescale_v` : should `v` be rescaled to unit length after each step
"""

@def dimer_shared begin
   tol_trans::Float64 = 1e-4
   tol_rot::Float64 = 1e-2
   maxnumdE::Int = 1000
   len::Float64 = 1e-3
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 1
   precon_rot::Bool = false
   rescale_v::Bool = false
end


"""
`StaticDimer`: the most basic dimer variant, simply taking alternating
steps with a fixed step-size.

### Parameters:
* `a_trans` : translation step
* `a_rot` : rotation step

$(dimer_shared_docs)
"""
@with_kw struct StaticDimer
   a_trans::Float64
   a_rot::Float64
   # ------ shared parameters ------
   @dimer_shared
end


"""
`BBDimer`: dimer method with Barzilai-Borwein step-size + an Armijo
type stability check.

### Parameters:
* `a0_trans` : initial translation step
* `a0_rot` : initial rotation step
* `ls` : linesearch

$(dimer_shared_docs)

## References

The method combines ideas from

[ZDZ] Optimization-based Shrinking Dimer Method for Finding Transition States
SIAM J. Sci. Comput., 38(1), A528–A544
Lei Zhang, Qiang Du, and Zhenzhen Zheng
DOI:10.1137/140972676

and

[GOP] A dimer-type saddle search algorithm with preconditioning and linesearch
Math. Comp. 85, 2016
N. Gould and C. Ortner and D. Packwood
http://arxiv.org/abs/1407.2817
"""
@with_kw struct BBDimer
   a0_trans::Float64
   a0_rot::Float64
   ls = StaticLineSearch()
   # ------ shared parameters ------
   @dimer_shared
end


"""
`SuperlinearDimer`: dimer variant based on Kastner's JCP 128, 014106 (2008)
article & ASE implementation

### Parameters
* maximum_translation : control step-size (actual, not the parameter)
* max_num_rot : maximum number of rotations at each step
* trial_angle : angle increment to predict change in energy after rotation
* trial_trans_step : step increment to predict change in energy after translation
* use_central_forces : ???
* extrapolate : whether or not to do the Kastner extrapolation
* translation_method : only "CG" is implemented

$(dimer_shared_docs)
"""
@with_kw struct SuperlinearDimer
   maximum_translation::Float64 = 0.001
   max_num_rot::Int = 1
   trial_angle::Float64 = pi / 4.0
   trial_trans_step::Float64 = 0.0001
   use_central_forces::Bool = false     # probably does not work well; maybe dont bother
   extrapolate::Bool = true
   translation_method::AbstractString = "CG"  # CG, LBFGS, SD
   # ------ shared parameters ------
   @dimer_shared
end


@with_kw struct ODEDimer
   ode::ODE12r = ODE12r()
   # order::Int = 1    # what is this???
   # damping::Float64 = 1.0   # TODO: add a relative damping for rotation vs translation
   # ------ shared parameters ------
   @dimer_shared
end

# @with_kw struct AccelDimer
#    a0::Float64 = 1e-1
#    b = nothing
#    fd_scheme = :central
#    redistrib = :canonical
#    # ------ shared parameters ------
#    @dimer_shared
# end
@with_kw struct AccelDimer
   h = nothing
   a0 = nothing
   b = nothing
   reltol::Float64
   fd_scheme = :central
   redistrib = :canonical
   # ------ shared parameters ------
   @dimer_shared
end


function Dimer(step=:ode; kwargs...)
   if step == :sd
      return StaticDimer(; kwargs...)
   elseif step == :bb
      return BBDimer(; kwargs...)
   elseif step == :cg
      return SuperlinearDimer(; translation_method="CG", kwargs...)
   elseif step == :ode
      return ODEDimer(; kwargs...)
  elseif step == :accel
     return AccelDimer(; kwargs...)
   else
      error("`Dimer`: unknown step selection mechanism $(step)")
   end
end
