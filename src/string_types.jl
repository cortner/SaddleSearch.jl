export StaticString, ODEString, StaticNEB, ODENEB

@def neb_string_params begin
   tol::Float64 = 1e-5
   maxnit::Int = 1000
   precon_scheme = localPrecon()
   path_traverse = serial()
   verbose::Int = 2
end

@def neb_params begin
   k::Float64 = 0.1
   interp = 1
end



@with_kw type StaticString
   alpha::Float64
   # ------ shared parameters ------
   @neb_string_params
end


@with_kw type ODEString
   reltol::Float64  # please think about reducing it to one tol parameter `odetol`
   threshold::Float64 = 1.0   # threshold for error estimate; we want to get rid of this
   a0 = nothing      # if a0 is not passed then use a default
   # ------ shared parameters ------
   @neb_string_params
end


@with_kw type StaticNEB
   alpha::Float64
   @neb_params
   # ------ shared parameters ------
   @neb_string_params
end


@with_kw type ODENEB
   reltol::Float64  # please think about reducing it to one tol parameter `odetol`
   threshold::Float64 = 1.0   # threshold for error estimate; we want to get rid of this
   a0 = nothing      # if a0 is not passed then use a default
   @neb_params
   # ------ shared parameters ------
   @neb_string_params
end

function String(step, args...; kwargs...)
   if step == :static
      return StaticString(args...; kwargs...)
  elseif step == :ode
      return ODEString(args...; kwargs...)
   else
      error("`String`: unknown step selection mechanism $(step)")
   end
end

function NEB(step, args...; kwargs...)
   if step == :static
      return StaticNEB(args...; kwargs...)
  elseif step == :ode
      return ODENEB(args...; kwargs...)
   else
      error("`NEB`: unknown step selection mechanism $(step)")
   end
end

solver(method::StaticString) = Euler(h=method.alpha)
solver(method::ODEString) = ODE12r(rtol=method.reltol, threshold=method.threshold)
solver(method::StaticNEB) = Euler(h=method.alpha)
solver(method::ODENEB) = ODE12r(rtol=method.reltol, threshold=method.threshold)


# TODO:
#  - write 3 test problems: muller, double-well and 2D vacancy
#  - get those 4 methods to run in your current implementation with and without precon
#  - rewrite and cleanup using above types
#  - delete all code that is not needed anymore
