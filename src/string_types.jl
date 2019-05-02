
export StaticString, ODEString, StaticNEB, ODENEB, AccelString

neb_string_shared_docs =  """
### Shared Parameters
* `tol` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon_scheme` : preconditioner scheme (localPrecon(), globalPrecon())
* `path_traverse` : how to travserse path to compute energy (serial(), palindrome())
* `fixed_ends` : true/false for fixing the ends of the path, default is false
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration, 3:file log)
"""

@def neb_string_params begin
   tol::Float64 = 1e-5
   maxnit::Int = 1000
   precon_scheme = localPrecon()
   path_traverse = serial()
   fixed_ends = false
   verbose::Int = 2
end

neb_only_docs = """
### NEB only Parameters:
* `k` : spring constant
* `interp` : degree of interpolant (1: central differences, >1: splines of degree interp)
"""
@def neb_params begin
   k::Float64 = 0.1
   interp = 1
end


"""
`StaticString`: the most basic string variant, integrating potential gradient
flow with a fixed step-size.

### Parameters:
* `alpha` : Euler method integration step length

$(neb_string_shared_docs)
"""
@with_kw type StaticString
   alpha::Float64
   # ------ shared parameters ------
   @neb_string_params
end


"""
`ODEString`: string method with adaptive ODE step size selection.

### Parameters:
* `reltol` : ode solver relative tolerance
* `threshold` : threshold for error estimate
* `a0` : initial step, if a0 is not passed then use a default

$(neb_string_shared_docs)
"""
@with_kw type ODEString
   reltol::Float64  # please think about reducing it to one tol parameter `odetol`
   threshold::Float64 = 1.0   # threshold for error estimate; we want to get rid of this
   a0 = nothing      # if a0 is not passed then use a default
   # ------ shared parameters ------
   @neb_string_params
end

"""
`AccelString`: string method using momentum descent to accelerate energy minimisation

### Parameters:
* `a0` : initial step, if a0 is not passed then use a default
* `b` : momentum term damping coefficient
* `finite_diff_scheme` : choice of discretisation of dumped wave equation

$(neb_string_shared_docs)
"""
@with_kw type AccelString
   a0 = nothing      # if a0 is not passed then use a default
   b = nothing      # if b is not passed then optimal value is used
   finite_diff_scheme = central_accel
   # ------ shared parameters ------
   @neb_string_params
end

"""
`StaticNEB`: the most basic NEB variant, integrating potential gradient
flow with a fixed step-size.

### Parameters:
* `alpha` : Euler method integration step length

$(neb_only_docs)

$(neb_string_shared_docs)
"""
@with_kw type StaticNEB
   alpha::Float64
   @neb_params
   # ------ shared parameters ------
   @neb_string_params
end


"""
`ODEString`: NEB method with adaptive ODE step size selection.

### Parameters:
* `reltol` : ode solver relative tolerance
* `threshold` : threshold for error estimate
* `a0` : initial step, if a0 is not passed then use a default

$(neb_only_docs)

$(neb_string_shared_docs)
"""
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
   elseif step == :accel
       return AccelString(args...; kwargs...)
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
solver(method::ODEString) = ODE12r(rtol=method.reltol, threshold=method.threshold, h=method.a0)
solver(method::AccelString) = momentum_descent(h=method.a0, b=method.b, finite_diff=method.finite_diff_scheme)
solver(method::StaticNEB) = Euler(h=method.alpha)
solver(method::ODENEB) = ODE12r(rtol=method.reltol, threshold=method.threshold, h=method.a0)
