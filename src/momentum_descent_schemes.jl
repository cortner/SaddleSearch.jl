"""
forward differences
xⁿ⁺¹ = (2-hb)xⁿ + (hb-1)xⁿ⁻¹ - h²∇E(xⁿ)
"""
function forward_accel(X, Fend, P, h, b)
    return X[end]*(2-h*b) + X[end-1]*(b*h-1) + h*h*Fend
end

"""
backward differences
(1+hb)xⁿ⁺¹ = (2+hb)xⁿ - xⁿ⁻¹ - h²∇E(xⁿ)
"""
function backward_accel(X, Fend, P, h, b)
    return X[end]*(2+h*b)/(1+h*b) + X[end-1]/(-1-h*b) + h*h*Fend/(b*h+1)
end

"""
central differences
(2+hb)xⁿ⁺¹ = 4xⁿ + (hb-2)xⁿ⁻¹ - 2h²∇E(xⁿ)
"""
function central_accel(X, Fend, P, h, b)
    return (4*X[end] + X[end-1]*(b*h-2) + 2*h*h*Fend)/(2+b*h)
end

length_n(x, n) = in(n,2:length(x)-1) ? norm(x[n+1] - x[n-1]) :
             error("l_n not defined for n<2 and n>N-1")
t_n(x, n) = in(n,2:length(x)-1) ? (x[n+1] - x[n-1])/length_n(x, n) : zeros(x[n])

∂Fⁿ⁺(x, n, ∇E) = (1/length_n(x, n)) * ( dot(t_n(x, n), ∇E(x[n])) * (I - kron(t_n(x, n), t_n(x, n)')) - kron(t_n(x, n), F(x, n, ∇E)') )
∂Fⁿ⁻(x, n, ∇E) = - ∂Fnplus(x, n, ∇E)

δFⁿ(x, n, Pn) = - (I - kron(t_n(x,n), t_n(x,n)')) * Pn[n]

∂Sⁿ(κ, x, n) = 2 * κ * kron(t_n(x, n), t_n(x, n)')

∂Sⁿ⁺(κ, x, n) = -κ * (kron(t_n(x, n), t_n(x, n)') +
(1/length_n(x, n)) * kron( x[n+1]-2*x[n]+x[n-1] - dot((x[n+1]-2*x[n]+x[n-1]), t_n(x, n))*t_n(x, n), t_n(x, n)' ) +
(1/length_n(x, n)) * dot(x[n+1]-2*x[n]+x[n-1], t_n(x, n)) * (I - kron(t_n(x,n), t_n(x,n)')))

∂Sⁿ⁻(κ, x, n) = -κ * (kron(t_n(x, n), t_n(x, n)') -
(1/length_n(x, n)) * kron( x[n+1]-2*x[n]+x[n-1] - dot((x[n+1]-2*x[n]+x[n-1]), t_n(x, n))*t_n(x, n), t_n(x, n)' ) -
(1/length_n(x, n)) * dot(x[n+1]-2*x[n]+x[n-1], t_n(x, n)) * (I - kron(t_n(x,n), t_n(x,n)')))

function string_jacobian(x, ∇E, Pn)
    N = length(x); M = length(x[1])
    O = zeros(M, M); J = fill(O,(N, N))
    [J[n,n-1] = ∂Fⁿ⁻(x, n, ∇E) for n=2:N-1]
    [J[n,n] = δFⁿ(x, n, Pn) for n=1:N]
    [J[n,n+1] = ∂Fⁿ⁺(x, n, ∇E) for n=2:N-1]
    return ref(J)
end

function neb_jacobian(x, ∇E, Pn)
    κ = kappa
    N = length(x); M = length(x[1])
    O = zeros(M, M); J = fill(O,(N, N))
    [J[n,n-1] = ∂Fⁿ⁻(x, n, ∇E) + ∂Sⁿ⁻(κ, x, n) for n=2:N-1]
    [J[n,n] = δFⁿ(x, n, Pn) + ∂Sⁿ(κ, x, n) for n=1:N]
    [J[n,n+1] = ∂Fⁿ⁺(x, n, ∇E) + ∂Sⁿ⁺(κ, x, n) for n=2:N-1]
    return ref(J)
end

function ref{T}(A::Array{Array{T,2},2})
    return cat(1,[cat(2,A[n,:]...) for n=1:size(A,1)]...)
end
