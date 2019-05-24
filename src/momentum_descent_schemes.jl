"""
forward differences
xⁿ⁺¹ = (2-hb)xⁿ + (hb-1)xⁿ⁻¹ - h²∇E(xⁿ)
"""
function forward_accel(X, Fend, Λ, b)
   h = 1.0; it = 1; it_max = 100
   while (it<=it_max && !minimum([forward_criterion(λ*h*h, b*h) for λ in Λ[real(Λ).>0.5]]))
       h = h/2
       it+=1
   end
    return X[end]*(2-h*b) + X[end-1]*(b*h-1) + h*h*Fend
end

"""
backward differences
(1+hb)xⁿ⁺¹ = (2+hb)xⁿ - xⁿ⁻¹ - h²∇E(xⁿ)
"""
function backward_accel(X, Fend, Λ, b)
   h = 1.0; it = 1; it_max = 100
   while (it<=it_max && !minimum([backward_criterion(λ*h*h, b*h) for λ in Λ[real(Λ).>0.5]]))
       h = h/2
       it+=1
   end
    return X[end]*(2+h*b)/(1+h*b) + X[end-1]/(-1-h*b) + h*h*Fend/(b*h+1)
end

"""
central differences
(2+hb)xⁿ⁺¹ = 4xⁿ + (hb-2)xⁿ⁻¹ - 2h²∇E(xⁿ)
"""
function central_accel(X, Fend, Λ, b)
   h = 1.0; it = 1; it_max = 100
   while (it<=it_max && !minimum([central_criterion(λ*h*h, b*h) for λ in Λ[real(Λ).>0.5]]))
       h = h/2
       it+=1
   end
    return (4*X[end] + X[end-1]*(b*h-2) + 2*h*h*Fend)/(2+b*h)
end

function objective(b, λ)
    return real(b-sqrt(complex(b*b-4*λ)))
end

function b_choice(λ)
    η = 4*real(λ); ξ = 4*imag(λ)
    f = b -> objective(b, λ)
    b = 2*sqrt(real(λ))
    if ξ!=0
        a = 4η; b = -3ξ^2; c = 6*η*ξ^2; d = (η*ξ)^2-4ξ^4
        Δ₀ = b^2-3a*c; Δ₁ = 2b^3 - 9a*b*c + 27d*a^2
        C = ( complex(0.5(Δ₁+sign(Δ₁)*sqrt(complex(Δ₁^2-4Δ₀^3)))) )^(1/3)
        x₁ = -(1/(3a))*(b+C+Δ₀/C)
        k = -0.5-0.5*sqrt(3)*im
        x₂ = -(1/(3a))*(b + k*C + Δ₀/(k*C))
        x₃ = -(1/(3a))*(b + k*k*C + Δ₀/(k*k*C))
        x = [x₁; x₂; x₃]
        x_real = real(x[findmin(abs.(imag(x)))[2]])
        b = sqrt(x_real+η)
    end

#     f_b = maximum([f(bᵢ) for bᵢ in b_all])
#     b = b_all[indmax([f(bᵢ) for bᵢ in b_all])]
    return b
end

function stability(λ)
    η = 4*real(λ); ξ = 4*imag(λ)
    f = b -> objective(b, λ)
    b = b_choice(λ)
    f_b = f(b)

    return f_b, b
end

function forward_criterion(λh², hb)
    condition_p = abs(2-hb-λh² + sqrt(complex(hb*hb + (λh²)^2 - 4λh² + λh²*hb))) <2
    condition_m = abs(2-hb-λh² - sqrt(complex(hb*hb + (λh²)^2 - 4λh² + λh²*hb))) <2
    return condition_p && condition_m
end

function backward_criterion(λh², hb)
    condition_p = abs( (2+hb-λh² + sqrt(complex((2+hb-λh²)^2-4(1+hb)))) / (1+hb) ) <2
    condition_m = abs( (2+hb-λh² - sqrt(complex((2+hb-λh²)^2-4(1+hb)))) / (1+hb) ) <2
    return condition_p && condition_m
end

function central_criterion(λh², hb)
    condition_p = abs((2-λh² + sqrt(complex( (2-λh²)^2 - 4 + hb*hb )))/(2+hb)) <1
    condition_m = abs((2-λh² - sqrt(complex( (2-λh²)^2 - 4 + hb*hb )))/(2+hb)) <1
    return condition_p && condition_m
end


length_n(x, n) = in(n,2:length(x)-1) ? norm(x[n+1] - x[n-1]) :
             error("l_n not defined for n<2 and n>N-1")
t_n(x, n) = in(n,2:length(x)-1) ? (x[n+1] - x[n-1])/length_n(x, n) : zeros(x[n])

F(x, n, ∇E, P) = - ∇E(x[n]) + dot(t_n(x, n), ∇E(x[n])) * P(n) * t_n(x, n)

∂Fⁿ⁺(x, n, ∇E, P) = (1/length_n(x, n)) * ( dot(t_n(x, n), ∇E(x[n])) * (I - kron(t_n(x, n), (P(n) * t_n(x, n))')) - kron(t_n(x, n), F(x, n, ∇E, P)') )
∂Fⁿ⁻(x, n, ∇E, P) = - ∂Fⁿ⁺(x, n, ∇E, P)

δFⁿ(x, n, ∇∇E, P⁻¹∇∇E) = - ( P⁻¹∇∇E(n) - kron(t_n(x,n), (∇∇E(n) * t_n(x, n))') )

∂Sⁿ(κ, x, n) = 2 * κ * kron(t_n(x, n), t_n(x, n)')

∂Sⁿ⁺(κ, x, n) = -κ * (kron(t_n(x, n), t_n(x, n)') +
(1/length_n(x, n)) * kron( x[n+1]-2*x[n]+x[n-1] - dot((x[n+1]-2*x[n]+x[n-1]), t_n(x, n))*t_n(x, n), t_n(x, n)' ) +
(1/length_n(x, n)) * dot(x[n+1]-2*x[n]+x[n-1], t_n(x, n)) * (I - kron(t_n(x,n), t_n(x,n)')))

∂Sⁿ⁻(κ, x, n) = -κ * (kron(t_n(x, n), t_n(x, n)') -
(1/length_n(x, n)) * kron( x[n+1]-2*x[n]+x[n-1] - dot((x[n+1]-2*x[n]+x[n-1]), t_n(x, n))*t_n(x, n), t_n(x, n)' ) -
(1/length_n(x, n)) * dot(x[n+1]-2*x[n]+x[n-1], t_n(x, n)) * (I - kron(t_n(x,n), t_n(x,n)')))


function ref{T}(A::Array{Array{T,2},2})
    return cat(1,[cat(2,A[n,:]...) for n=1:size(A,1)]...)
end
