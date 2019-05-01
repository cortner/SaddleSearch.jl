"""
forward differences
xⁿ⁺¹ = (2-hb)xⁿ + (hb-1)xⁿ⁻¹ - h²∇E(xⁿ)
"""
function forward_accel(X, dE, P, h, b)
    return X[end]*(2-h*b) + X[end-1]*(b*h-1) - h*h*dE(X[end])
end

"""
backward differences
(1+hb)xⁿ⁺¹ = (2+hb)xⁿ - xⁿ⁻¹ - h²∇E(xⁿ)
"""
function backward_accel(X, dE, P, h, b)
    return X[end]*(2+h*b)/(1+h*b) + X[end-1]/(-1-h*b) - h*h*dE(X[end])/(b*h+1)
end

"""
central differences
(2+hb)xⁿ⁺¹ = 4xⁿ + (hb-2)xⁿ⁻¹ - 2h²∇E(xⁿ)
"""
function central_accel(X, dE, P, h, b)
    return (4*X[end] + X[end-1]*(b*h-2) - 2*h*h*dE(X[end]))/(2+b*h)
end
