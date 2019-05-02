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
