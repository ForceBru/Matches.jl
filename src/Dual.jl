export Dual, TrackedDual, track!

"""
$(TYPEDEF)

A dual number.

$(TYPEDFIELDS)
"""
mutable struct Dual{T<:Real} <: Number
    "Real part of the dual number"
    re::T

    "Dual ('imaginary') part"
    im::T
end

# ===== Constructors =====
Dual{T}(re::T) where T<:Real = Dual{T}(re, zero(T))

Dual{T}(re::U) where {T<:Real, U<:Real} = 
    Dual{T}(convert(T, re), zero(T))

Dual(re::Real) = Dual{typeof(re)}(re)
Dual{T}(d::Dual{U}) where {T<:Real, U<:Real} =
    Dual{T}(convert(T, d.re), convert(T, d.im))

# ===== Conversion and promotion =====
Base.convert(::Type{Dual}, x::Dual) = x
Base.convert(::Type{Dual}, x::Real) = Dual(x)

Base.promote_rule(::Type{Dual{T}}, ::Type{S}) where {T<:Real, S<:Real} =
    Dual{promote_type(T, S)}

Base.promote_rule(a::Type{S}, b::Type{Dual{T}}) where {T<:Real, S<:Real} =
    promote_rule(b, a)

# ===== Element access =====
Base.real(d::Dual) = d.re
Base.imag(d::Dual) = d.im

# ===== Operations =====
Base.:+(x::Dual, y::Dual) = Dual(x.re + y.re, x.im + y.im)
Base.:-(x::Dual, y::Dual) = Dual(x.re - y.re, x.im - y.im)
Base.:*(x::Dual, y::Dual) = Dual(x.re * y.re, x.re * y.im + x.im * y.re)
Base.:/(x::Dual, y::Dual) = Dual(x.re / y.re, (-x.re * y.im + x.im * y.re) / y.re^2)
#Base.:^(x::Dual, p::Real) = Dual(x.re^p, p * x.re^(p - 1) * x.im)

Base.:-(x::Dual) = Dual(-x.re, -x.im)
Base.conj(x::Dual) = Dual(x.re, -x.im)

Base.sin(x::Dual) = Dual(sin(x.re), cos(x.re) * x.im)
Base.cos(x::Dual) = Dual(cos(x.re), -sin(x.re) * x.im)
Base.tanh(x::Dual) = Dual(tanh(x.re), (1 - tanh(x.re)^2) * x.im)

Base.log(x::Dual) = Dual(log(x.re), 1/x.re * x.im)
Base.exp(x::Dual) = Dual(exp(x.re), exp(x.re) * x.im)

Base.isless(x::Dual, y::Dual) = isless(x.re, y.re)
Base.:(==)(x::Dual, y::Dual) = x.re == y.re