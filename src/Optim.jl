export Descent, Adam, random_dual!, step!

abstract type Optimizer end

"""
$(TYPEDSIGNATURES)

Randomize dual ("imaginary") parts of model parameters.

Must be called _before_ calling the model.
"""
function random_dual!(opt::Optimizer)
    # Randomize dual part
    for par in opt.params
        @inbounds for i in eachindex(par)
            par[i].im = randn()
        end
    end
end

# ===== Gradient descent =====
"""
$(TYPEDEF)

State of gradient descent optimizer.

$(TYPEDFIELDS)
"""
struct Descent <: Optimizer
    "Parameters to be tweaked"
    params::Tuple{Vararg{AbstractVecOrMat}} # I bet this is highly type-unstabe

    "Learning rate"
    lr::Real
end

"""
$(TYPEDSIGNATURES)

Computes the forward gradient and updates model parameters.

See paper: <https://arxiv.org/abs/2202.08587>
"""
function step!(d::Descent, loss::Dual, lr::Real=d.lr)
    @assert lr > 0
    
    # One step of gradient descent
    for par in d.params
        @inbounds for i in eachindex(par)
            forward_grad_i = loss.im * par[i].im
            par[i].re -= lr * forward_grad_i
        end
    end
end

# ===== ADAM =====
"""
$(TYPEDEF)

State of the ADAM optimizer. See paper at https://arxiv.org/abs/1412.6980

$(TYPEDFIELDS)
"""
mutable struct Adam{T<:Real} <: Optimizer
    "Parameters to be tweaked"
    params::Tuple{Vararg{AbstractVecOrMat{Dual{T}}}} # I bet this is highly type-unstabe

    "Learning rate"
    lr::Real
    "Smoothing for 1st moment"
    β1::Real
    "Smoothing for 2nd moment"
    β2::Real
    "Very small positive constant"
    t::UInt

    "1st moment vector"
    m::Vector{T}
    "2nd moment vector"
    v::Vector{T}
end

function Adam{T}(params, lr::Real, β1::Real=0.9, β2::Real=0.999) where T<:Real
    N = nparams(params)
    m = zeros(T, N)
    v = zeros(T, N)

    Adam{T}(
        params,
        lr, β1, β2, 0x00,
        m, v
    )
end

Adam(params, lr::Real, β1::Real=0.9, β2::Real=0.999) = Adam{Float64}(params, lr, β1, β2)

function step!(opt::Adam, loss::Dual, lr::Real=opt.lr; ϵ::Real=1e-8)
    @assert lr > 0
    @assert ϵ > 0
    
    opt.t += one(opt.t)
    parid::Integer = 1
    for par in opt.params
        @inbounds for i in eachindex(par)
            forward_grad_i = loss.im * par[i].im

            opt.m[parid] = opt.β1 * opt.m[parid] + (1 - opt.β1) * forward_grad_i
            opt.v[parid] = opt.β2 * opt.v[parid] + (1 - opt.β2) * forward_grad_i^2
            m_hat = opt.m[parid] / (1 - opt.β1^opt.t)
            v_hat = opt.v[parid] / (1 - opt.β2^opt.t)

            par[i].re -= opt.lr * m_hat / (sqrt(v_hat) + ϵ)
            parid += 1
        end
    end
end