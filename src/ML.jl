export Activations, Losses, Linear, Sequential, params

"""
Activation functions.

$(EXPORTS)
"""
module Activations
import ..EXPORTS
export relu, softplus, sigmoid, tanh

relu(x) = max.(0, x)
softplus(x, β::Real=1.0) = @. log(1 + exp(β * x)) / β
sigmoid(x) = @. 1 / (1 + exp(-x))
tanh(x) = Base.tanh.(x)
end

"""
Loss functions.

$(EXPORTS)
"""
module Losses
export mse, mae
import ..mean, ..EXPORTS

mse(y_hat, y) = mean(@. (y_hat - y)^2)
mae(y_hat, y) = mean(@. abs(y_hat - y))
end

abstract type AbstractModule end

params(::Any) = AbstractVecOrMat[]

# ===== Linear =====
mutable struct Linear{T<:Real} <: AbstractModule
    W::Matrix{Dual{T}}
    b::Vector{Dual{T}}
end

"""
$(TYPEDSIGNATURES)

Callable linear module. Can be applied to matrix of shape `(n_observations, n_dims)`.

Same as <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>
"""
function Linear(in_::Integer, out::Integer)
    W = randn(out, in_)
    W ./= svd(W).S[1] # div by highest singular value
    b = zeros(out)

    Linear(Dual.(W), Dual.(b))
end

params(m::Linear)::Vector{AbstractVecOrMat} = [m.W, m.b]


(m::Linear)(x::AbstractVecOrMat) = x * transpose(m.W) .+ transpose(m.b)

# ===== Sequential =====
mutable struct Sequential <: AbstractModule
    modules::Vector{Union{AbstractModule, Function}}

    """
    $(TYPEDSIGNATURES)

    Callable module that chains calls to other modules.
    Can be applied to matrix of shape `(n_observations, n_dims)`.

    Same as <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>
    """
    Sequential(modules...) = new([modules...])
end

params(m::Sequential)::Vector{AbstractVecOrMat} =
    vcat(params.(m.modules)...)

function (m::Sequential)(x::AbstractVecOrMat)
    res = x
    for layer in m.modules
        res = layer(res)
    end

    res
end
