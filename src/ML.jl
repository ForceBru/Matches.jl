export Activations, Losses, Linear, Sequential, params, nparams

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

"""
Initializers for neural network weights.

$(EXPORTS)
"""
module Initializers
    import ..EXPORTS, TYPEDSIGNATURES

    "Random number from U[-r, r]"
    rand_uniform(r::Real, dims...) = 2r .* rand(dims...) .- r

    function glorot_normal(in::Integer, out::Integer)
        variance = 2 / (in + out)
        sqrt(variance) * randn(out, in)
    end

    function glorot_uniform(in::Integer, out::Integer)
        r = sqrt(6 / (in + out))
        rand_uniform(r, out, in)
    end

    """
    $(TYPEDSIGNATURES)

    Return a matrix with singular value `0 < r < 1`.
    Useful for Echo State networks.
    """
    function decaying(in::Integer, out::Integer, r::Real=0.9)
        W = randn(out, in)
        W ./= svd(W).S[1] # div by highest singular value
    end
end

abstract type AbstractModule end

params(::Any)::Tuple{Vararg{AbstractVecOrMat}} = ()
nparams(mod::Any)::Integer = 0
nparams(mod::AbstractModule)::Integer = mod |> params .|> length |> sum
nparams(params::Tuple) = params .|> length |> sum

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
function Linear(in_::Integer, out::Integer, init::Function=Initializers.glorot_uniform)
    W = init(in_, out)
    b = zeros(out)

    Linear(Dual.(W), Dual.(b))
end

params(m::Linear)::Tuple{AbstractMatrix, AbstractVector} = (m.W, m.b)


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

params(m::Sequential)::Tuple{Vararg{AbstractVecOrMat}} =
    tuplejoin(params.(m.modules)...)

function (m::Sequential)(x::AbstractVecOrMat)
    res = x
    for layer in m.modules
        res = layer(res)
    end

    res
end
