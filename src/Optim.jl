export Descent, random_dual!, step!

"""
$(TYPEDEF)

State of gradient descent optimizer.

$(TYPEDFIELDS)
"""
struct Descent
    "Parameters to be tweaked"
    params::Tuple{Vararg{AbstractVecOrMat}} # I bet this is highly type-unstabe

    "Learning rate"
    lr::Real
end

"""
$(TYPEDSIGNATURES)

Randomize dual ("imaginary") parts of model parameters.

Must be called _before_ calling the model.
"""
function random_dual!(d::Descent)
    # Randomize dual part
    for par in d.params
        @inbounds for i in eachindex(par)
            par[i].im = randn()
        end
    end
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
