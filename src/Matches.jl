"""
A simple machine-learning framework
that computes gradients without backpropagation.

Based on "Gradients without backpropagation"
<https://arxiv.org/abs/2202.08587>

$(EXPORTS)
"""
module Matches

import LinearAlgebra: svd
import Statistics: mean

using DocStringExtensions

DocStringExtensions

include("Dual.jl")
include("ML.jl")
include("Optim.jl")

end # module Matches
