# Join tuples: https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/9
@inline tuplejoin(x::Tuple) = x
@inline tuplejoin(x::Tuple, y::Tuple) = (x..., y...)
@inline tuplejoin(x::Tuple, y::Tuple, z...) = (x..., tuplejoin(y, z...)...)