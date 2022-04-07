# Join tuples: https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/9
@inline tuplejoin(x::Tuple)::Tuple = x
@inline tuplejoin(x::Tuple, y::Tuple)::Tuple = (x..., y...)
@inline tuplejoin(x::Tuple, y::Tuple, z...)::Tuple = (x..., tuplejoin(y, z...)...)
