# Matches.jl

The most barebones machine learning "framework" that can compute (unbiased estimates of) gradients without backpropagation.
Based on "Gradients without backpropagation" [1].

__NOTE__: I'm _not_ one of the authors and have no affiliation with the paper. I just found it interesting and tried to see whether it works.

The interface mimicks PyTorch, but since this package is just a toy, it's not a torch, but a bunch of matches :P

## Basic usage

```julia
using Matches

# 1. Get data.
n_observations, n_features = 100, 5
X, Y = cumsum(randn(n_observations, n_features)), cumsum(randn(n_observations, 1))

# 2. Build model, PyTorch-like.
model = Sequential(
    Linear(n_features, 7), Activations.sigmoid,
    Linear(7, 5), Activations.tanh,
    Linear(5, 1)
)

# 3. Set up gradient descent optimizer.
# It can compute the "forward gradient" from the paper.
optim = Descent(params(model), 1e-4)

# 4. Train.
for epoch in 1:50_000
    # Randomize the dual parts of model parameters
    random_dual!(optim)
    loss = Losses.mse(model(X), Y)
    # Estimate the gradient and use it in gradient descent
    step!(optim, loss)

    if epoch % 1000 == 0
        @show (epoch, real(loss))
    end
end

# 5. Predict!
Y_hat = real(model(X))
```

Also see [`example.jl`](./example.jl). Basic usage:

```sh
$ wget https://raw.githubusercontent.com/ForceBru/Matches.jl/master/example.jl
$ julia example.jl
```

## References

1. Baydin, A. G., Pearlmutter, B. A., Syme, D., Wood, F., & Torr, P. (2022). Gradients without Backpropagation (Version 1). arXiv. <https://doi.org/10.48550/ARXIV.2202.08587>
