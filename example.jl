begin
    @info "Setting up environment..."
    import Pkg
    Pkg.activate(temp=true)
    Pkg.add(url="https://github.com/ForceBru/Matches.jl", io=devnull)
    Pkg.add("UnicodePlots", io=devnull)
    Pkg.status()
end

using Matches, UnicodePlots

function to_windows(series::AbstractVector, win_size)
    vcat([
        series[off - win_size + 1:off]'
        for off in win_size:length(series)
    ]...)
end

# Data (n_observations, n_features)
time_series = cumsum(randn(200))
windows = to_windows(time_series, 20 + 1)

X, Y = windows[:, 1:end-1], windows[:, end:end]
@show size(X), size(Y)

# Model
model = Sequential(
    Linear(20, 7), Activations.sigmoid,
    Linear(7, 5), Activations.tanh,
    Linear(5, 1)
)

# Optimizer
optim = Descent(params(model), 1e-4)

# Train
epochs = 50_000
loss_history = Real[]
for epoch in 1:epochs
    random_dual!(optim)
    Y_hat = model(X)
    loss = Losses.mse(Y_hat, Y)
    step!(optim, loss)

    push!(loss_history, real(loss))

    if epoch % 500 == 0
        @show epoch, real(loss)
    end
end

# Plot the original time-series and the fit
plt = let 
    Y_original = Y[:, 1]
    Y_fitted = model(X)[:, 1] |> real
    
    plt = lineplot(Y_original, width=140, height=25)
    lineplot!(plt, Y_fitted)
    plt
end

display(plt)
