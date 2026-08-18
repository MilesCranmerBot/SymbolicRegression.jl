# Minimal in-house replacement for the subset of StatsBase.jl used by
# SymbolicRegression: weighted sampling and sampling without replacement.
# Implements the same standard algorithms (cumulative-search weighted
# sampling; partial Fisher-Yates shuffle) but is written from scratch,
# not vendored source.
module StatsBaseLite

using Random: AbstractRNG, default_rng

"""Probability weights for weighted sampling, like `StatsBase.Weights`."""
struct Weights{V<:AbstractVector{<:Real},S<:Real}
    values::V
    sum::S
end
Weights(values::AbstractVector{<:Real}) = Weights(values, sum(values))

"""Draw one index in `1:length(w)` with probability proportional to `w.values`."""
function sample(rng::AbstractRNG, w::Weights)
    t = rand(rng) * w.sum
    s = zero(w.sum)
    @inbounds for i in eachindex(w.values)
        s += w.values[i]
        if s >= t
            return i
        end
    end
    # Guard against floating-point accumulation undershooting w.sum
    return lastindex(w.values)
end
sample(w::Weights) = sample(default_rng(), w)

"""Draw one element of `items` with probability proportional to `w`."""
sample(rng::AbstractRNG, items, w::Weights) = items[sample(rng, w)]
sample(items, w::Weights) = sample(default_rng(), items, w)

"""
    sample([rng], items, k; replace=true)

Draw `k` elements from `items`. Without replacement, uses a partial
Fisher-Yates shuffle on a copy of `items`.
"""
function sample(rng::AbstractRNG, items, k::Integer; replace::Bool=true)
    n = length(items)
    if replace
        return [items[rand(rng, 1:n)] for _ in 1:k]
    end
    k > n &&
        throw(ArgumentError("cannot draw $k samples without replacement from $n items"))
    pool = collect(items)
    @inbounds for i in 1:k
        j = rand(rng, i:n)
        pool[i], pool[j] = pool[j], pool[i]
    end
    return pool[1:k]
end
sample(items, k::Integer; kws...) = sample(default_rng(), items, k; kws...)

end
