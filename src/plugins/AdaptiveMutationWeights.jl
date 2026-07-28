module AdaptiveMutationWeightsModule

using ..CoreModule:
    AbstractPlugin,
    AbstractOptions,
    AbstractMutation,
    SimplifyMutation,
    DoNothingMutation,
    MutationEvent
import ..CoreModule: init_plugin_state, fork_plugin_state, on_mutation_end!
import ..MutateModule: condition_mutation_weights!, _scale_weight!

"""
    AdaptiveMutationWeightsPlugin <: AbstractPlugin

Online-adapt per-mutation weights from the search's own success statistics.
For each mutation kind, the plugin tracks attempts and "strictly improving"
successes (`accepted && after_loss < before_loss`) and adjusts a
multiplicative factor applied to that mutation's base weight, updated each
mutation via an EMA over the smoothed success-ratio with a floor clamp.

Statistics are local to one worker dispatch — no cross-population
aggregation.

# Fields
- `smoothing::Float64 = 0.02`: EMA factor for the multiplier update.
- `floor::Float64 = 0.05`: clamp range for a single mutation's target
  multiplier (`[floor, 1/floor]`). Prevents necessary rare ops from being
  starved.

Mutation kinds excluded from accounting are declared by dispatch on
[`skip_in_adaptive_weights`](@ref); by default `SimplifyMutation` and `DoNothingMutation`
are skipped. Define `skip_in_adaptive_weights(::MyMutation) = true` to add
your own.

!!! warning "Experimental"
"""
Base.@kwdef struct AdaptiveMutationWeightsPlugin <: AbstractPlugin
    smoothing::Float64 = 0.02
    floor::Float64 = 0.05
end

"""
    skip_in_adaptive_weights(::AbstractMutation) -> Bool

Whether [`AdaptiveMutationWeightsPlugin`](@ref) should exclude a mutation
kind from its success/attempt accounting. Default `false`. Overridden to
`true` for `SimplifyMutation` and `DoNothingMutation` (they accept trivially / don't
represent real search moves).

Extend by dispatch:

```julia
SymbolicRegression.skip_in_adaptive_weights(::MyMutation) = true
```
"""
skip_in_adaptive_weights(::AbstractMutation) = false
skip_in_adaptive_weights(::SimplifyMutation) = true
skip_in_adaptive_weights(::DoNothingMutation) = true

"""
    AdaptiveMutationWeightsState

Per-dispatch (per-worker) mutable counters and multipliers, parallel to
`options.mutations`. Reset at each `fork_plugin_state` call.
"""
mutable struct AdaptiveMutationWeightsState
    attempts::Vector{Float64}
    successes::Vector{Float64}
    multipliers::Vector{Float64}
end

function init_plugin_state(::AdaptiveMutationWeightsPlugin, options, dataset)
    n = length(options.mutations)
    return AdaptiveMutationWeightsState(zeros(n), zeros(n), ones(n))
end

# Fresh stats per worker dispatch (per-population locality; no cross-pop merge).
function fork_plugin_state(
    head_state::AdaptiveMutationWeightsState, ::AdaptiveMutationWeightsPlugin, dataset
)
    n = length(head_state.multipliers)
    return AdaptiveMutationWeightsState(zeros(n), zeros(n), ones(n))
end

function on_mutation_end!(
    s::AdaptiveMutationWeightsState,
    p::AdaptiveMutationWeightsPlugin,
    mutation::AbstractMutation,
    event::MutationEvent,
    dataset,
    options::AbstractOptions,
)
    skip_in_adaptive_weights(mutation) && return nothing
    idx = event.mutation_idx
    s.attempts[idx] += 1.0
    if event.accepted && event.after_loss < event.before_loss
        s.successes[idx] += 1.0
    end
    # Recompute multipliers from current rates.
    total_successes = 0.0
    total_attempts = 0.0
    @inbounds for i in eachindex(s.attempts)
        total_successes += s.successes[i]
        total_attempts += s.attempts[i]
    end
    n = length(s.attempts)
    mean_rate = (total_successes + n) / (total_attempts + 2n)
    f = p.floor
    upper = f > 0 ? inv(f) : Inf
    @inbounds for i in eachindex(s.multipliers)
        rate = (s.successes[i] + 1.0) / (s.attempts[i] + 2.0)
        target = clamp(rate / mean_rate, f, upper)
        s.multipliers[i] = (1 - p.smoothing) * s.multipliers[i] + p.smoothing * target
    end
    return nothing
end

function condition_mutation_weights!(
    weights::AbstractVector,
    s::AdaptiveMutationWeightsState,
    ::AdaptiveMutationWeightsPlugin,
    member,
    options::AbstractOptions,
    curmaxsize,
    nfeatures,
)
    mutations = options.mutations
    @inbounds for i in eachindex(mutations)
        m, w = weights[i]
        weights[i] = m => w * s.multipliers[i]
    end
    return nothing
end

end  # module AdaptiveMutationWeightsModule
