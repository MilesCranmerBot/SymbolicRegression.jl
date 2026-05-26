module PluginModule

using DispatchDoctor: @unstable
using ..MutationsModule: AbstractMutation, ConstantMutation, ConstantMutationContext

# ────────────────────────────────────────────────────────────────────────────
# Hook naming taxonomy
# ────────────────────────────────────────────────────────────────────────────
#
# All plugin hooks belong to one of five categories. The name's verb/suffix
# tells you the contract; the docstring tells you the semantics.
#
#   `on_X_start!` / `on_X_end!`   — Observer. Engine fires; plugin reacts.
#                                   Return value ignored. `!` because state
#                                   may mutate. Composes by iteration.
#
#   `X_multiplier`                — Multiplier. Returns a `Real` the engine
#                                   multiplies into a base value. Reads like
#                                   a Julia getter (`length`, `size`). Plugins
#                                   compose multiplicatively in tuple order.
#
#   `condition_X!`                — Conditioner. Mutates a passed struct in
#                                   place. Engine layers plugin conditioning
#                                   on top of its own. Composes by sequential
#                                   in-place mutation in tuple order.
#
#   `init_X`                      — Factory (one-time). Called once per
#                                   (plugin, output) at startup. Returns a
#                                   new instance.
#
#   `prepare_X`                   — Factory (per-context). Called per dispatch
#                                   with the dataset. Returns a new instance
#                                   for the worker.
#
# When adding a new hook: pick a category, follow the verb shape. If the name
# feels ambiguous, the contract probably isn't one of these five — rethink
# whether you need a new category or are forcing the feature into the wrong
# shape.
#
# ── Hook argument-order convention ──────────────────────────────────────────
#
# All hooks follow the same positional convention:
#
#     (mutated_thing_if_any, state, plugin, ...other_context)
#
# - For `!` functions that mutate one specific thing (e.g.
#   `condition_X!(X, ...)`), the mutated thing is the FIRST arg, then
#   `state`, then `plugin`.
# - For observer `on_X_end!` hooks, the state is the mutated thing, so
#   ordering is `(state, plugin, ...)`.
# - For non-`!` hooks that read state without mutating (`*_multiplier`,
#   `init_member`, etc.), ordering is still `(state, plugin, ...)`.
# - For constructor hooks that *create* a state (`init_plugin_state`) — no
#   state exists yet, so plugin is the dispatch key and goes first.
#
# The verb/suffix in the name + this ordering rule together ARE the contract.
# ────────────────────────────────────────────────────────────────────────────

"""
    AbstractPlugin

Abstract type for a plugin's *configuration*. A plugin instance is an immutable
struct whose fields are the user-tunable settings of the plugin. Mutable
runtime data is held separately in a state object returned by
[`init_plugin_state`](@ref).

A search may have **any number of plugins active simultaneously**, supplied
via the `plugins = (Plugin1(), Plugin2(), ...)` keyword on `Options`. The
engine iterates the tuple at each lifecycle point and dispatches the
appropriate hook on the plugin type.

## Plugin state

State is duck-typed: it can be any object, including a `NamedTuple`, a `Dict`,
or a mutable struct of your own. Every hook dispatches on the *plugin* type,
so the state needs no particular supertype.

**Thread / Multiprocessing Safety**:
- `on_generation_end!` runs serially on the head node — safe to mutate.
- `on_cycle_end!` and `on_mutation_end!` run on workers, against per-dispatch
  copies built by [`fork_plugin_state`](@ref). Cross-worker
  communication must use `Channel` / `RemoteChannel`.
- `init_member` reads the head node's per-output state during initial
  population creation. In multithreading mode, multiple population-creation
  tasks may call it concurrently — keep it read-only or thread-safe.
- In multiprocessing mode, plugin config is serialized to workers via
  `options.plugins`, and state is forked on the head by
  [`fork_plugin_state`](@ref) and shipped with the dispatch. Workers never
  construct their own state.

!!! warning "Experimental"
    The plugin interface is experimental. Hook signatures may change in minor
    releases until validated by multiple in-tree plugins.
"""
abstract type AbstractPlugin end

"""
    init_plugin_state(plugin::AbstractPlugin, options, dataset) -> state

Create the mutable per-output state for `plugin`. Called once per
(plugin, output) pair at search start.

Override by dispatching on your plugin type:

```julia
SymbolicRegression.init_plugin_state(p::MyPlugin, options, dataset) =
    MyPluginState(p.config)
```

Default returns `nothing`.

!!! warning "Experimental"
"""
function init_plugin_state(::AbstractPlugin, options, dataset)
    return nothing
end

"""
    on_search_start!(plugin, state, dataset, options, ropt)

Lifecycle hook called on the head node after initialization, before warmup
and the main search loop. Called once per (plugin, output) pair.

Override by dispatching on your plugin type:

```julia
SymbolicRegression.on_search_start!(p::MyPlugin, s::MyPluginState, dataset, options, ropt) = ...
```

Default is a no-op.

!!! warning "Experimental"
"""
function on_search_start!(_, ::AbstractPlugin, dataset, options, ropt)
    return nothing
end

"""
    on_search_end!(plugin, state, search_state, dataset, options, ropt)

Lifecycle hook called on the head node after the main search loop exits and
before tearing down processes/threads. Multiprocessing cycles may still be
running when this hook is called. Called once per (plugin, output) pair.

Override by dispatching on your plugin type. Default is a no-op.

!!! warning "Experimental"
"""
function on_search_end!(_, ::AbstractPlugin, search_state, dataset, options, ropt)
    return nothing
end

"""
    on_generation_end!(plugin, state, search_state, dataset, options, ropt, returned_pop)

Lifecycle hook called on the HEAD NODE after each cycle's result has been
received from a worker. Runs serially; safe to mutate plugin state, update
concept databases, drain feedback channels, etc. Called once per (plugin,
output) pair per cycle. `state` is this output's state; `returned_pop` is
the population the worker produced.

Override by dispatching on your plugin type. Default is a no-op.

!!! warning "Experimental"
"""
function on_generation_end!(
    _, ::AbstractPlugin, search_state, dataset, options, ropt, returned_pop
)
    return nothing
end

"""
    on_cycle_end!(plugin, state, pop, dataset, hof, options)

Lifecycle hook called on the WORKER after each s_r_cycle finishes. May run
concurrently across workers. Use only worker-local state, or use `Channel` /
`RemoteChannel` for cross-worker communication. Called once per plugin per
worker cycle.

Override by dispatching on your plugin type. Default is a no-op.

!!! warning "Experimental"
"""
function on_cycle_end!(_, ::AbstractPlugin, pop, dataset, hof, options)
    return nothing
end

"""
    MutationEvent

Bundle of per-mutation observations passed to [`on_mutation_end!`](@ref)
once the accept/reject decision has been made inside `next_generation`.

# Fields

- `accepted::Bool`: `true` if the mutation was accepted (via
  `return_immediately` or annealing/fitness acceptance); `false` if rejected
  (constraint failure, NaN loss, or annealing/frequency rejection).
- `before_loss::L`: loss of the parent member before mutation.
- `after_loss::Union{L,Nothing}`: loss after mutation. `nothing` if no valid evaluation
  occurred (constraint failure or NaN loss).

`L` is the dataset's loss type (e.g. `Float32` or `Float64`).

The mutation kind itself is passed as a separate dispatch arg to
[`on_mutation_end!`](@ref), not stored on the event — that way plugin
authors can write type-specific methods.

!!! warning "Experimental"
"""
struct MutationEvent{L<:Real}
    accepted::Bool
    before_loss::L
    after_loss::Union{L,Nothing}
end

"""
    on_mutation_end!(state, plugin, mutation::AbstractMutation, event::MutationEvent, dataset, options)

Lifecycle hook called on the WORKER immediately before each return from
`next_generation`, after the final accept/reject decision for a mutation.
Called once per plugin per mutation. Plugins can dispatch on the mutation
type (e.g. `::ConstantMutation`) for type-specific handling, or
`::AbstractMutation` for a generic catch-all.

Default is a no-op.

!!! warning "Experimental"
"""
function on_mutation_end!(
    _, ::AbstractPlugin, ::AbstractMutation, ::MutationEvent, dataset, options
)
    return nothing
end

"""
    tournament_cost_multiplier(state, plugin, member, options) -> Real

Per-plugin multiplier applied to a candidate's `member.cost` during tournament
selection in `_best_of_sample`. Plugins compose multiplicatively: the
adjusted cost is `member.cost * ∏ tournament_cost_multiplier(s, p, ...)`
across all plugins in tuple order. Default returns `1.0` (no adjustment).

The shipped `AdaptiveParsimonyPlugin` is the canonical example; it reads
frequency statistics from its own state, not from an engine-passed arg.

!!! warning "Experimental"
"""
function tournament_cost_multiplier(_, ::AbstractPlugin, member, options)
    return 1.0
end

"""
    mutation_acceptance_multiplier(state, plugin, parent_member, new_tree, before_cost, after_cost, options) -> Real

Per-plugin multiplicative contribution to the engine's per-mutation accept
probability. The engine takes the product across all plugins and draws
**one** rand against it — so multiple plugins compose without
introducing independent rand draws. Default returns `1.0`.

Plugins that need cycle-progress / temperature in their multiplier should
maintain their own mutable state and update it in [`on_cycle_start!`](@ref).

!!! warning "Experimental"
"""
function mutation_acceptance_multiplier(
    _, ::AbstractPlugin, parent_member, new_tree, before_cost, after_cost, options
)
    return 1.0
end

"""
    fork_plugin_state(head_state, plugin, dataset) -> state

Build the worker-side plugin state for one cycle's dispatch, given the head
node's current plugin state for this output and the dataset the worker will
operate on. Called once per (plugin, output) pair per cycle dispatch.

Default returns `deepcopy(head_state)` (full snapshot).

!!! warning "Experimental"
"""
function fork_plugin_state(head_state, ::AbstractPlugin, dataset)
    return deepcopy(head_state)
end

"""
    init_member(state, plugin, dataset, options)

Called when initializing each population member's tree during **initial
population creation only**. Every plugin is asked; at most one may return a
non-`nothing` value, and two or more providers is an error. If all plugins
return `nothing`, the engine falls through to `gen_random_tree`.

Override by dispatching on your plugin type. Default returns `nothing`.

!!! note "State used"
    `init_member` is called with the **head node's** state instance, not a
    per-worker copy. In `:multithreading` mode, multiple population-creation
    tasks may call it concurrently — ensure your implementation is
    thread-safe or limit it to read-only access of the state.

!!! warning "Experimental"
"""
function init_member(_, ::AbstractPlugin, dataset, options)
    return nothing
end

# Ask every plugin, then require at most one provider. `map` over the
# heterogeneous tuple and `Base.tail` extraction keep this inferable.
@inline function resolve_init_member(states::Tuple, plugins::Tuple, dataset, options)
    candidates = map((s, p) -> init_member(s, p, dataset, options), states, plugins)
    count(!isnothing, candidates) > 1 && _init_member_conflict(plugins, candidates)
    return _first_non_nothing(candidates)
end

@inline _first_non_nothing(::Tuple{}) = nothing
@inline function _first_non_nothing(t::Tuple)
    return t[1] === nothing ? _first_non_nothing(Base.tail(t)) : t[1]
end

@noinline function _init_member_conflict(plugins::Tuple, candidates::Tuple)
    providers = [string(typeof(p)) for (p, c) in zip(plugins, candidates) if c !== nothing]
    throw(
        ArgumentError(
            "Plugins $(join(providers, ", ")) all returned a member from " *
            "`init_member`, but at most one plugin may provide initial members.",
        ),
    )
end

"""
    on_cycle_start!(state, plugin, cycle_idx, options)

Observer hook fired at the start of each evolution cycle (1-based
`cycle_idx`, total `options.ncycles_per_iteration` per outer iteration).
Plugins update their own mutable state here from the cycle position —
e.g. `SimulatedAnnealingPlugin` recomputes its `temperature` once per
cycle and consumes it later in `mutation_acceptance_multiplier` and
`condition_mutation!`.

Default is a no-op.

!!! warning "Experimental"
"""
function on_cycle_start!(_, ::AbstractPlugin, cycle_idx::Int, options)
    return nothing
end

"""
    prepare_mutation_context(mutation::AbstractMutation)

Build a fresh per-call mutable context for `mutation`, called inside the
function-barrier of `_next_generation` immediately **after** mutation
sampling — so only the selected mutation pays the construction cost.

Default returns `nothing` (no context — the mutation runs directly from
its immutable fields). Override for any mutation type that wants plugins
to layer per-call configuration on top of its base values.

Example:

```julia
mutable struct MyMutationContext
    perturbation_factor::Float64
end
SymbolicRegression.prepare_mutation_context(m::MyMutation) = MyMutationContext(m.perturbation_factor)
```

!!! warning "Experimental"
"""
prepare_mutation_context(::AbstractMutation) = nothing
function prepare_mutation_context(m::ConstantMutation)
    return ConstantMutationContext(m.perturbation_factor)
end

"""
    condition_mutation!(context, state, plugin, mutation, options)

In-place plugin modification of the per-call `context` produced by
[`prepare_mutation_context`](@ref). Only fires for the mutation that was
selected this cycle. Composes by sequential mutation in tuple order.

Default is a no-op.

!!! warning "Experimental"
"""
function condition_mutation!(::Any, _, ::AbstractPlugin, ::AbstractMutation, options)
    return nothing
end

"""
    wrap_mutation_step(state, plugin, parent_member, next_step) -> (member, accepted, num_evals)

Middleware-style hook around `next_generation`. The engine builds a thunk
`next_step(parent) -> (member, accepted, num_evals)` that runs one call to
`next_generation` against `parent`. The plugin may call `next_step` zero,
one, or many times — with the original parent (retry), with the previous
result (chain), with a different member (ensemble), or skip it entirely.
It returns the final tuple to its caller.

Plugins compose as nested middleware in `options.plugins` order:
plugin 1's `wrap` wraps plugin 2's wrap wraps... wraps `next_generation`.
Plugin 1 sees the composed behavior of plugins 2+ as `next_step`. Use this
shape for retry, compound bursts, MCMC-style acceptance criteria,
ensemble voting, or any control-flow extension that needs to call
`next_generation` more than once per cycle.

Default is a single pass-through: `next_step(parent_member)`.

!!! warning "Experimental"
"""
@unstable function wrap_mutation_step(
    _, ::AbstractPlugin, parent_member, next_step::F
) where {F}
    return next_step(parent_member)
end

# Forward-declared per kwarg→plugin migration. Plugin modules (loaded above
# Core) provide the only method. Each returns either a plugin instance or
# `nothing` (when the legacy kwarg is off).
function default_adaptive_parsimony_plugin end
function default_adaptive_mutation_weights_plugin end
function default_mutation_loop_plugin end
function default_simulated_annealing_plugin end


# Append defaults whose type isn't already in the user tuple. Each default
# may be `nothing` (auto-injected plugin disabled by the legacy kwarg) and
# is filtered out.
@unstable function _merge_with_default_plugins(
    @nospecialize(user_plugins::Tuple), @nospecialize(default_plugins...)
)
    actual_defaults = filter(!isnothing, default_plugins)
    novel = filter(dp -> !any(p -> p isa typeof(dp), user_plugins), actual_defaults)
    return (user_plugins..., novel...)
end

end  # module PluginModule
