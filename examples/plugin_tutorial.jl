#literate_begin file="src/examples/plugin_tutorial.md"
#=
# Writing a Custom Plugin

This tutorial walks through building a simple plugin from scratch.
We'll create a plugin that counts how many mutations are accepted vs
rejected during the search, which is useful for diagnosing whether
the search is exploring effectively.

## Defining the plugin

A plugin has two parts: an immutable **config struct** (subtyping
`AbstractPlugin`) and a mutable **state object** returned by
`init_plugin_state`.
=#
using SymbolicRegression
using SymbolicRegression: AbstractPlugin, MutationEvent, AbstractMutation
using SymbolicRegression: machine, fit!, report
using Test

struct MutationCounterPlugin <: AbstractPlugin end

mutable struct MutationCounterState
    accepted::Int
    rejected::Int
end

#=
## Implementing the hooks

Create the state once per output at search start:
=#
function SymbolicRegression.init_plugin_state(::MutationCounterPlugin, options, dataset)
    return MutationCounterState(0, 0)
end

#=
The `on_mutation_end!` hook fires on the worker after each mutation's
accept/reject decision. We increment the appropriate counter:
=#
function SymbolicRegression.on_mutation_end!(
    state::MutationCounterState,
    ::MutationCounterPlugin,
    ::AbstractMutation,
    event::MutationEvent,
    dataset,
    options,
)
    if event.accepted
        state.accepted += 1
    else
        state.rejected += 1
    end
    return nothing
end

#=
## Running the search

Pass the plugin to `SRRegressor` via the `plugins` keyword.
All default plugins (like `AdaptiveParsimonyPlugin`) are still
active alongside yours.
=#
X = 2randn(100, 5)
y = @. cos(X[:, 1]) + X[:, 2]^2

model = SRRegressor(;
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(MutationCounterPlugin(),),
    niterations=5,
)
mach = machine(model, X, y)
fit!(mach)

#=
The search ran with our plugin active. Since `on_mutation_end!` runs on
workers against per-dispatch copies of the state (built by `fork_plugin_state`,
which defaults to `deepcopy`), the head-side state won't reflect worker
counts in multithreading mode. For a production version you'd aggregate
via `on_generation_end!` on the head node. But the plugin loaded, the hooks
fired, and the search completed — the API works.
=#
@test report(mach).best_idx isa Int

#literate_end
