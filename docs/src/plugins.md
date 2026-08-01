# Plugins

Plugins let you hook into the search loop without modifying SymbolicRegression.jl itself.
A plugin is a small struct that opts into lifecycle hooks: observing mutations,
biasing selection, injecting initial population members, or tracking statistics
across generations.

## Using a plugin

Pass plugin instances to `Options` via the `plugins` keyword:

```julia
using SymbolicRegression

options = Options(;
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(AdaptiveParsimonyPlugin(; tournament=true, mutation_acceptance=true),),
)
```

Multiple plugins compose. The engine iterates the tuple at each lifecycle point
and dispatches the appropriate hook on each plugin type.

[`AdaptiveParsimonyPlugin`](@ref) ships with the package and is enabled by
default. It biases tournament selection and mutation acceptance away from
over-represented complexities, using a sliding window of recent equation
frequencies.

## Writing a custom plugin

Define a struct that subtypes [`AbstractPlugin`](@ref), then override whichever
hooks you need. The struct holds immutable configuration; mutable runtime state
lives in a separate object returned by [`init_plugin_state`](@ref).

Here is a plugin that tracks mutation acceptance rates per complexity level,
so you can see which expression sizes the search is exploring:

```julia
using SymbolicRegression
using SymbolicRegression: AbstractPlugin, MutationEvent, AbstractMutation
using SymbolicRegression: compute_complexity

struct MutationTrackerPlugin <: AbstractPlugin end

mutable struct MutationTrackerState
    accepted::Dict{Int,Int}
    rejected::Dict{Int,Int}
end
```

Create the state object once per output at search start:

```julia
function SymbolicRegression.init_plugin_state(::MutationTrackerPlugin, options, dataset)
    return MutationTrackerState(Dict{Int,Int}(), Dict{Int,Int}())
end
```

Track each mutation's accept/reject decision, keyed by the parent expression's
complexity:

```julia
function SymbolicRegression.on_mutation_end!(
    state::MutationTrackerState,
    ::MutationTrackerPlugin,
    ::AbstractMutation,
    event::MutationEvent,
    dataset,
    options,
)
    complexity = round(Int, event.before_loss)  # simplified; real use would key on member complexity
    if event.accepted
        state.accepted[complexity] = get(state.accepted, complexity, 0) + 1
    else
        state.rejected[complexity] = get(state.rejected, complexity, 0) + 1
    end
    return nothing
end
```

The default `fork_plugin_state` uses `deepcopy` to snapshot state for each
worker dispatch. If your state is large or contains non-serializable fields,
override it:

```julia
function SymbolicRegression.fork_plugin_state(
    head_state::MutationTrackerState, ::MutationTrackerPlugin, dataset
)
    return MutationTrackerState(copy(head_state.accepted), copy(head_state.rejected))
end
```

Pass it to the search:

```julia
X = 2randn(100, 5)
y = @. cos(X[:, 1]) + X[:, 2]^2

model = SRRegressor(
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(MutationTrackerPlugin(),),
    niterations=10,
)
mach = machine(model, X, y)
fit!(mach)
```

## Lifecycle hooks

Every hook dispatches on your plugin type. Default implementations are no-ops
(or return `1.0` for multipliers, `nothing` for factories). Override only what
you need.

Hooks fall into four categories:

| Category    | Name shape                 | Contract                                            |
| ----------- | -------------------------- | --------------------------------------------------- |
| Observer    | `on_X_start!`, `on_X_end!` | Engine fires, plugin reacts. Return value ignored.  |
| Multiplier  | `X_multiplier`             | Returns a `Real`. Plugins compose multiplicatively. |
| Conditioner | `condition_X!`             | Mutates a passed struct in place.                   |
| Factory     | `init_X`                   | Called once per (plugin, output) at startup.        |

### Initialization and teardown

```@docs
init_plugin_state
fork_plugin_state
refresh_worker_plugin_state
on_search_start!
on_search_end!
```

### Per-generation and per-cycle

```@docs
on_generation_end!
on_cycle_start!
on_cycle_end!
on_mutation_end!
MutationEvent
```

### Selection and acceptance biases

```@docs
tournament_cost_multiplier
mutation_acceptance_multiplier
```

### Mutation conditioning

```@docs
prepare_mutation_context
condition_mutation!
```

[`condition_mutation_weights!`](@ref) is a related hook that modifies the
mutation weight vector before sampling. See the [Customization](customization.md)
page for its full docstring.

### Population seeding

```@docs
init_member
```

## Working with expressions in plugins

Several hooks receive a `member` (a `PopMember`) or `new_tree` (an `Expression`).
To inspect the actual expression tree, use `get_contents(member.tree)` to get the
raw `Node`, then walk it. Each `Node` has fields `degree` (0 = leaf, 1 = unary,
2 = binary), `op` (index into the operator list), `feature` (for variable leaves),
`l` (left child), and `r` (right child). See the [Types](types.md) page for the
full `Node` API.

Here is a plugin that penalizes expressions containing division during
tournament selection:

```julia
using SymbolicRegression
using SymbolicRegression: AbstractPlugin, AbstractPopMember, AbstractOptions
using DynamicExpressions: get_contents

struct NoDivisionPlugin <: AbstractPlugin
    penalty::Float64
end
NoDivisionPlugin() = NoDivisionPlugin(10.0)

function has_op(node, op_idx::Int)
    node.degree == 0 && return false
    (node.degree == 2 && node.op == op_idx) && return true
    if node.degree >= 1 && has_op(node.l, op_idx)
        return true
    end
    return node.degree == 2 && has_op(node.r, op_idx)
end

function SymbolicRegression.tournament_cost_multiplier(
    state, p::NoDivisionPlugin, member::AbstractPopMember, options::AbstractOptions
)
    div_idx = findfirst(op -> op === (/), options.operators.binops)
    div_idx === nothing && return 1.0
    tree = get_contents(member.tree)
    return has_op(tree, div_idx) ? p.penalty : 1.0
end
```

This multiplies the tournament cost by 10x for any expression using `/`,
making the search strongly prefer division-free expressions. The plugin
struct is plain configuration, so it passes straight to `Options`:

```julia
model = SRRegressor(
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    plugins=(NoDivisionPlugin(),),
)
```

## Thread and process safety

- `on_generation_end!` runs serially on the head node. Safe to mutate state.
- `on_cycle_end!` and `on_mutation_end!` run on workers against per-dispatch
  copies built by `fork_plugin_state`. Cross-worker communication requires
  `Channel` / `RemoteChannel`.
- `init_member` reads head-node state. In multithreading mode, multiple
  population-creation tasks may call it concurrently, so keep it read-only
  or thread-safe.

## Dispatching on mutation type

`on_mutation_end!` receives the mutation as a typed argument. You can write
specific methods for individual mutation types:

```julia
function SymbolicRegression.on_mutation_end!(
    state::MyState,
    ::MyPlugin,
    ::ConstantMutation,
    event::MutationEvent,
    dataset,
    options,
)
    # handle constant mutations specifically
end
```

Available mutation types: `ConstantMutation`, `OperatorMutation`,
`FeatureMutation`, `SwapOperandsMutation`, `AddNodeMutation`,
`InsertNodeMutation`, `DeleteNodeMutation`, `FormConnectionMutation`,
`BreakConnectionMutation`, `RotateTreeMutation`, `BacksolveMutation`,
`SimplifyMutation`, `RandomizeMutation`, `OptimizeMutation`,
`DoNothingMutation`.

## Built-in plugins

```@docs
AdaptiveParsimonyPlugin
AdaptiveMutationWeightsPlugin
SimulatedAnnealingPlugin
MutationBurstPlugin
```

## Abstract type

```@docs
AbstractPlugin
```
