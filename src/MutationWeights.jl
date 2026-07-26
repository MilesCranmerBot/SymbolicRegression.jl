module MutationWeightsModule

import ..MutationsModule:
    AbstractMutation,
    MutateConstant,
    MutateOperator,
    MutateFeature,
    SwapOperands,
    AddNode,
    InsertNode,
    DeleteNode,
    FormConnection,
    BreakConnection,
    RotateTree,
    Backsolve,
    Simplify,
    Randomize,
    Optimize,
    DoNothing

using StatsBase: StatsBase

"""
    MutationWeights(;kws...)

This defines how often different mutations occur. These weightings
will be normalized to sum to 1.0 after initialization.

# Arguments

- `mutate_constant::Float64`: How often to mutate a constant.
- `mutate_operator::Float64`: How often to mutate an operator.
- `mutate_feature::Float64`: How often to mutate which feature a variable node references.
- `swap_operands::Float64`: How often to swap the operands of a binary operator.
- `rotate_tree::Float64`: How often to perform a tree rotation at a random node.
- `add_node::Float64`: How often to append a node to the tree.
- `insert_node::Float64`: How often to insert a node into the tree.
- `delete_node::Float64`: How often to delete a node from the tree.
- `simplify::Float64`: How often to simplify the tree.
- `randomize::Float64`: How often to create a random tree.
- `do_nothing::Float64`: How often to do nothing.
- `optimize::Float64`: How often to optimize the constants in the tree, as a mutation.
    Note that this is different from `optimizer_probability`, which is
    performed at the end of an iteration for all individuals.
- `backsolve::Float64`: How often to backsolve and rewrite a random subtree
    by inverting the evaluation path and fitting a replacement expression.
    **Experimental:** this mutation will change in minor version increments.
- `form_connection::Float64`: **Only used for `GraphNode`, not regular `Node`**.
    Otherwise, this will automatically be set to 0.0. How often to form a
    connection between two nodes.
- `break_connection::Float64`: **Only used for `GraphNode`, not regular `Node`**.
    Otherwise, this will automatically be set to 0.0. How often to break a
    connection between two nodes.

# See Also

- [`AbstractMutation`](@ref): Use to define custom mutation types.
"""
Base.@kwdef mutable struct MutationWeights
    mutate_constant::Float64 = 0.0353
    mutate_operator::Float64 = 3.63
    mutate_feature::Float64 = 0.1
    swap_operands::Float64 = 0.00608
    rotate_tree::Float64 = 1.42
    add_node::Float64 = 0.0771
    insert_node::Float64 = 2.44
    delete_node::Float64 = 0.369
    simplify::Float64 = 0.00148
    randomize::Float64 = 0.00695
    do_nothing::Float64 = 0.431
    optimize::Float64 = 0.0
    backsolve::Float64 = 0.0
    form_connection::Float64 = 0.5
    break_connection::Float64 = 0.1
end

const mutations = fieldnames(MutationWeights)
const v_mutations = Symbol[mutations...]

# For some reason it's much faster to write out the fields explicitly:
let contents = [Expr(:., :w, QuoteNode(field)) for field in mutations]
    @eval begin
        function Base.convert(::Type{Vector}, w::MutationWeights)::Vector{Float64}
            return $(Expr(:vect, contents...))
        end
        function Base.copy(w::MutationWeights)
            return $(Expr(:call, :MutationWeights, contents...))
        end
    end
end

const _MUTATION_FROM_SYMBOL = Dict{Symbol,AbstractMutation}(
    :mutate_constant => MutateConstant(),
    :mutate_operator => MutateOperator(),
    :mutate_feature => MutateFeature(),
    :swap_operands => SwapOperands(),
    :rotate_tree => RotateTree(),
    :add_node => AddNode(),
    :insert_node => InsertNode(),
    :delete_node => DeleteNode(),
    :simplify => Simplify(),
    :randomize => Randomize(),
    :do_nothing => DoNothing(),
    :optimize => Optimize(),
    :backsolve => Backsolve(),
    :form_connection => FormConnection(),
    :break_connection => BreakConnection(),
)

"""
    _mutations_from_weights(w) -> Vector{Pair{AbstractMutation,Float64}}

Convert built-in mutation weights to the mutation list used by `Options`.
"""
function _mutations_from_weights(w::MutationWeights)
    return Pair{AbstractMutation,Float64}[
        _MUTATION_FROM_SYMBOL[k] => Float64(getfield(w, k)) for k in fieldnames(typeof(w))
    ]
end

using DispatchDoctor: @unstable

"""
    sample_mutation(mutations) -> AbstractMutation

Pick a mutation kind by weight. Returns the singleton instance.

Marked `@unstable` because the return type is `AbstractMutation` — the
concrete subtype is selected at runtime by weighted sampling. The caller
hands the result to `mutate!`, which dispatches per concrete type, so the
instability is contained.
"""
@unstable function sample_mutation(
    mutations::AbstractVector{Pair{AbstractMutation,Float64}}
)
    idx = StatsBase.sample(eachindex(mutations), StatsBase.Weights(map(last, mutations)))
    return mutations[idx].first
end

end
