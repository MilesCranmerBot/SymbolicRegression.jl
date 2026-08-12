module MutateModule

using DispatchDoctor: @unstable
using DynamicExpressions:
    AbstractExpression,
    copy_into!,
    get_tree,
    preserve_sharing,
    count_constant_nodes,
    count_scalar_constants,
    simplify_tree!,
    combine_operators,
    allocate_container
using ..CoreModule:
    AbstractOptions,
    AbstractMutation,
    ConstantMutation,
    OperatorMutation,
    FeatureMutation,
    SwapOperandsMutation,
    AddNodeMutation,
    InsertNodeMutation,
    DeleteNodeMutation,
    FormConnectionMutation,
    BreakConnectionMutation,
    RotateTreeMutation,
    BacksolveMutation,
    SimplifyMutation,
    RandomizeMutation,
    OptimizeMutation,
    DoNothingMutation,
    BUILTIN_MUTATION_TYPES,
    Dataset,
    MaybeTrace,
    max_features,
    dataset_fraction,
    AbstractPlugin,
    MutationEvent,
    on_mutation_end!,
    mutation_acceptance_multiplier,
    MutationAcceptanceContext,
    ConstantMutationContext,
    prepare_mutation_context,
    condition_mutation!
using ..ComplexityModule: compute_complexity
using ..LossFunctionsModule: eval_cost, loss_to_cost
using ..CheckConstraintsModule: check_constraints
using ..PopMemberModule: AbstractPopMember, PopMember, create_child
using ..UtilsModule: strictmap
using ..MutationFunctionsModule:
    mutate_constant,
    mutate_operator,
    mutate_feature,
    swap_operands,
    append_random_op,
    prepend_random_op,
    insert_random_op,
    delete_random_op!,
    form_random_connection!,
    break_random_connection!,
    randomly_rotate_tree!,
    randomize_tree,
    backsolve_rewrite_random_node
using ..ConstantOptimizationModule: optimize_constants
using ..TracingModule:
    trace_identity_mutation!, trace_mutation_result!, trace_mutation_type!

abstract type AbstractMutationResult{N<:AbstractExpression,P<:AbstractPopMember} end

"""
    MutationResult{N<:AbstractExpression,P<:AbstractPopMember}

Represents the result of a mutation operation in the genetic programming algorithm. This struct is used to return values from `mutate!` functions.

# Fields

- `tree::Union{N, Nothing}`: The mutated expression tree, if applicable. Either `tree` or `member` must be set, but not both.
- `member::Union{P, Nothing}`: The mutated population member, if applicable. Either `member` or `tree` must be set, but not both.
- `num_evals::Float64`: The number of evaluations performed during the mutation, which is automatically set to `0.0`. Only used for things like `optimize`.
- `return_immediately::Bool`: If `true`, the mutation process should return immediately, bypassing further checks, used for things like `simplify` or `optimize` where you already know the loss value of the result.

# Usage

This struct encapsulates the result of a mutation operation. Either a new expression tree or a new population member is returned, but not both.

Return the `member` if you want to return immediately, and have
computed the loss value as part of the mutation.
"""
struct MutationResult{N<:AbstractExpression,P<:AbstractPopMember} <:
       AbstractMutationResult{N,P}
    tree::Union{N,Nothing}
    member::Union{P,Nothing}
    num_evals::Float64
    return_immediately::Bool

    # Explicit constructor with keyword arguments
    function MutationResult{_N,_P}(;
        tree::Union{_N,Nothing}=nothing,
        member::Union{_P,Nothing}=nothing,
        num_evals::Float64=0.0,
        return_immediately::Bool=false,
    ) where {_N<:AbstractExpression,_P<:AbstractPopMember}
        @assert(
            (tree === nothing) ⊻ (member === nothing),
            "Mutation result must return either a tree or a pop member, not both"
        )
        return new{_N,_P}(tree, member, num_evals, return_immediately)
    end
end

"""
    _update_weight!(op, weights, ::Type{M})

For every entry whose mutation is `<: M`, replace its weight with `op(weight)`.
Core helper used by `condition_mutation_weights!` plugin authors via the
`_set_weight!` / `_scale_weight!` wrappers below.
"""
function _update_weight!(
    op::F, weights::AbstractVector, ::Type{M}
) where {F,M<:AbstractMutation}
    for i in eachindex(weights)
        m, w = weights[i]
        if m isa M
            weights[i] = m => op(w)
        end
    end
    return nothing
end

"""Set the weight of every `<: M` entry to `value` (in place)."""
function _set_weight!(
    weights::AbstractVector, ::Type{M}, value::Real
) where {M<:AbstractMutation}
    return _update_weight!(Returns(Float64(value)), weights, M)
end

"""Multiply the weight of every `<: M` entry by `factor` (in place)."""
function _scale_weight!(
    weights::AbstractVector, ::Type{M}, factor::Real
) where {M<:AbstractMutation}
    return _update_weight!(w -> w * Float64(factor), weights, M)
end

"""
    condition_mutation_weights!(weights, member::AbstractPopMember, options, curmaxsize, nfeatures)

Adjust the mutation `weights` (a `Vector{Pair{AbstractMutation,Float64}}`)
based on the properties of the current member and options — e.g. disable
operator-mutation when the tree has no operators, disable simplify when
`options.should_simplify` is false, etc.

Plugin overloads should use `_set_weight!(weights, MyMutation, w)` to
modify the per-mutation weight in place.
"""
function condition_mutation_weights!(
    weights::AbstractVector,
    member::P,
    options::AbstractOptions,
    curmaxsize::Int,
    nfeatures::Int,
) where {T,L,N<:AbstractExpression,P<:AbstractPopMember{T,L,N}}
    tree = get_tree(member.tree)
    if !preserve_sharing(typeof(member.tree))
        _set_weight!(weights, FormConnectionMutation, 0.0)
        _set_weight!(weights, BreakConnectionMutation, 0.0)
    end
    if tree.degree == 0
        _set_weight!(weights, OperatorMutation, 0.0)
        _set_weight!(weights, SwapOperandsMutation, 0.0)
        _set_weight!(weights, DeleteNodeMutation, 0.0)
        _set_weight!(weights, SimplifyMutation, 0.0)
        if !tree.constant
            _set_weight!(weights, OptimizeMutation, 0.0)
            _set_weight!(weights, ConstantMutation, 0.0)
        else
            _set_weight!(weights, FeatureMutation, 0.0)
        end
        return nothing
    end

    if !any(node -> node.degree == 2, tree)
        _set_weight!(weights, SwapOperandsMutation, 0.0)
    end

    condition_mutate_constant!(typeof(member.tree), weights, member, options, curmaxsize)

    if nfeatures <= 1
        _set_weight!(weights, FeatureMutation, 0.0)
    end

    complexity = compute_complexity(member, options)
    if complexity >= curmaxsize
        _set_weight!(weights, AddNodeMutation, 0.0)
        _set_weight!(weights, InsertNodeMutation, 0.0)
    end

    if !options.should_simplify
        _set_weight!(weights, SimplifyMutation, 0.0)
    end

    return nothing
end

"""
    condition_mutation_weights!(weights, state, plugin, member, options, curmaxsize, nfeatures)

Plugin-dispatched method: called once per plugin in tuple order after the
engine's legality conditioning. Default is a no-op.

!!! warning "Experimental"
"""
function condition_mutation_weights!(
    weights::AbstractVector, _, ::AbstractPlugin, member, options, curmaxsize, nfeatures
)
    return nothing
end

"""
Use this to modify how `mutate_constant` changes for an expression type.
"""
function condition_mutate_constant!(
    ::Type{<:AbstractExpression},
    weights::AbstractVector,
    member::AbstractPopMember,
    options::AbstractOptions,
    curmaxsize::Int,
)
    n_constants = max(
        count_constant_nodes(member.tree), count_scalar_constants(member.tree)
    )
    _scale_weight!(weights, ConstantMutation, min(8, n_constants) / 8.0)
    return nothing
end

@unstable function _sample_mutation(mutations::AbstractVector{<:Pair{<:Any,<:Real}})
    total_weight = 0.0
    for (_, weight) in mutations
        weight >= 0.0 || throw(ArgumentError("Weights must be nonnegative."))
        total_weight += weight
    end
    total_weight > 0.0 || throw(ArgumentError("At least one weight must be positive."))

    threshold = rand() * total_weight
    cumulative_weight = 0.0
    for (i, (_, weight)) in enumerate(mutations)
        cumulative_weight += weight
        threshold < cumulative_weight && return i
    end
    return lastindex(mutations)
end

# Go through one simulated options.annealing mutation cycle
@inline function _fire_on_mutation_end!(
    options::AbstractOptions,
    plugin_states::Tuple,
    mutation::AbstractMutation,
    event::MutationEvent,
    dataset,
)
    strictmap(options.plugins, plugin_states) do plugin, pstate
        on_mutation_end!(pstate, plugin, mutation, event, dataset, options)
    end
    return nothing
end

let mutation_types = BUILTIN_MUTATION_TYPES
    @eval @inline function _dispatch_next_generation(mutation::AbstractMutation, args...)
        Base.Cartesian.@nif(
            $(length(mutation_types) + 1),
            i -> mutation isa $(mutation_types)[i],  # COV_EXCL_LINE
            i -> _next_generation(mutation::$(mutation_types)[i], args...),  # COV_EXCL_LINE
            i -> _next_generation(mutation, args...),  # COV_EXCL_LINE
        )
    end
end

@unstable function next_generation(
    dataset::D,
    member::P,
    curmaxsize::Int,
    options::AbstractOptions;
    tmp_trace::MaybeTrace,
    plugin_states::Tuple,
    eval_context=nothing,
    population_for_backsolve=nothing,
)::Tuple{
    P,Bool,Float64
} where {T,L,D<:Dataset{T,L},N<:AbstractExpression{T},P<:AbstractPopMember{T,L,N}}
    parent_ref = member.ref
    num_evals = 0.0

    #TODO - reconsider this
    before_cost, before_loss = member.cost, member.loss

    nfeatures = max_features(dataset, options)

    weights = copy(options.mutations)

    condition_mutation_weights!(weights, member, options, curmaxsize, nfeatures)
    strictmap(options.plugins, plugin_states) do plugin, pstate
        condition_mutation_weights!(
            weights, pstate, plugin, member, options, curmaxsize, nfeatures
        )
    end

    mutation_idx = _sample_mutation(weights)
    mutation_choice = weights[mutation_idx].first

    # Preserve concrete mutation dispatch through the hot path.
    return _dispatch_next_generation(
        mutation_choice,
        mutation_idx,
        dataset,
        member,
        curmaxsize,
        nfeatures,
        before_cost,
        before_loss,
        parent_ref,
        options,
        tmp_trace,
        plugin_states,
        eval_context,
        population_for_backsolve,
        num_evals,
    )
end

function _next_generation(
    mutation_choice::M,
    mutation_idx::Int,
    dataset::D,
    member::P,
    curmaxsize::Int,
    nfeatures::Int,
    before_cost,
    before_loss,
    parent_ref,
    options::AbstractOptions,
    tmp_trace::MaybeTrace,
    plugin_states::Tuple,
    eval_context,
    population_for_backsolve,
    num_evals::Float64,
)::Tuple{
    P,Bool,Float64
} where {
    T,
    L,
    D<:Dataset{T,L},
    N<:AbstractExpression{T},
    P<:AbstractPopMember{T,L,N},
    M<:AbstractMutation,
}
    successful_mutation = false
    attempts = 0
    max_attempts = 10
    node_storage = allocate_container(member.tree)

    mut_context = prepare_mutation_context(mutation_choice)
    if !isnothing(mut_context)
        strictmap(options.plugins, plugin_states) do plugin, pstate
            return condition_mutation!(
                mut_context, pstate, plugin, mutation_choice, options
            )
        end
    end

    #############################################
    # Mutations
    #############################################
    # local tree
    rtree = Ref{N}()
    while (!successful_mutation) && attempts < max_attempts
        rtree[] = copy_into!(node_storage, member.tree)

        mutation_result = mutate!(
            rtree[],
            member,
            mutation_choice,
            options;
            trace=tmp_trace,
            context=mut_context,
            attempt=attempts + 1,
            dataset,
            cost=before_cost,
            loss=before_loss,
            parent_ref,
            curmaxsize,
            nfeatures,
            plugin_states,
            population_for_backsolve,
        )
        mutation_result::AbstractMutationResult{N,P}
        num_evals += mutation_result.num_evals::Float64

        if mutation_result.return_immediately
            @assert(
                mutation_result.member isa P,
                "Mutation result must return a `PopMember` if `return_immediately` is true"
            )
            _fire_on_mutation_end!(
                options,
                plugin_states,
                mutation_choice,
                MutationEvent(
                    true,
                    before_cost,
                    mutation_result.member.cost,
                    before_loss,
                    mutation_result.member.loss,
                    mutation_idx,
                ),
                dataset,
            )
            return mutation_result.member::P, true, num_evals
        else
            @assert(
                mutation_result.tree isa N,
                "Mutation result must return a tree if `return_immediately` is false"
            )
            rtree[] = mutation_result.tree::N
            successful_mutation = check_constraints(rtree[], options, curmaxsize)
            attempts += 1
        end
    end

    tree = rtree[]

    if !successful_mutation
        trace_mutation_result!(tmp_trace, "reject", "failed_constraint_check")
        mutation_accepted = false
        _fire_on_mutation_end!(
            options,
            plugin_states,
            mutation_choice,
            MutationEvent(false, before_cost, nothing, before_loss, nothing, mutation_idx),
            dataset,
        )
        return (
            create_child(
                member,
                copy_into!(node_storage, member.tree),
                before_cost,
                before_loss,
                options;
                parent_ref=parent_ref,
                mutation_choice=mutation_choice,
            ),
            mutation_accepted,
            num_evals,
        )
    end

    after_cost, after_loss = eval_cost(dataset, tree, options; eval_context)
    num_evals += dataset_fraction(dataset)

    if isnan(after_cost)
        trace_mutation_result!(tmp_trace, "reject", "nan_loss")
        mutation_accepted = false
        _fire_on_mutation_end!(
            options,
            plugin_states,
            mutation_choice,
            MutationEvent(false, before_cost, nothing, before_loss, nothing, mutation_idx),
            dataset,
        )
        return (
            create_child(
                member,
                copy_into!(node_storage, member.tree),
                before_cost,
                before_loss,
                options;
                parent_ref=parent_ref,
                mutation_choice=mutation_choice,
            ),
            mutation_accepted,
            num_evals,
        )
    end

    acceptance_ctx = MutationAcceptanceContext(member, tree, before_cost, after_cost)
    probChange = prod(
        strictmap(options.plugins, plugin_states) do plugin, pstate
            return mutation_acceptance_multiplier(pstate, plugin, acceptance_ctx, options)
        end,
    )

    if probChange < rand()
        trace_mutation_result!(tmp_trace, "reject", "acceptance")
        _fire_on_mutation_end!(
            options,
            plugin_states,
            mutation_choice,
            MutationEvent(
                false, before_cost, after_cost, before_loss, after_loss, mutation_idx
            ),
            dataset,
        )
        return (
            create_child(
                member,
                copy_into!(node_storage, member.tree),
                before_cost,
                before_loss,
                options;
                parent_ref=parent_ref,
            ),
            false,
            num_evals,
        )
    end

    trace_mutation_result!(tmp_trace, "accept", "pass")
    new_member = create_child(
        member, tree, after_cost, after_loss, options; parent_ref=parent_ref
    )
    _fire_on_mutation_end!(
        options,
        plugin_states,
        mutation_choice,
        MutationEvent(true, before_cost, after_cost, before_loss, after_loss, mutation_idx),
        dataset,
    )
    return (new_member, true, num_evals)
end

"""
    mutate!(
        new_tree::N,
        parent_member::P,
        mutation::AbstractMutation,
        options::AbstractOptions;
        kws...,
    ) where {N<:AbstractExpression,P<:AbstractPopMember}

Perform `mutation` on the offspring `new_tree` (a fresh scratch copy of the
parent's tree). `parent_member` carries parent metadata (cost, loss, ref, etc.)
that some mutations need.

Add a new mutation by defining a struct subtyping
[`AbstractMutation`](@ref) and a matching `mutate!` method.

# Keywords

- `dataset::Dataset`: The dataset used for scoring.
- `cost`: The cost of `parent_member` before mutation.
- `loss`: The loss of `parent_member` before mutation.
- `curmaxsize`: The current maximum size constraint, which may differ from `options.maxsize`.
- `nfeatures`: The number of features in the dataset.
- `parent_ref`: Reference to `parent_member`'s parent (used for lineage logging).
- `attempt::Int`: 1-based attempt number within the engine's constraint-retry
  loop. Expensive mutations can use this to behave differently on retries.
- `trace::MaybeTrace`: Mutation tracing state, or `nothing` when tracing is disabled.
- `context`: per-call mutable context for the selected mutation type (built by
  [`prepare_mutation_context`](@ref) and conditioned by plugins via
  [`condition_mutation!`](@ref)); `nothing` for mutations without one.
- `plugin_states::Tuple`: The active worker plugin states, in tuple order matching
  `options.plugins`.

# Returns

A `MutationResult{N,P}` object containing the mutated tree or member (but not both),
the number of evaluations performed, if any, and whether to return immediately from
the mutation function, or to let the `next_generation` function handle accepting or
rejecting the mutation. For example, a `simplify` operation will not change the loss,
so it can always return immediately.
"""
function mutate!(new_tree, parent_member, m::AbstractMutation, options; kws...)
    return error("Unknown mutation type: $(typeof(m))")
end

function mutate!(
    new_tree::N,
    parent_member::P,
    m::ConstantMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    context::Union{Nothing,ConstantMutationContext}=nothing,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    scale = isnothing(context) ? 1.0 : context.scale
    new_tree = mutate_constant(new_tree, scale, options, m)
    trace_mutation_type!(trace, "mutate_constant")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::OperatorMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = mutate_operator(new_tree, options)
    trace_mutation_type!(trace, "mutate_operator")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::FeatureMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    nfeatures,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = mutate_feature(new_tree, nfeatures)
    trace_mutation_type!(trace, "mutate_feature")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::SwapOperandsMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = swap_operands(new_tree)
    trace_mutation_type!(trace, "swap_operands")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::AddNodeMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    nfeatures,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    if rand() < 0.5
        new_tree = append_random_op(new_tree, options, nfeatures)
        trace_mutation_type!(trace, "add_node:append")
    else
        new_tree = prepend_random_op(new_tree, options, nfeatures)
        trace_mutation_type!(trace, "add_node:prepend")
    end
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::InsertNodeMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    nfeatures,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = insert_random_op(new_tree, options, nfeatures)
    trace_mutation_type!(trace, "insert_node")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::DeleteNodeMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = delete_random_op!(new_tree)
    trace_mutation_type!(trace, "delete_node")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::FormConnectionMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = form_random_connection!(new_tree)
    trace_mutation_type!(trace, "form_connection")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::BreakConnectionMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = break_random_connection!(new_tree)
    trace_mutation_type!(trace, "break_connection")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::RotateTreeMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = randomly_rotate_tree!(new_tree)
    trace_mutation_type!(trace, "rotate_tree")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    m::BacksolveMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    dataset::Dataset,
    population_for_backsolve=nothing,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    new_tree = backsolve_rewrite_random_node(
        new_tree,
        dataset,
        options;
        backsolve_options=m,
        population_for_backsolve=population_for_backsolve,
    )
    trace_mutation_type!(trace, "backsolve")
    return MutationResult{N,P}(; tree=new_tree)
end

# Handle mutations that require early return
function mutate!(
    new_tree::N,
    parent_member::P,
    ::SimplifyMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    dataset::Dataset,
    parent_ref,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    @assert options.should_simplify
    simplify_tree!(new_tree, options.operators)
    new_tree = combine_operators(new_tree, options.operators)
    simplified_complexity = compute_complexity(new_tree, options)
    simplified_cost = loss_to_cost(
        parent_member.loss,
        dataset.use_baseline,
        dataset.baseline_loss,
        new_tree,
        options,
        simplified_complexity,
    )
    trace_mutation_type!(trace, "simplify")
    new_member = create_child(
        parent_member,
        new_tree,
        simplified_cost,
        parent_member.loss,
        options;
        complexity=simplified_complexity,
        parent_ref=parent_ref,
    )
    return MutationResult{N,P}(; member=new_member, return_immediately=true)
end

function mutate!(
    new_tree::N,
    ::P,
    ::RandomizeMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    curmaxsize,
    nfeatures,
    kws...,
) where {T,N<:AbstractExpression{T},P<:AbstractPopMember}
    new_tree = randomize_tree(new_tree, curmaxsize, options, nfeatures)
    trace_mutation_type!(trace, "randomize")
    return MutationResult{N,P}(; tree=new_tree)
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::OptimizeMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    dataset::Dataset,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    cur_member, new_num_evals = optimize_constants(dataset, parent_member, options)
    trace_mutation_type!(trace, "optimize")
    return MutationResult{N,P}(;
        member=cur_member, num_evals=new_num_evals, return_immediately=true
    )
end

function mutate!(
    new_tree::N,
    parent_member::P,
    ::DoNothingMutation,
    options::AbstractOptions;
    trace::MaybeTrace,
    parent_ref,
    kws...,
) where {N<:AbstractExpression,P<:AbstractPopMember}
    trace_identity_mutation!(trace)
    return MutationResult{N,P}(;
        member=create_child(
            parent_member,
            new_tree,
            parent_member.cost,
            parent_member.loss,
            options;
            parent_ref=parent_ref,
        ),
        return_immediately=true,
    )
end

end  # module MutateModule
