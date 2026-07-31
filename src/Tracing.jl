module TracingModule

using DynamicExpressions: string_tree
using ..CoreModule: AbstractOptions, MaybeTrace, TraceType
using ..ComplexityModule: compute_complexity
using ..UtilsModule: json3_write, recursive_merge

@inline new_trace(options::AbstractOptions) = new_trace(options.use_tracing)
@inline new_trace(::Val{false}) = nothing
@inline new_trace(::Val{true}) = TraceType()
@inline new_trace(::Nothing) = nothing
@inline new_trace(::TraceType) = TraceType()

@inline new_step_trace(::Nothing) = nothing
@inline new_step_trace(steps) = TraceType()

@inline new_traced_steps(::Nothing, ::Type) = nothing
@inline function new_traced_steps(::TraceType, ::Type{P}) where {P}
    return Tuple{P,P,TraceType}[]
end

@inline trace_mutation_step!(::Nothing, parent, member, trace) = nothing
@inline function trace_mutation_step!(
    steps::AbstractVector, parent, member, trace::TraceType
)
    push!(steps, (copy(parent), copy(member), trace))
    return nothing
end

@inline reset_traced_steps!(::Nothing) = nothing
@inline reset_traced_steps!(steps) = empty!(steps)

@inline trace_mutation_type!(::Nothing, type) = nothing
@inline function trace_mutation_type!(trace::TraceType, type)
    trace["type"] = type
    return nothing
end

@inline trace_mutation_result!(::Nothing, result, reason) = nothing
@inline function trace_mutation_result!(trace::TraceType, result, reason)
    trace["result"] = result
    trace["reason"] = reason
    return nothing
end

@inline trace_identity_mutation!(::Nothing) = nothing
@inline function trace_identity_mutation!(trace::TraceType)
    trace["type"] = "identity"
    trace["result"] = "accept"
    trace["reason"] = "identity"
    return nothing
end

@inline trace_search_options!(::Nothing, options) = nothing
function trace_search_options!(trace::TraceType, options)
    trace["options"] = string(options)
    return nothing
end

@inline trace_worker!(::Nothing, out, pop) = nothing
function trace_worker!(trace::TraceType, out, pop)
    trace["out$(out)_pop$(pop)"] = TraceType()
    return nothing
end

@inline merge_traces(::Nothing, incoming) = nothing
@inline merge_traces(trace::TraceType, incoming::TraceType) = recursive_merge(
    trace, incoming
)

@inline write_trace(::Nothing, filename) = nothing
function write_trace(trace::TraceType, filename)
    json3_write(trace, filename)
    return nothing
end

@inline next_trace_iteration(::Nothing, out, pop) = 0
function next_trace_iteration(trace::TraceType, out, pop)
    key = "out$(out)_pop$(pop)"
    return find_iteration_from_trace(key, trace) + 1
end

function find_iteration_from_trace(key::String, trace::TraceType)
    iteration = 0
    while haskey(trace[key], "iteration$(iteration)")
        iteration += 1
    end
    return iteration - 1
end

@inline trace_iteration_start!(::Nothing, out, pop, iteration, population, options) =
    nothing
function trace_iteration_start!(trace::TraceType, out, pop, iteration, population, options)
    trace["out$(out)_pop$(pop)"] = TraceType(
        "iteration$(iteration)" => _trace_population(population, options)
    )
    return nothing
end

function _trace_population(population, options)
    return TraceType(
        "population" => [
            TraceType(
                "tree" => string_tree(member.tree, options; pretty=false),
                "loss" => member.loss,
                "cost" => member.cost,
                "complexity" => compute_complexity(member, options),
                "birth" => member.birth,
                "ref" => member.ref,
                "parent" => member.parent,
            ) for member in population.members
        ],
        "time" => time(),
    )
end

function _trace_member!(mutations::TraceType, member, options)
    key = string(member.ref)
    if !haskey(mutations, key)
        mutations[key] = TraceType(
            "events" => Vector{TraceType}(),
            "tree" => string_tree(member.tree, options),
            "cost" => member.cost,
            "loss" => member.loss,
            "parent" => member.parent,
        )
    end
    return nothing
end

@inline trace_optimization!(::Nothing, member, old_ref, new_ref, optimized, options) =
    nothing
function trace_optimization!(trace::TraceType, member, old_ref, new_ref, optimized, options)
    @assert haskey(trace, "mutations")
    mutations = trace["mutations"]::TraceType
    _trace_member!(mutations, member, options)

    mutation_type = if optimized && options.should_optimize_constants
        "simplification_and_optimization"
    else
        "simplification"
    end
    tuning_event = TraceType(
        "type" => "tuning",
        "time" => time(),
        "child" => new_ref,
        "mutation" => TraceType("type" => mutation_type),
    )
    death_event = TraceType("type" => "death", "time" => time())
    push!(mutations[string(old_ref)]["events"], tuning_event, death_event)
    return nothing
end

@inline function trace_mutation_attempts!(
    ::Nothing, steps, population, oldest, should_replace, selected_attempt_idx, options
)
    return nothing
end
function trace_mutation_attempts!(
    trace::TraceType,
    steps,
    population,
    oldest,
    should_replace,
    selected_attempt_idx,
    options,
)
    mutations = get!(TraceType, trace, "mutations")
    members = should_replace ? [population.members[oldest]] : eltype(population.members)[]
    for (parent, child, _) in steps
        push!(members, parent, child)
    end
    for member in members
        _trace_member!(mutations, member, options)
    end
    for (attempt_idx, (parent, child, step_trace)) in enumerate(steps)
        mutation_event = TraceType(
            "type" => "mutate",
            "time" => time(),
            "child" => child.ref,
            "selected" => attempt_idx == selected_attempt_idx,
            "mutation" => step_trace,
        )
        push!(mutations[string(parent.ref)]["events"], mutation_event)
    end
    if should_replace
        death_event = TraceType("type" => "death", "time" => time())
        push!(mutations[string(population.members[oldest].ref)]["events"], death_event)
    end
    return nothing
end

@inline function trace_crossover!(
    ::Nothing,
    parent1,
    parent2,
    child1,
    child2,
    population,
    oldest1,
    oldest2,
    crossover_trace,
    options,
)
    return nothing
end
function trace_crossover!(
    trace::TraceType,
    parent1,
    parent2,
    child1,
    child2,
    population,
    oldest1,
    oldest2,
    crossover_trace::MaybeTrace,
    options,
)
    @assert crossover_trace isa TraceType
    mutations = get!(TraceType, trace, "mutations")
    for member in (
        parent1,
        parent2,
        child1,
        child2,
        population.members[oldest1],
        population.members[oldest2],
    )
        _trace_member!(mutations, member, options)
    end

    crossover_event = TraceType(
        "type" => "crossover",
        "time" => time(),
        "parent1" => parent1.ref,
        "parent2" => parent2.ref,
        "child1" => child1.ref,
        "child2" => child2.ref,
        "details" => crossover_trace,
    )
    death_event1 = TraceType("type" => "death", "time" => time())
    death_event2 = TraceType("type" => "death", "time" => time())
    push!(mutations[string(parent1.ref)]["events"], crossover_event)
    push!(mutations[string(parent2.ref)]["events"], crossover_event)
    push!(mutations[string(population.members[oldest1].ref)]["events"], death_event1)
    push!(mutations[string(population.members[oldest2].ref)]["events"], death_event2)
    return nothing
end

end
