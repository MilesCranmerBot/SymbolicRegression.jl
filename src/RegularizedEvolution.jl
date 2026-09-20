module RegularizedEvolutionModule

using DynamicExpressions: Expression, Node, get_tree
using ..PopMemberModule: PopMember
using ..CoreModule:
    BUILTIN_MUTATION_TYPES,
    SubtreeCrossover,
    default_adaptive_parsimony_plugin,
    default_adaptive_mutation_weights_plugin,
    default_simulated_annealing_plugin

# Only these built-in hooks leave population members and their trees unaliased.
function can_recycle(pop, options)
    member = first(pop.members)
    member isa PopMember && member.tree isa Expression && get_tree(member.tree) isa Node ||
        return false
    isnothing(options.loss_function) && isnothing(options.loss_function_expression) ||
        return false
    options.complexity_mapping isa Function && return false
    all(pair -> any(T -> pair.first isa T, BUILTIN_MUTATION_TYPES), options.mutations) ||
        return false
    all(pair -> pair.first isa SubtreeCrossover, options.crossovers) || return false
    parsimony = default_adaptive_parsimony_plugin(; use_frequency=true, use_frequency_in_tournament=true)
    weights = default_adaptive_mutation_weights_plugin()
    annealing = default_simulated_annealing_plugin(; annealing=true, alpha=0.1)
    return all(options.plugins) do plugin
        plugin isa typeof(parsimony) || plugin isa typeof(weights) || plugin isa typeof(annealing)
    end
end
using ..CoreModule:
    AbstractOptions,
    Dataset,
    MaybeTrace,
    DATA_TYPE,
    LOSS_TYPE,
    MutationStepResult,
    wrap_mutation_step
using ..PopulationModule: Population, best_of_sample
using ..HallOfFameModule: HallOfFame, update_hall_of_fame!, _update_hall_of_fame_unchecked!
using ..ComplexityModule: compute_complexity
using ..MutateModule: next_generation, MutationWorkspace, recycle!
using ..CrossoverModule: crossover_generation
using ..TracingModule:
    new_trace,
    new_step_trace,
    new_traced_steps,
    reset_traced_steps!,
    trace_crossover!,
    trace_mutation_attempts!,
    trace_mutation_step!
using ..UtilsModule: strictmap

function mutation_workspace(pop, options, curmaxsize)
    return can_recycle(pop, options) ? MutationWorkspace(
        first(pop.members).tree,
        curmaxsize,
        options.mutations,
        options.tournament_selection_n,
        typeof(first(pop.members).cost),
    ) : nothing
end
function oldest_member(pop, skip::Int=0)
    BT = typeof(first(pop.members).birth)
    oldest = 1
    oldest_birth = typemax(BT)
    @inbounds for i in 1:(pop.n)
        birth = i == skip ? typemax(BT) : pop.members[i].birth
        if birth < oldest_birth
            oldest = i
            oldest_birth = birth
        end
    end
    return oldest
end

"""
One precomposed mutation-middleware layer.
"""
struct MutationStepLayer{W,F}
    wrapper::W
    next_step::F
end
@inline function (layer::MutationStepLayer)(parent)
    return layer.wrapper(parent, layer.next_step)
end

build_mutation_step(::Tuple{}, base_step) = base_step
function build_mutation_step(wrappers::Tuple, base_step::F) where {F}
    inner = build_mutation_step(Base.tail(wrappers), base_step)
    return _add_mutation_step_layer(first(wrappers), inner)
end
_add_mutation_step_layer(::Nothing, inner) = inner
function _add_mutation_step_layer(wrapper, inner)  # COV_EXCL_LINE
    return MutationStepLayer(wrapper, inner)
end

"""
Engine-owned state for one mutation step. Mutable contents accumulate every
middleware attempt so evaluation counts, Hall-of-Fame updates, and tracing
stay under engine control.
"""
struct MutationStep{D,P,O,S,E,H,A,M,R,W}
    dataset::D
    population::P
    curmaxsize::Int
    options::O
    plugin_states::S
    eval_context::E
    best_seen::H
    attempted_results::A
    attempted_members::M
    traced_steps::R
    workspace::W
end

function (step::MutationStep)(parent)
    step_trace = new_step_trace(step.traced_steps)
    member, accepted, num_evals = next_generation(
        step.dataset,
        parent,
        step.curmaxsize,
        step.options;
        tmp_trace=step_trace,
        plugin_states=step.plugin_states,
        eval_context=step.eval_context,
        population_for_backsolve=step.population,
        workspace=step.workspace,
    )
    attempt_id = isnothing(step.attempted_results) ? 1 : length(step.attempted_results) + 1
    result = MutationStepResult(member, accepted, attempt_id, num_evals)
    !isnothing(step.attempted_results) && push!(step.attempted_results, result)
    !isnothing(step.attempted_members) && push!(step.attempted_members, copy(member))
    trace_mutation_step!(step.traced_steps, parent, member, step_trace)
    accepted &&
        !isnothing(step.attempted_members) &&
        update_hall_of_fame!(step.best_seen, member, step.options)
    return result
end

function reset!(step::MutationStep)
    !isnothing(step.attempted_results) && empty!(step.attempted_results)
    !isnothing(step.attempted_members) && empty!(step.attempted_members)
    reset_traced_steps!(step.traced_steps)
    return nothing
end

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function reg_evol_cycle(
    dataset::Dataset{T,L},
    pop::P,
    curmaxsize::Int,
    options::AbstractOptions,
    trace::MaybeTrace;
    plugin_states::Tuple,
    best_seen::HallOfFame,
    eval_context=nothing,
    workspace=nothing,
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}}
    num_evals = 0.0
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n)
    mutation_wrappers = strictmap(wrap_mutation_step, plugin_states, options.plugins)
    traced_steps = new_traced_steps(trace, eltype(pop.members))
    has_mutation_wrappers = any(!isnothing, mutation_wrappers)
    attempted_results =
        has_mutation_wrappers ? MutationStepResult{eltype(pop.members)}[] : nothing
    attempted_members = has_mutation_wrappers ? eltype(pop.members)[] : nothing
    base_step = MutationStep(
        dataset,
        pop,
        curmaxsize,
        options,
        plugin_states,
        eval_context,
        best_seen,
        attempted_results,
        attempted_members,
        traced_steps,
        workspace,
    )
    wrapped_step = build_mutation_step(mutation_wrappers, base_step)

    borrow_parents = !isnothing(workspace) && !has_mutation_wrappers
    for i in 1:n_evol_cycles
        if rand() > options.crossover_probability
            allstar = best_of_sample(pop, options; plugin_states, workspace)
            borrow_parents || (allstar = copy(allstar))
            reset!(base_step)
            result = wrapped_step(allstar)
            selected_attempt_idx = result.attempt_id
            selected_result = if isnothing(base_step.attempted_results)
                num_evals += result.num_evals
                result
            else
                checkbounds(Bool, base_step.attempted_results, selected_attempt_idx) ||
                    throw(
                        ArgumentError(
                            "Mutation middleware must return a result from `next_step`."
                        ),
                    )
                num_evals += sum(attempt -> attempt.num_evals, base_step.attempted_results)
                base_step.attempted_results[selected_attempt_idx]
            end
            baby = if isnothing(base_step.attempted_members)
                selected_result.member
            else
                base_step.attempted_members[selected_attempt_idx]
            end
            mutation_accepted = selected_result.accepted

            should_replace = mutation_accepted || !options.skip_mutation_failures
            oldest = if should_replace
                oldest_member(pop)
            else
                0
            end

            trace_mutation_attempts!(
                trace,
                traced_steps,
                pop,
                oldest,
                should_replace,
                selected_attempt_idx,
                options,
            )

            should_replace || continue
            mutation_accepted || (baby = copy(baby))
            !isnothing(workspace) && recycle!(workspace, pop.members[oldest].tree)
            pop.members[oldest] = baby

        else # Crossover
            allstar1 = best_of_sample(pop, options; plugin_states, workspace)
            allstar2 = best_of_sample(pop, options; plugin_states, workspace)
            if !borrow_parents
                allstar1, allstar2 = copy(allstar1), copy(allstar2)
            elseif allstar1 === allstar2
                allstar2 = copy(allstar2)
            end

            crossover_trace = new_trace(trace)
            baby1, baby2, crossover_accepted, tmp_num_evals = crossover_generation(
                allstar1,
                allstar2,
                dataset,
                curmaxsize,
                options;
                trace=crossover_trace,
                plugin_states,
                eval_context,
            )
            num_evals += tmp_num_evals
            if crossover_accepted
                _update_hall_of_fame_unchecked!(
                    best_seen, baby1, compute_complexity(baby1, options)
                )
                _update_hall_of_fame_unchecked!(
                    best_seen, baby2, compute_complexity(baby2, options)
                )
            end

            if !crossover_accepted && options.skip_mutation_failures
                continue
            end
            if !crossover_accepted
                baby1, baby2 = copy(baby1), copy(baby2)
            end

            # Find the oldest members to replace:
            oldest1 = oldest_member(pop)
            oldest2 = oldest_member(pop, oldest1)

            trace_crossover!(
                trace,
                allstar1,
                allstar2,
                baby1,
                baby2,
                pop,
                oldest1,
                oldest2,
                crossover_trace,
                options,
            )

            # Replace old members with new ones:
            pop.members[oldest1] = baby1
            pop.members[oldest2] = baby2
        end
    end

    return (pop, num_evals)
end

end
