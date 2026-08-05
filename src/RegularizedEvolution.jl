module RegularizedEvolutionModule

using DynamicExpressions: string_tree
using ..CoreModule:
    AbstractOptions,
    Dataset,
    RecordType,
    DATA_TYPE,
    LOSS_TYPE,
    MutationStepResult,
    wrap_mutation_step
using ..PopulationModule: Population, best_of_sample
using ..HallOfFameModule: HallOfFame, update_hall_of_fame!, _update_hall_of_fame_unchecked!
using ..ComplexityModule: compute_complexity
using ..MutateModule: next_generation, crossover_generation
using ..RecorderModule: @recorder
using ..UtilsModule: argmin_fast, strictmap

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
middleware attempt so evaluation counts, Hall-of-Fame updates, and recording
stay under engine control.
"""
struct MutationStep{D,P,O,S,E,H,A,M,R}
    dataset::D
    population::P
    curmaxsize::Int
    options::O
    plugin_states::S
    eval_options::E
    best_seen::H
    attempted_results::A
    attempted_members::M
    recorded_steps::R
end

function (step::MutationStep)(parent)
    step_recorder = RecordType()
    member, accepted, num_evals = next_generation(
        step.dataset,
        parent,
        step.curmaxsize,
        step.options;
        tmp_recorder=step_recorder,
        plugin_states=step.plugin_states,
        eval_options=step.eval_options,
        population_for_backsolve=step.population,
    )
    attempt_id = isnothing(step.attempted_results) ? 1 : length(step.attempted_results) + 1
    result = MutationStepResult(member, accepted, attempt_id, num_evals)
    !isnothing(step.attempted_results) && push!(step.attempted_results, result)
    !isnothing(step.attempted_members) && push!(step.attempted_members, copy(member))
    if !isnothing(step.recorded_steps)
        push!(step.recorded_steps, (copy(parent), copy(member), step_recorder))
    end
    accepted &&
        !isnothing(step.attempted_members) &&
        update_hall_of_fame!(step.best_seen, member, step.options)
    return result
end

function reset!(step::MutationStep)
    !isnothing(step.attempted_results) && empty!(step.attempted_results)
    !isnothing(step.attempted_members) && empty!(step.attempted_members)
    !isnothing(step.recorded_steps) && empty!(step.recorded_steps)
    return nothing
end

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function reg_evol_cycle(
    dataset::Dataset{T,L},
    pop::P,
    curmaxsize::Int,
    options::AbstractOptions,
    record::RecordType;
    plugin_states::Tuple,
    best_seen::HallOfFame,
    eval_options=nothing,
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}}
    num_evals = 0.0
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n)
    mutation_wrappers = strictmap(wrap_mutation_step, plugin_states, options.plugins)
    recorded_steps = if options.use_recorder isa Val{true}
        Tuple{eltype(pop.members),eltype(pop.members),RecordType}[]
    else
        nothing
    end
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
        eval_options,
        best_seen,
        attempted_results,
        attempted_members,
        recorded_steps,
    )
    wrapped_step = build_mutation_step(mutation_wrappers, base_step)

    for i in 1:n_evol_cycles
        if rand() > options.crossover_probability
            allstar = best_of_sample(pop, options; plugin_states)
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
                argmin_fast([pop.members[member].birth for member in 1:(pop.n)])
            else
                0
            end

            @recorder begin
                steps = something(recorded_steps)

                if !haskey(record, "mutations")
                    record["mutations"] = RecordType()
                end
                members_to_record =
                    should_replace ? [pop.members[oldest]] : eltype(pop.members)[]
                for (parent, child, _) in steps
                    push!(members_to_record, parent, child)
                end
                for member in members_to_record
                    if !haskey(record["mutations"], "$(member.ref)")
                        record["mutations"]["$(member.ref)"] = RecordType(
                            "events" => Vector{RecordType}(),
                            "tree" => string_tree(member.tree, options),
                            "cost" => member.cost,
                            "loss" => member.loss,
                            "parent" => member.parent,
                        )
                    end
                end
                for (attempt_idx, (parent, child, step_recorder)) in enumerate(steps)
                    mutate_event = RecordType(
                        "type" => "mutate",
                        "time" => time(),
                        "child" => child.ref,
                        "selected" => attempt_idx == selected_attempt_idx,
                        "mutation" => step_recorder,
                    )
                    push!(record["mutations"]["$(parent.ref)"]["events"], mutate_event)
                end
                if should_replace
                    death_event = RecordType("type" => "death", "time" => time())
                    push!(
                        record["mutations"]["$(pop.members[oldest].ref)"]["events"],
                        death_event,
                    )
                end
            end

            should_replace || continue
            pop.members[oldest] = baby

        else # Crossover
            allstar1 = best_of_sample(pop, options; plugin_states)
            allstar2 = best_of_sample(pop, options; plugin_states)

            crossover_recorder = RecordType()
            baby1, baby2, crossover_accepted, tmp_num_evals = crossover_generation(
                allstar1,
                allstar2,
                dataset,
                curmaxsize,
                options;
                recorder=crossover_recorder,
                eval_options,
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

            # Find the oldest members to replace:
            oldest1 = argmin_fast([pop.members[member].birth for member in 1:(pop.n)])
            BT = typeof(first(pop.members).birth)
            oldest2 = argmin_fast([
                i == oldest1 ? typemax(BT) : pop.members[i].birth for i in 1:(pop.n)
            ])

            @recorder begin
                if !haskey(record, "mutations")
                    record["mutations"] = RecordType()
                end
                for member in [
                    allstar1,
                    allstar2,
                    baby1,
                    baby2,
                    pop.members[oldest1],
                    pop.members[oldest2],
                ]
                    if !haskey(record["mutations"], "$(member.ref)")
                        record["mutations"]["$(member.ref)"] = RecordType(
                            "events" => Vector{RecordType}(),
                            "tree" => string_tree(member.tree, options),
                            "cost" => member.cost,
                            "loss" => member.loss,
                            "parent" => member.parent,
                        )
                    end
                end
                crossover_event = RecordType(
                    "type" => "crossover",
                    "time" => time(),
                    "parent1" => allstar1.ref,
                    "parent2" => allstar2.ref,
                    "child1" => baby1.ref,
                    "child2" => baby2.ref,
                    "details" => crossover_recorder,
                )
                death_event1 = RecordType("type" => "death", "time" => time())
                death_event2 = RecordType("type" => "death", "time" => time())

                push!(record["mutations"]["$(allstar1.ref)"]["events"], crossover_event)
                push!(record["mutations"]["$(allstar2.ref)"]["events"], crossover_event)
                push!(
                    record["mutations"]["$(pop.members[oldest1].ref)"]["events"],
                    death_event1,
                )
                push!(
                    record["mutations"]["$(pop.members[oldest2].ref)"]["events"],
                    death_event2,
                )
            end

            # Replace old members with new ones:
            pop.members[oldest1] = baby1
            pop.members[oldest2] = baby2
        end
    end

    return (pop, num_evals)
end

end
