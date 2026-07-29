module RegularizedEvolutionModule

using DynamicExpressions: string_tree
using ..CoreModule:
    AbstractOptions,
    Dataset,
    RecordType,
    DATA_TYPE,
    LOSS_TYPE,
    MutationStepResult,
    wraps_mutation_step,
    wrap_mutation_step
using ..PopulationModule: Population, best_of_sample
using ..HallOfFameModule: HallOfFame, update_hall_of_fame!
using ..MutateModule: next_generation, crossover_generation
using ..RecorderModule: @recorder
using ..UtilsModule: argmin_fast

struct MutationStepLayer{S,P,F}
    state::S
    plugin::P
    next_step::F
end
@inline function (layer::MutationStepLayer)(parent)
    return wrap_mutation_step(layer.state, layer.plugin, parent, layer.next_step)
end

build_mutation_step(::Tuple{}, ::Tuple{}, base_step) = base_step
build_mutation_step(::Tuple{}, ::Tuple, base_step) = _plugin_state_mismatch()
build_mutation_step(::Tuple, ::Tuple{}, base_step) = _plugin_state_mismatch()
function build_mutation_step(plugins::Tuple, states::Tuple, base_step::F) where {F}
    inner = build_mutation_step(Base.tail(plugins), Base.tail(states), base_step)
    plugin = first(plugins)
    state = first(states)
    return _add_mutation_step_layer(wraps_mutation_step(plugin), state, plugin, inner)
end
_add_mutation_step_layer(::Val{false}, state, plugin, inner) = inner
function _add_mutation_step_layer(::Val{true}, state, plugin, inner)  # COV_EXCL_LINE
    return MutationStepLayer(state, plugin, inner)
end
@noinline function _plugin_state_mismatch()
    throw(ArgumentError("`options.plugins` and `plugin_states` must have the same length."))
end

mutable struct MutationStep{D,P,O,S,H,A,M,R}
    dataset::D
    population::P
    curmaxsize::Int
    options::O
    plugin_states::S
    best_seen::H
    attempted_results::A
    attempted_members::M
    recorded_steps::R
    num_evals::Float64
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
        population_for_backsolve=step.population,
    )
    step.num_evals += num_evals
    result = MutationStepResult(member, accepted, UInt(length(step.attempted_results) + 1))
    push!(step.attempted_results, result)
    step.attempted_members !== nothing && push!(step.attempted_members, copy(member))
    if step.recorded_steps !== nothing
        push!(step.recorded_steps, (copy(parent), copy(member), step_recorder))
    end
    accepted && update_hall_of_fame!(step.best_seen, member, step.options)
    return result
end

function reset!(step::MutationStep)
    step.num_evals = 0.0
    empty!(step.attempted_results)
    step.attempted_members !== nothing && empty!(step.attempted_members)
    step.recorded_steps !== nothing && empty!(step.recorded_steps)
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
    best_seen::Union{Nothing,HallOfFame}=nothing,
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}}
    num_evals = 0.0
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n)
    actual_best_seen = best_seen === nothing ? HallOfFame(options, dataset) : best_seen
    mutation_steps = if options.use_recorder
        Tuple{eltype(pop.members),eltype(pop.members),RecordType}[]
    else
        nothing
    end
    attempted_members =
        if any(plugin -> wraps_mutation_step(plugin) === Val(true), options.plugins)
            eltype(pop.members)[]
        else
            nothing
        end
    base_step = MutationStep(
        dataset,
        pop,
        curmaxsize,
        options,
        plugin_states,
        actual_best_seen,
        MutationStepResult{eltype(pop.members)}[],
        attempted_members,
        mutation_steps,
        0.0,
    )
    wrapped_step = build_mutation_step(options.plugins, plugin_states, base_step)

    for i in 1:n_evol_cycles
        if rand() > options.crossover_probability
            allstar = best_of_sample(pop, options; plugin_states)
            reset!(base_step)
            result = wrapped_step(allstar)
            selected_attempt = findfirst(
                attempt ->
                    result.attempt_id != 0 && attempt.attempt_id == result.attempt_id,
                base_step.attempted_results,
            )
            selected_attempt === nothing && throw(
                ArgumentError("Mutation middleware must return a result from `next_step`."),
            )
            selected_attempt_idx = selected_attempt::Int
            selected_result = base_step.attempted_results[selected_attempt_idx]
            baby = if base_step.attempted_members === nothing
                selected_result.member
            else
                base_step.attempted_members[selected_attempt_idx]
            end
            mutation_accepted = selected_result.accepted
            num_evals += base_step.num_evals

            should_replace = mutation_accepted || !options.skip_mutation_failures
            oldest = if should_replace
                argmin_fast([pop.members[member].birth for member in 1:(pop.n)])
            else
                0
            end

            @recorder begin
                recorded_steps = something(mutation_steps)

                if !haskey(record, "mutations")
                    record["mutations"] = RecordType()
                end
                members_to_record =
                    should_replace ? [pop.members[oldest]] : eltype(pop.members)[]
                for (parent, child, _) in recorded_steps
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
                for (attempt_idx, (parent, child, step_recorder)) in
                    enumerate(recorded_steps)
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
            )
            num_evals += tmp_num_evals
            if crossover_accepted
                update_hall_of_fame!(actual_best_seen, baby1, options)
                update_hall_of_fame!(actual_best_seen, baby2, options)
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
