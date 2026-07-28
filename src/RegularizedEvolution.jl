module RegularizedEvolutionModule

using DynamicExpressions: string_tree
using ..CoreModule:
    AbstractOptions, Dataset, RecordType, DATA_TYPE, LOSS_TYPE, wrap_mutation_step
using ..PopulationModule: Population, best_of_sample
using ..MutateModule: next_generation, crossover_generation
using ..RecorderModule: @recorder
using ..UtilsModule: argmin_fast

# Compose plugins around `next_generation` as middleware via
# `wrap_mutation_step(state, plugin, parent, next_step)`, recursing over the
# plugin tuple so the composed step is fully inferable. Plugin tuple order is
# the outer-to-inner middleware order.
build_mutation_step(::Tuple{}, ::Tuple{}, base_step) = base_step
build_mutation_step(::Tuple{}, ::Tuple, base_step) = _plugin_state_mismatch()
build_mutation_step(::Tuple, ::Tuple{}, base_step) = _plugin_state_mismatch()
function build_mutation_step(plugins::Tuple, states::Tuple, base_step::F) where {F}
    inner = build_mutation_step(Base.tail(plugins), Base.tail(states), base_step)
    plugin, state = first(plugins), first(states)
    return parent -> wrap_mutation_step(state, plugin, parent, inner)
end
@noinline function _plugin_state_mismatch()
    throw(
        ArgumentError("`options.plugins` and `plugin_states` must have the same length.")
    )
end

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function reg_evol_cycle(
    dataset::Dataset{T,L},
    pop::P,
    curmaxsize::Int,
    options::AbstractOptions,
    record::RecordType;
    plugin_states::Tuple=(),
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}}
    num_evals = 0.0
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n)

    for i in 1:n_evol_cycles
        if rand() > options.crossover_probability
            allstar = best_of_sample(pop, options; plugin_states)
            mutation_recorder = RecordType()
            mutation_steps = if options.use_recorder
                Tuple{eltype(pop.members),eltype(pop.members),RecordType}[]
            else
                nothing
            end

            base_step = if mutation_steps === nothing
                parent -> next_generation(
                    dataset,
                    parent,
                    curmaxsize,
                    options;
                    tmp_recorder=mutation_recorder,
                    plugin_states,
                    population_for_backsolve=pop,
                )
            else
                parent -> begin
                    step_recorder = RecordType()
                    result = next_generation(
                        dataset,
                        parent,
                        curmaxsize,
                        options;
                        tmp_recorder=step_recorder,
                        plugin_states,
                        population_for_backsolve=pop,
                    )
                    push!(mutation_steps, (parent, result[1], step_recorder))
                    return result
                end
            end
            wrapped_step = build_mutation_step(options.plugins, plugin_states, base_step)
            baby, mutation_accepted, tmp_num_evals = wrapped_step(allstar)
            num_evals += tmp_num_evals

            if !mutation_accepted && options.skip_mutation_failures
                # Skip this mutation rather than replacing oldest member with unchanged member
                continue
            end

            oldest = argmin_fast([pop.members[member].birth for member in 1:(pop.n)])

            @recorder begin
                recorded_steps = something(mutation_steps)

                if !haskey(record, "mutations")
                    record["mutations"] = RecordType()
                end
                members_to_record = [pop.members[oldest]]
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
                death_event = RecordType("type" => "death", "time" => time())

                for (parent, child, step_recorder) in recorded_steps
                    mutate_event = RecordType(
                        "type" => "mutate",
                        "time" => time(),
                        "child" => child.ref,
                        "mutation" => step_recorder,
                    )
                    push!(record["mutations"]["$(parent.ref)"]["events"], mutate_event)
                end
                push!(
                    record["mutations"]["$(pop.members[oldest].ref)"]["events"], death_event
                )
            end

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
