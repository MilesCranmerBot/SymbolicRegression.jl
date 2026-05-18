module RegularizedEvolutionModule

using ..CoreModule: AbstractOptions, Dataset, RecordType, DATA_TYPE, LOSS_TYPE
using ..RecorderModule: JSONLRecorder
using ..PopulationModule: Population, best_of_sample
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..MutateModule: next_generation, crossover_generation
using ..RecorderModule: @recorder, ensure_member_recorded!, record_member_event!
using ..UtilsModule: argmin_fast

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function reg_evol_cycle(
    dataset::Dataset{T,L},
    pop::P,
    temperature,
    curmaxsize::Int,
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions,
    record::JSONLRecorder,
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:Population{T,L}}
    num_evals = 0.0
    n_evol_cycles = ceil(Int, pop.n / options.tournament_selection_n)

    for i in 1:n_evol_cycles
        if rand() > options.crossover_probability
            allstar = best_of_sample(pop, running_search_statistics, options)
            mutation_recorder = RecordType()
            baby, mutation_accepted, tmp_num_evals = next_generation(
                dataset,
                allstar,
                temperature,
                curmaxsize,
                running_search_statistics,
                options;
                tmp_recorder=mutation_recorder,
                population_for_backsolve=pop,
            )
            num_evals += tmp_num_evals

            if !mutation_accepted && options.skip_mutation_failures
                # Skip this mutation rather than replacing oldest member with unchanged member
                continue
            end

            oldest = argmin_fast([pop.members[member].birth for member in 1:(pop.n)])

            @recorder begin
                ensure_member_recorded!.(Ref(record), [allstar, baby, pop.members[oldest]], Ref(options))
                mutate_event = RecordType(
                    "type" => "mutate",
                    "time" => time(),
                    "child" => baby.ref,
                    "mutation" => mutation_recorder,
                )
                death_event = RecordType("type" => "death", "time" => time())

                record_member_event!(record, allstar.ref, mutate_event)
                record_member_event!(record, pop.members[oldest].ref, death_event)
            end

            pop.members[oldest] = baby

        else # Crossover
            allstar1 = best_of_sample(pop, running_search_statistics, options)
            allstar2 = best_of_sample(pop, running_search_statistics, options)

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
                ensure_member_recorded!.(
                    Ref(record),
                    [allstar1, allstar2, baby1, baby2, pop.members[oldest1], pop.members[oldest2]],
                    Ref(options),
                )
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

                record_member_event!(record, allstar1.ref, crossover_event)
                record_member_event!(record, allstar2.ref, crossover_event)
                record_member_event!(record, pop.members[oldest1].ref, death_event1)
                record_member_event!(record, pop.members[oldest2].ref, death_event2)
            end

            # Replace old members with new ones:
            pop.members[oldest1] = baby1
            pop.members[oldest2] = baby2
        end
    end

    return (pop, num_evals)
end

end
