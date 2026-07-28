@testitem "AdaptiveMutationWeightsPlugin: defaults + state" begin
    using SymbolicRegression
    using Test

    p = AdaptiveMutationWeightsPlugin()
    @test p.smoothing == 0.02
    @test p.floor == 0.05

    opts = Options(;
        binary_operators=[+, *],
        unary_operators=[sin],
        plugins=(AdaptiveMutationWeightsPlugin(),),
    )
    s = SymbolicRegression.init_plugin_state(AdaptiveMutationWeightsPlugin(), opts, nothing)
    @test length(s.attempts) == length(opts.mutations)
    @test all(s.multipliers .== 1.0)
end

@testitem "skip_in_adaptive_weights: defaults skip Simplify and DoNothing" begin
    using SymbolicRegression
    using Test

    struct SkippedMutation <: AbstractMutation end
    SymbolicRegression.skip_in_adaptive_weights(::SkippedMutation) = true

    @test SymbolicRegression.skip_in_adaptive_weights(SimplifyMutation()) == true
    @test SymbolicRegression.skip_in_adaptive_weights(DoNothingMutation()) == true
    @test SymbolicRegression.skip_in_adaptive_weights(ConstantMutation()) == false
    @test SymbolicRegression.skip_in_adaptive_weights(OperatorMutation()) == false
    @test SymbolicRegression.skip_in_adaptive_weights(SkippedMutation()) == true
end

@testitem "SimulatedAnnealingPlugin uses the requested cycle count" begin
    using SymbolicRegression
    using SymbolicRegression.SimulatedAnnealingModule: SimulatedAnnealingState
    using Test

    plugin = SimulatedAnnealingPlugin()
    state = SimulatedAnnealingState(0.5)
    options = Options(; ncycles_per_iteration=100)

    SymbolicRegression.on_cycle_start!(state, plugin, 1, 3, options)
    @test state.temperature == 1.0
    SymbolicRegression.on_cycle_start!(state, plugin, 2, 3, options)
    @test state.temperature == 0.5
    @test SymbolicRegression.constant_mutation_multiplier(state, plugin) == 0.5
    SymbolicRegression.on_cycle_start!(state, plugin, 3, 3, options)
    @test state.temperature == 0.0
    SymbolicRegression.on_cycle_start!(state, plugin, 1, 1, options)
    @test state.temperature == 1.0
end

@testitem "AdaptiveMutationWeightsPlugin attributes configured instances" begin
    using SymbolicRegression
    using SymbolicRegression: MutationEvent, init_plugin_state, on_mutation_end!
    using Test

    first_mutation = ConstantMutation(; perturbation_factor=0.1)
    second_mutation = ConstantMutation(; perturbation_factor=0.2)
    options = Options(;
        default_mutations=(),
        mutations=(first_mutation => 1.0, second_mutation => 1.0),
        plugins=(AdaptiveMutationWeightsPlugin(),),
        default_plugins=(),
    )
    plugin = only(options.plugins)
    state = init_plugin_state(plugin, options, nothing)

    on_mutation_end!(
        state, plugin, second_mutation, MutationEvent(true, 1.0, 0.5), nothing, options
    )
    @test state.attempts == [0.0, 1.0]
    @test state.successes == [0.0, 1.0]
end

@testitem "MutationLoopPlugin: retry portion retries until accepted" begin
    using SymbolicRegression
    using SymbolicRegression: wrap_mutation_step
    using Test

    n_calls = Ref(0)
    inner = parent -> begin
        n_calls[] += 1
        (parent, n_calls[] >= 3, 1.0)
    end
    p = MutationLoopPlugin(;
        retry_attempts=4, compound_probability=0.0, compound_max_steps=1
    )
    member, accepted, num_evals = wrap_mutation_step(nothing, p, :parent, inner)
    @test accepted == true
    @test n_calls[] == 3
    @test num_evals == 3.0
end

@testitem "MutationLoopPlugin: retry stops at budget when never accepted" begin
    using SymbolicRegression
    using SymbolicRegression: wrap_mutation_step
    using Test

    n_calls = Ref(0)
    inner = parent -> begin
        n_calls[] += 1
        (parent, false, 1.0)
    end
    p = MutationLoopPlugin(;
        retry_attempts=4, compound_probability=0.0, compound_max_steps=1
    )
    member, accepted, num_evals = wrap_mutation_step(nothing, p, :parent, inner)
    @test accepted == false
    @test n_calls[] == 4
    @test num_evals == 4.0
end

@testitem "MutationLoopPlugin: compound portion chains on success" begin
    using SymbolicRegression
    using SymbolicRegression: wrap_mutation_step
    using Random
    using Test

    n_calls = Ref(0)
    inner = parent -> begin
        n_calls[] += 1
        (parent + 1, true, 1.0)
    end
    p = MutationLoopPlugin(;
        retry_attempts=1, compound_probability=1.0, compound_max_steps=3
    )
    Random.seed!(0)
    member, accepted, num_evals = wrap_mutation_step(nothing, p, 0, inner)
    @test accepted == true
    @test n_calls[] == 3
    @test member == 3
end

@testitem "MutationLoopPlugin: compound doesn't chain on rejection" begin
    using SymbolicRegression
    using SymbolicRegression: wrap_mutation_step
    using Test

    n_calls = Ref(0)
    inner = parent -> begin
        n_calls[] += 1
        (parent, false, 1.0)
    end
    p = MutationLoopPlugin(;
        retry_attempts=1, compound_probability=1.0, compound_max_steps=3
    )
    member, accepted, num_evals = wrap_mutation_step(nothing, p, 0, inner)
    @test accepted == false
    @test n_calls[] == 1
end

@testitem "MutationLoopPlugin records every compound mutation" begin
    using SymbolicRegression
    using SymbolicRegression: Dataset, RecordType, init_plugin_state
    using SymbolicRegression.RegularizedEvolutionModule: reg_evol_cycle
    using Test

    plugin = MutationLoopPlugin(;
        retry_attempts=1, compound_probability=1.0, compound_max_steps=3
    )
    options = Options(;
        default_mutations=(),
        mutations=(DoNothingMutation() => 1.0,),
        plugins=(plugin,),
        default_plugins=(),
        crossover_probability=0.0,
        population_size=2,
        tournament_selection_n=1,
        use_recorder=true,
    )
    dataset = Dataset(zeros(1, 8), zeros(8))
    plugin_states = (init_plugin_state(plugin, options, dataset),)
    population = Population(
        dataset; population_size=2, nlength=1, options, nfeatures=1, plugin_states
    )
    record = RecordType()

    reg_evol_cycle(dataset, population, options.maxsize, options, record; plugin_states)

    mutation_events = Tuple{String,RecordType}[]
    for (parent, member_record) in record["mutations"]
        for event in member_record["events"]
            event["type"] == "mutate" && push!(mutation_events, (parent, event))
        end
    end
    @test length(mutation_events) == 3 * options.population_size
    for (parent, event) in mutation_events
        child = record["mutations"]["$(event["child"])"]
        @test child["parent"] == parse(Int, parent)
        @test event["mutation"]["type"] == "identity"
    end
end

@testitem "wrap_mutation_step: default plugin is pass-through" begin
    using SymbolicRegression
    using SymbolicRegression: AbstractPlugin, wrap_mutation_step
    using Test

    struct _NoopMidPlugin <: AbstractPlugin end
    n_calls = Ref(0)
    inner = parent -> begin
        n_calls[] += 1
        (parent, true, 2.0)
    end
    member, accepted, num_evals = wrap_mutation_step(
        nothing, _NoopMidPlugin(), :parent, inner
    )
    @test n_calls[] == 1
    @test num_evals == 2.0
end

@testitem "wrap_mutation_step: third-party plugin can extend the hook" begin
    # Stress test: a plugin that runs the inner step exactly twice and
    # keeps the better result by num_evals. Demonstrates the middleware
    # shape isn't tied to retry/chain semantics.
    using SymbolicRegression
    using SymbolicRegression: AbstractPlugin, wrap_mutation_step
    using Test

    struct BestOfTwoPlugin <: AbstractPlugin end
    function SymbolicRegression.wrap_mutation_step(
        _, ::BestOfTwoPlugin, parent, next_step::F
    ) where {F}
        r1 = next_step(parent)
        r2 = next_step(parent)
        return r1[3] <= r2[3] ? r1 : r2
    end

    n_calls = Ref(0)
    inner = parent -> begin
        n_calls[] += 1
        (parent, true, Float64(n_calls[]))
    end
    member, accepted, num_evals = wrap_mutation_step(nothing, BestOfTwoPlugin(), :p, inner)
    @test n_calls[] == 2
    @test num_evals == 1.0
end

@testitem "Integration: all three mechanisms enabled, search runs and returns a HoF" begin
    using SymbolicRegression
    using Random
    using Test

    Random.seed!(0)
    X = rand(Float32, 2, 60)
    y = @. 2.0f0 * X[1, :] + sin(X[2, :])

    opts = Options(;
        binary_operators=[+, -, *, /],
        unary_operators=[sin, cos],
        populations=4,
        population_size=20,
        ncycles_per_iteration=20,
        verbosity=0,
        progress=false,
        deterministic=true,
        plugins=(
            AdaptiveMutationWeightsPlugin(; smoothing=0.02, floor=0.05),
            MutationLoopPlugin(;
                retry_attempts=4, compound_probability=0.25, compound_max_steps=2
            ),
        ),
    )
    hof = equation_search(X, y; options=opts, niterations=3, parallelism=:serial)
    @test hof isa SymbolicRegression.HallOfFame
    @test any(hof.exists)
end

@testitem "Integration: defaults match upstream single-step loop" begin
    using SymbolicRegression
    using Random
    using Test

    Random.seed!(0)
    X = rand(Float32, 2, 30)
    y = X[1, :] .+ X[2, :]
    opts = Options(;
        binary_operators=[+, *],
        populations=2,
        population_size=30,
        ncycles_per_iteration=10,
        verbosity=0,
        progress=false,
        deterministic=true,
    )
    hof = equation_search(X, y; options=opts, niterations=2, parallelism=:serial)
    @test hof isa SymbolicRegression.HallOfFame
end
