@testitem "Custom mutation dispatch" begin
    using SymbolicRegression
    using SymbolicRegression: Dataset, MutationResult, RecordType, mutate!, sample_mutation
    using SymbolicRegression.MutateModule: _sample_mutation, next_generation
    using Random: seed!

    struct CustomMutation <: AbstractMutation
        calls::Base.RefValue{Int}
    end

    function SymbolicRegression.mutate!(
        tree::N, ::P, mutation::CustomMutation, options; kws...
    ) where {N,P}
        mutation.calls[] += 1
        return MutationResult{N,P}(; tree)
    end

    calls = Ref(0)
    options = Options(;
        binary_operators=(+, *),
        default_mutations=(),
        mutations=(CustomMutation(calls) => 1.0,),
    )
    dataset = Dataset(randn(2, 16), randn(16))
    member = PopMember(dataset, Node(Float64; feature=1), options; deterministic=false)

    next_generation(
        dataset, member, 1.0, options.maxsize, options; tmp_recorder=RecordType()
    )

    @test calls[] == 1

    struct MissingMutation <: AbstractMutation end
    @test_throws ErrorException mutate!(
        copy(member.tree), member, MissingMutation(), options
    )

    weighted_mutations = [DoNothingMutation() => 1.0]
    @test sample_mutation(weighted_mutations) isa DoNothingMutation

    seed!(4)
    @test _sample_mutation([DoNothingMutation() => nextfloat(0.0)]) isa DoNothingMutation
end

@testitem "Legacy next_generation preserves positional temperature" begin
    using SymbolicRegression
    using SymbolicRegression:
        AbstractPlugin, Dataset, RecordType, init_plugin_state, set_temperature!
    using SymbolicRegression.MutateModule: next_generation
    using Random
    using Test

    struct TemperatureConsumer <: AbstractPlugin end

    function SymbolicRegression.set_temperature!(
        state::Base.RefValue, ::TemperatureConsumer, temperature
    )
        state[] = Float64(temperature)
        return true
    end

    function constant_setup(; annealing)
        options = Options(;
            default_mutations=(),
            mutations=(ConstantMutation(; probability_negate=0.0) => 1.0,),
            annealing,
            use_frequency=false,
            use_frequency_in_tournament=false,
        )
        dataset = Dataset(zeros(1, 8), fill(1.0, 8))
        member = PopMember(dataset, Node(Float64; val=1.0), options; deterministic=false)
        return options, dataset, member
    end

    options, dataset, member = constant_setup(; annealing=true)
    temperature = 0.25
    plugin_states = map(
        plugin -> init_plugin_state(plugin, options, dataset), options.plugins
    )
    for (plugin, state) in zip(options.plugins, plugin_states)
        set_temperature!(state, plugin, temperature)
    end

    Random.seed!(12)
    expected = next_generation(
        dataset, member, options.maxsize, options; tmp_recorder=RecordType(), plugin_states
    )
    Random.seed!(12)
    actual = next_generation(
        dataset, member, temperature, options.maxsize, options; tmp_recorder=RecordType()
    )
    @test actual[1].tree == expected[1].tree
    @test actual[2:3] == expected[2:3]

    options, dataset, member = constant_setup(; annealing=false)
    Random.seed!(21)
    cold = next_generation(
        dataset, member, 0.0, options.maxsize, options; tmp_recorder=RecordType()
    )
    Random.seed!(21)
    hot = next_generation(
        dataset, member, 1.0, options.maxsize, options; tmp_recorder=RecordType()
    )
    @test cold[1].tree != hot[1].tree

    tracking_options = Options(;
        default_mutations=(),
        mutations=(DoNothingMutation() => 1.0,),
        plugins=(TemperatureConsumer(), TemperatureConsumer()),
        default_plugins=(),
        use_frequency=false,
        use_frequency_in_tournament=false,
    )
    tracking_member = PopMember(
        dataset, Node(Float64; val=1.0), tracking_options; deterministic=false
    )
    temperature_states = (Ref(NaN), Ref(NaN))
    next_generation(
        dataset,
        tracking_member,
        0.4,
        tracking_options.maxsize,
        tracking_options;
        tmp_recorder=RecordType(),
        plugin_states=temperature_states,
    )
    @test getindex.(temperature_states) == (0.4, 0.4)
end
