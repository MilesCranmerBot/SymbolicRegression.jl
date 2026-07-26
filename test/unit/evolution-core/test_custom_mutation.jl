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
