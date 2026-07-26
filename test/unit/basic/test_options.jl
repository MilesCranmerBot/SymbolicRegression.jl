@testitem "Test options" begin
    using SymbolicRegression
    using Optim: Optim

    # testing types
    op = Options(; optimizer_options=(iterations=16, f_calls_limit=100, x_abstol=1e-16))
    @test isa(op.optimizer_options, Optim.Options)

    op = Options(;
        optimizer_options=Dict(:iterations => 32, :g_calls_limit => 50, :f_reltol => 1e-16)
    )
    @test isa(op.optimizer_options, Optim.Options)

    optim_op = Optim.Options(; iterations=16)
    op = Options(; optimizer_options=optim_op)
    @test isa(op.optimizer_options, Optim.Options)

    # testing loss_scale parameter
    op_log = Options(; loss_scale=:log)
    @test op_log.loss_scale == :log

    op_linear = Options(; loss_scale=:linear)
    @test op_linear.loss_scale == :linear

    # test that invalid loss_scale values are caught
    @test_throws AssertionError Options(; loss_scale=:invalid)
    @test_throws AssertionError Options(; loss_scale=:cubic)
end

@testitem "Test backsolve options" begin
    using SymbolicRegression
    using SymbolicRegression.BacksolveModule: configured_backsolve

    # Backsolve configuration lives on BacksolveMutation.
    default_backsolve = first(
        p.first for p in Options().mutations if p.first isa BacksolveMutation
    )
    @test default_backsolve == BacksolveMutation()

    custom = Options(;
        default_mutations=(), mutations=[BacksolveMutation(; lambda=0.2) => 1.0]
    )
    @test length(custom.mutations) == 1
    @test custom.mutations[1].first.lambda == 0.2

    with_override = Options(; mutations=(BacksolveMutation(; lambda=0.3) => 1.0,))
    @test length(with_override.mutations) == length(default_mutations())
    @test first(with_override.mutations) == (BacksolveMutation(; lambda=0.3) => 1.0)
    @test count(p -> p.first isa BacksolveMutation, with_override.mutations) == 1
    @test configured_backsolve(with_override).lambda == 0.3
    @test_throws ArgumentError configured_backsolve(Options(; default_mutations=()))

    @test isempty(Options(; default_mutations=()).mutations)
    @test default_mutations() == SymbolicRegression.default_mutations()

    @test_throws ArgumentError Options(;
        mutation_weights=MutationWeights(), default_mutations=()
    )
end

@testitem "Mutation collections mirror plugin override semantics" begin
    using SymbolicRegression

    struct ProbeMutation <: AbstractMutation
        value::Int
    end

    disabled = Options(; mutations=[ConstantMutation() => 0.0])
    @test first(disabled.mutations) == (ConstantMutation() => 0.0)
    @test count(p -> p.first isa ConstantMutation, disabled.mutations) == 1
    @test length(disabled.mutations) == length(default_mutations())

    added = Options(; mutations=[ProbeMutation(1) => 0.5])
    @test first(added.mutations) == (ProbeMutation(1) => 0.5)
    @test length(added.mutations) == length(default_mutations()) + 1

    no_defaults = Options(; default_mutations=(), mutations=[ProbeMutation(2) => 1.0])
    @test no_defaults.mutations == [ProbeMutation(2) => 1.0]

    custom_defaults = Options(;
        mutations=[ProbeMutation(3) => 0.25],
        default_mutations=[ProbeMutation(4) => 0.75, ConstantMutation() => 1.0],
    )
    @test custom_defaults.mutations == [ProbeMutation(3) => 0.25, ConstantMutation() => 1.0]
end

@testitem "Test operators parameter conflicts" begin
    using SymbolicRegression
    using DynamicExpressions: OperatorEnum

    # Test that when operators is provided, we can't also provide individual sets
    operators = OperatorEnum(1 => (sin, cos), 2 => (+, *, -))
    @test_throws AssertionError Options(; operators, binary_operators=(+, *))
    @test_throws AssertionError Options(; operators, unary_operators=(sin,))

    # Test that when operators is provided, operator_enum_constructor should be nothing
    @test_throws AssertionError Options(; operators, operator_enum_constructor=OperatorEnum)

    # Test that providing operators alone works fine (should not throw)
    @test_nowarn Options(; operators)
end

@testitem "Test operators stored globally" begin
    using SymbolicRegression
    using DynamicExpressions.OperatorEnumConstructionModule: LATEST_OPERATORS

    operators = OperatorEnum(1 => [sin, cos], 2 => [+, -, *], 3 => [fma], 5 => [max])
    options = Options(; operators)

    @test LATEST_OPERATORS[] == operators
end

@testitem "Test with_max_degree_from_context" begin
    using SymbolicRegression

    operators = OperatorEnum(1 => (sin, cos), 2 => (+, *, -))
    @test Options(; node_type=GraphNode, operators).node_type <: GraphNode{<:Any,2}
    @test Options(; node_type=Node, operators).node_type <: Node{<:Any,2}

    operators = OperatorEnum(1 => (sin, cos), 2 => ())
    @test Options(; node_type=Node{<:Any,1}, operators).node_type <: Node{<:Any,1}

    @test Options().node_type <: Node{<:Any,2}

    operators = OperatorEnum(1 => (sin, cos), 2 => (+, *, -), 3 => (fma, max))
    options = Options(; operators)
    @test options.node_type <: Node{<:Any,3}
    @test options.op_constraints ==
        ([-1, -1], [(-1, -1), (-1, -1), (-1, -1)], [(-1, -1, -1), (-1, -1, -1)])
    @test options.nops == (2, 3, 2)
end

@testitem "Test operator appears in multiple degrees error" begin
    using SymbolicRegression

    operators = OperatorEnum(1 => (+, sin), 2 => (+, *))  # + appears in both degrees

    @test_throws(
        "Operator + appears in multiple degrees. You can't use nested constraints.",
        Options(; operators, nested_constraints=[(+) => [(+) => 0]])
    )

    @test_throws(
        "Operator + appears in multiple degrees. You can't use constraints.",
        Options(; operators, constraints=[(+) => -1])
    )
end

@testitem "Test build_constraints with pre-processed vector format" begin
    using SymbolicRegression
    using SymbolicRegression.CoreModule.OptionsModule: build_constraints
    using DynamicExpressions: OperatorEnum

    operators = OperatorEnum(1 => (sin, cos), 2 => (+, *, -), 5 => (max,))

    constraints_processed = (
        [-1, -1], [(-1, -1), (-1, -1), (-1, -1)], nothing, nothing, [(-1, -1, -1, -1, -1)]
    )

    result = build_constraints(;
        constraints=constraints_processed, operators_by_degree=operators.ops
    )

    # Verify the result matches expected format (fills empty slots with default values)
    @test result == (
        [-1, -1],
        [(-1, -1), (-1, -1), (-1, -1)],
        NTuple{3,Int}[],
        NTuple{4,Int}[],
        [(-1, -1, -1, -1, -1)],
    )
end
