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

@testitem "Test versioned default profile selection" begin
    using SymbolicRegression
    using SymbolicRegression.CoreModule.OptionsModule: default_options

    v1_defaults = default_options(v"1.0.0")
    v2_defaults = default_options(v"2.0.0-alpha")
    current_defaults = default_options()

    @test keys(v1_defaults) == keys(v2_defaults) == keys(current_defaults)
    for field in keys(v1_defaults)
        v1_value = getproperty(v1_defaults, field)
        v2_value = getproperty(v2_defaults, field)
        current_value = getproperty(current_defaults, field)
        if field == :mutation_weights
            v1_value = convert(Vector, v1_value)
            v2_value = convert(Vector, v2_value)
            current_value = convert(Vector, current_value)
        end

        @test isequal(current_value, v2_value)
        if field == :crossover_probability
            @test v1_value == 0.0259
            @test v2_value == 0.20
        else
            @test isequal(v1_value, v2_value)
        end
    end

    @test Options().crossover_probability == 0.20
    @test Options(; defaults=v"1.0.0").crossover_probability == 0.0259
    @test Options(; defaults=v"2.0.0-alpha").adaptive_parsimony_scaling == 20.0
end

@testitem "Test backsolve options" begin
    using SymbolicRegression

    @test Options().backsolve == BacksolveOptions()
    @test Options(; backsolve=nothing).backsolve == BacksolveOptions()
    @test Options(; backsolve=BacksolveOptions(; lambda=0.2)).backsolve.lambda == 0.2
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
