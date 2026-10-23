@testitem "Test complexity evaluation" begin
    using SymbolicRegression

    x1, x2, x3 = Node("x1"), Node("x2"), Node("x3")

    # First, test regular complexities:
    function make_options(; kw...)
        return Options(;
            binary_operators=(+, -, *, /, ^), unary_operators=(cos, sin), kw...
        )
    end
    options = make_options()
    @extend_operators options
    tree = sin((x1 + x2 + x3)^2.3)
    @test compute_complexity(tree, options) == 8

    options = make_options(; complexity_of_operators=[sin => 3])
    @test compute_complexity(tree, options) == 10
    options = make_options(; complexity_of_operators=[sin => 3, (+) => 2])
    @test compute_complexity(tree, options) == 12

    # Real numbers:
    options = make_options(; complexity_of_operators=[sin => 3, (+) => 2, (^) => 3.2])
    @test compute_complexity(tree, options) == round(Int, 12 + (3.2 - 1))
end

@testitem "Test other things about complexity" begin
    using SymbolicRegression

    x1, x2, x3 = Node("x1"), Node("x2"), Node("x3")

    function make_options(; kw...)
        return Options(;
            binary_operators=(+, -, *, /, ^), unary_operators=(cos, sin), kw...
        )
    end

    options = make_options(;
        complexity_of_operators=[sin => 3, (+) => 2], complexity_of_variables=2
    )
    tree = sin((x1 + x2 + x3)^2.3)
    @test compute_complexity(tree, options) == 12 + 3 * 1
    options = make_options(;
        complexity_of_operators=[sin => 3, (+) => 2],
        complexity_of_variables=2,
        complexity_of_constants=2,
    )
    @test compute_complexity(tree, options) == 12 + 3 * 1 + 1
    options = make_options(;
        complexity_of_operators=[sin => 3, (+) => 2],
        complexity_of_variables=2,
        complexity_of_constants=2.6,
    )
    @test compute_complexity(tree, options) == 12 + 3 * 1 + 1 + 1

    # Custom variables
    options = make_options(;
        complexity_of_variables=[1, 2, 3], complexity_of_operators=[(+) => 5, (*) => 2]
    )
    x1, x2, x3 = [Node{Float64}(; feature=i) for i in 1:3]
    tree = x1 + x2 * x3
    @test compute_complexity(tree, options) == 1 + 5 + 2 + 2 + 3
    options = make_options(;
        complexity_of_variables=2, complexity_of_operators=[(+) => 5, (*) => 2]
    )
    @test compute_complexity(tree, options) == 2 + 5 + 2 + 2 + 2
end

@testitem "Custom complexity mapping" begin
    using SymbolicRegression

    function custom_complexity(tree)
        @test tree isa AbstractExpression
        return 10
    end

    options = Options(; complexity_mapping=custom_complexity)
    variable_names = ["x1"]
    x1 = Expression(Node{Float64}(; feature=1); options.operators, variable_names)
    @test compute_complexity(x1, options) == 10
end

@testitem "Operator-keyed options with opmap-replaced operators" begin
    # https://github.com/astroautomata/SymbolicRegression.jl/issues/781
    using SymbolicRegression
    using SymbolicRegression: safe_pow, safe_sqrt, safe_log

    operators = OperatorEnum(1 => (cos, sqrt, log), 2 => (+, *, ^))
    for (pow_op, sqrt_op, log_op) in ((^, sqrt, log), (safe_pow, safe_sqrt, safe_log))
        complexity_of_operators = [pow_op => 3, (*) => 2, sqrt_op => 4, log_op => 5]
        options = Options(;
            operators,
            complexity_of_operators,
            constraints=Dict(1 => [sqrt_op => 5], 2 => [pow_op => (-1, 1)]),
            nested_constraints=[pow_op => [pow_op => 0], log_op => [log_op => 0]],
        )
        @test options.operators.ops == ((cos, safe_sqrt, safe_log), (+, *, safe_pow))
        @test options.complexity_mapping.op_complexities == ([1, 4, 5], [1, 2, 3])
        @test options.op_constraints == ([-1, 5, -1], [(-1, -1), (-1, -1), (-1, 1)])
        @test sort(options.nested_constraints) == [(1, 3, [(1, 3, 0)]), (2, 3, [(2, 3, 0)])]

        legacy = Options(;
            unary_operators=(cos, sqrt, log),
            binary_operators=(+, *, ^),
            complexity_of_operators,
            constraints=[sqrt_op => 5, pow_op => (-1, 1)],
        )
        @test legacy.complexity_mapping.op_complexities == ([1, 4, 5], [1, 2, 3])
        @test legacy.op_constraints == options.op_constraints
    end
end
