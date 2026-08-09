# This file tests particular functionality of ExpressionBuilderModule
@testitem "NamedTuple support in parse_expression" begin
    using SymbolicRegression
    using DynamicExpressions

    # Test basic NamedTuple parsing for template expressions
    operators = OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[cos, sin])
    variable_names = ["x1", "x2"]

    # Create a simple template using @template_spec macro
    template = @template_spec(expressions = (f, g)) do x1, x2
        f(x1, x2) * g(x1, x2)  # Simple multiplication combination
    end
    options = Options(; operators, expression_spec=template)

    # Test NamedTuple parsing with expression_options using #N placeholder syntax
    named_tuple_input = (; f="#1 + 1.0", g="#2 - 0.5")
    result = parse_expression(
        named_tuple_input;
        options.expression_options,
        operators,
        expression_type=TemplateExpression,
        node_type=Node{Float64,2},
    )

    @test result isa TemplateExpression
    @test result.trees.f isa ComposableExpression
    @test result.trees.g isa ComposableExpression
    @test length(result.trees) == 2
    @test keys(result.trees) == (:f, :g)

    # Test NamedTuple parsing with expression_spec
    result_with_spec = parse_expression(
        named_tuple_input; expression_spec=template, operators, node_type=Node{Float64,2}
    )

    @test result_with_spec isa TemplateExpression
    @test typeof(result_with_spec) == typeof(result)

    # Test that different expression strings create different expressions using #N syntax
    different_input = (; f="cos(#1)", g="sin(#2)")
    different_result = parse_expression(
        different_input;
        options.expression_options,
        operators,
        expression_type=TemplateExpression,
        node_type=Node{Float64},
    )

    @test different_result isa TemplateExpression
    @test different_result.trees.f isa ComposableExpression
    @test different_result.trees.g isa ComposableExpression
end
