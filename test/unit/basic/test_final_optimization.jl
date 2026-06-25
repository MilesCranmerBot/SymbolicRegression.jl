@testitem "Test final_optimizer_iterations default" begin
    using SymbolicRegression

    options = Options()
    @test options.final_optimizer_iterations == 0
end

@testitem "Test final_optimizer_iterations=0 is no-op" begin
    using SymbolicRegression
    using Random

    Random.seed!(42)
    X = randn(2, 30)
    y = 2.0 .* X[1, :] .+ X[2, :]

    options = Options(;
        binary_operators=[+, -, *],
        unary_operators=[],
        populations=3,
        population_size=33,
        final_optimizer_iterations=0,
        verbosity=0,
    )
    hof = equation_search(X, reshape(y, 1, :); niterations=1, options=options)
    @test sum(hof.exists) > 0
    for i in eachindex(hof.members, hof.exists)
        hof.exists[i] || continue
        @test isfinite(hof.members[i].loss)
    end
end

@testitem "Test final_optimizer_iterations>0 runs final pass" begin
    using SymbolicRegression
    using Random

    Random.seed!(42)
    X = randn(2, 30)
    y = 2.0 .* X[1, :] .+ X[2, :]

    options = Options(;
        binary_operators=[+, -, *],
        unary_operators=[],
        populations=3,
        population_size=33,
        final_optimizer_iterations=40,
        verbosity=0,
    )
    hof = equation_search(X, reshape(y, 1, :); niterations=1, options=options)
    @test sum(hof.exists) > 0
    for i in eachindex(hof.members, hof.exists)
        hof.exists[i] || continue
        @test isfinite(hof.members[i].loss)
    end
end
