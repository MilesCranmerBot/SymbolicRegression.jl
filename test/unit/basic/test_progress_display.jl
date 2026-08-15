@testitem "print_search_state does not round partial iterations up" begin
    using SymbolicRegression
    using SymbolicRegression: print_search_state
    using Suppressor: @capture_out

    populations = 31
    options = Options(; binary_operators=(+,), populations=populations)

    X = [1.0 2.0 3.0]
    y = [2.0, 3.0, 4.0]
    dataset = Dataset(X, y)
    hof = HallOfFame(options, dataset)

    # One iteration = `populations` population-cycles; niterations=1.
    total_cycles = populations
    print_state(cycles_remaining) = @capture_out print_search_state(
        [hof],
        [dataset];
        options,
        equation_speed=Float32[1.0],
        total_cycles=total_cycles,
        cycles_remaining=cycles_remaining,
        head_node_occupation=0.0,
    )

    # A single completed population-cycle is only a partial iteration,
    # so it must not display as a completed iteration.
    partial = print_state([total_cycles - 1])
    @test occursin("Progress: 0 / 1 total iterations", partial)
    @test occursin("(3.226%)", partial)

    # The iteration only counts once all of its population-cycles are done.
    complete = print_state([0])
    @test occursin("Progress: 1 / 1 total iterations", complete)
    @test occursin("(100.000%)", complete)
end
