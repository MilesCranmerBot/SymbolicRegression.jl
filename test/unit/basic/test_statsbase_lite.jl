@testitem "StatsBaseLite" begin
    using SymbolicRegression.StatsBaseLite
    using StatsBase
    using StableRNGs
    using Test

    @testset "parity with StatsBase (seeded draws)" begin
        # (n, k) covering every replace=false poly-algorithm branch:
        # k == 1, k == 2 (samplepair), n < 24k (Fisher-Yates), n >= 24k (self-avoidance)
        for (n, k) in (
            (1, 1),
            (5, 1),
            (2, 2),
            (5, 2),
            (10, 2),
            (23, 2),
            (24, 2),
            (100, 5),
            (30, 30),
            (1000, 40),
        )
            a = collect(1:n)
            @test StatsBaseLite.sample(StableRNG(42), a, k; replace=false) ==
                StatsBase.sample(StableRNG(42), a, k; replace=false)
        end

        # UnitRange input (used for parameter index mutation)
        @test StatsBaseLite.sample(StableRNG(1), 1:97, 9; replace=false) ==
            StatsBase.sample(StableRNG(1), 1:97, 9; replace=false)

        # With replacement
        @test StatsBaseLite.sample(StableRNG(7), [3.5, 2.5, 1.5], 12) ==
            StatsBase.sample(StableRNG(7), [3.5, 2.5, 1.5], 12)

        # Single draw
        @test StatsBaseLite.sample(StableRNG(3), ['a', 'b', 'c']) ==
            StatsBase.sample(StableRNG(3), ['a', 'b', 'c'])

        # Weighted single draws, including zero-weight entries
        w = [0.1, 0.0, 5.0, 2.0, 0.0, 0.3, 1.0]
        r_ours, r_theirs = StableRNG(11), StableRNG(11)
        ours = [StatsBaseLite.sample(r_ours, StatsBaseLite.Weights(w)) for _ in 1:200]
        theirs = [StatsBase.sample(r_theirs, StatsBase.Weights(w)) for _ in 1:200]
        @test ours == theirs
        @test all(i -> i ∉ (2, 5), ours)

        # Weighted draw from a collection
        @test StatsBaseLite.sample(StableRNG(5), 'a':'g', StatsBaseLite.Weights(w)) ==
            StatsBase.sample(StableRNG(5), 'a':'g', StatsBase.Weights(w))
    end

    @testset "contracts" begin
        rng = StableRNG(0)
        a = string.('a':'z')
        s = StatsBaseLite.sample(rng, a, 8; replace=false)
        @test length(s) == 8
        @test length(unique(s)) == 8
        @test all(in(a), s)

        @test_throws ErrorException StatsBaseLite.sample(rng, a, 27; replace=false)
        @test_throws ArgumentError StatsBaseLite.sample(rng, a, 2; ordered=true)
        @test_throws ArgumentError StatsBaseLite.Weights([1.0, Inf])
        @test_throws ArgumentError StatsBaseLite.Weights([1.0, 2.0], Inf)

        w = [0.5, 1.5, 2.0]
        @test StatsBaseLite.Weights(w).sum == 4.0
        @test StatsBaseLite.Weights(w).values === w
    end
end
