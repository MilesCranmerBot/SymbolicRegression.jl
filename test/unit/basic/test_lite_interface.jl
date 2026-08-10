@testitem "MMI.fit/predict without MLJBase - matrix input" begin
    using SymbolicRegression
    const MMI = SymbolicRegression.MLJInterfaceModule.MMI
    using Test

    @test !any(m -> nameof(m) === :MLJBase, values(Base.loaded_modules))

    X = randn(60, 3)
    y = @. 2 * X[:, 1] + X[:, 2]
    model = SRRegressor(;
        binary_operators=[+, *],
        niterations=2,
        populations=4,
        population_size=20,
        parallelism=:serial,
        progress=false,
        deterministic=true,
        seed=0,
    )
    fitresult, cache, rep = MMI.fit(model, 0, X, y)
    @test rep.equations isa Vector
    @test length(rep.equations) >= 1
    pred = MMI.predict(model, fitresult, X)
    @test pred isa AbstractVector
    @test length(pred) == 60
end

@testitem "MMI.fit/predict without MLJBase - NamedTuple input" begin
    using SymbolicRegression
    const MMI = SymbolicRegression.MLJInterfaceModule.MMI
    using Test

    @test !any(m -> nameof(m) === :MLJBase, values(Base.loaded_modules))

    n = 60
    Xnt = (alpha=randn(n), beta=randn(n))
    y = @. 2 * Xnt.alpha + Xnt.beta
    model = SRRegressor(;
        binary_operators=[+, *],
        niterations=2,
        populations=4,
        population_size=20,
        parallelism=:serial,
        progress=false,
        deterministic=true,
        seed=0,
    )
    fitresult, _, rep = MMI.fit(model, 0, Xnt, y)
    @test any(s -> occursin("alpha", s), rep.equation_strings)
    pred = MMI.predict(model, fitresult, Xnt)
    @test length(pred) == n
end

@testitem "MMI.fit/predict without MLJBase - multitarget NamedTuple y" begin
    using SymbolicRegression
    const MMI = SymbolicRegression.MLJInterfaceModule.MMI
    using Test

    @test !any(m -> nameof(m) === :MLJBase, values(Base.loaded_modules))

    n = 60
    X = randn(n, 2)
    ynt = (p=X[:, 1] .* 2, q=X[:, 2] .+ 1)
    model = MultitargetSRRegressor(;
        binary_operators=[+, *],
        niterations=2,
        populations=4,
        population_size=20,
        parallelism=:serial,
        progress=false,
        deterministic=true,
        seed=0,
    )
    fitresult, _, rep = MMI.fit(model, 0, X, ynt)
    @test length(rep.equations) == 2
    pred = MMI.predict(model, fitresult, X)
    @test pred isa NamedTuple
    @test keys(pred) == (:p, :q)
    @test length(pred.p) == n
end

@testitem "lite machine interface without MLJBase" begin
    using SymbolicRegression
    using SymbolicRegression: machine, fit!, predict, report
    using Test

    @test !any(m -> nameof(m) === :MLJBase, values(Base.loaded_modules))

    n = 60
    X = randn(n, 3)
    y = @. 2 * X[:, 1] + X[:, 2]
    model = SRRegressor(;
        binary_operators=[+, *],
        niterations=2,
        populations=4,
        population_size=20,
        parallelism=:serial,
        progress=false,
        deterministic=true,
        seed=0,
    )

    mach = machine(model, X, y)

    @test_throws ArgumentError predict(mach)
    @test_throws ArgumentError report(mach)

    @test fit!(mach; verbosity=0) === mach
    rep = report(mach)
    @test rep.equations isa Vector
    @test rep.best_idx isa Integer
    @test length(predict(mach)) == n
    @test length(predict(mach, X[1:10, :])) == 10

    # warm start: second fit! goes through MMI.update, reusing cached
    # types (identity) and extending the search to the new niterations
    types_before = mach.fitresult.types
    mach.model.niterations = 4
    fit!(mach; verbosity=0)
    @test mach.fitresult.types === types_before
    @test mach.fitresult.niterations == 4
end

@testitem "lite machine interface - weights and NamedTuple" begin
    using SymbolicRegression
    using SymbolicRegression: machine, fit!, predict, report
    using Test

    @test !any(m -> nameof(m) === :MLJBase, values(Base.loaded_modules))

    n = 60
    Xnt = (alpha=randn(n), beta=randn(n))
    y = @. 2 * Xnt.alpha + Xnt.beta
    w = ones(n)
    model = SRRegressor(;
        binary_operators=[+, *],
        niterations=2,
        populations=4,
        population_size=20,
        parallelism=:serial,
        progress=false,
        deterministic=true,
        seed=0,
    )
    mach = machine(model, Xnt, y, w)
    fit!(mach; verbosity=0)
    @test any(s -> occursin("alpha", s), report(mach).equation_strings)
    @test length(predict(mach, Xnt)) == n
end

@testitem "lite machine interface - row table (Vector of NamedTuples)" begin
    using SymbolicRegression
    using SymbolicRegression: machine, fit!, predict, report
    using Test

    @test !any(m -> nameof(m) === :MLJBase, values(Base.loaded_modules))

    n = 60
    rows = [(alpha=randn(), beta=randn()) for _ in 1:n]
    y = [2 * r.alpha + r.beta for r in rows]
    model = SRRegressor(;
        binary_operators=[+, *],
        niterations=2,
        populations=4,
        population_size=20,
        parallelism=:serial,
        progress=false,
        deterministic=true,
        seed=0,
    )
    mach = machine(model, rows, y)
    fit!(mach; verbosity=0)
    @test any(s -> occursin("alpha", s), report(mach).equation_strings)
    @test length(predict(mach, rows)) == n
end
