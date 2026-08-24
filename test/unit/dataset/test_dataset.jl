@testitem "Dataset mismatched X/y dimensions" begin
    using SymbolicRegression

    # Transposed X (nsamples × nfeatures) should fail early with a clear error
    @test_throws DimensionMismatch Dataset(randn(32, 3), randn(32))
    @test_throws DimensionMismatch Dataset(randn(3, 32), randn(31))
    # Correct shapes still work
    @test Dataset(randn(3, 32), randn(32)) isa Dataset
    @test Dataset(randn(3, 32)) isa Dataset
end

@testitem "Dataset construction" begin
    using SymbolicRegression

    dataset = Dataset(randn(3, 32), randn(Float32, 32); weights=randn(Float32, 32))
    @test typeof(dataset.X) == Matrix{Float64}
    @test typeof(dataset.y) == Vector{Float32}
    @test typeof(dataset.weights) == Vector{Float32}
end

@testitem "Dataset with deprecated kwarg" begin
    using SymbolicRegression
    using DispatchDoctor: allow_unstable
    dataset = allow_unstable() do
        Dataset(randn(ComplexF32, 3, 32), randn(ComplexF32, 32); loss_type=Float64)
    end
    @test dataset isa Dataset{ComplexF32,Float64}
end

@testitem "Vector output dataset" begin
    using SymbolicRegression

    X = randn(Float64, 3, 32)
    y = [ntuple(_ -> randn(Float64), 3) for _ in 1:32]
    dataset = Dataset(X, y)
    @test dataset isa Dataset{Float64,Float64}
    @test dataset.y isa Vector{NTuple{3,Float64}}
end

@testitem "Large dataset warning" begin
    using SymbolicRegression: Dataset, Options, test_dataset_configuration

    options = Options()
    dataset_50000 = Dataset(zeros(1, 50000), zeros(50000))
    dataset_50001 = Dataset(zeros(1, 50001), zeros(50001))

    @test_logs test_dataset_configuration(dataset_50000, options, 1)
    @test_logs test_dataset_configuration(dataset_50001, options, 0)
    @test_logs (:warn, r"more than 50,000") begin
        test_dataset_configuration(dataset_50001, options, 1)
        test_dataset_configuration(dataset_50001, options, 1)
    end
end
