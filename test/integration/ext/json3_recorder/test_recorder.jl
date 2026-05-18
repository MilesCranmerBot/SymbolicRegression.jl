@testitem "Test JSON3 recorder" begin
    using SymbolicRegression
    using SymbolicRegression.UtilsModule: recursive_merge
    using JSON3
    include(joinpath(@__DIR__, "..", "..", "..", "test_params.jl"))

    base_dir = mktempdir()
    recorder_file = joinpath(base_dir, "pysr_recorder.jsonl")

    let
        stream_file = joinpath(base_dir, "stream_probe.jsonl")
        head = SymbolicRegression.RecorderModule.JSONLRecorder(stream_file)
        worker = SymbolicRegression.RecorderModule.JSONLRecorder()
        member_entry = SymbolicRegression.CoreModule.RecordType(
            "kind" => "member",
            "id" => "42",
            "tree" => "x1",
            "cost" => 1.0,
            "loss" => 1.0,
            "parent" => nothing,
        )
        push!(SymbolicRegression.RecorderModule.recording_entries(worker), member_entry)
        SymbolicRegression.RecorderModule.record_member_event!(
            worker,
            "42",
            SymbolicRegression.CoreModule.RecordType("type" => "death", "time" => 0.0),
        )
        SymbolicRegression.RecorderModule.append_recordings!(head, worker)
        @test isempty(SymbolicRegression.RecorderModule.recording_entries(head))
        @test length(readlines(stream_file)) == 2

        duplicate = SymbolicRegression.RecorderModule.JSONLRecorder()
        push!(SymbolicRegression.RecorderModule.recording_entries(duplicate), member_entry)
        SymbolicRegression.RecorderModule.append_recordings!(head, duplicate)
        @test length(readlines(stream_file)) == 2
    end
    X = 2 .* randn(Float32, 2, 1000)
    y = 3 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

    options = SymbolicRegression.Options(;
        binary_operators=(+, *, /, -),
        unary_operators=(cos,),
        use_recorder=true,
        recorder_file=recorder_file,
        populations=2,
        population_size=100,
        maxsize=20,
        complexity_of_operators=[cos => 2],
    )

    hall_of_fame = equation_search(
        X, y; niterations=5, options=options, parallelism=:multithreading
    )

    lines = readlines(options.recorder_file)
    data = [JSON3.read(line; allow_inf=true) for line in lines]

    option_entries = filter(entry -> entry.kind == "options", data)
    population_entries = filter(entry -> entry.kind == "population", data)
    member_entries = filter(entry -> entry.kind == "member", data)
    member_event_entries = filter(entry -> entry.kind == "member_event", data)

    @test length(option_entries) == 1
    @test any(entry -> entry.key == "out1_pop1", population_entries)
    @test any(entry -> entry.key == "out1_pop2", population_entries)
    @test !isempty(member_entries)
    @test !isempty(member_event_entries)

    # Test that "Options" is part of the string in the options entry:
    @test contains(only(option_entries).options, "Options")
    @test length(member_entries) > 1000

    for member in Iterators.take(member_entries, 10)
        @test haskey(member, :id)
        @test haskey(member, :cost)
        @test haskey(member, :tree)
        @test haskey(member, :loss)
        @test haskey(member, :parent)
    end

    @test_throws ErrorException recursive_merge()
end
