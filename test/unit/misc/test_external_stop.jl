@testitem "check_stop_fd drains and filters trigger bytes" begin
    using SymbolicRegression.SearchUtilsModule:
        stop_requested,
        stop_fd,
        stop_fd_trigger,
        check_stop_fd,
        check_external_stop,
        _last_stop_poll_ns

    # The poll is throttled; force it for each check below.
    unthrottled_check() = (_last_stop_poll_ns[] = 0; check_external_stop())

    if Sys.iswindows()
        @test !check_stop_fd(Cint(-1))
    else
        fds = Vector{Cint}(undef, 2)
        @test ccall(:pipe, Cint, (Ptr{Cint},), fds) == 0
        r, w = fds
        write_byte(b) =
            ccall(:write, Cssize_t, (Cint, Ref{UInt8}, Csize_t), w, Ref(UInt8(b)), 1)

        try
            # Any byte stops with the default trigger, and the queue drains fully.
            stop_fd_trigger[] = 0x00
            write_byte(14)
            write_byte(14)
            @test check_stop_fd(r)
            @test !check_stop_fd(r)

            # Non-matching bytes are consumed but ignored.
            stop_fd_trigger[] = 0x02
            write_byte(14)
            write_byte(14)
            @test !check_stop_fd(r)

            # A matching byte among noise stops, and nothing is left after.
            write_byte(14)
            write_byte(2)
            write_byte(14)
            @test check_stop_fd(r)
            @test !check_stop_fd(r)

            # check_external_stop latches, and keeps draining while latched.
            stop_requested[] = false
            stop_fd[] = r
            write_byte(2)
            @test unthrottled_check()
            @test stop_requested[]
            write_byte(2)  # second request while latched
            @test unthrottled_check()
            stop_requested[] = false
            @test !unthrottled_check()  # latched-window byte was drained
        finally
            stop_requested[] = false
            stop_fd[] = Cint(-1)
            stop_fd_trigger[] = 0x00
            ccall(:close, Cint, (Cint,), r)
            ccall(:close, Cint, (Cint,), w)
        end
    end
end

@testitem "external stop requests end a search at a cycle boundary" begin
    using SymbolicRegression
    using SymbolicRegression.SearchUtilsModule: stop_requested, stop_fd

    if !Sys.iswindows()
        X = randn(Float32, 2, 64)
        y = X[1, :] .* X[2, :]

        fds = Vector{Cint}(undef, 2)
        @test ccall(:pipe, Cint, (Ptr{Cint},), fds) == 0
        r, w = fds
        try
            stop_fd[] = r
            # Byte written before the search begins: latched at search start
            # (covers stops requested during host-side input preparation).
            ccall(:write, Cssize_t, (Cint, Ref{UInt8}, Csize_t), w, Ref(UInt8(1)), 1)
            options = Options(;
                binary_operators=[+, *], population_size=20, verbosity=0, progress=false
            )
            hof = equation_search(
                X, y; niterations=1_000_000, options, parallelism=:serial
            )
            @test hof isa SymbolicRegression.HallOfFame
            @test stop_requested[]
        finally
            stop_requested[] = false
            stop_fd[] = Cint(-1)
            ccall(:close, Cint, (Cint,), r)
            ccall(:close, Cint, (Cint,), w)
        end
    end
end
