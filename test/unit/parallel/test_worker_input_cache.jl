@testitem "Multiprocessing reuses caller-owned workers with fresh search inputs" begin
    using Distributed
    using SymbolicRegression
    using Test

    defs = quote
        function worker_cache_loss(ex::AbstractExpression, dataset::Dataset, options::Options)
            return sum(abs2, dataset.y) / dataset.n
        end
    end
    if (@__MODULE__) != Core.Main
        Core.eval(Core.Main, :(using SymbolicRegression))
        Core.eval(Core.Main, defs)
        eval(:(using Main: worker_cache_loss))
    else
        eval(defs)
    end

    procs = addprocs(2)
    try
        @everywhere procs using SymbolicRegression
        X = reshape(Float32.(1:16), 1, :)
        options = Options(;
            binary_operators=(+, *),
            loss_function_expression=worker_cache_loss,
            populations=2,
            population_size=12,
            tournament_selection_n=3,
            ncycles_per_iteration=1,
            maxsize=5,
            save_to_file=false,
            verbosity=0,
            progress=false,
        )
        for scale in (1.0f0, 3.0f0)
            y = scale .* vec(X)
            populations, _ = equation_search(
                X,
                y;
                options,
                niterations=1,
                parallelism=:multiprocessing,
                procs,
                return_state=true,
            )
            expected_loss = sum(abs2, y) / length(y)
            @test minimum(
                member.loss for pop in only(populations) for member in pop.members
            ) ≈ expected_loss
        end
    finally
        rmprocs(procs)
    end
end
