module SymbolicRegressionJSON3Ext

using JSON3: JSON3
import SymbolicRegression.UtilsModule: json3_write

function json3_write(trace, tracing_file)
    open(tracing_file, "w") do io
        JSON3.write(io, trace; allow_inf=true)
    end
end

end
