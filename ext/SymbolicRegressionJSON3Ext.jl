module SymbolicRegressionJSON3Ext

using JSON3: JSON3
import SymbolicRegression.CoreModule: RecordType
import SymbolicRegression.UtilsModule: json3_write
import SymbolicRegression.RecorderModule: JSONLRecorder, recording_entries

json3_write(io::IO, entry::RecordType) = JSON3.write(io, entry; allow_inf=true)

function json3_write(record::JSONLRecorder, recorder_file; append::Bool=false)
    open(recorder_file, append ? "a" : "w") do io
        for entry in recording_entries(record)
            json3_write(io, entry)
            write(io, '\n')
        end
    end
end

end
