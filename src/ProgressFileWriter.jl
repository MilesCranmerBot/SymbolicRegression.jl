"""
    ProgressFileWriter

Module for writing progress updates to a file that Python can monitor.
"""
module ProgressFileWriter

using JSON

mutable struct ProgressWriter
    file_path::String
    current::Int
    total::Int
    
    function ProgressWriter(file_path::String, total::Int)
        writer = new(file_path, 0, total)
        # Initialize file
        write_progress(writer, 0)
        return writer
    end
end

"""Write current progress to file."""
function write_progress(writer::ProgressWriter, current::Int)
    writer.current = current
    try
        # Write atomically: avoid readers observing partially-written JSON.
        dir = dirname(writer.file_path)
        mkpath(dir)
        (tmp, io) = mktemp(dir)
        try
            JSON.print(io, Dict("current" => current, "total" => writer.total))
            flush(io)
            close(io)
            mv(tmp, writer.file_path; force=true)
        catch
            try
                close(io)
            catch
            end
            try
                isfile(tmp) && rm(tmp)
            catch
            end
            rethrow()
        end
    catch
        # Ignore write errors
    end
end

"""Update progress based on search state."""
function update_progress!(
    writer::ProgressWriter,
    cycles_elapsed::Int,
    total_cycles::Int,
)
    current = min(cycles_elapsed, writer.total)
    write_progress(writer, current)
end

"""Close and cleanup."""
function close!(writer::ProgressWriter)
    try
        if isfile(writer.file_path)
            rm(writer.file_path)
        end
    catch
        # Ignore cleanup errors
    end
end

end # module
