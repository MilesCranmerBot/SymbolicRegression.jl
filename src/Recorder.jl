module RecorderModule

using DynamicExpressions: string_tree
using ..CoreModule: RecordType
using ..UtilsModule: json3_write

abstract type AbstractRecorder end

mutable struct JSONLRecorder <: AbstractRecorder
    entries::Vector{RecordType}
    seen_members::Set{String}
    iteration_counts::Dict{String,Int}
    recorder_file::Union{Nothing,String}
end

JSONLRecorder() = JSONLRecorder(RecordType[], Set{String}(), Dict{String,Int}(), nothing)

function JSONLRecorder(recorder_file::AbstractString)
    mkpath(dirname(recorder_file))
    open(recorder_file, "w") do io
        nothing
    end
    return JSONLRecorder(
        RecordType[], Set{String}(), Dict{String,Int}(), String(recorder_file)
    )
end

"Assumes that `options` holds the user options::AbstractOptions"
macro recorder(ex)
    quote
        if $(esc(:options)).use_recorder
            $(esc(ex))
        end
    end
end

recording_entries(recorder::JSONLRecorder) = recorder.entries

function write_recording_entries!(recorder::JSONLRecorder, entries)
    isempty(entries) && return recorder
    if isnothing(recorder.recorder_file)
        append!(recorder.entries, entries)
    else
        open(recorder.recorder_file, "a") do io
            for entry in entries
                json3_write(io, entry)
                write(io, '\n')
            end
        end
    end
    return recorder
end

function record_entry!(recorder::JSONLRecorder, entry::RecordType)
    write_recording_entries!(recorder, (entry,))
end

close_recording!(::JSONLRecorder) = nothing

function next_iteration!(recorder::JSONLRecorder, key::String)
    iteration = get(recorder.iteration_counts, key, -1) + 1
    recorder.iteration_counts[key] = iteration
    return iteration
end

function append_recordings!(dest::JSONLRecorder, src::JSONLRecorder)
    entries_to_write = RecordType[]
    for entry in src.entries
        if get(entry, "kind", nothing) == "member"
            member_id = entry["id"]
            if member_id in dest.seen_members
                continue
            end
            push!(dest.seen_members, member_id)
        end
        push!(entries_to_write, entry)
    end
    write_recording_entries!(dest, entries_to_write)
    union!(dest.seen_members, src.seen_members)
    for (key, iteration) in src.iteration_counts
        dest.iteration_counts[key] = max(get(dest.iteration_counts, key, -1), iteration)
    end
    return dest
end

function record_options!(recorder::JSONLRecorder, options)
    record_entry!(recorder, RecordType("kind" => "options", "options" => "$(options)"))
    return recorder
end

function record_population_iteration!(
    recorder::JSONLRecorder, out::Int, pop::Int, iteration::Int, population_record
)
    key = "out$(out)_pop$(pop)"
    recorder.iteration_counts[key] = max(get(recorder.iteration_counts, key, -1), iteration)
    record_entry!(
        recorder,
        RecordType(
            "kind" => "population",
            "key" => key,
            "iteration" => iteration,
            "data" => population_record,
        ),
    )
    return recorder
end

function ensure_member_recorded!(
    recorder::JSONLRecorder, ref, tree, cost, loss, parent, options
)
    key = string(ref)
    if key ∉ recorder.seen_members
        record_entry!(
            recorder,
            RecordType(
                "kind" => "member",
                "id" => key,
                "tree" => string_tree(tree, options),
                "cost" => cost,
                "loss" => loss,
                "parent" => parent,
            ),
        )
        push!(recorder.seen_members, key)
    end
    return recorder
end

function ensure_member_recorded!(recorder::JSONLRecorder, member, options)
    return ensure_member_recorded!(
        recorder, member.ref, member.tree, member.cost, member.loss, member.parent, options
    )
end

function record_member_event!(recorder::JSONLRecorder, member_ref, event::RecordType)
    record_entry!(
        recorder,
        RecordType(
            "kind" => "member_event", "member_id" => string(member_ref), "event" => event
        ),
    )
    return recorder
end

end
