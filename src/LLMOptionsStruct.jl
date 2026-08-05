module LLMOptionsStructModule

using PromptingTools: aigenerate
using SymbolicRegression: AbstractMutation, AbstractOptions, AbstractPlugin, Options
using ..LaSRMutationWeightsModule: LaSRMutationWeights
using ..LoggingModule: LaSRLogger

const DEFAULT_PROMPTS_DIR = joinpath(pkgdir(parentmodule(@__MODULE__)), "prompts") * "/"

Base.@kwdef mutable struct LLMOperationWeights
    llm_crossover::Float64 = 0.0
    llm_mutate::Float64 = 0.0
    llm_randomize::Float64 = 0.0
end

Base.@kwdef mutable struct LLMOptions
    use_llm::Bool = true
    use_concepts::Bool = false
    use_concept_evolution::Bool = false
    mutation_weights::LaSRMutationWeights = LaSRMutationWeights()
    llm_operation_weights::LLMOperationWeights = LLMOperationWeights()
    num_pareto_context::Int = 5
    num_generated_equations::Int = 5
    num_generated_concepts::Int = 5
    num_concept_crossover::Int = 2
    max_concepts::Int = 30
    is_parametric::Bool = false
    llm_context::String = ""
    variable_names::Union{Dict,Nothing} = nothing
    prompts_dir::String = DEFAULT_PROMPTS_DIR
    idea_database::Vector{AbstractString} = AbstractString[]
    api_key::Union{String,Nothing} = nothing
    model::Union{String,Nothing} = nothing
    api_kwargs::Dict = Dict("max_tokens" => 1000)
    http_kwargs::Dict = Dict("retries" => 3, "readtimeout" => 3600)
    lasr_logger::Union{LaSRLogger,Nothing} = nothing
    verbose::Bool = true
    llm_generate::Function = aigenerate
end

struct LLMMutateMutation <: AbstractMutation end
struct LLMRandomizeMutation <: AbstractMutation end

"""
    LaSRPlugin(; llm_options=LLMOptions(), llm_mutate_weight, llm_randomize_weight,
                 llm_crossover_probability)

Library-augmented symbolic regression plugin. Pass it through
`Options(; plugins=(LaSRPlugin(...),), ...)`. The two mutation weights are
unnormalized, like all entries in `Options.mutations`; crossover is a
probability conditional on SR selecting crossover.
"""
struct LaSRPlugin <: AbstractPlugin
    llm_options::LLMOptions
    llm_mutate_weight::Float64
    llm_randomize_weight::Float64
    llm_crossover_probability::Float64
    function LaSRPlugin(;
        llm_options::LLMOptions=LLMOptions(),
        llm_mutate_weight::Real=llm_options.mutation_weights.llm_mutate,
        llm_randomize_weight::Real=llm_options.mutation_weights.llm_randomize,
        llm_crossover_probability::Real=llm_options.llm_operation_weights.llm_crossover,
    )
        mutate_weight = Float64(llm_mutate_weight)
        randomize_weight = Float64(llm_randomize_weight)
        crossover_probability = Float64(llm_crossover_probability)
        mutate_weight >= 0 || throw(ArgumentError("`llm_mutate_weight` must be nonnegative."))
        randomize_weight >= 0 || throw(ArgumentError("`llm_randomize_weight` must be nonnegative."))
        0 <= crossover_probability <= 1 ||
            throw(ArgumentError("`llm_crossover_probability` must be between 0 and 1."))
        return new(llm_options, mutate_weight, randomize_weight, crossover_probability)
    end
end

mutable struct LaSRPluginState
    idea_database::Vector{AbstractString}
    lasr_logger::Union{LaSRLogger,Nothing}
    variable_names::Dict
    generations::Int
    worst_members::Vector{Any}
end

struct LaSRContext{O<:Options,S} <: AbstractOptions
    sr_options::O
    llm_options::LLMOptions
    state::S
end

const LLM_OPTIONS_KEYS = fieldnames(LLMOptions)

function Base.getproperty(context::LaSRContext, key::Symbol)
    if key in (:sr_options, :llm_options, :state)
        return getfield(context, key)
    elseif key === :idea_database && !isnothing(getfield(context, :state))
        return getfield(context, :state).idea_database
    elseif key === :lasr_logger && !isnothing(getfield(context, :state))
        return getfield(context, :state).lasr_logger
    elseif key === :variable_names && !isnothing(getfield(context, :state))
        return getfield(context, :state).variable_names
    elseif key in LLM_OPTIONS_KEYS
        return getproperty(getfield(context, :llm_options), key)
    else
        return getproperty(getfield(context, :sr_options), key)
    end
end

end
