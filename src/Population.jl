module PopulationModule

using StatsBase: StatsBase
using DispatchDoctor: @unstable
using DynamicExpressions: AbstractExpression, Expression, Node, constructorof
using Serialization: Serialization
using Distributed: ClusterSerializer
using ..CoreModule:
    AbstractOptions,
    Options,
    Dataset,
    DATA_TYPE,
    LOSS_TYPE,
    init_member,
    resolve_init_member,
    tournament_cost_multiplier,
    use_batching
using ..LossFunctionsModule: eval_cost, update_baseline_loss!
using ..MutationFunctionsModule: gen_random_tree
using ..PopMemberModule: AbstractPopMember, PopMember, _pack_node!, _unpack_node
import ..PopMemberModule: popmember_type
using ..UtilsModule: bottomk_fast, PerTaskCache, strictmap
# A list of members of the population, with easy constructors,
#  which allow for random generation of new populations
struct Population{
    T<:DATA_TYPE,L<:LOSS_TYPE,N<:AbstractExpression{T},PM<:AbstractPopMember{T,L,N}
}
    members::Array{PM,1}
    n::Int
end
"""
    Population(pop::Array{<:AbstractPopMember, 1})

Create population from list of PopMembers.
"""
function Population(pop::Vector{<:AbstractPopMember})
    return Population(pop, size(pop, 1))
end

"""
    _init_tree(dataset, options, nlength, nfeatures, ::Type{T}, plugin_states)

Initialize a tree for a new population member. Asks every plugin via
[`resolve_init_member`](@ref); at most one may return a non-`nothing`
expression (two or more providers throw). If all plugins return `nothing`
(the common case — no plugin overrides `init_member`), falls back to
`gen_random_tree`.
"""
function _init_tree(
    dataset, options, nlength::Int, nfeatures::Int, ::Type{T}, plugin_states::Tuple
) where {T}
    return @something(
        resolve_init_member(plugin_states, options.plugins, dataset, options),
        gen_random_tree(nlength, options, nfeatures, T),
    )
end

"""
    Population(dataset::Dataset{T,L};
               population_size, nlength::Int=3, options::AbstractOptions,
               nfeatures::Int, plugin_states::Tuple)

Create random population and evaluate them on the dataset.
"""
function Population(
    dataset::Dataset{T,L};
    options::AbstractOptions,
    population_size=nothing,
    nlength::Int=3,
    nfeatures::Int,
    npop=nothing,
    plugin_states::Tuple,
) where {T,L}
    @assert (population_size !== nothing) ⊻ (npop !== nothing)
    population_size = something(population_size, npop)
    PM = options.popmember_type

    # Create first member to get concrete type
    first_member = constructorof(PM)(
        dataset,
        _init_tree(dataset, options, nlength, nfeatures, T, plugin_states),
        options;
        parent=-1,
        deterministic=options.deterministic,
    )

    # Use the concrete type for the array
    members = typeof(first_member)[
        if i == 1
            first_member
        else
            constructorof(PM)(
                dataset,
                _init_tree(dataset, options, nlength, nfeatures, T, plugin_states),
                options;
                parent=-1,
                deterministic=options.deterministic,
            )
        end for i in 1:population_size
    ]

    return Population(members, population_size)
end

function _population_without_plugins(
    dataset::Dataset{T,L}; options::AbstractOptions, nlength::Int=3, nfeatures::Int
) where {T,L}
    PM = options.popmember_type
    member = constructorof(PM)(
        dataset,
        gen_random_tree(nlength, options, nfeatures, T),
        options;
        parent=-1,
        deterministic=options.deterministic,
    )
    return Population([member])
end

"""
    Population(X::AbstractMatrix{T}, y::AbstractVector{T};
               population_size, nlength::Int=3,
               options::AbstractOptions, nfeatures::Int,
               loss_type::Type=Nothing, plugin_states::Tuple)

Create random population and score them on the dataset.
"""
@unstable function Population(
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    population_size=nothing,
    nlength::Int=3,
    options::AbstractOptions,
    nfeatures::Int,
    loss_type::Type{L}=Nothing,
    npop=nothing,
    plugin_states::Tuple,
) where {T<:DATA_TYPE,L}
    @assert (population_size !== nothing) ⊻ (npop !== nothing)
    population_size = if npop === nothing
        population_size
    else
        npop
    end
    dataset = Dataset(X, y, L)
    update_baseline_loss!(dataset, options)
    return Population(dataset; population_size, options, nfeatures, plugin_states)
end

function Base.copy(pop::P)::P where {T,L,N,PM,P<:Population{T,L,N,PM}}
    copied_members = Vector{PM}(undef, pop.n)
    for i in 1:(pop.n)
        copied_members[i] = copy(pop.members[i])
    end
    return Population(copied_members)
end

# `n` distinct indices in `1:N` by rejection; allocation-free for tournament sizes.
const TOURNAMENT_INDICES_SCRATCH = PerTaskCache{Vector{Int}}()
function _sample_indices!(idxs::Vector{Int}, N::Int, n::Int)
    resize!(idxs, n)
    for i in 1:n
        candidate = rand(1:N)
        while candidate in view(idxs, 1:(i - 1))
            candidate = rand(1:N)
        end
        idxs[i] = candidate
    end
    return idxs
end

"""
    best_of_sample(pop, options; plugin_states)

Sample a tournament from the population and return its winner. The winner is
the population's own member, not a copy: callers that insert it back into a
population must copy it themselves.
"""
function best_of_sample(
    pop::Population{T,L,N}, options::AbstractOptions; plugin_states::Tuple
) where {T,L,N}
    idxs = _sample_indices!(
        TOURNAMENT_INDICES_SCRATCH[], pop.n, options.tournament_selection_n
    )
    return _best_of_sample(pop.members, idxs, options; plugin_states)
end
function _best_of_sample(
    members::Vector{P}, idxs::Vector{Int}, options::AbstractOptions; plugin_states::Tuple
) where {T,L,N,P<:AbstractPopMember{T,L,N}}
    p = options.tournament_selection_p
    n = length(idxs)  # == tournament_selection_n
    function adjusted_cost(i)
        member = members[idxs[i]]
        multipliers = strictmap(options.plugins, plugin_states) do plugin, pstate
            return L(tournament_cost_multiplier(pstate, plugin, member, options))
        end
        L(member.cost) * prod(multipliers)
    end

    # First, decide what place we take (usually 1st place wins):
    tournament_winner =
        p == 1.0 ? 1 : StatsBase.sample(get_tournament_selection_weights(options))
    chosen_idx = if tournament_winner == 1
        # Strict `<`: the first minimum wins, and a NaN cost never does.
        best_i, best_cost = 1, adjusted_cost(1)
        for i in 2:n
            cost = adjusted_cost(i)
            cost < best_cost && ((best_i, best_cost) = (i, cost))
        end
        best_i
    else
        # Then, find the member that won that place, given their fitness:
        bottomk_fast(L[adjusted_cost(i) for i in 1:n], tournament_winner)[2][end]
    end
    return members[idxs[chosen_idx]]
end
_get_cost(member::AbstractPopMember) = member.cost

const CACHED_WEIGHTS =
    let init_k = collect(0:5),
        init_prob_each = 0.5 * (1 - 0.5) .^ init_k,
        test_weights = StatsBase.Weights(init_prob_each, sum(init_prob_each))

        PerTaskCache{Dict{Tuple{Int,Float64},typeof(test_weights)}}()
    end

@unstable function get_tournament_selection_weights(@nospecialize(options::AbstractOptions))
    n = options.tournament_selection_n::Int
    p = options.tournament_selection_p::Float64
    # Computing the weights for the tournament becomes quite expensive,
    return get!(CACHED_WEIGHTS[], (n, p)) do
        k = collect(0:(n - 1))
        prob_each = p * ((1 - p) .^ k)

        return StatsBase.Weights(prob_each, sum(prob_each))
    end
end

function finalize_costs(
    dataset::Dataset{T,L}, pop::P, options::AbstractOptions
)::Tuple{P,Float64} where {T,L,P<:Population{T,L}}
    need_recalculate = use_batching(options, dataset)
    num_evals = 0.0
    if need_recalculate
        for member in 1:(pop.n)
            cost, loss = eval_cost(dataset, pop.members[member], options)
            pop.members[member].cost = cost
            pop.members[member].loss = loss
        end
        num_evals += pop.n
    end
    return (pop, num_evals)
end

# Return best 10 examples
function best_sub_pop(pop::P; topn::Int=10)::P where {P<:Population}
    best_idx = sortperm([pop.members[member].cost for member in 1:(pop.n)])
    # Ensure we don't try to access more elements than exist in the population
    actual_topn = min(topn, pop.n)
    return Population(pop.members[best_idx[1:actual_topn]])
end

# Type accessor for Population
popmember_type(::Type{<:Population{T,L,N,PM}}) where {T,L,N,PM} = PM

function _can_pack_members(s::ClusterSerializer, members::Vector{PM}) where {PM}
    PM <: PopMember || return false
    E = fieldtype(PM, :tree)
    E <: Expression{<:Any,<:Node} || return false
    isempty(members) && return false
    haskey(s.table, members) && return false
    metadata = getfield(getfield(first(members), :tree), :metadata)
    for i in eachindex(members)
        member = members[i]
        getfield(getfield(member, :tree), :metadata) === metadata || return false
        haskey(s.table, member) && return false
        for j in 1:(i - 1)
            member === members[j] && return false
        end
    end
    return true
end

function _serialize_packed_members(s::ClusterSerializer, members::Vector{PM}) where {PM}
    write(s.io, length(members))
    # Reserve the reference ids that ordinary Vector and PopMember transfers would use.
    Serialization.serialize_cycle(s, members) && error("Repeated packed member vector")
    metadata = getfield(getfield(first(members), :tree), :metadata)
    Serialization.serialize(s, metadata)
    T = PM.parameters[1]
    degrees = UInt8[]
    leaf_types = UInt8[]
    constants = T[]
    features = UInt16[]
    ops = UInt8[]
    for member in members
        _pack_node!(
            getfield(getfield(member, :tree), :tree),
            degrees,
            leaf_types,
            constants,
            features,
            ops,
        )
    end
    for values in (degrees, leaf_types, constants, features, ops)
        Serialization.serialize(s, values)
    end
    for field in fieldnames(PM)
        field === :tree && continue
        values = Vector{fieldtype(PM, field)}(undef, length(members))
        for i in eachindex(members)
            values[i] = getfield(members[i], field)
        end
        Serialization.serialize(s, values)
    end
    for member in members
        Serialization.serialize_cycle(s, member) && error("Repeated packed member")
    end
    return nothing
end

function _deserialize_packed_members(s::ClusterSerializer, ::Type{PM}) where {PM}
    PM <: PopMember || throw(ArgumentError("Unsupported packed member type $PM"))
    E = fieldtype(PM, :tree)
    E <: Expression{<:Any,<:Node} ||
        throw(ArgumentError("Unsupported packed expression type $E"))
    n = read(s.io, Int)
    n >= 0 || throw(ArgumentError("Negative packed member count"))
    members = Vector{PM}(undef, n)
    Serialization.resolve_ref_immediately(s, members)
    metadata = Serialization.deserialize(s)
    degrees, leaf_types, constants, features, ops = (
        Serialization.deserialize(s) for _ in 1:5
    )
    columns = Vector{Any}(undef, fieldcount(PM) - 1)
    column = 0
    for field in fieldnames(PM)
        field === :tree && continue
        column += 1
        values = Serialization.deserialize(s)
        length(values) == n || throw(ArgumentError("Invalid packed member field length"))
        columns[column] = values
    end
    N = fieldtype(E, :tree)
    positions = ones(Int, 5)
    for i in eachindex(members)
        member = ccall(:jl_new_struct_uninit, Any, (Any,), PM)::PM
        tree = _unpack_node(N, degrees, leaf_types, constants, features, ops, positions)
        setfield!(member, :tree, E(tree, metadata))
        column = 0
        for field in fieldnames(PM)
            field === :tree && continue
            column += 1
            setfield!(member, field, columns[column][i])
        end
        members[i] = member
        Serialization.resolve_ref_immediately(s, member)
    end
    for (position, values) in
        zip(positions, (degrees, leaf_types, constants, features, ops))
        position == length(values) + 1 || throw(ArgumentError("Trailing packed Node data"))
    end
    return members
end

function _serialize_member_collection(s::ClusterSerializer, value, tag::UInt8)
    P = typeof(value)
    Serialization.serialize_type(s, P)
    write(s.io, tag)
    members = getfield(value, :members)
    packed = _can_pack_members(s, members)
    write(s.io, UInt8(packed))
    for field in fieldnames(P)
        if field === :members && packed
            _serialize_packed_members(s, members)
        else
            Serialization.serialize(s, getfield(value, field))
        end
    end
    return nothing
end

function _deserialize_member_collection(
    s::ClusterSerializer, ::Type{P}, tag::UInt8
) where {P}
    read(s.io, UInt8) == tag || throw(ArgumentError("Unsupported $P encoding"))
    mode = read(s.io, UInt8)
    mode in (0x00, 0x01) || throw(ArgumentError("Unsupported member collection mode $mode"))
    fields = Vector{Any}(undef, fieldcount(P))
    for (i, field) in enumerate(fieldnames(P))
        fields[i] = if field === :members && mode == 0x01
            _deserialize_packed_members(s, eltype(fieldtype(P, :members)))
        else
            Serialization.deserialize(s)
        end
    end
    return ccall(
        :jl_new_structv, Any, (Any, Ptr{Any}, UInt32), P, fields, UInt32(length(fields))
    )::P
end

function Serialization.serialize(s::ClusterSerializer, pop::Population)
    _serialize_member_collection(s, pop, 0xa5)
end
function Serialization.deserialize(s::ClusterSerializer, ::Type{P}) where {P<:Population}
    _deserialize_member_collection(s, P, 0xa5)
end
end
