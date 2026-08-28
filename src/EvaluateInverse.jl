module EvaluateInverseModule

using DynamicExpressions:
    OperatorEnum,
    AbstractExpressionNode,
    eval_tree_array,
    get_child,
    preserve_sharing,
    EvalContext

using ..InverseFunctionsModule: approx_inverse

# Helper struct for returning results
struct ResultOk{T}
    x::T
    ok::Bool
end

# Helper struct for masked results: `valid[i]` marks rows where the
# inversion stayed on-domain through the entire path from the root.
struct ResultMasked{T}
    x::T
    valid::BitVector
end

@inline function _masked_eval_context(eval_context::EvalContext)
    return EvalContext(;
        turbo=eval_context.turbo,
        bumper=eval_context.bumper,
        early_exit=false,
        buffer=eval_context.buffer,
        use_fused=eval_context.use_fused,
    )
end

@inline function _masked_eval_kws(eval_kws::NamedTuple)
    eval_context = get(eval_kws, :eval_context, nothing)
    if eval_context !== nothing
        return merge(eval_kws, (; eval_context=_masked_eval_context(eval_context)))
    end

    eval_options = get(eval_kws, :eval_options, nothing)
    if eval_options !== nothing
        masked_eval_context = _masked_eval_context(eval_options)
        if haskey(eval_kws, :turbo) || haskey(eval_kws, :bumper)
            return merge(eval_kws, (; eval_context=masked_eval_context))
        end
        remaining_kws = Base.structdiff(eval_kws, (; eval_options=nothing))
        return merge(remaining_kws, (; eval_context=masked_eval_context))
    end

    masked_eval_context = EvalContext(;
        turbo=get(eval_kws, :turbo, Val(false)),
        bumper=get(eval_kws, :bumper, Val(false)),
        early_exit=false,
    )
    remaining_kws = Base.structdiff(eval_kws, (; turbo=nothing, bumper=nothing))
    return merge(remaining_kws, (; eval_context=masked_eval_context))
end

@inline function _reset_eval_buffer!(eval_kws::NamedTuple)
    buffer = eval_kws.eval_context.buffer
    isnothing(buffer) || (buffer.index[] = 0)
    return nothing
end

"""
Inverse the tree evaluation at some `node_to_invert_at` in the `tree`,
given some output of the `tree`, `y` and feature values `X`.

For example, inverting `y = cos(x) * 2.1` with `x` as
`node_to_invert_at` would return an evaluation of the
tree `acos(y / 2.1)`.

!!! warning
    This API supports an experimental mutation and will change in minor version
    increments.
"""
function eval_inverse_tree_array(
    tree::N,
    X::AbstractMatrix{T},
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T};
    eval_kws...,
) where {T,D,N<:AbstractExpressionNode{T,D}}
    if preserve_sharing(tree)
        throw(
            ArgumentError(
                "eval_inverse_tree_array does not currently support shared-node expressions.",
            ),
        )
    end
    masked_eval_kws = _masked_eval_kws((; eval_kws...))
    result = _eval_inverse_tree_array_masked(
        tree, X, operators, node_to_invert_at, copy(y), isfinite.(y), masked_eval_kws
    )
    return (result.x, all(result.valid))
end

"""
    eval_inverse_tree_array_masked(tree, X, operators, node_to_invert_at, y)
        -> (values, valid::BitVector)

Like [`eval_inverse_tree_array`](@ref), but instead of failing when any row
leaves the domain of an inverse operator, it returns a per-row validity mask.
Row `i` is valid only if every inversion step from the root down to
`node_to_invert_at` produced a finite value for that row (and every sibling
subtree evaluated to a finite value there). Rows outside the principal branch
of a non-injective operator (e.g. `log` of a negative value) are marked
invalid rather than poisoning the whole inversion.

A structural failure (an operator with no inverse, a sibling subtree that
fails to evaluate, or the node not being present) yields an all-false mask.

!!! warning
    This API supports an experimental mutation and will change in minor
    version increments.
"""
function eval_inverse_tree_array_masked(
    tree::N,
    X::AbstractMatrix{T},
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T};
    eval_kws...,
) where {T,D,N<:AbstractExpressionNode{T,D}}
    if preserve_sharing(tree)
        throw(
            ArgumentError(
                "eval_inverse_tree_array does not currently support shared-node expressions.",
            ),
        )
    end
    masked_eval_kws = _masked_eval_kws((; eval_kws...))
    result = _eval_inverse_tree_array_masked(
        tree, X, operators, node_to_invert_at, copy(y), isfinite.(y), masked_eval_kws
    )
    return (result.x, result.valid)
end

function _eval_inverse_tree_array(
    tree::N,
    X::AbstractMatrix{T},
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T},
    eval_kws::NamedTuple,
) where {T,D,N<:AbstractExpressionNode{T,D}}
    result = _eval_inverse_tree_array_masked(
        tree, X, operators, node_to_invert_at, y, isfinite.(y), _masked_eval_kws(eval_kws)
    )
    return ResultOk(result.x, all(result.valid))
end

@generated function _eval_inverse_tree_array_masked(
    tree::N,
    X::AbstractMatrix{T},
    operators::O,
    node_to_invert_at::N,
    y::AbstractVector{T},
    valid::BitVector,
    eval_kws::NamedTuple,
)::ResultMasked where {T,D,N<:AbstractExpressionNode{T,D},O<:OperatorEnum}
    # `Tuple{Tuple{degree-1 ops...}, Tuple{degree-2 ops...}, ...}`
    op_type = O.parameters[1]
    max_degree = min(D, length(op_type.parameters))
    nops = ntuple(d -> length(op_type.parameters[d].parameters), max_degree)

    dispatch = :(throw(
        ArgumentError(
            "eval_inverse_tree_array cannot invert node degree $(tree.degree) with the configured operators.",
        ),
    ))
    for degree in max_degree:-1:1
        nops[degree] == 0 && continue
        dispatch = :(
            if tree.degree == $degree
                op_idx = tree.op
                Base.Cartesian.@nif(
                    $(nops[degree]),
                    i -> i == op_idx,
                    i -> let op = operators.ops[$degree][i]
                        return dispatch_degn_masked(
                            tree,
                            X,
                            Val($degree),
                            op,
                            operators,
                            node_to_invert_at,
                            y,
                            valid,
                            eval_kws,
                        )
                    end
                )
            else
                $dispatch
            end
        )
    end

    quote
        tree === node_to_invert_at && return ResultMasked(y, valid)

        tree.degree == 0 && return ResultMasked(y, falses(length(y)))

        $dispatch
    end
end

function dispatch_degn_masked(
    tree::N,
    X::AbstractMatrix{T},
    ::Val{1},
    op::F,
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T},
    valid::BitVector,
    eval_kws::NamedTuple,
) where {F,T,D,N<:AbstractExpressionNode{T,D}}
    complete_inv = deg1_invert!(y, op)
    !complete_inv && return ResultMasked(y, falses(length(y)))
    valid .&= isfinite.(y)
    any(valid) || return ResultMasked(y, valid)
    return _eval_inverse_tree_array_masked(
        get_child(tree, 1), X, operators, node_to_invert_at, y, valid, eval_kws
    )
end

function dispatch_degn_masked(
    tree::N,
    X::AbstractMatrix{T},
    ::Val{2},
    op::F,
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T},
    valid::BitVector,
    eval_kws::NamedTuple,
) where {F,T,D,N<:AbstractExpressionNode{T,D}}
    left_child = get_child(tree, 1)
    right_child = get_child(tree, 2)

    if any(Base.Fix1(===, node_to_invert_at), right_child)
        _reset_eval_buffer!(eval_kws)
        (result_l, complete_l) = eval_tree_array(left_child, X, operators; eval_kws...)
        !complete_l && return ResultMasked(y, falses(length(y)))
        valid .&= isfinite.(result_l)
        any(valid) || return ResultMasked(y, valid)
        complete_inv_r = deg2_invert_right!(y, result_l, op)
        !complete_inv_r && return ResultMasked(y, falses(length(y)))
        valid .&= isfinite.(y)
        any(valid) || return ResultMasked(y, valid)
        return _eval_inverse_tree_array_masked(
            right_child, X, operators, node_to_invert_at, y, valid, eval_kws
        )
    else  # any(===(node_to_invert_at), left_child)
        _reset_eval_buffer!(eval_kws)
        (result_r, complete_r) = eval_tree_array(right_child, X, operators; eval_kws...)
        !complete_r && return ResultMasked(y, falses(length(y)))
        valid .&= isfinite.(result_r)
        any(valid) || return ResultMasked(y, valid)
        complete_inv_l = deg2_invert_left!(y, result_r, op)
        !complete_inv_l && return ResultMasked(y, falses(length(y)))
        valid .&= isfinite.(y)
        any(valid) || return ResultMasked(y, valid)
        return _eval_inverse_tree_array_masked(
            left_child, X, operators, node_to_invert_at, y, valid, eval_kws
        )
    end
end

@generated function dispatch_degn_masked(
    tree::N,
    X::AbstractMatrix{T},
    ::Val{degree},
    op::F,
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T},
    valid::BitVector,
    eval_kws::NamedTuple,
) where {degree,F,T,D,N<:AbstractExpressionNode{T,D}}
    quote
        n = length(y)

        child_to_invert = 0
        Base.Cartesian.@nexprs $degree i -> begin
            if child_to_invert == 0 &&
                any(Base.Fix1(===, node_to_invert_at), get_child(tree, i))
                child_to_invert = i
            end
        end
        child_to_invert == 0 && return ResultMasked(y, falses(n))

        # A single reset for the whole sibling group: each sibling has to keep
        # its own slice of the evaluation buffer while the others are computed.
        _reset_eval_buffer!(eval_kws)
        siblings = Base.Cartesian.@ntuple $degree i -> if i == child_to_invert
            y  # placeholder; `degn_invert!` substitutes `y[row]` at position `i`
        else
            (result_i, complete_i) = eval_tree_array(
                get_child(tree, i), X, operators; eval_kws...
            )
            !complete_i && return ResultMasked(y, falses(n))
            valid .&= isfinite.(result_i)
            result_i
        end
        any(valid) || return ResultMasked(y, valid)

        Base.Cartesian.@nif(
            $degree,
            i -> i == child_to_invert,
            i -> begin
                complete_inv = degn_invert!(y, siblings, op, Val($degree), Val(i))
                !complete_inv && return ResultMasked(y, falses(n))
                valid .&= isfinite.(y)
                any(valid) || return ResultMasked(y, valid)
                return _eval_inverse_tree_array_masked(
                    get_child(tree, i),
                    X,
                    operators,
                    node_to_invert_at,
                    y,
                    valid,
                    eval_kws,
                )
            end
        )
    end
end

function deg1_invert!(y::AbstractVector, op::F) where {F}
    op_inv = approx_inverse(op)
    op_inv === nothing && return false
    @inbounds @simd for i in eachindex(y)
        y[i] = op_inv(y[i])
    end
    return true
end

function deg2_invert_right!(y::AbstractVector, l::AbstractVector, op::F) where {F}
    @inbounds for i in eachindex(y, l)
        op_inv = approx_inverse(Base.Fix1(op, l[i]))
        op_inv === nothing && return false
        y[i] = op_inv(y[i])
    end
    return true
end

function deg2_invert_left!(y::AbstractVector, r::AbstractVector, op::F) where {F}
    @inbounds for i in eachindex(y, r)
        op_inv = approx_inverse(Base.Fix2(op, r[i]))
        op_inv === nothing && return false
        y[i] = op_inv(y[i])
    end
    return true
end

function degn_invert!(
    y::AbstractVector, siblings::Tuple, op::F, ::Val{degree}, ::Val{I}
) where {F,degree,I}
    @inbounds for row in eachindex(y)
        args = ntuple(j -> j == I ? y[row] : siblings[j][row], Val(degree))
        op_inv = approx_inverse(op, Val(I), args)
        op_inv === nothing && return false
        y[row] = op_inv(y[row])
    end
    return true
end

end
