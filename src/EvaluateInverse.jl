module EvaluateInverseModule

using DynamicExpressions:
    OperatorEnum, AbstractExpressionNode, eval_tree_array, get_children, preserve_sharing

using ..InverseFunctionsModule: approx_inverse, PartialFunction

# Helper struct for returning results
struct ResultOk{T}
    x::T
    ok::Bool
end

is_bad_array(x) = any(isnan, x) || any(isinf, x)

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
    result = _eval_inverse_tree_array(
        tree, X, operators, node_to_invert_at, copy(y), (; eval_kws...)
    )
    return (result.x, result.ok && !is_bad_array(result.x))
end

@generated function _eval_inverse_tree_array(
    tree::N,
    X::AbstractMatrix{T},
    operators::O,
    node_to_invert_at::N,
    y::AbstractVector{T},
    eval_kws::NamedTuple,
)::ResultOk where {T,D,N<:AbstractExpressionNode{T,D},O<:OperatorEnum}
    op_type = O.parameters[1]
    nops = ntuple(Val(D)) do degree
        if degree <= length(op_type.parameters)
            length(op_type.parameters[degree].parameters)
        else
            0
        end
    end
    dispatch_cases = ntuple(Val(D)) do degree
        if nops[degree] == 0
            quote
                throw(
                    ArgumentError(
                        "eval_inverse_tree_array cannot invert node degree $(tree.degree) with the configured operators.",
                    ),
                )
            end
        else
            quote
                op_idx = tree.op
                Base.Cartesian.@nif(
                    $(nops[degree]),
                    j -> j == op_idx,
                    j -> let op = operators[$degree][j]
                        return dispatch_degn(
                            tree,
                            Val($degree),
                            X,
                            op,
                            operators,
                            node_to_invert_at,
                            y,
                            eval_kws,
                        )
                    end,
                )
            end
        end
    end
    degree_dispatch = quote
        throw(
            ArgumentError(
                "eval_inverse_tree_array cannot invert node degree $(tree.degree) with the configured operators.",
            ),
        )
    end
    for degree in D:-1:1
        degree_dispatch = quote
            if degree == $degree
                $(dispatch_cases[degree])
            else
                $degree_dispatch
            end
        end
    end
    quote
        tree === node_to_invert_at && return ResultOk(y, true)

        tree.degree == 0 && return ResultOk(y, false)

        degree = tree.degree
        $degree_dispatch
    end
end

function dispatch_degn(
    tree::N,
    ::Val{degree},
    X::AbstractMatrix{T},
    op::F,
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T},
    eval_kws::NamedTuple,
) where {degree,F,T,D,N<:AbstractExpressionNode{T,D}}
    children = get_children(tree, Val(degree))
    target_child_index = findfirst(
        child -> any(Base.Fix1(===, node_to_invert_at), child), children
    )
    target_child_index === nothing && return ResultOk(y, false)
    return dispatch_degn_target(
        children, Val(target_child_index), X, op, operators, node_to_invert_at, y, eval_kws
    )
end

function dispatch_degn_target(
    children::NTuple{degree,N},
    ::Val{target_child_index},
    X::AbstractMatrix{T},
    op::F,
    operators::OperatorEnum,
    node_to_invert_at::N,
    y::AbstractVector{T},
    eval_kws::NamedTuple,
) where {degree,target_child_index,F,T,N<:AbstractExpressionNode}
    inputs = ntuple(Val(degree)) do i
        if i == target_child_index
            y
        else
            result, complete = eval_tree_array(children[i], X, operators; eval_kws...)
            !complete && return ResultOk(result, complete)
            result
        end
    end

    complete_inv = degn_invert!(y, inputs, Val(target_child_index), op)
    (!complete_inv || is_bad_array(y)) && return ResultOk(y, false)
    return _eval_inverse_tree_array(
        children[target_child_index], X, operators, node_to_invert_at, y, eval_kws
    )
end

partial_function(op::F, args::Tuple, ::Val{1}, ::Val{1}) where {F} = op
partial_function(op::F, args::Tuple, ::Val{1}, ::Val{2}) where {F} = Base.Fix2(op, args[2])
partial_function(op::F, args::Tuple, ::Val{2}, ::Val{2}) where {F} = Base.Fix1(op, args[1])
function partial_function(op::F, args::Tuple, ::Val{I}, ::Val{degree}) where {I,degree,F}
    PartialFunction{I}(op, args)
end

function degn_invert!(
    y::AbstractVector, inputs::NTuple{degree,AbstractVector}, ::Val{target_idx}, op::F
) where {degree,target_idx,F}
    @inbounds for i in eachindex(inputs...)
        args = ntuple(j -> inputs[j][i], Val(degree))
        op_inv = approx_inverse(partial_function(op, args, Val(target_idx), Val(degree)))
        op_inv === nothing && return false
        y[i] = op_inv(y[i])
    end
    return true
end

end
