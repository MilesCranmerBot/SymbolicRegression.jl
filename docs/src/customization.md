# Customization

Many parts of SymbolicRegression.jl are designed to be customizable.

The normal way to do this in Julia is to define a new type that subtypes
an abstract type from a package, and then define new methods for the type,
extending internal methods on that type.

## Custom Options

For example, you can define a custom options type:

```@docs
AbstractOptions
```

Any function in SymbolicRegression.jl you can generally define a new method
on your custom options type, to define custom behavior.

## Custom Mutations

Define a custom mutation by subtyping `AbstractMutation`, implementing `mutate!`,
and passing it with a weight through `Options(; mutations=...)`.

Here is a mutation that replaces a random subtree with a single variable:

```julia
using SymbolicRegression
using SymbolicRegression: AbstractMutation, MutationResult
using DynamicExpressions: get_contents, with_contents, AbstractExpression, AbstractExpressionNode

struct PruneMutation <: AbstractMutation end

function SymbolicRegression.mutate!(
    new_tree::N, parent_member::P, ::PruneMutation, options; nfeatures, kws...
) where {N<:AbstractExpression,P}
    tree = get_contents(new_tree)
    # Find a random non-leaf node and replace it with a variable
    nodes = filter(n -> n.degree > 0, collect(tree))
    if !isempty(nodes)
        target = rand(nodes)
        target.degree = 0
        target.feature = rand(1:nfeatures)
    end
    return MutationResult{N,P}(; tree=new_tree)
end
```

Pass it to `Options` with a weight. New mutation types are added alongside the
defaults; to replace or remove a default, pass `default_mutations=()`:

```julia
model = SRRegressor(
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    mutations=[PruneMutation() => 0.1],
)
```

```@docs
mutate!
AbstractMutation
condition_mutation_weights!
sample_mutation
MutationResult
```

## Custom Expressions

You can create your own expression types by defining a new type that extends `AbstractExpression`.

```@docs
AbstractExpression
```

The interface is fairly flexible, and permits you define specific functional forms,
extra parameters, etc. See the documentation of DynamicExpressions.jl for more details on what
methods you need to implement. You can test the implementation of a given interface by using
`ExpressionInterface` which makes use of `Interfaces.jl`:

```@docs
ExpressionInterface
```

Then, for SymbolicRegression.jl, you would
pass `expression_type` to the `Options` constructor, as well as any
`expression_options` you need (as a `NamedTuple`).

If needed, you may need to overload `SymbolicRegression.ExpressionBuilder.extra_init_params` in
case your expression needs additional parameters. See `src/TemplateExpression.jl` for an example.

You can also look at `src/TemplateExpression.jl` for a custom expression type used by
SymbolicRegression.jl.

## Plugins

See the [Plugins](plugins.md) page for how to hook into the search loop
with custom lifecycle callbacks, selection biases, and population seeding.

## Other Customizations

Other internal abstract types include the following:

```@docs
AbstractRuntimeOptions
AbstractSearchState
```

These let you include custom state variables and runtime options.
