# StatsBaseLite.jl — vendored subset of StatsBase.jl v0.34 (MIT license below)
#
# > Copyright (c) 2012-2016: Dahua Lin, Simon Byrne, Andreas Noack,
# > Douglas Bates, John Myles White, Simon Kornblith, and other contributors.
# >
# > Permission is hereby granted, free of charge, to any person obtaining
# > a copy of this software and associated documentation files (the
# > "Software"), to deal in the Software without restriction, including
# > without limitation the rights to use, copy, modify, merge, publish,
# > distribute, sublicense, and/or sell copies of the Software, and to
# > permit persons to whom the Software is furnished to do so, subject to
# > the following conditions:
# >
# > The above copyright notice and this permission notice shall be
# > included in all copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# > NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# > LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# > OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# > WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
module StatsBaseLite

using Random: AbstractRNG, Sampler, default_rng

#####
##### Ported from StatsBase.jl src/weights.jl; made immutable (upstream: mutable)
#####

struct Weights{S<:Real,T<:Real,V<:AbstractVector{T}}
    values::V
    sum::S
    function Weights{S,T,V}(values, sum) where {S<:Real,T<:Real,V<:AbstractVector{T}}
        isfinite(sum) || throw(ArgumentError("weights cannot contain Inf or NaN values"))
        return new{S,T,V}(values, sum)
    end
end
function Weights(values::AbstractVector{T}, sum::S) where {S<:Real,T<:Real}
    return Weights{S,T,typeof(values)}(values, sum)
end
Weights(values::AbstractVector{<:Real}) = Weights(values, sum(values))

Base.length(wv::Weights) = length(wv.values)
Base.sum(wv::Weights) = wv.sum
Base.firstindex(wv::Weights) = firstindex(wv.values)
Base.@propagate_inbounds Base.getindex(wv::Weights, i::Integer) = wv.values[i]

#####
##### Ported from StatsBase.jl src/sampling.jl
#####

### draw a pair of distinct integers in [1:n]

function samplepair(rng::AbstractRNG, n::Integer)
    i1 = rand(rng, one(n):n)
    i2 = rand(rng, one(n):(n - one(n)))
    return (i1, ifelse(i2 == i1, n, i2))
end
samplepair(n::Integer) = samplepair(default_rng(), n)

function samplepair(rng::AbstractRNG, a::AbstractArray)
    i1, i2 = samplepair(rng, length(a))
    return a[i1], a[i2]
end
samplepair(a::AbstractArray) = samplepair(default_rng(), a)

### Algorithms for sampling without replacement

function fisher_yates_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    1 == firstindex(a) == firstindex(x) ||
        throw(ArgumentError("non 1-based arrays are not supported"))
    Base.mightalias(a, x) &&
        throw(ArgumentError("output array x must not share memory with input array a"))
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")

    inds = Vector{Int}(undef, n)
    for i in 1:n
        inds[i] = i
    end

    for i in 1:k
        j = rand(rng, i:n)
        t = inds[j]
        inds[j] = inds[i]
        inds[i] = t
        x[i] = a[t]
    end
    return x
end
function fisher_yates_sample!(a::AbstractArray, x::AbstractArray)
    return fisher_yates_sample!(default_rng(), a, x)
end

function self_avoid_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    1 == firstindex(a) == firstindex(x) ||
        throw(ArgumentError("non 1-based arrays are not supported"))
    Base.mightalias(a, x) &&
        throw(ArgumentError("output array x must not share memory with input array a"))
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")

    s = Set{Int}()
    sizehint!(s, k)
    rgen = Sampler(rng, 1:n)

    # first one
    idx = rand(rng, rgen)
    x[1] = a[idx]
    push!(s, idx)

    # remaining
    for i in 2:k
        idx = rand(rng, rgen)
        while idx in s
            idx = rand(rng, rgen)
        end
        x[i] = a[idx]
        push!(s, idx)
    end
    return x
end
function self_avoid_sample!(a::AbstractArray, x::AbstractArray)
    return self_avoid_sample!(default_rng(), a, x)
end

### Interface functions (poly-algorithms)

sample(rng::AbstractRNG, a::AbstractArray) = a[rand(rng, 1:length(a))]
sample(a::AbstractArray) = sample(default_rng(), a)

function direct_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    1 == firstindex(a) == firstindex(x) ||
        throw(ArgumentError("non 1-based arrays are not supported"))
    Base.mightalias(a, x) &&
        throw(ArgumentError("output array x must not share memory with input array a"))
    s = Sampler(rng, 1:length(a))
    for i in 1:length(x)
        x[i] = a[rand(rng, s)]
    end
    return x
end

function sample!(
    rng::AbstractRNG,
    a::AbstractArray,
    x::AbstractArray;
    replace::Bool=true,
    ordered::Bool=false,
)
    1 == firstindex(a) == firstindex(x) ||
        throw(ArgumentError("non 1-based arrays are not supported"))
    ordered && throw(ArgumentError("ordered sampling is not supported by StatsBaseLite"))
    n = length(a)
    k = length(x)
    k == 0 && return x

    if replace  # with replacement
        direct_sample!(rng, a, x)
    else  # without replacement
        k <= n || error("Cannot draw more samples without replacement.")
        if k == 1
            x[1] = sample(rng, a)
        elseif k == 2
            (x[1], x[2]) = samplepair(rng, a)
        elseif n < k * 24
            fisher_yates_sample!(rng, a, x)
        else
            self_avoid_sample!(rng, a, x)
        end
    end
    return x
end
function sample!(
    a::AbstractArray, x::AbstractArray; replace::Bool=true, ordered::Bool=false
)
    return sample!(default_rng(), a, x; replace=replace, ordered=ordered)
end

function sample(
    rng::AbstractRNG,
    a::AbstractArray{T},
    n::Integer;
    replace::Bool=true,
    ordered::Bool=false,
) where {T}
    return sample!(rng, a, Vector{T}(undef, n); replace=replace, ordered=ordered)
end
function sample(a::AbstractArray, n::Integer; replace::Bool=true, ordered::Bool=false)
    return sample(default_rng(), a, n; replace=replace, ordered=ordered)
end

################################################################
#  Weighted sampling
################################################################

function sample(rng::AbstractRNG, wv::Weights)
    1 == firstindex(wv) || throw(ArgumentError("non 1-based arrays are not supported"))
    wsum = sum(wv)
    isfinite(wsum) || throw(ArgumentError("only finite weights are supported"))
    t = rand(rng) * wsum
    n = length(wv)
    i = 1
    cw = wv[1]
    while cw < t && i < n
        i += 1
        cw += wv[i]
    end
    return i
end
sample(wv::Weights) = sample(default_rng(), wv)

sample(rng::AbstractRNG, a::AbstractArray, wv::Weights) = a[sample(rng, wv)]
sample(a::AbstractArray, wv::Weights) = sample(default_rng(), a, wv)

end
