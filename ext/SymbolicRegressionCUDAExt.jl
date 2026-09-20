module SymbolicRegressionCUDAExt

using CUDA
using Distributed: myid
using SymbolicRegression: SymbolicRegression as SR, AbstractOptions, Dataset, BasicDataset, SubDataset
using DynamicExpressions: DynamicExpressions as DE, AbstractExpression, AbstractExpressionNode, OperatorEnum, get_tree, count_nodes
using DynamicExpressions.EvaluateModule: reset_index!, get_nops
using SymbolicRegression.CoreModule: get_full_dataset, get_indices, DATA_TYPE, LOSS_TYPE
using SymbolicRegression.LossFunctionsModule: _eval_loss, dimensional_regularization
using LossFunctions: SupervisedLoss

struct CuDataMatrix{T} <: AbstractMatrix{T}
    host::Matrix{T}
    device_t::CuMatrix{T}
    services::IdDict{Any,Any}
    lock::ReentrantLock
    min_gpu_rows::Int
end
function SR.gpu_matrix(X::AbstractMatrix{T}; min_gpu_rows::Integer=16384) where {T<:Union{Float32,Float64}}
    myid() == 1 || error("CUDA loss evaluation does not support multiprocessing")
    return CuDataMatrix(Matrix(X), CuMatrix(permutedims(X)), IdDict(), ReentrantLock(), Int(min_gpu_rows))
end
Base.size(m::CuDataMatrix) = size(m.host)
Base.IndexStyle(::Type{<:CuDataMatrix}) = IndexCartesian()
Base.getindex(m::CuDataMatrix, i::Int, j::Int) = m.host[i, j]
Base.similar(m::CuDataMatrix, ::Type{S}, dims::Dims) where {S} = similar(m.host, S, dims)

for A in (:(CuDataMatrix{T}), :(SubArray{T,2,<:CuDataMatrix})), E in (:AbstractExpressionNode, :AbstractExpression)
    @eval function DE.eval_tree_array(tree::$E{T}, X::$A, operators::OperatorEnum; kws...) where {T}
        host = X isa CuDataMatrix ? X.host : view(parent(X).host, parentindices(X)...)
        return DE.eval_tree_array(tree, host, operators; kws...)
    end
end

mutable struct Slot{N,L}
    tree::N
    rows::Union{Nothing,Vector{Int}}
    @atomic state::Int
    result::L
    error::Any
    expression::Any
    options::AbstractOptions
end
mutable struct GPULossService{T,L,N,O<:OperatorEnum,F,W,D}
    X::CuMatrix{T}
    y::CuVector{T}
    w::W
    w_host::Union{Nothing,Vector{T}}
    operators::O
    loss::F
    pending::Vector{Slot{N,L}}
    batch::Vector{Slot{N,L}}
    pending_lock::Threads.SpinLock
    combine_lock::Threads.SpinLock
    w_sum::Float64
    host_dataset::D
    words::CuVector{UInt32}
    dvals::CuVector{T}
    tree_starts::CuVector{Int32}
    row_index::CuMatrix{Int32}
    sums::CuVector{Float64}
    host_memory::Vector{CUDA.HostMemory}
    host_words::Vector{UInt32}
    host_vals::Vector{T}
    host_starts::Vector{Int32}
    host_rows::Matrix{Int32}
    host_sums::Vector{Float64}
    kernel_rows::Any
    kernel_full::Any
    launches::Int
    requests::Int
    cpu_fallbacks::Int
    max_depth::Int
end

function host_buffer!(memory, index, ::Type{T}, dims...) where {T}
    mem = CUDA.alloc(CUDA.HostMemory, prod(dims) * sizeof(T))
    buffer = unsafe_wrap(Array, convert(Ptr{T}, mem), dims; own=false)
    if index <= length(memory)
        CUDA.synchronize(; blocking=true)
        CUDA.free(memory[index])
        memory[index] = mem
    else
        push!(memory, mem)
    end
    return buffer
end

function service!(dataset::Dataset{T,L}, options, node::N, operators) where {T,L,N}
    full = dataset isa SubDataset ? get_full_dataset(dataset) : dataset
    X = full.X
    return lock(X.lock) do
        svc = get(X.services, full.y, nothing)
        if svc === nothing || svc.operators !== operators || svc.loss !== options.elementwise_loss
            w_host = full.weights === nothing ? nothing : Vector{T}(full.weights)
            w = w_host === nothing ? nothing : CuVector(w_host)
            memory = CUDA.HostMemory[]
            host_dataset = Dataset(X.host, full.y, L; weights=full.weights, extra=full.extra,
                variable_names=full.variable_names, display_variable_names=full.display_variable_names,
                y_variable_name=full.y_variable_name, X_units=full.X_units, y_units=full.y_units)
            svc = GPULossService(full.X.device_t, CuVector{T}(full.y), w, w_host,
                operators, options.elementwise_loss, Slot{N,L}[], Slot{N,L}[],
                Threads.SpinLock(), Threads.SpinLock(), w_host === nothing ? 0.0 : Float64(sum(w_host)),
                host_dataset, CuVector{UInt32}(undef, 0), CuVector{T}(undef, 0),
                CuVector{Int32}(undef, 0), CuMatrix{Int32}(undef, 0, 0),
                CuVector{Float64}(undef, 0), memory, host_buffer!(memory, 1, UInt32, 1),
                host_buffer!(memory, 2, T, 1), host_buffer!(memory, 3, Int32, 1),
                host_buffer!(memory, 4, Int32, 1, 1), host_buffer!(memory, 5, Float64, 1), nothing, nothing, 0, 0, 0, 0)
            finalizer(svc) do service
                for mem in service.host_memory
                    CUDA.free(mem)
                end
            end
            sizehint!(svc.pending, 4096)
            sizehint!(svc.batch, 4096)
            X.services[full.y] = svc
        end
        return svc
    end
end

function wait_slot!(svc, slot)
    spins = 0
    while true
        if !islocked(svc.combine_lock) && trylock(svc.combine_lock)
            try
                combine!(svc)
            finally
                unlock(svc.combine_lock)
            end
        end
        state = @atomic :acquire slot.state
        state == 2 && throw(slot.error)
        state == 1 && break
        spins += 1
        if spins <= 1000
            ccall(:jl_cpu_pause, Cvoid, ())
            spins % 64 == 0 && GC.safepoint()
        else
            yield()
        end
    end
    return nothing
end

function gpu_route(dataset, tree, options)
    operators = DE.get_operators(tree, options)
    rows = dataset isa SubDataset ? get_indices(dataset) : nothing
    X = (dataset isa SubDataset ? get_full_dataset(dataset) : dataset).X
    nrows = rows === nothing ? size(X, 2) : length(rows)
    eligible = nrows >= X.min_gpu_rows && operators isa OperatorEnum &&
        options.elementwise_loss isa Union{SupervisedLoss,Function}
    return eligible, rows, operators
end

function SR.LossFunctionsModule._eval_loss(
    tree::Union{AbstractExpression{T},AbstractExpressionNode{T}},
    dataset::Union{BasicDataset{T,L,<:CuDataMatrix{T}},SubDataset{T,L,<:BasicDataset{T,L,<:CuDataMatrix{T}}}},
    options::AbstractOptions, regularization::Bool, eval_context,
)::L where {T<:DATA_TYPE,L<:LOSS_TYPE}
    myid() == 1 || error("CUDA loss evaluation does not support multiprocessing")
    eval_context === nothing || reset_index!(eval_context.buffer)
    eligible, rows, operators = gpu_route(dataset, tree, options)
    if !eligible
        return invoke(_eval_loss, Tuple{typeof(tree),Dataset{T,L},typeof(options),Bool,typeof(eval_context)}, tree, dataset, options, regularization, eval_context)
    end
    node = tree isa AbstractExpression ? get_tree(tree) : tree
    svc = service!(dataset, options, node, operators)
    slot = Slot{typeof(node),L}(node, rows, 0, zero(L), nothing, tree, options)
    lock(svc.pending_lock) do
        push!(svc.pending, slot)
    end
    wait_slot!(svc, slot)
    loss_val = slot.result
    regularization && (loss_val += dimensional_regularization(tree, dataset, options))
    return loss_val
end

function SR.LossFunctionsModule.eval_losses(
    trees::AbstractVector,
    dataset::Union{BasicDataset{T,L,<:CuDataMatrix{T}},SubDataset{T,L,<:BasicDataset{T,L,<:CuDataMatrix{T}}}},
    options::AbstractOptions;
    regularization::Bool=true,
)::Vector{L} where {T<:DATA_TYPE,L<:LOSS_TYPE}
    fallback() = invoke(
        SR.LossFunctionsModule.eval_losses,
        Tuple{typeof(trees),Dataset{T,L},typeof(options)},
        trees,
        dataset,
        options;
        regularization,
    )
    isempty(trees) && return fallback()
    options.loss_function === nothing || return fallback()
    options.loss_function_expression === nothing || return fallback()
    tree = first(trees)
    eligible, rows, operators = gpu_route(dataset, tree, options)
    eligible || return fallback()
    myid() == 1 || error("CUDA loss evaluation does not support multiprocessing")
    first_node = tree isa AbstractExpression ? get_tree(tree) : tree
    svc = service!(dataset, options, first_node, operators)
    slots = map(trees) do expression
        node = expression isa AbstractExpression ? get_tree(expression) : expression
        Slot{typeof(node),L}(node, rows, 0, zero(L), nothing, expression, options)
    end
    lock(svc.pending_lock) do
        append!(svc.pending, slots)
    end
    for slot in slots
        wait_slot!(svc, slot)
    end
    return L[
        (regularization ? slot.result + dimensional_regularization(expression, dataset, options) : slot.result) for
        (slot, expression) in zip(slots, trees)
    ]
end

struct BatchedFiniteDifference{E} <: Function
    e::E
end

function SR.ConstantOptimizationModule.finite_difference_objective(
    f::SR.ConstantOptimizationModule.Evaluator,
    dataset::Union{BasicDataset{T,L,<:CuDataMatrix{T}},SubDataset{T,L,<:BasicDataset{T,L,<:CuDataMatrix{T}}}},
) where {T<:DATA_TYPE,L<:LOSS_TYPE}
    options = f.ctx.options
    eligible, _, _ = gpu_route(dataset, f.tree, options)
    eligible && options.loss_function === nothing && options.loss_function_expression === nothing || return f
    return SR.ConstantOptimizationModule.NLSolversBase.only_fg!(BatchedFiniteDifference(f))
end

function (g::BatchedFiniteDifference)(F, G, x::AbstractVector)
    G === nothing && return g.e(x)
    e = g.e
    relstep = cbrt(eps(real(eltype(x))))
    xi = copy(x)
    trees = Vector{typeof(e.tree)}(undef, 2length(x) + 1)
    SR.ConstantOptimizationModule.set_optimizable_parameters!(e.tree, x, e.refs)
    trees[1] = copy(e.tree)
    for i in eachindex(x)
        epsilon = max(relstep * abs(x[i]), relstep)
        xi[i] += epsilon
        SR.ConstantOptimizationModule.set_optimizable_parameters!(e.tree, xi, e.refs)
        trees[2i] = copy(e.tree)
        xi[i] = x[i] - epsilon
        SR.ConstantOptimizationModule.set_optimizable_parameters!(e.tree, xi, e.refs)
        trees[2i + 1] = copy(e.tree)
        xi[i] = x[i]
    end
    SR.ConstantOptimizationModule.set_optimizable_parameters!(e.tree, x, e.refs)
    losses = SR.LossFunctionsModule.eval_losses(trees, e.ctx.dataset, e.ctx.options; regularization=false)
    for i in eachindex(x)
        epsilon = max(relstep * abs(x[i]), relstep)
        dfi = losses[2i]
        dfi -= losses[2i + 1]
        G[i] = real(dfi / (2 * epsilon))
    end
    return losses[1]
end

function combine!(svc)
    while true
        lock(svc.pending_lock) do
            svc.pending, svc.batch = svc.batch, svc.pending
        end
        isempty(svc.batch) && return nothing
        process!(svc, svc.batch)
        empty!(svc.batch)
    end
end

function process!(svc::GPULossService, reqs)
    isempty(reqs) && return nothing
    full_rows = size(svc.X, 1)
    if all(req -> req.rows === nothing, reqs)
        process_group!(svc, reqs, 1, length(reqs), full_rows)
    else
        nrows(req) = req.rows === nothing ? full_rows : length(req.rows)
        sort!(reqs; by=nrows, alg=QuickSort)
        first = 1
        while first <= length(reqs)
            rows = nrows(reqs[first])
            last = first
            while last < length(reqs) && nrows(reqs[last + 1]) == rows
                last += 1
            end
            process_group!(svc, reqs, first, last, rows)
            first = last + 1
        end
    end
    return nothing
end

function process_group!(svc::GPULossService{T}, reqs, first, last, nrows) where {T}
    try
        nodes = sum(i -> count_nodes(reqs[i].tree; break_sharing=Val(true)), first:last)
        process_chunk!(svc, reqs, first, last, nrows, nodes)
    catch err
        try
            CUDA.synchronize(; blocking=true)
        catch
            # Preserve the group error if synchronization also fails.
        finally
            for i in first:last
                req = reqs[i]
                if (@atomic :acquire req.state) == 0
                    req.error = err
                    @atomic :release req.state = 2
                end
            end
        end
    end
    return nothing
end

const MAX_STACK = 16

function flatten_node!(words, vals, node::N, k, sp, peak) where {N}
    @inbounds begin
        degree = node.degree
        if degree > 0
            if degree == 2
                k, sp, peak = flatten_node!(words, vals, node.r, k, sp, peak)
            end
            k, sp, peak = flatten_node!(words, vals, node.l, k, sp, peak)
        end
        constant = degree == 0 && node.constant
        index = constant ? k : degree == 0 ? node.feature : node.op
        1 <= index <= 0x1fffffff || error("Packed index out of range: $index")
        words[k] = UInt32(degree) | (UInt32(constant)<<2) | (UInt32(index)<<3)
        constant && (vals[k] = node.val)
        sp += 1 - Int(degree)
        return k - 1, sp, max(peak, sp)
    end
end

function flatten!(svc::GPULossService{T,L}, slots, first, last) where {T,L}
    k = 1
    gpu_last = first - 1
    svc.max_depth = 0
    for i in first:last
        req = slots[i]
        n = count_nodes(req.tree; break_sharing=Val(true))
        _, _, depth = flatten_node!(svc.host_words, svc.host_vals, req.tree, k + n - 1, 0, 0)
        svc.max_depth = max(svc.max_depth, depth)
        if depth > MAX_STACK
            dataset = req.rows === nothing ? svc.host_dataset : SR.batch(svc.host_dataset, req.rows)
            value = invoke(_eval_loss,
                Tuple{typeof(req.expression),Dataset{T,L},typeof(req.options),Bool,Nothing},
                req.expression, dataset, req.options, false, nothing)
            req.result = isfinite(value) ? L(value) : L(Inf)
            svc.cpu_fallbacks += 1
            svc.requests += 1
            @atomic :release req.state = 1
        else
            gpu_last += 1
            slots[i], slots[gpu_last] = slots[gpu_last], req
            svc.host_starts[gpu_last - first + 1] = k
            k += n
        end
    end
    svc.host_starts[gpu_last - first + 2] = k
    return gpu_last, k - 1
end

function process_chunk!(svc::GPULossService{T,L}, reqs, first, last, nrows, num_nodes) where {T,L}
    ntrees = last - first + 1
    length(svc.host_words) < num_nodes && (svc.host_words = host_buffer!(svc.host_memory, 1, UInt32, num_nodes))
    if length(svc.words) < num_nodes
        svc.words = CuVector{UInt32}(undef, num_nodes)
        svc.dvals = CuVector{T}(undef, num_nodes)
    end
    length(svc.tree_starts) < ntrees + 1 && (svc.tree_starts = CuVector{Int32}(undef, ntrees + 1))
    length(svc.sums) < ntrees && (svc.sums = CuVector{Float64}(undef, ntrees))
    length(svc.host_vals) < num_nodes && (svc.host_vals = host_buffer!(svc.host_memory, 2, T, num_nodes))
    length(svc.host_starts) < ntrees + 1 && (svc.host_starts = host_buffer!(svc.host_memory, 3, Int32, ntrees + 1))
    length(svc.host_sums) < ntrees && (svc.host_sums = host_buffer!(svc.host_memory, 5, Float64, ntrees))
    last, num_nodes = flatten!(svc, reqs, first, last)
    ntrees = last - first + 1
    ntrees == 0 && return nothing
    rowidx = nothing
    if any(i -> reqs[i].rows !== nothing, first:last)
        if size(svc.row_index, 1) < nrows || size(svc.row_index, 2) < ntrees
            svc.row_index = CuMatrix{Int32}(undef, nrows, ntrees)
        end
        if size(svc.host_rows, 1) < nrows || size(svc.host_rows, 2) < ntrees
            svc.host_rows = host_buffer!(svc.host_memory, 4, Int32, nrows, ntrees)
        end
        for i in first:last
            rows = reqs[i].rows
            t = i - first + 1
            for r in 1:nrows
                svc.host_rows[r, t] = rows === nothing ? r : rows[r]
            end
        end
        rowidx = svc.row_index
    end
    host_words, host_vals, host_starts, host_rows, host_sums =
        svc.host_words, svc.host_vals, svc.host_starts, svc.host_rows, svc.host_sums
    GC.@preserve svc host_words host_vals host_starts host_rows host_sums begin
        stream = CUDA.stream()
        CUDA.unsafe_copyto!(pointer(svc.words), pointer(host_words), num_nodes; async=true, stream)
        CUDA.unsafe_copyto!(pointer(svc.tree_starts), pointer(host_starts), ntrees + 1; async=true, stream)
        CUDA.unsafe_copyto!(pointer(svc.dvals), pointer(host_vals), num_nodes; async=true, stream)
        if rowidx !== nothing
            for t in 1:ntrees
                CUDA.unsafe_copyto!(pointer(svc.row_index, (t - 1) * size(svc.row_index, 1) + 1),
                    pointer(host_rows, (t - 1) * size(host_rows, 1) + 1), nrows; async=true, stream)
            end
        end
        CUDA.memset(reinterpret(CuPtr{UInt32}, pointer(svc.sums)), UInt32(0), 2 * ntrees; stream)
        args = (svc.words, svc.dvals, svc.tree_starts, nrows, svc.X, svc.y, svc.w, rowidx, svc.sums, svc.loss)
        kernel = rowidx === nothing ? svc.kernel_full : svc.kernel_rows
        if kernel === nothing
            nuna = get_nops(typeof(svc.operators), Val(1))
            nbin = get_nops(typeof(svc.operators), Val(2))
            (nuna > 10 || nbin > 10) && error("Too many operators. Kernels are only compiled up to 10.")
            kernel! = create_gpu_kernel(svc.operators, Val(nuna), Val(nbin))
            kernel = @cuda launch=false kernel!(args...)
            if rowidx === nothing
                svc.kernel_full = kernel
            else
                svc.kernel_rows = kernel
            end
        end
        threads = kernel_threads(T)
        kernel(args...; threads, blocks=(cld(nrows, 2 * threads), ntrees))
        svc.launches += 1
        CUDA.unsafe_copyto!(pointer(host_sums), pointer(svc.sums), ntrees; async=true, stream)
        CUDA.synchronize(; blocking=true)
    end
    for i in first:last
        t = i - first + 1
        req = reqs[i]
        denom = svc.w_host === nothing ? nrows : req.rows === nothing ? svc.w_sum : sum(view(svc.w_host, req.rows))
        value = svc.host_sums[t] > floatmax(L) ? Inf : svc.host_sums[t] / denom
        svc.requests += 1
        req.result = isfinite(value) ? L(value) : L(Inf)
        @atomic :release req.state = 1
    end
end

kernel_threads(::Type{Float32}) = 256
kernel_threads(::Type{Float64}) = 128

for nuna in 0:10, nbin in 0:10
    R, TOP, depth = 2, true, MAX_STACK
    @eval function create_gpu_kernel(operators::OperatorEnum,::Val{$nuna},::Val{$nbin})
        return function(words,vals,starts,nrows,X,y,w::Union{Nothing,CuDeviceVector},rowidx::Union{Nothing,CuDeviceMatrix{Int32}},sums,loss)
            tree=blockIdx().y; tid=threadIdx().x
            elem=(blockIdx().x-1)*blockDim().x+tid
            stride=blockDim().x*gridDim().x
            stack=CuStaticSharedArray(eltype(X),(kernel_threads(eltype(X)),$R,$depth))
            total=0.0
            @inbounds if elem<=nrows
                Base.Cartesian.@nexprs $R j -> begin
                    elem_j=elem+(j-1)*stride
                    active_j=elem_j<=nrows
                    row_j=active_j ? (rowidx===nothing ? elem_j : rowidx[elem_j,tree]) : one(Int32)
                    top_j=zero(eltype(X)); bad_j=false
                end
                sp=Int32(0)
                for node in (starts[tree+1]-Int32(1)):-1:starts[tree]
                    word=words[node]
                    degree=word&UInt32(3); index=Int32(word>>3)
                    if degree==0
                        Base.Cartesian.@nexprs $R j -> begin
                            if $TOP && sp>0
                                stack[tid,j,sp]=top_j
                            end
                            value_j=word&UInt32(4)!=0 ? vals[index] : X[row_j,index]
                        end
                        sp+=Int32(1)
                    elseif degree==1 && $nuna>0
                        Base.Cartesian.@nif($nuna,i->i==index,i->let op=operators.unaops[i]
                            Base.Cartesian.@nexprs $R j -> (value_j=op($TOP ? top_j : stack[tid,j,sp]))
                        end)
                    elseif $nbin>0
                        Base.Cartesian.@nif($nbin,i->i==index,i->let op=operators.binops[i]
                            Base.Cartesian.@nexprs $R j -> (value_j=op($TOP ? top_j : stack[tid,j,sp],stack[tid,j,sp-Int32(1)]))
                        end)
                        sp-=Int32(1)
                    end
                    Base.Cartesian.@nexprs $R j -> begin
                        if $TOP
                            top_j=value_j
                        else
                            stack[tid,j,sp]=value_j
                        end
                        bad_j|=!isfinite(value_j)
                    end
                end
                Base.Cartesian.@nexprs $R j -> begin
                    if active_j
                        pred=bad_j ? eltype(X)(NaN) : ($TOP ? top_j : stack[tid,j,1])
                        target=y[row_j]
                        contribution=if w===nothing
                            loss(pred,target)
                        elseif loss isa SupervisedLoss
                            w[row_j]*loss(pred,target)
                        else
                            loss(pred,target,w[row_j])
                        end
                        total+=Float64(contribution)
                    end
                end
            end
            for offset in (16,8,4,2,1)
                total+=shfl_down_sync(0xffffffff,total,offset)
            end
            lane=(tid-1)%32; warp=(tid-1)÷32+1
            shared=CuStaticSharedArray(Float64,kernel_threads(eltype(X))÷32)
            lane==0 && (shared[warp]=total)
            sync_threads()
            if warp==1
                total=lane<kernel_threads(eltype(X))÷32 ? shared[lane+1] : 0.0
                for offset in (16,8,4,2,1)
                    total+=shfl_down_sync(0xffffffff,total,offset)
                end
                lane==0 && CUDA.@atomic sums[tree]+=total
            end
            return nothing
        end
    end
end

end
