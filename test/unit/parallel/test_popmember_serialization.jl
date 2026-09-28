@testitem "PopMember serialization preserves disk format and worker transfers" begin
    using SymbolicRegression
    using Serialization
    using Distributed
    using Distributed: ClusterSerializer
    using Random
    using Test
    using DynamicExpressions: get_child

    function same_tree(a, b)
        @test typeof(a) === typeof(b)
        @test a.degree == b.degree
        if a.degree == 0
            @test a.constant == b.constant
            if a.constant
                @test bitstring(a.val) == bitstring(b.val)
            else
                @test a.feature == b.feature
            end
        else
            @test a.op == b.op
            for i in 1:a.degree
                same_tree(get_child(a, i), get_child(b, i))
            end
        end
    end

    function check_member(member, copied)
        @test typeof(copied) === typeof(member)
        for field in fieldnames(typeof(member))
            if field == :tree
                @test typeof(copied.tree) === typeof(member.tree)
                @test isequal(
                    getfield(copied.tree, :metadata), getfield(member.tree, :metadata)
                )
                same_tree(get_tree(member.tree), get_tree(copied.tree))
            else
                @test isequal(getfield(copied, field), getfield(member, field))
            end
        end
        return copied
    end

    function roundtrip(member)
        io = IOBuffer()
        serialize(io, member)
        seekstart(io)
        return check_member(member, deserialize(io))
    end

    proc = only(addprocs(1))
    try
        Distributed.remotecall_eval(Main, [proc], :(using SymbolicRegression))
        rng = MersenneTwister(714)
        for T in (Float32, Float64)
            operators = OperatorEnum(1 => (cos, exp), 2 => (+, -, *, /))
            options = Options(; binary_operators=[+, -, *, /], unary_operators=[cos, exp])
            nan = if T === Float32
                reinterpret(Float32, 0x7fc01234)
            else
                reinterpret(Float64, 0x7ff8000000001234)
            end
            trees = [
                Node(T; feature=1),
                Node(T; val=(-zero(T))),
                Node(T; val=nan),
                Node(T; val=T(Inf)),
                Node(T; val=T(-Inf)),
                Node(; op=1, children=(Node(T; val=T(1.25)),)),
                Node(; op=1, children=(Node(T; feature=2), Node(T; val=T(2.5)))),
            ]
            append!(
                trees,
                [
                    gen_random_tree_fixed_size(rand(rng, 1:25), options, 5, T, rng) for
                    _ in 1:25
                ],
            )
            members = PopMember[]
            for (i, tree) in enumerate(trees)
                member = PopMember(
                    Expression(tree; operators, variable_names=["x1", "x2"]),
                    T(i) / T(8),
                    T(i) / T(16),
                    nothing,
                    count_nodes(tree);
                    deterministic=false,
                    ref=100 + i,
                    parent=i - 1,
                )
                if i == 1
                    disk_serializer = Serializer(IOBuffer())
                    @test which(serialize, (typeof(disk_serializer), typeof(member))).module ===
                        Serialization
                    @test which(
                        deserialize, (typeof(disk_serializer), Type{typeof(member)})
                    ).module === Serialization
                end
                roundtrip(member)
                io = IOBuffer()
                serialize(ClusterSerializer(io), member)
                seekstart(io)
                check_member(member, deserialize(ClusterSerializer(io)))
                push!(members, member)
            end
            for member in (members[2], members[3], members[6], members[7])
                check_member(member, remotecall_fetch(identity, proc, member))
            end

            other = copy(members[2])
            original_tree = getfield(other, :tree)
            setfield!(
                other, :tree,
                typeof(original_tree)(
                    get_tree(original_tree), getfield(members[1].tree, :metadata)
                )
            )
            io = IOBuffer()
            serialize(ClusterSerializer(io), (members[1], other, members[1]))
            seekstart(io)
            first_copy, other_copy, again = deserialize(ClusterSerializer(io))
            @test first_copy === again
            @test getfield(first_copy.tree, :metadata) ===
                getfield(other_copy.tree, :metadata)
            check_member(members[1], first_copy)
            check_member(other, other_copy)
            population = Population(typeof(members[1])[members[1:7]...])
            copied_population = remotecall_fetch(identity, proc, population)
            @test copied_population.n == population.n
            for (original, copied) in zip(population.members, copied_population.members)
                check_member(original, copied)
            end

            shared_metadata = getfield(members[1].tree, :metadata)
            shared_members = typeof(members[1])[]
            for i in 1:33
                member = copy(members[mod1(i, length(members))])
                ex = getfield(member, :tree)
                setfield!(member, :tree, typeof(ex)(get_tree(ex), shared_metadata))
                push!(shared_members, member)
            end
            shared_population = Population(shared_members)
            disk_serializer = Serializer(IOBuffer())
            @test which(serialize, (typeof(disk_serializer), typeof(shared_population))).module ===
                Serialization
            @test which(
                deserialize, (typeof(disk_serializer), Type{typeof(shared_population)})
            ).module === Serialization
            shared_io = IOBuffer()
            serialize(ClusterSerializer(shared_io), shared_population)
            shared_bytes = position(shared_io)
            seekstart(shared_io)
            shared_copy = deserialize(ClusterSerializer(shared_io))
            @test shared_copy.n == 33
            for (original, copied) in zip(shared_population.members, shared_copy.members)
                check_member(original, copied)
            end
            @test all(
                getfield(member.tree, :metadata) ===
                getfield(shared_copy.members[1].tree, :metadata) for
                member in shared_copy.members
            )
            shared_worker_copy = remotecall_fetch(identity, proc, shared_population)
            for (original, copied) in
                zip(shared_population.members, shared_worker_copy.members)
                check_member(original, copied)
            end
            legacy_io = IOBuffer()
            serialize(ClusterSerializer(legacy_io), (shared_members, shared_population.n))
            @test shared_bytes < position(legacy_io)

            shared_hall = HallOfFame(shared_members[1:25], [isodd(i) for i in 1:25])
            hall_io = IOBuffer()
            serialize(ClusterSerializer(hall_io), shared_hall)
            hall_bytes = position(hall_io)
            seekstart(hall_io)
            hall_copy = deserialize(ClusterSerializer(hall_io))
            @test hall_copy.exists == shared_hall.exists
            for (original, copied) in zip(shared_hall.members, hall_copy.members)
                check_member(original, copied)
            end
            hall_worker_copy = remotecall_fetch(identity, proc, shared_hall)
            @test hall_worker_copy.exists == shared_hall.exists
            for (original, copied) in zip(shared_hall.members, hall_worker_copy.members)
                check_member(original, copied)
            end
            legacy_hall_io = IOBuffer()
            serialize(
                ClusterSerializer(legacy_hall_io), (shared_hall.members, shared_hall.exists)
            )
            @test hall_bytes < position(legacy_hall_io)
            @test_throws ArgumentError deserialize(
                ClusterSerializer(IOBuffer(UInt8[0xff])), typeof(shared_hall)
            )
            joined_io = IOBuffer()
            serialize(ClusterSerializer(joined_io), (shared_population, shared_hall))
            seekstart(joined_io)
            joined_pop, joined_hall = deserialize(ClusterSerializer(joined_io))
            @test joined_pop.members[1] === joined_hall.members[1]
            member_vector_io = IOBuffer()
            serialize(
                ClusterSerializer(member_vector_io),
                (shared_population, shared_population.members),
            )
            seekstart(member_vector_io)
            joined_pop, joined_members = deserialize(ClusterSerializer(member_vector_io))
            @test joined_pop.members === joined_members

            changed_member = copy(members[2])
            setfield!(
                changed_member,
                :tree,
                Expression(
                    get_tree(changed_member.tree);
                    operators,
                    variable_names=["different", "names"],
                ),
            )
            distinct_population = Population([members[1], changed_member])
            distinct_io = IOBuffer()
            serialize(ClusterSerializer(distinct_io), distinct_population)
            seekstart(distinct_io)
            distinct_copy = deserialize(ClusterSerializer(distinct_io))
            for (original, copied) in
                zip(distinct_population.members, distinct_copy.members)
                check_member(original, copied)
            end
            @test getfield(distinct_copy.members[1].tree, :metadata) !=
                getfield(distinct_copy.members[2].tree, :metadata)
            repeated_population = Population([shared_members[1], shared_members[1]])
            repeated_io = IOBuffer()
            serialize(ClusterSerializer(repeated_io), repeated_population)
            seekstart(repeated_io)
            repeated_copy = deserialize(ClusterSerializer(repeated_io))
            @test repeated_copy.members[1] === repeated_copy.members[2]
            empty_population = Population(typeof(members[1])[])
            empty_io = IOBuffer()
            serialize(ClusterSerializer(empty_io), empty_population)
            seekstart(empty_io)
            @test isempty(deserialize(ClusterSerializer(empty_io)).members)
            @test_throws ArgumentError deserialize(
                ClusterSerializer(IOBuffer(UInt8[0xff])), typeof(shared_population)
            )
            @test_throws ArgumentError deserialize(
                ClusterSerializer(IOBuffer(UInt8[0xa5, 0xff])), typeof(shared_population)
            )
            disk_io = IOBuffer()
            serialize(disk_io, shared_population)
            seekstart(disk_io)
            disk_copy = deserialize(disk_io)
            @test disk_copy.n == 33
            for (original, copied) in zip(shared_population.members, disk_copy.members)
                check_member(original, copied)
            end

            shared = GraphNode(T; feature=1)
            graph = GraphNode(; op=1, children=(shared, shared))
            fallback = PopMember(
                Expression(graph; operators, variable_names=["x1", "x2"]),
                T(2),
                T(3),
                nothing,
                3;
                deterministic=false,
                ref=789,
                parent=123,
            )
            recovered = roundtrip(fallback)
            @test get_child(get_tree(recovered.tree), 1) ===
                get_child(get_tree(recovered.tree), 2)
            returned_graph = check_member(
                fallback, remotecall_fetch(identity, proc, fallback)
            )
            @test get_child(get_tree(returned_graph.tree), 1) ===
                get_child(get_tree(returned_graph.tree), 2)

            pair = (fallback, fallback)
            io = IOBuffer()
            serialize(ClusterSerializer(io), pair)
            seekstart(io)
            recovered_pair = deserialize(ClusterSerializer(io))
            @test recovered_pair[1] === recovered_pair[2]
            check_member(fallback, recovered_pair[1])

            graph_population = Population([fallback, fallback])
            graph_io = IOBuffer()
            serialize(ClusterSerializer(graph_io), graph_population)
            seekstart(graph_io)
            copied_graph_population = deserialize(ClusterSerializer(graph_io))
            @test copied_graph_population.members[1] === copied_graph_population.members[2]
            @test get_child(get_tree(copied_graph_population.members[1].tree), 1) ===
                get_child(get_tree(copied_graph_population.members[1].tree), 2)

            pair = [fallback, fallback]
            io = IOBuffer()
            serialize(io, pair)
            seekstart(io)
            recovered_pair = deserialize(io)
            @test recovered_pair[1] === recovered_pair[2]
        end
    finally
        rmprocs(proc)
    end
end
