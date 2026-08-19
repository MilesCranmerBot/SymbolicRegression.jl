import json
import networkx as nx
from tqdm import tqdm


def load_pysr_graph(jsonl_path, progress=True):
    """Load a PySR trace JSONL file into a NetworkX directed graph.

    Args:
        jsonl_path: Path to pysr_trace.jsonl
        progress: Show progress bars

    Returns:
        NetworkX DiGraph with:
        - Node attributes: tree, cost, loss, parent
        - Edge attributes: type, time, mutation_details
    """
    G = nx.DiGraph()
    event_counts = {"mutate": 0, "crossover": 0, "tuning": 0, "other": 0}
    edge_counts = {"parent": 0, "mutate": 0, "crossover": 0, "tuning": 0}
    sample = None

    with open(jsonl_path) as trace_file:
        records = tqdm(
            trace_file,
            desc="Processing trace records" if progress else None,
            disable=not progress,
        )
        for line in records:
            record = json.loads(line)
            if record.get("record_type") != "iteration":
                continue

            for member_id_str, member_data in record.get("mutations", {}).items():
                member_id = int(member_id_str)
                previous_events = (
                    G.nodes[member_id].get("events", []) if member_id in G else []
                )
                node_data = dict(member_data)
                node_data["events"] = previous_events + member_data.get("events", [])
                G.add_node(member_id, **node_data)
                sample = sample or (member_id, member_data)

                parent_id = member_data.get("parent")
                if parent_id is not None:
                    parent_id = int(parent_id)
                    if parent_id in G and not G.has_edge(parent_id, member_id):
                        G.add_edge(parent_id, member_id, type="parent")
                        edge_counts["parent"] += 1

                for event in member_data.get("events", []):
                    event_type = event.get("type")
                    event_counts[event_type if event_type in event_counts else "other"] += 1

                    if event_type == "mutate":
                        child_id = event.get("child")
                        if child_id is None:
                            continue
                        G.add_edge(
                            member_id,
                            int(child_id),
                            type="mutate",
                            time=event.get("time"),
                            details=event.get("mutation", {}),
                        )
                        edge_counts["mutate"] += 1

                    elif event_type == "crossover":
                        parent1 = event.get("parent1")
                        parent2 = event.get("parent2")
                        child1 = event.get("child1")
                        child2 = event.get("child2")
                        parent1 = int(parent1) if parent1 is not None else None
                        parent2 = int(parent2) if parent2 is not None else None

                        for parent, partner in ((parent1, parent2), (parent2, parent1)):
                            if parent is None:
                                continue
                            for child in (child1, child2):
                                if child is None:
                                    continue
                                G.add_edge(
                                    parent,
                                    int(child),
                                    type="crossover",
                                    time=event.get("time"),
                                    partner=partner,
                                    details=event.get("details", {}),
                                )
                                edge_counts["crossover"] += 1

                    elif event_type == "tuning":
                        child_id = event.get("child")
                        if child_id is None:
                            continue
                        G.add_edge(
                            member_id,
                            int(child_id),
                            type="tuning",
                            time=event.get("time"),
                            details=event.get("mutation", {}),
                        )
                        edge_counts["tuning"] += 1

    if sample is not None:
        sample_id, sample_data = sample
        print("\nSample member data:")
        print(f"ID: {sample_id}")
        print(f"Parent: {sample_data.get('parent')}")
        print(f"Events: {len(sample_data.get('events', []))}")
        if sample_data.get("events"):
            print("First event:", sample_data["events"][0])

    print("\nEvent counts:", event_counts)
    print("Edge counts:", edge_counts)
    return G


def simplify_graph(G):
    """Create a simplified version with only essential attributes"""
    simple_G = nx.DiGraph()

    for node, data in G.nodes(data=True):
        # Keep full tree and add a truncated version for display
        tree = data.get("tree", "No equation")
        display_tree = str(tree)
        if len(display_tree) > 30:  # Truncate long equations
            display_tree = display_tree[:27] + "..."

        simple_G.add_node(
            node,
            cost=data.get("cost"),
            loss=data.get("loss"),
            tree=tree,
            display_tree=display_tree,
        )

    for u, v, data in G.edges(data=True):
        simple_G.add_edge(u, v, type=data.get("type"), time=data.get("time"))

    return simple_G


if __name__ == "__main__":
    # Example usage
    G = load_pysr_graph("pysr_trace.jsonl")

    # Basic stats
    print(f"Loaded graph with {len(G)} nodes and {G.size()} edges")

    # Save simplified version
    simple_G = simplify_graph(G)
    nx.write_graphml(simple_G, "pysr_graph.graphml")
