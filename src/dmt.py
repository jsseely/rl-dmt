import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def environment_to_graph(env, optimal_q_function):
    # Create a directed graph
    G = nx.Graph()

    sorted_tuple = lambda x: tuple(sorted(x))

    node_values = {}
    edge_values = {}

    # Add vertices (states) to the graph
    for state in env._state_to_grid.keys():
        G.add_node(state)
        node_values[state] = np.max(optimal_q_function[state])

    # Add edges (actions) to the graph
    for state in G.nodes:
        if state in env.terminal_states:
            continue
        for action in range(env.action_space.n):
            next_state, reward = env.get_next_state_and_reward(state, action)
            if next_state == state:
                continue
            if sorted_tuple((state, next_state)) in edge_values:
                current_q_value = edge_values[sorted_tuple((state, next_state))]
            else:
                current_q_value = -np.inf

            G.add_edge(state, next_state)
            edge_values[sorted_tuple((state, next_state))] = np.max(
                [optimal_q_function[state][action], current_q_value]
            )

    nx.set_node_attributes(G, node_values, "node_value")
    nx.set_edge_attributes(G, edge_values, "edge_value")

    return G


def classify_simplices(G):
    critical_nodes = []
    regular_nodes = []
    critical_edges = []
    regular_edges = []

    # Function to classify nodes
    for node, data in G.nodes(data=True):
        node_value = -data["node_value"]  # take negative of value fn to get morse fn
        edges = G.edges(node, data=True)
        exception_count = sum(
            -edge_data["edge_value"] <= node_value for _, _, edge_data in edges
        )  # take negative of edge value to get morse fn

        if exception_count == 0:
            critical_nodes.append(node)
        elif exception_count == 1:
            regular_nodes.append(node)
        else:
            print(node, data, exception_count)
            return None, None, None, None  # Not a discrete Morse function

    # Function to classify edges
    for edge in G.edges(data=True):
        edge_value = -edge[2]["edge_value"]
        faces = [edge[0], edge[1]]
        exception_count = sum(
            -G.nodes[face]["node_value"] >= edge_value for face in faces
        )

        if exception_count == 0:
            critical_edges.append(edge)
        elif exception_count == 1:
            regular_edges.append(edge)
        else:
            print(edge, edge_value, exception_count)
            return None, None, None, None  # Not a discrete Morse function

    return critical_nodes, regular_nodes, critical_edges, regular_edges


def visualize_graph(G, env, critical_nodes, critical_edges):
    # Get node positions from env._state_to_grid
    pos = {state: env._state_to_grid[state] for state in G.nodes}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color="black", node_size=50, ax=ax)

    # Highlight critical nodes and edges in red
    nx.draw_networkx_nodes(
        G, pos, nodelist=critical_nodes, node_color="red", node_size=50, ax=ax
    )
    nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="red", ax=ax)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Show the plot
    plt.show()


def get_induced_gradient_vector_field(G):
    V = []

    for edge in G.edges(data=True):
        edge_value = -edge[2]["edge_value"]
        faces = [edge[0], edge[1]]

        for face in faces:
            if -G.nodes[face]["node_value"] >= edge_value:
                V.append((face, edge))

    return V


def visualize_induced_vector_field(G, env, critical_nodes, critical_edges, V):
    # Get node positions from env._state_to_grid
    pos = {state: env._state_to_grid[state] for state in G.nodes}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color="black", node_size=30, ax=ax)

    # Highlight critical nodes and edges in red
    nx.draw_networkx_nodes(
        G, pos, nodelist=critical_nodes, node_color="red", node_size=30, ax=ax
    )
    nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="red", ax=ax)

    # Draw arrows for induced vector field
    for v, e in V:
        if v in G and G.has_edge(e[0], e[1]):
            v_pos = pos[v]
            e_pos = np.mean([pos[e[0]], pos[e[1]]], axis=0)
            ax.arrow(
                v_pos[0],
                v_pos[1],
                e_pos[0] - v_pos[0],
                e_pos[1] - v_pos[1],
                head_width=0.25,
                head_length=0.15,
                fc="black",
                linewidth=0,
                length_includes_head=True,
            )

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Show the plot
    plt.show()
