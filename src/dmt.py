import networkx as nx
import numpy as np
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
    for node in G.nodes(data=True):
        node_value = -node[1]["node_value"]  # take negative of value fn to get morse fn
        edges = G.edges(node[0], data=True)
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
    nodes_inds = [c[0] for c in critical_nodes]
    pos = {state: env._state_to_grid[state] for state in G.nodes}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color="black", node_size=50, ax=ax)

    # Highlight critical nodes and edges in red
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes_inds, node_color="red", node_size=50, ax=ax
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
    nodes_inds = [c[0] for c in critical_nodes]
    # Get node positions from env._state_to_grid
    pos = {state: env._state_to_grid[state] for state in G.nodes}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color="black", node_size=30, ax=ax)

    # Highlight critical nodes and edges in red
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes_inds, node_color="red", node_size=30, ax=ax
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


# TODO: function that gets all maximal V-paths. -- start from critical nodes,
# follow reverse gradient until you hit a critical edge or a leaf node.
# Implement via depth-first search.

# TODO: visualize as Hasse diagram (not sure if useful enough to be worth the effort)

# TODO: Homology calculations. betti numbers as a fn of:
# - level sets
# - learning process
# -

### HOMOLOGY


def boundary_operator_f2(G):
    num_edges = G.number_of_edges()
    num_vertices = G.number_of_nodes()
    boundary_operator = np.zeros((num_edges, num_vertices))

    for i, edge in enumerate(G.edges()):
        for j, vertex in enumerate(G.nodes()):
            if vertex in edge:
                boundary_operator[i, j] = 1

    return boundary_operator


def boundary_operator_r(G):
    num_edges = G.number_of_edges()
    num_vertices = G.number_of_nodes()
    boundary_operator = np.zeros((num_edges, num_vertices))

    for i, edge in enumerate(G.edges()):
        for j, vertex in enumerate(G.nodes()):
            if vertex == edge[0]:
                boundary_operator[i, j] = 1
            elif vertex == edge[1]:
                boundary_operator[i, j] = -1

    return boundary_operator


def betti_numbers(G):
    D = boundary_operator_r(G)
    rank = np.linalg.matrix_rank(D)
    b_0 = D.shape[1] - rank
    b_1 = D.shape[0] - rank
    return b_0, b_1


# TODO: homological sequences
# TODO: homological persistence
# TODO: level subcomplexes.
