import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def environment_to_graph(env, optimal_q_function):
    """
    Convert an environment and its optimal Q-function into a graph representation.

    Uses the optimal Q-function to determine the edge weights.

    Namely: for each state, the node value is the value function at that state
    (equivalently, the max of the Q function across actions at that state).
    and for edge (s, s') the edge value is q(s, a) where a takes us to s'.

    Since it's an undirected graph, we let edge (s1, s2) be the max of:
    q(s1, the a that takes us to s2) and q(s2, the a that takes us to s1).

    Parameters:
    - env: The environment object.
    - optimal_q_function: The optimal Q-function for the environment. Shape: (num_states, num_actions)

    Returns:
    - G: The graph representation of the environment.
    """

    G = nx.Graph()
    node_values = {}
    edge_values = {}

    for state in env._state_to_grid.keys():
        if state in env.terminal_states:
            _, r = env.get_next_state_and_reward(state, 0)
            G.add_node(state)
            node_values[state] = r
            continue

        next_states = []
        actions = []
        for a in range(env.action_space.n):
            ns, r = env.get_next_state_and_reward(state, a)
            if ns == state:
                continue

            G.add_edge(state, ns)
            key = tuple(sorted((state, ns)))
            if key in edge_values:
                edge_values[key] = np.max([optimal_q_function[state][a], edge_values[key]])
            else:
                edge_values[key] = optimal_q_function[state][a]

            next_states.append(ns)
            actions.append(a)

        if len(next_states) > 0:
            G.add_node(state)
            node_values[state] = np.max(optimal_q_function[state][actions])

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
            critical_nodes.append(node[0])
        elif exception_count == 1:
            regular_nodes.append(node[0])
        else:
            return None, None, None, None  # Not a discrete Morse function

    # Function to classify edges
    for edge in G.edges(data=True):
        edge_value = -edge[2]["edge_value"]
        faces = [edge[0], edge[1]]
        exception_count = sum(-G.nodes[face]["node_value"] >= edge_value for face in faces)

        if exception_count == 0:
            critical_edges.append(edge[:2])
        elif exception_count == 1:
            regular_edges.append(edge[:2])
        else:
            return None, None, None, None  # Not a discrete Morse function

    return critical_nodes, regular_nodes, critical_edges, regular_edges


def visualize_graph(
    G,
    env,
    critical_nodes,
    critical_edges,
    figax=None,
):
    """
    Visualizes a graph with highlighted critical nodes and edges.

    Parameters:
    - G (networkx.Graph): The graph to be visualized.
    - env (Environment): The environment object containing node positions.
    - critical_nodes (list): List of critical nodes to be highlighted.
    - critical_edges (list): List of critical edges to be highlighted.

    Returns:
    - None
    """
    # Get node positions from env._state_to_grid
    nodes_inds = [c for c in critical_nodes]

    pos = {state: env._state_to_grid[state] for state in G.nodes}

    # Create a figure and axis
    if figax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = figax

    # Draw the graph

    label_offset = (0.3, 0.3)
    nx.draw(G, pos, with_labels=False, node_color="black", node_size=50, ax=ax)
    label_pos = {k: (v[0] + label_offset[0], v[1] + label_offset[1]) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=8, font_color="black")

    # Highlight critical nodes and edges in red
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_inds, node_color="red", node_size=50, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="red", ax=ax)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Show the plot
    plt.show()

    return fig, ax


def get_induced_gradient_vector_field(G):
    """
    Compute the induced gradient vector field for a given graph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    list: A list of tuples representing the induced gradient vector field.
          Each tuple contains a face and an edge.

    """
    V = []

    for edge in G.edges(data=True):
        edge_value = -edge[2]["edge_value"]
        faces = [edge[0], edge[1]]

        for face in faces:
            if -G.nodes[face]["node_value"] >= edge_value:
                V.append((face, edge))

    return V


def visualize_induced_vector_field(G, env, critical_nodes, critical_edges, V, figax=None):
    """
    Visualizes the induced vector field on a graph.

    Parameters:
    - G (networkx.Graph): The graph on which the vector field is induced.
    - env: The environment object containing state-to-grid mapping.
    - critical_nodes (list): List of critical nodes.
    - critical_edges (list): List of critical edges.
    - V (list): List of tuples representing the induced vector field.

    Returns:
    - None
    """
    nodes_inds = [c for c in critical_nodes]
    # Get node positions from env._state_to_grid
    pos = {state: env._state_to_grid[state] for state in G.nodes}

    # Create a figure and axis
    if figax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = figax

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color="black", node_size=30, ax=ax)

    # Highlight critical nodes and edges in red
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_inds, node_color="red", node_size=30, ax=ax)
    label_offset = (0.15, 0.15)
    label_pos = {k: (v[0] + label_offset[0], v[1] + label_offset[1]) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=8)

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

    return fig, ax


from collections import deque


def get_vector_field(G, cn=None, ce=None, mode="bfs"):
    """
    Computes the vector field of a graph based on a given set of critical nodes and critical edges.

    Parameters:
        G (networkx.Graph): The input graph.
        cn (list or set, optional): The set of critical nodes. Defaults to an empty set.
        ce (list or set, optional): The set of critical edges. Defaults to an empty set.
        mode (str, optional): The mode of traversal. Can be either 'bfs' (breadth-first search) or 'dfs' (depth-first search). Defaults to 'bfs'.

    Returns:
        tuple: A tuple containing the vector field, updated set of critical nodes, and updated set of critical edges.
            - The vector field is represented as a list of tuples, where each tuple contains a node and its corresponding edge.
            - The updated set of critical nodes includes the original critical nodes and any unused nodes in the graph.
            - The updated set of critical edges includes the original critical edges and any unused edges in the graph.
    """
    if cn is None:
        cn = set()
    if ce is None:
        ce = set()

    if type(cn) == list:
        cn = set(cn)
    if type(ce) == list:
        ce = set(map(tuple, map(sorted, ce)))

    vectors = {"nodes": [], "edges": []}
    visited_nodes = set()

    def dfs(node):
        visited_nodes.add(node)
        for neighbor in G.neighbors(node):
            edge = tuple(sorted((node, neighbor)))
            if neighbor not in vectors["nodes"] and edge not in vectors["edges"] and neighbor not in cn and edge not in ce:
                vectors["nodes"].append(neighbor)
                vectors["edges"].append(edge)
                dfs(neighbor)

    def bfs(start_node):
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            if node not in visited_nodes:
                visited_nodes.add(node)
                for neighbor in G.neighbors(node):
                    edge = tuple(sorted((node, neighbor)))
                    if neighbor not in vectors["nodes"] and edge not in vectors["edges"] and neighbor not in cn and edge not in ce:
                        vectors["nodes"].append(neighbor)
                        vectors["edges"].append(edge)
                        queue.append(neighbor)

    for critical_node in cn:
        if mode == "bfs":
            bfs(critical_node)
        elif mode == "dfs":
            dfs(critical_node)
        else:
            raise ValueError("Invalid mode. Please choose either 'bfs' or 'dfs'.")

    v = [(vectors["nodes"][i], vectors["edges"][i]) for i in range(len(vectors["nodes"]))]

    unused_nodes = set(G.nodes) - set(cn) - set(vectors["nodes"])
    unused_edges = set(map(tuple, map(sorted, G.edges))) - set(ce) - set(map(tuple, map(sorted, vectors["edges"])))

    cn = list(cn.union(unused_nodes))
    ce = list(ce.union(unused_edges))
    return v, cn, ce


# TODO: function that gets all maximal V-paths. -- start from critical nodes,
# follow reverse gradient until you hit a critical edge or a leaf node.
# Implement via depth-first search.


def boundary_operator_f2(G):
    """
    Compute the boundary operator matrix for a given graph G using F2 coefficients.

    Parameters:
    - G (networkx.Graph): The input graph.

    Returns:
    - boundary_operator (numpy.ndarray): The boundary operator matrix.
    """
    num_edges = G.number_of_edges()
    num_vertices = G.number_of_nodes()
    boundary_operator = np.zeros((num_edges, num_vertices))

    for i, edge in enumerate(G.edges()):
        for j, vertex in enumerate(G.nodes()):
            if vertex in edge:
                boundary_operator[i, j] = 1

    return boundary_operator


def boundary_operator_r(G):
    """
    Compute the boundary operator matrix for a given graph G using real coefficients.

    Parameters:
    - G: NetworkX graph object

    Returns:
    - boundary_operator: numpy array representing the boundary operator matrix
    """
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


def euler_characteristic(G):
    """
    Calculate the Euler characteristic of a graph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    int: The Euler characteristic of the graph.

    """
    V = G.number_of_nodes()
    E = G.number_of_edges()
    return V - E


def betti_numbers(G):
    """
    Calculate the Betti numbers of a graph G.

    Parameters:
    - G: The input graph.

    Returns:
    - b_0: The Betti number of dimension 0.
    - b_1: The Betti number of dimension 1.
    """
    D = boundary_operator_r(G)
    rank = np.linalg.matrix_rank(D)
    b_0 = D.shape[1] - rank
    b_1 = D.shape[0] - rank
    return b_0, b_1


def subcomplex(G, c):
    """
    Compute the subcomplex of a graph G induced by nodes and edges with Morse function values less than or equal to c.

    Parameters:
    G (networkx.Graph): The input graph.
    c (float): The threshold value for the Morse function.

    Returns:
    networkx.Graph: The subgraph of G induced by nodes and edges with Morse function values less than or equal to c.
    """
    # Get the nodes whose Morse function value is less than or equal to c
    nodes = [node[0] for node in G.nodes(data=True) if -node[1]["node_value"] <= c]
    # Get the edges whose Morse function value is less than or equal to c
    edges = [edge[:2] for edge in G.edges(data=True) if -edge[2]["edge_value"] <= c]
    # Return the subgraph of G induced by these nodes and edges
    return G.edge_subgraph(edges).subgraph(nodes)


# TODO: homological persistence


def summary_morse_analysis(env, q_function):
    """
    Perform Morse theory analysis on a graph G using a q_function.

    Parameters:
    - G: The input graph.
    - q_function: The q table (q_function) used for analysis.
    """
    G = environment_to_graph(env, q_function)
    (
        critical_nodes,
        regular_nodes,
        critical_edges,
        regular_edges,
    ) = classify_simplices(G)
    if critical_nodes is not None:
        V = get_induced_gradient_vector_field(G)
        m0 = len(critical_nodes)
        m1 = len(critical_edges)
        critical_values = [-G.nodes[v]["node_value"] for v in critical_nodes] + [-G.edges[e]["edge_value"] for e in critical_edges]
        critical_values = sorted(set(critical_values))
    else:
        V, m0, m1, critical_values = None, None, None, None

    summary = {
        "G": G,
        "critical_nodes": critical_nodes,
        "regular_nodes": regular_nodes,
        "critical_edges": critical_edges,
        "regular_edges": regular_edges,
        "induced_vector_field": V,
        "m0": m0,
        "m1": m1,
        "q": q_function,
    }

    return summary
