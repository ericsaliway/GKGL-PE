import json
import networkx as nx
import dgl
import torch

def load_graph_data(file_path):
    # Load data from JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a directed graph
    G_nx = nx.DiGraph()

    # Create a mapping for edge types to numerical values
    edge_type_mapping = {}

    # Iterate over the data and add nodes and edges
    for item in data:
        source = item['n']
        target = item['m']
        relationship_type = item['r']['type']

        # Add source and target nodes
        source_name = source['properties']['name']
        target_name = target['properties']['name']
        G_nx.add_node(source_name, **source['properties'])
        G_nx.add_node(target_name, **target['properties'])

        # Add edge with numerical type
        if relationship_type not in edge_type_mapping:
            edge_type_mapping[relationship_type] = len(edge_type_mapping)
        G_nx.add_edge(source_name, target_name, type=edge_type_mapping[relationship_type])  

    # Convert the NetworkX graph to a DGL graph
    G_dgl = dgl.from_networkx(G_nx, edge_attrs=['type'])

    # Extract node features
    node_features = torch.tensor([node[1]['embedding'] for node in G_nx.nodes(data=True)], dtype=torch.float32)
    G_dgl.ndata['feat'] = node_features

    return G_dgl, node_features
