import os
import pickle
import urllib.request
import json
from collections import defaultdict, namedtuple
from datetime import datetime
import networkx as nx
import pandas as pd
from py2neo import Graph, Node, Relationship
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import marker
import network
import dataset
import model
import train
from sklearn.model_selection import train_test_split

def get_stid_mapping(graph):
    stid_mapping = {}  # Mapping of node_id to stId
    for node_id, data in graph.graph_nx.nodes(data=True):
        stId = data['stId']
        stid_mapping[node_id] = stId  # Store the mapping
    return stid_mapping  # Return the stId mapping

def save_graph_to_neo4j(graph):
    from py2neo import Graph, Node, Relationship

    neo4j_url = "neo4j+s://7ffb183d.databases.neo4j.io"
    user = "neo4j"
    password = "BGc2jKUI44h_awhU5gEp8NScyuyx-iSSkTbFHEHJRpY"
    
    neo4j_graph = Graph(neo4j_url, auth=(user, password))

    # Clear the existing graph
    neo4j_graph.delete_all()

    # Create nodes
    nodes = {}
    for node_id, data in graph.graph_nx.nodes(data=True):
        stId = data['stId']
        node = Node("Pathway", stId=stId, name=data['name'], weight=data['weight'], significance=data['significance'])
        nodes[node_id] = node
        neo4j_graph.create(node)

    # Create relationships
    for source, target in graph.graph_nx.edges():
        relationship = Relationship(nodes[source], "parent-child", nodes[target])
        neo4j_graph.create(relationship)

def save_graph_to_neo4j_(graph):
    from py2neo import Graph, Node, Relationship

    neo4j_url = "neo4j+s://7ffb183d.databases.neo4j.io"
    user = "neo4j"
    password = "BGc2jKUI44h_awhU5gEp8NScyuyx-iSSkTbFHEHJRpY"
    
    neo4j_graph = Graph(neo4j_url, auth=(user, password))

    # Clear the existing graph
    neo4j_graph.delete_all()

    # Create nodes
    nodes = {}
    stid_mapping = {}  # Mapping of node_id to stId
    for node_id, data in graph.graph_nx.nodes(data=True):
        stId = data['stId']
        node = Node("Pathway", stId=stId, name=data['name'], weight=data['weight'], significance=data['significance'])
        nodes[node_id] = node
        stid_mapping[node_id] = stId  # Store the mapping
        neo4j_graph.create(node)

    # Create relationships
    for source, target in graph.graph_nx.edges():
        relationship = Relationship(nodes[source], "parent-child", nodes[target])
        neo4j_graph.create(relationship)

    return stid_mapping  # Return the stId mapping

def create_network_from_markers(marker_list, p_value, kge):
    enrichment_analysis = marker.Marker(marker_list, p_value)
    graph = network.Network(enrichment_analysis.result, kge)
    return graph


def save_to_disk(graph, save_dir):
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    save_path = os.path.join(save_dir, graph.kge + '.pkl')
    pickle.dump(graph.graph_nx, open(save_path, 'wb'))

def save_stid_to_csv(graph, save_dir):
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    stid_data = {'stId': [node['stId'] for node in graph.graph_nx.nodes.values()]}
    df = pd.DataFrame(stid_data)
    csv_path = os.path.join(save_dir, 'stId_nodes.csv')
    df.to_csv(csv_path, index=False)


def create_embedding_with_markers(p_value=0.05, save=True, data_dir='embedding/data/emb'):
    # Read symbols from the CSV file
    csv_path = 'data/genes_pathways.csv'
    data = pd.read_csv(csv_path)
    symbols = data['symbol'].tolist()
    
    # Split the symbols into train and test sets
    emb_train, emb_test = train_test_split(symbols, test_size=0.3, random_state=42)

    # Create networks for train and test sets
    graph_train = create_network_from_markers(emb_train, p_value, 'emb_train')
    graph_test = create_network_from_markers(emb_test, p_value, 'emb_test')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

def create_embeddings(load_model=True, save=True, data_dir='embedding/data/emb', hyperparams=None, plot=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = dataset.PathwayDataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    os.makedirs(emb_dir, exist_ok=True)

    ##in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams.get('num_heads', 2)  # Default to 2 heads if not specified

    net = model.GATModel(out_feats=out_feats, num_heads=num_heads, num_layers=num_layers).to(device)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    
    for idx in tqdm(range(len(data))):
        graph, name = data[idx]
        graph = graph.to(device)  # Move graph to the same device as net
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding.cpu()
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)

    return embedding_dict

