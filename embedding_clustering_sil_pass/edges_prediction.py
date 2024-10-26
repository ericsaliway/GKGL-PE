import json
import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import itertools
import scipy.sparse as sp
from dgl.nn import GATConv
import os
import argparse

# Define the GATModel
class GATModel(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=1, num_heads=4, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, out_feats // num_heads, num_heads, feat_drop=0.6, attn_drop=0.6, activation=F.elu, allow_zero_in_degree=True))
        
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_feats, out_feats // num_heads, num_heads, feat_drop=0.6, attn_drop=0.6, activation=F.elu, allow_zero_in_degree=True))
        
        self.predict = nn.Linear(out_feats, 1)

    def forward(self, graph, features):
        h = features
        for layer in self.layers:
            h = layer(graph, h).flatten(1)
        
        if not self.do_train:
            return h.detach()
        
        logits = self.predict(h)
        return logits

# Load data from JSON file
with open('gat/data/neo4j_graph.json', 'r') as file:
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

# Add self-loops to the DGL graph
##G_dgl = dgl.add_self_loop(G_dgl)

# Extract node features
node_features = torch.tensor([node[1]['embedding'] for node in G_nx.nodes(data=True)], dtype=torch.float32)
G_dgl.ndata['feat'] = node_features

# Display graph information
print(f'NumNodes: {G_dgl.num_nodes()}')
print(f'NumEdges: {G_dgl.num_edges()}')
print(f'NumFeats: {node_features.size(1)}')

# Split edge set for training and testing
u, v = G_dgl.edges()
eids = np.arange(G_dgl.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = G_dgl.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

train_g = dgl.remove_edges(G_dgl, eids[:test_size])

# Create positive and negative graphs for training and testing
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=G_dgl.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=G_dgl.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=G_dgl.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=G_dgl.number_of_nodes())

# Define the edge predictor models
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]

class MLPPredictor_(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(2 * h_feats, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

# Define the loss function and AUC computation
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_f1(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    preds = np.concatenate([pos_score.numpy(), neg_score.numpy()])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return f1_score(labels, preds_binary, zero_division=1) 

def compute_precision(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    preds = np.concatenate([pos_score.numpy(), neg_score.numpy()])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return precision_score(labels, preds_binary, zero_division=1) 

def compute_recall(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    preds = np.concatenate([pos_score.numpy(), neg_score.numpy()])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return recall_score(labels, preds_binary, zero_division=1) 

# Define command-line arguments
parser = argparse.ArgumentParser(description='MLP Predictor')
parser.add_argument('--in-feats', type=int, default=16, help='Dimension of the first layer')
parser.add_argument('--out-feats', type=int, default=128, help='Dimension of the final layer')
parser.add_argument('--num-heads', type=int, default=1, help='Number of heads')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')  # New argument
parser.add_argument('--input-size', type=int, default=2, help='Input size for the first linear layer')
parser.add_argument('--hidden-size', type=int, default=16, help='Hidden size for the first linear layer')
args = parser.parse_args()

# Adjust the input sizes of the linear layers based on command-line arguments
class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

# Initialize and train the model
k = 10
out_feats = k

model = GATModel(node_features.shape[1], out_feats, num_layers=2, num_heads=1, do_train=True)
predictor = MLPPredictor(args.input_size, args.hidden_size)

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=args.lr)

for epoch in range(args.epochs):
    h = model(train_g, train_g.ndata['feat'])
    pos_score = predictor(train_pos_g, h)
    neg_score = predictor(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        h = model(G_dgl, G_dgl.ndata['feat'])
        pos_score = predictor(test_pos_g, h)
        neg_score = predictor(test_neg_g, h)
        auc = compute_auc(pos_score, neg_score)
        f1 = compute_f1(pos_score, neg_score)
        precision = compute_precision(pos_score, neg_score)
        recall = compute_recall(pos_score, neg_score)

        print(f'Epoch {epoch:05d} | Loss {loss.item():.4f} | AUC {auc:.4f} | F1 {f1:.4f} | Precision {precision:.4f} | Recall {recall:.4f}')


# Save the model
model_path = 'gat/data/emb/predictor_model.pth'
torch.save(predictor.state_dict(), model_path)

# Save output to the results folder
output = {'AUC': auc, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}
output_path = 'gat/results/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(os.path.join(output_path, 'output.json'), 'w') as f:
    json.dump(output, f)

