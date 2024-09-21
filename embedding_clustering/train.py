import copy
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import dataset
import model, utils, network
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure the input and target have the same shape
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(dim=-1)
        elif targets.dim() > inputs.dim():
            targets = targets.squeeze(dim=-1)

        # Check if the shapes match after squeezing
        if inputs.size() != targets.size():
            raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({inputs.size()})")

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def train(hyperparams=None, data_path='gat/data/emb', plot=True):
    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    neo4j_uri = "neo4j+s://bb7d3bb8.databases.neo4j.io"
    neo4j_user = "neo4j"
    neo4j_password = "0vZCoYqO6E9YkZRSFsdKPwHcziXu1-b0h8O9edAzWjM"

    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
  
    ds = dataset.PathwayDataset(data_path)
    ds_train = [ds[0]]
    ds_valid = [ds[1]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)


    net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads, do_train=True)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    ##criterion = nn.BCEWithLogitsLoss(reduction='none')
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0

    max_f1_scores_train = []
    max_f1_scores_valid = []
    
    ##results_path = os.path.join(data_path, 'results')
    results_path = 'gat/results/node_embeddings'

    all_embeddings_initial, cluster_labels_initial = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)  # Flatten 
    save_path_heatmap_initial= os.path.join(results_path, f'heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial = os.path.join(results_path, f'pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial = os.path.join(results_path, f't-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
        
    for data in dl_train:
        graph, _ = data
        node_embeddings_initial= best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels_initial) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}


        stid_dic_initial= {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            stid_dic_initial[nx_graph.nodes[node]['stId']] = node_embeddings_initial[node_to_index[node]]
            
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if cluster not in first_node_stId_in_cluster_initial:
                first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index[node]]
                ##print("nx_graph.nodes[node]['stId']------------------\n", nx_graph.nodes[node]['stId'])

        print('first_node_stId_in_cluster_initial-------------------------------\n', first_node_stId_in_cluster_initial)
        stid_list = list(first_node_stId_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        create_heatmap_with_stid(embedding_list_initial, stid_list, save_path_heatmap_initial)

        break

    visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_t_SNE_initial)
    visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_pca_initial)
    silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)
        
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = [] 
            net.train()
            for data in dl_train:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                optimizer.zero_grad()
                loss_weighted.backward()
                optimizer.step()

                loss_per_graph.append(loss_weighted.item())
                preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                labels = labels.squeeze(1).int()
                f1 = metrics.f1_score(labels, preds)
                f1_per_graph.append(f1)

            running_loss = np.array(loss_per_graph).mean()
            running_f1 = np.array(f1_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1)


            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    loss_per_graph.append(loss_weighted.item())
                    preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                    labels = labels.squeeze(1).int()
                    f1 = metrics.f1_score(labels, preds)
                    f1_per_graph.append(f1)

                running_loss = np.array(loss_per_graph).mean()
                running_f1 = np.array(f1_per_graph).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1)
                
                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)

                if running_loss < best_valid_loss:
                    best_train_loss = running_loss
                    best_valid_loss = running_loss
                    best_f1_score = running_f1
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"Best F1 Score: {best_f1_score}")

            pbar.update(1)
            print(f"Epoch {epoch + 1} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}")

    
    all_embeddings, cluster_labels = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten 
    print('cluster_labels=========================\n', cluster_labels)

    cos_sim = np.dot(all_embeddings, all_embeddings.T)
    norms = np.linalg.norm(all_embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    if plot:
        plot_path = os.path.join(data_path, 'results')
        loss_path = os.path.join(plot_path, f'loss_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(plot_path, f'f1_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        max_f1_path = os.path.join(plot_path, f'max_f1_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        matrix_path = os.path.join(plot_path, f'matrix_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
 
        display_and_save_similarity_matrix(cos_sim, matrix_path)
        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    torch.save(best_model.state_dict(), model_path)


    ##results_path = os.path.join(data_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_path_pca = os.path.join(results_path, f'pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE = os.path.join(results_path, f't-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_heatmap = os.path.join(results_path, f'heatmap_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_heatmap_= os.path.join(results_path, f'heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_heatmap__= os.path.join(results_path, f'heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_.png')
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}


        stid_dic = {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            stid_dic[nx_graph.nodes[node]['stId']] = node_embeddings[node_to_index[node]]
            
        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = nx_graph.nodes[node]['stId']
                first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                ##print("nx_graph.nodes[node]['stId']------------------\n", nx_graph.nodes[node]['stId'])

        print(first_node_stId_in_cluster)
        stid_list = list(first_node_stId_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())
        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        break

    visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    visualize_embeddings_pca(all_embeddings, cluster_labels, stid_list, save_path_pca)
    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"

    save_file = os.path.join(results_path, f'dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)

    '''graph_train, graph_test = utils.create_embedding_with_markers()  # 创建训练和测试的图

    # 从 save_graph_to_neo4j 获取 stid_mapping
    stid_mapping = utils.get_stid_mapping(graph_train)
    print('node_embeddings------------------\n', node_embeddings)

    # 使用 stid_mapping 将嵌入和聚类标签保存到 Neo4j
    save_to_neo4j(graph_train, stid_dic, stid_mapping, neo4j_uri, neo4j_user, neo4j_password)'''
    ##save_to_neo4j(graph_train, node_embeddings, cluster_labels, stid_list, stid_dic, stid_mapping, neo4j_uri, neo4j_user, neo4j_password)
    
    return model_path


def save_to_neo4j_cluster(graph, node_embeddings, cluster_labels, stid_list, stid_dic, stid_mapping, uri, user, password):
    from neo4j import GraphDatabase

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()

    # Clean the database
    session.run("MATCH (n) DETACH DELETE n")

    try:
        # Create nodes with embeddings and cluster labels
        for node_id in stid_dic:
            if node_id not in node_embeddings or node_id not in cluster_labels:
                continue  # Skip if node_id is not in embeddings or cluster labels

            embedding = node_embeddings[node_id]
            cluster = cluster_labels[node_id]
            stId = graph.nodes[node_id].get('stId', None)  # Use .get() to avoid KeyError
            name = graph.nodes[node_id].get('name', None)
            weight = graph.nodes[node_id].get('weight', None)
            significance = graph.nodes[node_id].get('significance', None)
            node_labels = stid_list[cluster] if cluster in stid_list else []  # Get labels for cluster

            session.run(
                "CREATE (n:Node {embedding: $embedding, cluster: $cluster, stId: $stId, name: $name, weight: $weight, significance: $significance}) "
                "SET n :PathwayNode" + (":%s" % (":".join(node_labels)) if node_labels else ""),  # Add labels to node
                embedding=embedding.tolist(), cluster=cluster, stId=stId, name=name, weight=weight, significance=significance
            )

        # Create relationships using the stId mapping
        for source, target in graph.graph_nx.edges():
            source_stId = stid_mapping[source]
            target_stId = stid_mapping[target]
            session.run(
                "MATCH (a {stId: $source_stId}), (b {stId: $target_stId}) "
                "CREATE (a)-[:CONNECTED]->(b)",
                source_stId=source_stId, target_stId=target_stId
            )

    finally:
        session.close()
        driver.close()


def save_to_neo4j(graph, stid_dic, stid_mapping, uri, user, password):
    from neo4j import GraphDatabase

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()

    # Clean the database
    session.run("MATCH (n) DETACH DELETE n")

    try:
        # Create nodes with embeddings and additional attributes
        for node_id in stid_dic:
            embedding = stid_dic[node_id].tolist()  
            stId = stid_mapping[node_id]  # Access stId based on node_id
            name = graph.graph_nx.nodes[node_id]['name']
            weight = graph.graph_nx.nodes[node_id]['weight']
            significance = graph.graph_nx.nodes[node_id]['significance']
            session.run(
                "CREATE (n:Pathway {embedding: $embedding, stId: $stId, name: $name, weight: $weight, significance: $significance})",
                embedding=embedding, stId=stId, name=name, weight=weight, significance=significance
            )

        # Create relationships using the stId mapping
        for source, target in graph.graph_nx.edges():
            source_stId = stid_mapping[source]
            target_stId = stid_mapping[target]
            session.run(
                "MATCH (a {stId: $source_stId}), (b {stId: $target_stId}) "
                "CREATE (a)-[:CONNECTED]->(b)",
                source_stId=source_stId, target_stId=target_stId
            )

    finally:
        session.close()
        driver.close()


def create_heatmap_with_stid(embedding_list, stid_list, save_path):
    # Create DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
    
    # Transpose DataFrame to rotate 90 degrees clockwise
    heatmap_data = heatmap_data.T

    # Generate distinct colors using Seaborn color palette palette = sns.color_palette('dark', num_colors)

    num_colors = len(stid_list)
    palette = sns.color_palette('dark', num_colors)  # Using 'hsv' palette for distinct colors
    color_dict = {stid: palette[i] for i, stid in enumerate(stid_list)}

    # Create heatmap
    ##sns.heatmap(data, cmap='cividis')
    ##sns.heatmap(data, cmap='Blues') 'Greens' sns.heatmap(data, cmap='Spectral') 'coolwarm') 'YlGnBu') viridis cubehelix inferno


    plt.figure(figsize=(10, 10))  # Square figure size
    ax = sns.heatmap(heatmap_data, cmap='tab20', cbar_kws={'label': 'Mean embedding value', 'orientation': 'horizontal', 'fraction': 0.046, 'pad': 0.04})
    '''
    ax = sns.heatmap(heatmap_data, cmap='plasma', cbar_kws={
                                    'label': 'Mean embedding value',
                                    'orientation': 'vertical',
                                    'fraction': 0.046,
                                    'pad': 0.04
                                })
    sns.set(style="white")
    ax = sns.heatmap(heatmap_data, cmap='cubehelix', cbar_kws={
                                    'label': 'Mean embedding value',
                                    'orientation': 'horizontal',
                                    'fraction': 0.046,
                                    'pad': 0.04
                                })
    
    sns.set(style="white")
    ax = sns.heatmap(heatmap_data, cmap='plasma', cbar_kws={
                                    'label': 'Mean embedding value',
                                    'orientation': 'horizontal',
                                    'fraction': 0.1,
                                    'pad': 0.05,
                                    ##'ticks': [0, 0.25, 0.5, 0.75, 1]  # Example of custom ticks
                                })
    '''

    plt.xlabel('Human pathways')
    plt.ylabel('Dimension of the embeddings')
    
    # Customize the color bar to be small and on top
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(size=0, labelsize=8)
    cbar.set_label('Mean embedding value', size=10)
    cbar.ax.set_position([0.2, 0.92, 0.6, 0.03])  # [left, bottom, width, height]

    # Add custom color patches above each column
    for i, stid in enumerate(stid_list):
        color = color_dict[stid]
        rect = mpatches.Rectangle((i, heatmap_data.shape[0]), 1, 2, color=color, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.add_patch(rect)

    # Adjust the axis limits to make space for the patches
    ax.set_xlim(0, len(stid_list))
    ax.set_ylim(0, heatmap_data.shape[0] + 1.5)

    # Custom x-axis labels
    ax.set_xticks(np.arange(len(stid_list)) + 1.0)
    ax.set_xticklabels(stid_list, rotation=30, fontsize=8, ha='right')

    # Custom y-axis labels
    ax.set_yticks(np.arange(heatmap_data.shape[0]) + 1.0)
    ax.set_yticklabels(heatmap_data.index, fontsize=6)

    plt.savefig(save_path)
    plt.close()
    

def calculate_cluster_labels(net, dataloader, device, num_clusters=20):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            embeddings = net.get_node_embeddings(graph.to(device))
            all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    return all_embeddings, cluster_labels


def visualize_embeddings_pca(embeddings, cluster_labels, stid_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i])

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    
def visualize_embeddings_tsne(embeddings, cluster_labels, stid_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def export_to_cytoscape(node_embeddings, cluster_labels, stid_list, output_path):
    # Create a DataFrame for Cytoscape export
    data = {
        'Node': stid_list,
        'Cluster': cluster_labels,
        'Embedding': list(node_embeddings)
    }
    df = pd.DataFrame(data)
    
    # Expand the embedding column into separate columns
    embeddings_df = pd.DataFrame(node_embeddings, columns=[f'Embed_{i}' for i in range(node_embeddings.shape[1])])
    df = df.drop('Embedding', axis=1).join(embeddings_df)

    # Save to CSV for Cytoscape import
    df.to_csv(output_path, index=False)
    print(f"Data exported to {output_path} for Cytoscape visualization.")

    
def display_and_save_similarity_matrix(cos_sim, matrix_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(cos_sim, cmap='viridis', origin='lower', vmin=0, vmax=1.0)
    plt.colorbar()
    plt.title('Similarity Matrix')
    plt.xlabel('Pathway Entities')
    plt.ylabel('Pathway Entities')

    plt.savefig(matrix_path)
    plt.close()

def draw_loss_plot(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

if __name__ == '__main__':
    hyperparams = {
        'num_epochs': 100,
        'out_feats': 128,
        'num_layers': 2,
        'lr': 0.001,
        'batch_size': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    train(hyperparams=hyperparams)
