import os
import torch
import numpy as np
from sklearn.cross_decomposition import CCA


def get_embedding(name, data_dir='gat/data/emb/'):
    """Load and return the embedding of the graph with specified index."""
    emb_path = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    embedding = torch.load(os.path.join(emb_path, f'{name}.pth'))
    return embedding

def load_name_to_id(file_path):
    """Load name_to_id mapping from a text file."""
    name_to_id = {}
    with open(file_path, 'r') as file:
        for line in file:
            name, id = line.rsplit(':', 1)
            name_to_id[name.strip()] = id.strip()
    return name_to_id

def load_stids(file_path):
    """Load stids from a text file."""
    with open(file_path, 'r') as file:
        stids = [line.strip() for line in file]
    return stids

def fit_cca_on_toy_data(data_dir='gat/data/emb/'):
    """Perform canonical correlation analysis on the toy datasets.
    The trained CCA model which can be used on other embeddings
    """

    stids = load_stids(os.path.join(data_dir, 'info/stids.txt'))
    name_to_id = load_name_to_id(os.path.join(data_dir, 'info/name_to_id.txt'))
    
    indices_A = [stids.index(id) for name, id in name_to_id.items()]
    indices_B = [stids.index(id) for name, id in name_to_id.items()]
    indices_C = [stids.index(id) for name, id in name_to_id.items()]
    indices_D = [stids.index(id) for name, id in name_to_id.items()]

    y_A = torch.tensor([1.0 if i in indices_A else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_B = torch.tensor([1.0 if i in indices_B else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_C = torch.tensor([1.0 if i in indices_C else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_D = torch.tensor([1.0 if i in indices_D else 0.0 for i in range(len(stids))]).unsqueeze(-1)

    emb_A = get_embedding('gcn', data_dir).reshape(len(stids), -1)
    emb_B = get_embedding('gin', data_dir).reshape(len(stids), -1)
    emb_C = get_embedding('gat', data_dir).reshape(len(stids), -1)
    emb_D = get_embedding('sage', data_dir).reshape(len(stids), -1)
    
    cca = CCA(n_components=1)
    cca.fit(emb_A, y_A)
    emb_A_t = cca.transform(emb_A)
    
    cca.fit(emb_B, y_B)
    emb_B_t = cca.transform(emb_B)
    
    cca.fit(emb_C, y_C)
    emb_C_t = cca.transform(emb_C)
    
    cca.fit(emb_D, y_D)
    emb_D_t = cca.transform(emb_D)
    
    stid = name_to_id['WNT ligand biogenesis and trafficking']
    idx = stids.index(stid)

    print("Transformed coordinates of 'WNT ligand biogenesis and trafficking':")
    print(emb_A_t[idx], emb_B_t[idx], emb_C_t[idx], emb_D_t[idx])
    
    # Canonical correlations
    canonical_correlations = np.corrcoef(emb_A_t.T, y_A.T)[0, 1]
    print("\nCanonical Correlation between emb_A and y_A:", canonical_correlations)
    
    # Loadings
    loadings_A = np.corrcoef(emb_A.T, emb_A_t.T)[0, 1]
    loadings_B = np.corrcoef(emb_B.T, emb_B_t.T)[0, 1]
    loadings_C = np.corrcoef(emb_C.T, emb_C_t.T)[0, 1]
    loadings_D = np.corrcoef(emb_D.T, emb_D_t.T)[0, 1]
    
    print("\nLoadings for emb_A:", loadings_A)
    print("Loadings for emb_B:", loadings_B)
    print("Loadings for emb_C:", loadings_C)
    print("Loadings for emb_D:", loadings_D)
    
    return cca


def fit_cca_on_toy_data_(data_dir='gat/data/emb/'):
    """Perform canonical correlation analysis on the toy datasets.
    The trained CCA model which can be used on other embeddings
    """

    stids = load_stids(os.path.join(data_dir, 'info/stids.txt'))
    name_to_id = load_name_to_id(os.path.join(data_dir, 'info/name_to_id.txt'))
    
    indices_A = [stids.index(id) for name, id in name_to_id.items()]
    indices_B = [stids.index(id) for name, id in name_to_id.items()]
    indices_C = [stids.index(id) for name, id in name_to_id.items()]
    indices_D = [stids.index(id) for name, id in name_to_id.items()]

    y_A = torch.tensor([1.0 if i in indices_A else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_B = torch.tensor([1.0 if i in indices_B else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_C = torch.tensor([1.0 if i in indices_C else 0.0 for i in range(len(stids))]).unsqueeze(-1)
    y_D = torch.tensor([1.0 if i in indices_D else 0.0 for i in range(len(stids))]).unsqueeze(-1)

    emb_A = get_embedding('gcn', data_dir).reshape(len(stids), -1)
    emb_B = get_embedding('gin', data_dir).reshape(len(stids), -1)
    emb_C = get_embedding('gat', data_dir).reshape(len(stids), -1)
    emb_D = get_embedding('sage', data_dir).reshape(len(stids), -1)
    
    cca = CCA(n_components=1)
    cca.fit(emb_A, y_A)
    cca.fit(emb_B, y_B)
    cca.fit(emb_C, y_C)
    cca.fit(emb_D, y_D)

    emb_A_t = cca.transform(emb_A)
    emb_B_t = cca.transform(emb_B)
    emb_C_t = cca.transform(emb_C)
    emb_D_t = cca.transform(emb_D)
    
    stid = name_to_id['WNT ligand biogenesis and trafficking']
    idx = stids.index(stid)

    print("Transformed coordinates of 'WNT ligand biogenesis and trafficking':")
    print(emb_A_t[idx], emb_B_t[idx], emb_C_t[idx], emb_D_t[idx])
    
    # Canonical correlations
    canonical_correlations = np.corrcoef(emb_A_t.T, y_A.T)[0, 1]
    print("\nCanonical Correlation between emb_A and y_A:", canonical_correlations)
    
    # Loadings
    loadings_A = np.corrcoef(emb_A.T, emb_A_t.T)[0, 1]
    loadings_B = np.corrcoef(emb_B.T, emb_B_t.T)[0, 1]
    loadings_C = np.corrcoef(emb_C.T, emb_C_t.T)[0, 1]
    loadings_D = np.corrcoef(emb_D.T, emb_D_t.T)[0, 1]
    
    print("\nLoadings for emb_A:", loadings_A)
    print("Loadings for emb_B:", loadings_B)
    print("Loadings for emb_C:", loadings_C)
    print("Loadings for emb_D:", loadings_D)
    
    return cca

fit_cca_on_toy_data(data_dir='gat/data/emb/')
