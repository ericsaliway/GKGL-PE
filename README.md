## GKGL-PE: a GNN-based Knowledge Graph Learning framework for Pathway Embedding

This repository provides the code for our research project "GKGL-PE: a GNN-based Knowledge Graph Learning framework for Pathway Embedding".

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

-) The built knowledge graph including pathway-pathway and pathway-protein relationships.


## Get start
python GKGL-PE/embedding_clustering/gat_embedding.py --in_feats 128 --out_feats 128 --num_layers 4 --num_heads 1 --batch_size 1 --lr 0.01 --num_epochs 203

python embedding_clustering/gat_embedding.py --in_feats 20 --out_feats 128 --num_layers 2 --num_heads 1 --batch_size 1 --lr 0.0001 --num_epochs 20000

python embedding/embedding.py --out_feats 128 --num_layers 4 --num_heads 2 --batch_size 1 --lr 0.01 --num_epochs 1000

python prediction/main.py --out-feats 128 --num-heads 4 --num-layers 6 --lr 0.02 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 200
