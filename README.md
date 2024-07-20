This repository provides the code for our research project "GKGL-PE: a GNN-based Knowledge Graph Learning framework for Pathway Embedding".

python auc_sagegraph_bkg/main.py --learning-rate 0.0005 --input-size 2 --hidden-size 128 --epochs 20000

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

-) The built knowledge graph including pathway-pathway and pathway-protein relationships.


## Scripts
The code directory contains the following scripts:

-)The scripts for processing data download from Reactome

-)The scripts for building KG for training embeddings and edge prediction.


## Setup
-)conda create -n kg python=3.9 -y

-)conda activate kg

-)pip install -r requirements.txt


## Get start
python embedding/embedding.py --out_feats 128 --num_layers 4 --num_heads 2 --batch_size 1 --lr 0.01 --num_epochs 1000

python prediction/main.py --out-feats 128 --num-heads 4 --num-layers 6 --lr 0.02 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 200
