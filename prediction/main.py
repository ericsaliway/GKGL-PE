import argparse
from src.data_loader import load_graph_data
from src.train import train_and_evaluate

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='MLP Predictor')
    parser.add_argument('--in-feats', type=int, default=128, help='Dimension of the first layer')
    parser.add_argument('--out-feats', type=int, default=128, help='Dimension of the final layer')
    parser.add_argument('--num-heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--input-size', type=int, default=2, help='Input size for the first linear layer')
    parser.add_argument('--hidden-size', type=int, default=16, help='Hidden size for the first linear layer')
    parser.add_argument('--feat-drop', type=float, default=0.0, help='Feature dropout rate')
    parser.add_argument('--attn-drop', type=float, default=0.0, help='Attention dropout rate')
    args = parser.parse_args()

    G_dgl, node_features = load_graph_data('data/neo4j_graph_pass.json')

    train_and_evaluate(args, G_dgl, node_features)



## python link_prediction_gat/main.py --in-feats 128 --out-feats 128 --num-heads 4 --num-layers 2 --lr 0.01 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 2001
