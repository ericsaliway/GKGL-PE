import os
import pickle
import torch
import dgl
import utils
import model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create embeddings and save to disk.')
    parser.add_argument('--data_dir', type=str, default='gat/data/emb', help='Directory to save the data.')
    parser.add_argument('--output-file', type=str, default='gat/data/emb/embeddings.pkl', help='File to save the embeddings')
    parser.add_argument('--p_value', type=float, default=0.05, help='P-value threshold for creating embeddings.')
    parser.add_argument('--save', type=bool, default=True, help='Flag to save embeddings.')
    parser.add_argument('--num_epochs', type=int, default=20000, help='Number of epochs for training.')
    parser.add_argument('--in_feats', type=int, default=20, help='Number of input features.')
    parser.add_argument('--out_feats', type=int, default=128, help='Number of output features.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for GAT model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--print-embeddings', action='store_true', help='Print the embeddings dictionary')

    args = parser.parse_args()

    # Main script to create embeddings and save to disk
    graph_train, graph_test = utils.create_embedding_with_markers(
        p_value=args.p_value, 
        save=args.save, 
        data_dir=args.data_dir
    )

    hyperparameters = {
        'num_epochs': args.num_epochs,
        'in_feats': args.in_feats,
        'out_feats': args.out_feats,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,  # Added num_heads to hyperparameters
        'batch_size': args.batch_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': args.lr,
    }

    embedding_dict = utils.create_embeddings(
        data_dir=args.data_dir, 
        load_model=False, 
        hyperparams=hyperparameters
    )
    
    # Print the embeddings dictionary if required
    if args.print_embeddings:
        print(embedding_dict)

    # Save embeddings to file
    with open(args.output_file, 'wb') as f:
        pickle.dump(embedding_dict, f)
    print(f"Embeddings saved to {args.output_file}")
    

if __name__ == '__main__':
    main()

## python gat_copy_8/gat_embedding.py --in_feats 20 --out_feats 128 --num_layers 2 --num_heads 1 --batch_size 1 --lr 0.0001 --num_epochs 1011
## python gat/gat_embedding.py --in_feats 20 --out_feats 128 --num_layers 2 --num_heads 1 --batch_size 1 --lr 0.0001 --num_epochs 20002