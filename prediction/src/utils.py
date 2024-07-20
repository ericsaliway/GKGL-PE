import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils import resample
from matplotlib import pyplot as plt


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
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return f1_score(labels, preds_binary, zero_division=1) 

def compute_accuracy(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return accuracy_score(labels, preds_binary)

def compute_precision(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return precision_score(labels, preds_binary, zero_division=1) 

def compute_recall(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    pos_labels = np.ones(pos_score.shape[0])
    neg_labels = np.zeros(neg_score.shape[0])
    labels = np.concatenate([pos_labels, neg_labels])
    threshold = 0.5  # Define threshold for binary classification
    preds_binary = (scores > threshold).astype(int)
    return recall_score(labels, preds_binary, zero_division=1) 


def compute_hits_k(pos_score, neg_score, k=10):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    ranked_scores = np.argsort(-scores)  # Rank in descending order
    top_k = ranked_scores[:k]
    return np.mean(labels[top_k])

def compute_map(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    ranked_indices = np.argsort(-scores)  # Rank in descending order
    sorted_labels = labels[ranked_indices]

    precisions = []
    relevant_docs = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            relevant_docs += 1
            precisions.append(relevant_docs / (i + 1))
    
    if len(precisions) == 0:
        return 0.0
    
    return np.mean(precisions)


def compute_map_k(pos_score, neg_score, k=None):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    ranked_indices = np.argsort(-scores)  # Rank in descending order
    sorted_labels = labels[ranked_indices]

    if k is not None:
        sorted_labels = sorted_labels[:k]

    precisions = []
    relevant_docs = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            relevant_docs += 1
            precisions.append(relevant_docs / (i + 1))

    if len(precisions) == 0:
        return 0.0

    return np.mean(precisions)



# Define metric functions with confidence intervals

def compute_accuracy_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_accuracy(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return accuracy_score(labels, preds_binary)
    
    initial_accuracy = compute_accuracy(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_accuracy, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_accuracy, error_range

def compute_precision_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_precision(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return precision_score(labels, preds_binary, zero_division=1)
    
    initial_precision = compute_precision(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_precision, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_precision, error_range

def compute_f1_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_f1(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return f1_score(labels, preds_binary, zero_division=1)
    
    initial_f1 = compute_f1(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_f1, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_f1, error_range


def compute_focalloss(pos_score, neg_score, alpha=1, gamma=2):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    BCE_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean().item()

def compute_focalloss_with_symmetrical_confidence(pos_score, neg_score, alpha=1, gamma=2, n_bootstraps=1000, confidence_level=0.95):
    initial_focal_loss = compute_focalloss(pos_score, neg_score, alpha, gamma)
    error_range = bootstrap_confidence_interval(
        lambda pos, neg: compute_focalloss(pos, neg, alpha, gamma),
        pos_score, neg_score, n_bootstraps, confidence_level
    )
    return initial_focal_loss, error_range

def compute_loss_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_loss(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return loss_score(labels, preds_binary)
    
    initial_f1 = compute_loss(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_loss, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_f1, error_range

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_auc(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
        return roc_auc_score(labels, scores)
    
    initial_auc = compute_auc(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_auc, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_auc, error_range

def compute_recall_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_recall(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        pos_labels = np.ones(pos_score.shape[0])
        neg_labels = np.zeros(neg_score.shape[0])
        labels = np.concatenate([pos_labels, neg_labels])
        threshold = 0.5
        preds_binary = (scores > threshold).astype(int)
        return recall_score(labels, preds_binary, zero_division=1)
    
    initial_recall = compute_recall(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_recall, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_recall, error_range

def compute_map_with_symmetrical_confidence(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    def compute_map(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).detach().numpy()
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
        ranked_indices = np.argsort(-scores)
        sorted_labels = labels[ranked_indices]

        precisions = []
        relevant_docs = 0
        for i, label in enumerate(sorted_labels):
            if label == 1:
                relevant_docs += 1
                precisions.append(relevant_docs / (i + 1))

        if len(precisions) == 0:
            return 0.0

        return np.mean(precisions)
    
    initial_map = compute_map(pos_score, neg_score)
    error_range = bootstrap_confidence_interval(compute_map, pos_score, neg_score, n_bootstraps, confidence_level)
    
    return initial_map, error_range

# Helper function to perform bootstrap resampling and calculate error range
def bootstrap_confidence_interval(metric_func, pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
    metric_scores = []
    for _ in range(n_bootstraps):
        pos_sampled = resample(pos_score.numpy())
        neg_sampled = resample(neg_score.numpy())
        metric_scores.append(metric_func(torch.tensor(pos_sampled), torch.tensor(neg_sampled)))
    
    lower_bound = np.percentile(metric_scores, ((1 - confidence_level) / 2) * 100)
    upper_bound = np.percentile(metric_scores, (confidence_level + (1 - confidence_level) / 2) * 100)
    error_range = (upper_bound - lower_bound) / 2
    
    return error_range


def plot_scores(epochs, train_f1_scores, val_f1_scores, train_focal_loss_scores, val_focal_loss_scores, train_auc_scores, val_auc_scores, 
                train_map_scores, val_map_scores, train_recall_scores, val_recall_scores,
                train_acc_scores, val_acc_scores, train_precision_scores, val_precision_scores,
                output_path, args):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_f1_scores, label='Training F1 Score')
    plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Scores over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'f1_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))

    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_focal_loss_scores, label='Training FocalLoss Score')
    plt.plot(epochs, val_focal_loss_scores, label='Validation FocalLoss Score')
    plt.xlabel('Epochs')
    plt.ylabel('FocalLoss Score')
    plt.title('Training and Validation FocalLoss Scores over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'loss_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))
    
    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_auc_scores, label='Training AUC')
    plt.plot(epochs, val_auc_scores, label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'auc_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))

    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_map_scores, label='Training mAP')
    plt.plot(epochs, val_map_scores, label='Validation mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Training and Validation mAP over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'mAP_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))


    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_recall_scores, label='Training Recall')
    plt.plot(epochs, val_recall_scores, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'recall_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))


    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_acc_scores, label='Training Accuracy')
    plt.plot(epochs, val_acc_scores, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'acc_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))
    ##plt.figure(figsize=(15, 5))

    ##plt.subplot(1, 2, 1)
    plt.figure()
    plt.plot(epochs, train_precision_scores, label='Training Precision')
    plt.plot(epochs, val_precision_scores, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision over Epochs')
    plt.legend()
    ##plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(os.path.join(output_path, f'precision_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png'))

    plt.show()


