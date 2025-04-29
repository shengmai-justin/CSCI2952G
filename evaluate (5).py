import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr  # 添加这个导入用于计算Spearman相关系数
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from dataset import RiboSeqDataset
from translatomer_cl import TranslatomerCL

def parse_args():
    parser = argparse.ArgumentParser(description='Translatomer-CL Evaluation')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the test data')
    parser.add_argument('--output_dir', type=str, default='./evaluation', help='Directory to save evaluation results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    
    # Model arguments
    parser.add_argument('--num_genomic_features', type=int, default=6, help='Number of genomic features')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    
    # Fold arguments (for chromosome-based splitting)
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation (0-10)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def calculate_metrics(pred, target):
    """Calculate evaluation metrics"""
    # Flatten predictions and targets
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Remove any NaN values
    mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    # Pearson correlation
    pearson = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Spearman correlation
    spearman, _ = spearmanr(pred_flat, target_flat)
    
    # Mean Squared Error
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    # Adjusted Rand Index (requires clustering the data)
    # For simplicity, we'll use a basic clustering by binning the values
    pred_bins = np.digitize(pred_flat, np.linspace(min(pred_flat), max(pred_flat), 10))
    target_bins = np.digitize(target_flat, np.linspace(min(target_flat), max(target_flat), 10))
    
    ari = adjusted_rand_score(target_bins, pred_bins)
    
    return {
        'pearson': pearson,
        'spearman': spearman,  # 添加Spearman相关系数
        'mse': mse,
        'ari': ari
    }

def visualize_predictions(pred, target, output_dir):
    """Visualize predictions vs targets"""
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Remove any NaN values
    mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    # Scatter plot with density heatmap
    plt.figure(figsize=(10, 8))
    
    # Create a joint plot
    g = sns.jointplot(
        x=target_flat,
        y=pred_flat,
        kind="hex",
        color="#4CB391",
        height=8,
        ratio=3,
        space=0,
        xlim=(min(target_flat), max(target_flat)),
        ylim=(min(pred_flat), max(pred_flat))
    )
    
    # Add pearson correlation coefficient
    pearson = np.corrcoef(target_flat, pred_flat)[0, 1]
    spearman, _ = spearmanr(target_flat, pred_flat)  # 计算Spearman相关系数
    
    # 在图上添加两种相关系数
    plt.text(0.1, 0.9, f'Pearson r = {pearson:.4f}', transform=g.fig.transFigure, fontsize=15)
    plt.text(0.1, 0.85, f'Spearman r = {spearman:.4f}', transform=g.fig.transFigure, fontsize=15)  # 添加Spearman相关系数显示
    
    # Add diagonal line
    lims = [
        np.min([g.ax_joint.get_xlim(), g.ax_joint.get_ylim()]),
        np.max([g.ax_joint.get_xlim(), g.ax_joint.get_ylim()]),
    ]
    g.ax_joint.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
    
    # Set labels
    g.set_axis_labels('Ground Truth', 'Predicted', fontsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=300)
    plt.close()
    
    # Additionally create a sample-wise visualization for a few samples
    num_samples = min(5, pred.size(0))
    
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        
        sample_pred = pred[i].cpu().numpy()
        sample_target = target[i].cpu().numpy()
        
        plt.plot(sample_target, label='Ground Truth', alpha=0.7)
        plt.plot(sample_pred, label='Predicted', alpha=0.7)
        
        if i == 0:
            plt.legend()
            
        plt.title(f'Sample {i+1}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'), dpi=300)
    plt.close()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    fold_output_dir = os.path.join(args.output_dir, f'fold{args.fold}')
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define chromosome list
    chrlist = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9',
               'chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17',
               'chr18','chr19','chr20','chr21','chr22']
    
    # Shuffle chromosomes (using fixed seed for reproducibility)
    random.seed(args.seed)
    random.shuffle(chrlist)
    
    # Get fold-specific test chromosomes
    n_fold = args.fold
    print(f'Using FOLD {n_fold} for evaluation')
    print('--------------------------------')
    
    # Get test chromosomes for this fold
    test_chrlist = chrlist[2*n_fold+2:2*n_fold+4]
    if not test_chrlist:  # Handle last fold
        test_chrlist = chrlist[0:2]
    
    print("Test chromosomes:", test_chrlist)
    
    # Load test dataset with chromosome filtering
    print('Loading test dataset...')
    try:
        test_dataset = RiboSeqDataset(args.data_dir, chrlist=test_chrlist)
        print(f"Successfully loaded test dataset with {len(test_dataset)} samples")
        
        # Print sample shapes for debugging
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            print(f"Gene sequence shape: {sample['gene_seq'].shape}")
            print(f"RNA-seq shape: {sample['rna_seq'].shape}")
            print(f"Target shape: {sample['target'].shape}")
        else:
            print("Warning: Test dataset is empty!")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = TranslatomerCL(
        num_genomic_features=args.num_genomic_features,
        mid_hidden=args.hidden_dim
    ).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Loaded trained model from {args.model_path}')
    except Exception as e:
        print(f"Error loading trained model: {e}")
        raise
    
    # Evaluate model
    print('Evaluating model...')
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            # Get data
            gene_seq = batch['gene_seq'].to(device)
            rna_seq = batch['rna_seq'].to(device)
            targets = batch['target'].to(device)
            
            # Print shapes for first batch
            if batch_idx == 0:
                print(f"Gene sequence shape: {gene_seq.shape}")
                print(f"RNA-seq shape: {rna_seq.shape}")
                print(f"Target shape: {targets.shape}")
            
            try:
                # Find dimension with size 5 (one-hot encoding dimension)
                dim_5 = None
                for i, dim_size in enumerate(gene_seq.shape):
                    if dim_size == 5:
                        dim_5 = i
                        break
                
                # Create inputs based on tensor shape
                if dim_5 == 1:  # If gene_seq shape is [batch, 5, seq_len]
                    inputs = torch.cat([gene_seq, rna_seq.unsqueeze(1)], dim=1)
                elif dim_5 == 2:  # If gene_seq shape is [batch, seq_len, 5]
                    inputs = torch.cat([gene_seq, rna_seq.unsqueeze(2)], dim=2)
                else:
                    # Default case - try reshaping
                    print(f"Warning: Could not find dimension with size 5, trying alternative method")
                    gene_seq_reshaped = gene_seq.permute(0, 2, 1) if gene_seq.dim() == 3 else gene_seq
                    inputs = torch.cat([gene_seq_reshaped, rna_seq.unsqueeze(1)], dim=1)
                    
            except RuntimeError as e:
                print(f"Error connecting tensors:")
                print(f"gene_seq shape: {gene_seq.shape}")
                print(f"rna_seq shape: {rna_seq.shape}")
                raise e
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and targets
            all_preds.append(outputs)
            all_targets.append(targets)
    
    # Check if we have any predictions
    if not all_preds:
        print("Error: No predictions generated. Test dataset might be empty.")
        return
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    # Print metrics
    print('\nEvaluation Results:')
    print(f"Pearson Correlation: {metrics['pearson']:.4f}")
    print(f"Spearman Correlation: {metrics['spearman']:.4f}")  # 打印Spearman相关系数
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Adjusted Rand Index: {metrics['ari']:.4f}")
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(fold_output_dir, f'evaluation_metrics_fold{n_fold}.csv'), index=False)
    
    # Visualize predictions
    print('Generating visualizations...')
    visualize_predictions(all_preds, all_targets, fold_output_dir)
    
    print(f'Evaluation completed. Results saved to {fold_output_dir}')

if __name__ == '__main__':
    main()