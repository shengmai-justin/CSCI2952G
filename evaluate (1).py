import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def calculate_metrics(pred, target):
    """Calculate evaluation metrics"""
    # Pearson correlation
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Remove any NaN values
    mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    pearson = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # Mean Squared Error
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    # Adjusted Rand Index (requires clustering the data)
    # For simplicity, we'll use a basic clustering by binning the values
    pred_bins = np.digitize(pred_flat, np.linspace(min(pred_flat), max(pred_flat), 10))
    target_bins = np.digitize(target_flat, np.linspace(min(target_flat), max(target_flat), 10))
    
    ari = adjusted_rand_score(target_bins, pred_bins)
    
    return {
        'pearson': pearson,
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
    plt.text(0.1, 0.9, f'Pearson r = {pearson:.4f}', 
             transform=g.fig.transFigure, fontsize=15)
    
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    print('Loading test dataset...')
    test_dataset = RiboSeqDataset(args.data_dir)
    
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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f'Loaded trained model from {args.model_path}')
    
    # Evaluate model
    print('Evaluating model...')
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Get data
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and targets
            all_preds.append(outputs)
            all_targets.append(targets)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    # Print metrics
    print('\nEvaluation Results:')
    print(f"Pearson Correlation: {metrics['pearson']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Adjusted Rand Index: {metrics['ari']:.4f}")
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, 'evaluation_metrics.csv'), index=False)
    
    # Visualize predictions
    print('Generating visualizations...')
    visualize_predictions(all_preds, all_targets, args.output_dir)
    
    print(f'Evaluation completed. Results saved to {args.output_dir}')

if __name__ == '__main__':
    main()
