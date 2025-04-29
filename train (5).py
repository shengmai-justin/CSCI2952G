import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from dataset import RiboSeqDataset
from translatomer_cl import TranslatomerCL

def parse_args():
    parser = argparse.ArgumentParser(description='Translatomer-CL Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save models and logs')
    parser.add_argument('--pretrained_model', type=str, required=True, 
                        help='Path to pretrained model (stage 2 teacher model)')
    
    # Model arguments
    parser.add_argument('--num_genomic_features', type=int, default=6, help='Number of genomic features')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    
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
    
    return pearson, mse, ari

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Get fold-specific chromosomes
    n_fold = args.fold
    print(f'Using FOLD {n_fold}')
    print('--------------------------------')
    
    # Each fold takes 2 chromosomes for validation and 2 for testing
    val_chrlist = chrlist[2*n_fold:2*n_fold+2]
    test_chrlist = chrlist[2*n_fold+2:2*n_fold+4]
    if not test_chrlist:  # Handle last fold
        test_chrlist = chrlist[0:2]
    
    # The rest are for training
    train_chrlist = list(set(chrlist) - set(val_chrlist) - set(test_chrlist))
    
    print("Training chromosomes:", train_chrlist)
    print("Validation chromosomes:", val_chrlist)
    print("Test chromosomes:", test_chrlist)
    
    # Load datasets with chromosome filtering
    print('Loading datasets...')
    try:
        train_dataset = RiboSeqDataset(args.data_dir, chrlist=train_chrlist)
        val_dataset = RiboSeqDataset(args.data_dir, chrlist=val_chrlist)
        
        print(f"Loaded training dataset with {len(train_dataset)} samples")
        print(f"Loaded validation dataset with {len(val_dataset)} samples")
        
        # Print sample shapes for debugging
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Gene sequence shape: {sample['gene_seq'].shape}")
            print(f"RNA-seq shape: {sample['rna_seq'].shape}")
            print(f"Target shape: {sample['target'].shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
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
    
    # Load pretrained weights
    try:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print(f'Loaded pretrained model from {args.pretrained_model}')
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        raise
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    metrics_history = []
    
    # Training loop
    print('Starting training...')
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, metrics = validate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        val_losses.append(val_loss)
        metrics_history.append(metrics)
        
        pearson, mse, ari = metrics
        print(f'Epoch {epoch+1}/{args.epochs}, '
              f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
              f'Pearson: {pearson:.4f}, MSE: {mse:.6f}, ARI: {ari:.4f}')
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(args.output_dir, f'translatomer_cl_best_fold{n_fold}.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f'Early stopping after {epoch+1} epochs')
                break
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_fold{n_fold}_epoch{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
        }, checkpoint_path)
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'pearson': [m[0] for m in metrics_history],
        'mse': [m[1] for m in metrics_history],
        'ari': [m[2] for m in metrics_history]
    }
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.output_dir, f'training_history_fold{n_fold}.csv'), index=False)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot([m[0] for m in metrics_history])
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    
    plt.subplot(1, 3, 3)
    plt.plot([m[2] for m in metrics_history])
    plt.xlabel('Epoch')
    plt.ylabel('Adjusted Rand Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'training_curves_fold{n_fold}.png'))
    
    print('Training completed!')

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc='Training')):
        # Get data
        gene_seq = batch['gene_seq'].to(device)
        rna_seq = batch['rna_seq'].to(device)
        targets = batch['target'].to(device)
        
        # Print shapes for first batch
        if batch_idx == 0:
            print(f"Gene sequence shape: {gene_seq.shape}")
            print(f"RNA-seq shape: {rna_seq.shape}")
            print(f"Targets shape: {targets.shape}")
        
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
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Validating')):
            # Get data
            gene_seq = batch['gene_seq'].to(device)
            rna_seq = batch['rna_seq'].to(device)
            targets = batch['target'].to(device)
            
            try:
                # Use same tensor connection logic as in training
                dim_5 = None
                for i, dim_size in enumerate(gene_seq.shape):
                    if dim_size == 5:
                        dim_5 = i
                        break
                
                if dim_5 == 1:
                    inputs = torch.cat([gene_seq, rna_seq.unsqueeze(1)], dim=1)
                elif dim_5 == 2:
                    inputs = torch.cat([gene_seq, rna_seq.unsqueeze(2)], dim=2)
                else:
                    gene_seq_reshaped = gene_seq.permute(0, 2, 1) if gene_seq.dim() == 3 else gene_seq
                    inputs = torch.cat([gene_seq_reshaped, rna_seq.unsqueeze(1)], dim=1)
                    
            except RuntimeError as e:
                print(f"Error connecting tensors during validation:")
                print(f"gene_seq shape: {gene_seq.shape}")
                print(f"rna_seq shape: {rna_seq.shape}")
                raise e
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            all_preds.append(outputs)
            all_targets.append(targets)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    return total_loss / len(loader), metrics

if __name__ == '__main__':
    main()