import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import RiboSeqDataset, ContrastiveRiboSeqDataset
from translatomer_cl import TranslatomerCL, RiboSeqAugmenter, nt_xent_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Translatomer-CL Stage 1 Pretraining')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save models and logs')
    
    # Model arguments
    parser.add_argument('--num_genomic_features', type=int, default=6, help='Number of genomic features')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for NT-Xent loss')
    parser.add_argument('--negative_ratio', type=float, default=0.3, 
                        help='Ratio of negative samples to positive samples')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    print('Loading dataset...')
    try:
        base_dataset = RiboSeqDataset(args.data_dir)
        print(f"Successfully loaded dataset with {len(base_dataset)} samples")
        
        # Print sample shapes for debugging
        sample = base_dataset[0]
        print(f"Gene sequence shape: {sample['gene_seq'].shape}")
        print(f"RNA-seq shape: {sample['rna_seq'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Create augmenter
    augmenter = RiboSeqAugmenter(low_expr_threshold=0.2, high_expr_threshold=0.8)
    
    # Create contrastive dataset
    contrastive_dataset = ContrastiveRiboSeqDataset(
        base_dataset=base_dataset,
        augmenter=augmenter,
        negative_ratio=args.negative_ratio
    )
    print(f"Created contrastive dataset with {len(contrastive_dataset)} samples")
    
    # Create data loader
    train_loader = DataLoader(
        contrastive_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = TranslatomerCL(
        num_genomic_features=args.num_genomic_features,
        mid_hidden=args.hidden_dim
    ).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print('Starting training...')
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            epoch=epoch
        )
        
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.6f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_stage1_epoch{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'translatomer_cl_stage1.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'Model saved to {final_model_path}')

def train_epoch(model, loader, optimizer, device, temperature, epoch):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc=f'Epoch {epoch+1}'):
        # Get data
        original = batch['original'].to(device)
        positive = batch['positive'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        try:
            # Get embeddings
            z_original = model.get_embedding(original)
            z_positive = model.get_embedding(positive)
            
            # Compute loss
            loss = nt_xent_loss(z_original, z_positive, temperature=temperature)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        except Exception as e:
            print(f"Error during training: {e}")
            print(f"Original shape: {original.shape}")
            print(f"Positive shape: {positive.shape}")
            raise
    
    return total_loss / len(loader)

if __name__ == '__main__':
    main()