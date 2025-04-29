import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from dataset import RiboSeqDataset, ContrastiveRiboSeqDataset
from translatomer_cl import TranslatomerCL, RiboSeqAugmenter, matching_loss, EMA

def parse_args():
    parser = argparse.ArgumentParser(description='Translatomer-CL Stage 2 Pretraining')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save models and logs')
    
    # Model arguments
    parser.add_argument('--num_genomic_features', type=int, default=6, help='Number of genomic features')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--stage1_model', type=str, required=True, 
                        help='Path to stage 1 pretrained model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate for teacher model')
    parser.add_argument('--negative_ratio', type=float, default=0.5, 
                        help='Ratio of negative samples to positive samples')
    
    # Curriculum learning arguments
    parser.add_argument('--curriculum_epochs', type=str, default='0:0.0,5:0.2,15:0.5,25:0.8',
                        help='Epochs and ratios for curriculum learning of type 2 hard negatives, format: epoch:ratio')
    
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
    
def parse_curriculum_epochs(curriculum_str):
    """Parse curriculum epochs string into list of (epoch, ratio) pairs"""
    pairs = []
    for pair in curriculum_str.split(','):
        epoch, ratio = pair.split(':')
        pairs.append((int(epoch), float(ratio)))
    return pairs

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Parse curriculum epochs
    curriculum_epochs = parse_curriculum_epochs(args.curriculum_epochs)
    
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
    
    # For pretraining, we use all chromosomes except test chromosomes
    pretrain_chrlist = list(set(chrlist) - set(test_chrlist))
    
    print("Pretraining chromosomes:", pretrain_chrlist)
    print("Excluded chromosomes (test):", test_chrlist)
    
    # Load dataset with chromosome filtering
    print('Loading dataset...')
    try:
        base_dataset = RiboSeqDataset(args.data_dir, chrlist=pretrain_chrlist)
        print(f"Successfully loaded dataset with {len(base_dataset)} samples")
        
        # Print sample shapes for debugging
        if len(base_dataset) > 0:
            sample = base_dataset[0]
            print(f"Gene sequence shape: {sample['gene_seq'].shape}")
            print(f"RNA-seq shape: {sample['rna_seq'].shape}")
            print(f"Target shape: {sample['target'].shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Create augmenter
    augmenter = RiboSeqAugmenter(low_expr_threshold=0.2, high_expr_threshold=0.8)
    
    # Create contrastive dataset with curriculum learning
    contrastive_dataset = ContrastiveRiboSeqDataset(
        base_dataset=base_dataset,
        augmenter=augmenter,
        negative_ratio=args.negative_ratio,
        hard_negative_type2_ratio=0.0,  # Start with 0, will be updated during training
        curriculum_epochs=curriculum_epochs
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
    
    # Initialize student model (from stage 1)
    student_model = TranslatomerCL(
        num_genomic_features=args.num_genomic_features,
        mid_hidden=args.hidden_dim
    ).to(device)
    
    # Load stage 1 pretrained weights
    try:
        student_model.load_state_dict(torch.load(args.stage1_model, map_location=device))
        print(f'Loaded stage 1 model from {args.stage1_model}')
    except Exception as e:
        print(f"Error loading stage 1 model: {e}")
        raise
    
    # Initialize teacher model with EMA
    teacher_model = EMA(student_model, decay=args.ema_decay)
    
    # Define optimizer
    optimizer = optim.Adam(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print('Starting training...')
    for epoch in range(args.epochs):
        # Update curriculum
        contrastive_dataset.update_curriculum(epoch)
        current_type2_ratio = contrastive_dataset.hard_negative_type2_ratio
        print(f'Epoch {epoch+1}: Type 2 hard negative ratio = {current_type2_ratio:.2f}')
        
        train_loss = train_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.6f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_stage2_fold{n_fold}_epoch{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'student_state_dict': student_model.state_dict(),
            'teacher_state_dict': teacher_model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
    
    # Save final models
    student_model_path = os.path.join(args.output_dir, f'translatomer_cl_student_fold{n_fold}.pt')
    teacher_model_path = os.path.join(args.output_dir, f'translatomer_cl_teacher_fold{n_fold}.pt')
    
    torch.save(student_model.state_dict(), student_model_path)
    torch.save(teacher_model.model.state_dict(), teacher_model_path)
    
    print(f'Student model saved to {student_model_path}')
    print(f'Teacher model saved to {teacher_model_path}')

def train_epoch(student_model, teacher_model, loader, optimizer, device, epoch):
    student_model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}')):
        # Get data
        original = batch['original'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)
        is_positive = batch['is_positive'].to(device)
        
        # Print shapes for first batch in first epoch
        if batch_idx == 0 and epoch == 0:
            print(f"Original shape: {original.shape}")
            print(f"Positive shape: {positive.shape}")
            print(f"Negative shape: {negative.shape}")
        
        # Clear gradients
        optimizer.zero_grad()
        
        try:
            # Process positive and negative pairs
            positive_scores = student_model.compute_matching_score(original, positive)
            negative_scores = student_model.compute_matching_score(original, negative)
            
            # Combine scores and labels
            scores = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([torch.ones_like(positive_scores), is_positive])
            
            # Compute matching loss
            loss = matching_loss(scores, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update teacher model with EMA
            teacher_model.update(student_model)
            
            total_loss += loss.item()
            
        except Exception as e:
            print(f"Error during training: {e}")
            print(f"Original shape: {original.shape}")
            print(f"Positive shape: {positive.shape}")
            print(f"Negative shape: {negative.shape}")
            raise
    
    return total_loss / len(loader)

if __name__ == '__main__':
    main()