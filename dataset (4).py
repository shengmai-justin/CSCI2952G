import os
import glob
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class RiboSeqDataset(Dataset):
    """
    Dataset for Ribosome Profiling data with chromosome-based filtering
    """
    def __init__(self, data_dir, chrlist=None):
        """
        Args:
            data_dir: Directory containing the preprocessed data
            chrlist: List of chromosomes to include (if None, include all)
        """
        self.data_dir = data_dir
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by chromosomes if specified
        if chrlist is not None:
            self.metadata = self.metadata[self.metadata['chr'].isin(chrlist)].reset_index(drop=True)
            print(f"Filtered to {len(self.metadata)} regions from chromosomes: {chrlist}")
        
        # Load gene sequences
        gene_seqs_path = os.path.join(data_dir, 'gene_seqs', 'gene_sequences.pt')
        all_gene_seqs = torch.load(gene_seqs_path)
        
        # Find RNA-seq and Ribo-seq files
        rna_seq_files = glob.glob(os.path.join(data_dir, 'rna_seqs', '*.pt'))
        ribo_seq_files = glob.glob(os.path.join(data_dir, 'ribo_seqs', '*.pt'))
        
        if not rna_seq_files or not ribo_seq_files:
            raise FileNotFoundError("No RNA-seq or Ribo-seq files found")
            
        # Load the first RNA-seq and Ribo-seq file
        all_rna_seqs = torch.load(rna_seq_files[0])
        all_ribo_seqs = torch.load(ribo_seq_files[0])
        
        print(f"Loaded RNA-seq file: {rna_seq_files[0]} with shape {all_rna_seqs.shape}")
        print(f"Loaded Ribo-seq file: {ribo_seq_files[0]} with shape {all_ribo_seqs.shape}")
        
        # Get indices of the regions that match our chromosome filter
        if chrlist is not None:
            # Get original indices from the full metadata
            original_metadata = pd.read_csv(metadata_path)
            filtered_indices = []
            
            for _, row in self.metadata.iterrows():
                # Find matching rows in the original metadata
                matches = original_metadata[
                    (original_metadata['chr'] == row['chr']) & 
                    (original_metadata['start'] == row['start']) & 
                    (original_metadata['end'] == row['end'])
                ]
                if not matches.empty:
                    filtered_indices.append(matches.index[0])
            
            # Extract the filtered data
            self.gene_seqs = all_gene_seqs[filtered_indices]
            self.rna_seqs = all_rna_seqs[filtered_indices]
            self.ribo_seqs = all_ribo_seqs[filtered_indices]
        else:
            # Use all data
            self.gene_seqs = all_gene_seqs
            self.rna_seqs = all_rna_seqs
            self.ribo_seqs = all_ribo_seqs
        
        # Ensure all tensors have the same first dimension (number of regions)
        assert len(self.metadata) == len(self.gene_seqs) == len(self.rna_seqs) == len(self.ribo_seqs), \
            f"Mismatch in data dimensions: metadata={len(self.metadata)}, gene_seqs={len(self.gene_seqs)}, " \
            f"rna_seqs={len(self.rna_seqs)}, ribo_seqs={len(self.ribo_seqs)}"
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get data for the specified index
        gene_seq = self.gene_seqs[idx]
        rna_seq = self.rna_seqs[idx]
        ribo_seq = self.ribo_seqs[idx]
        
        return {
            'gene_seq': gene_seq,
            'rna_seq': rna_seq,
            'target': ribo_seq
        }


class ContrastiveRiboSeqDataset(Dataset):
    """
    Dataset for contrastive learning with Ribosome Profiling data
    """
    def __init__(self, base_dataset, augmenter, negative_ratio=0.8, 
                 hard_negative_type2_ratio=0.0, curriculum_epochs=None):
        """
        Args:
            base_dataset: the base RiboSeqDataset
            augmenter: RiboSeqAugmenter for creating positive and negative samples
            negative_ratio: ratio of negative samples to positive samples
            hard_negative_type2_ratio: ratio of type2 hard negative samples to all negative samples
            curriculum_epochs: list of (epoch, ratio) pairs for curriculum learning of type2 hard negatives
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.negative_ratio = negative_ratio
        self.hard_negative_type2_ratio = hard_negative_type2_ratio
        self.curriculum_epochs = curriculum_epochs
        self.current_epoch = 0
    
    def __len__(self):
        return len(self.base_dataset)
    
    def update_curriculum(self, epoch):
        """
        Update the ratio of type2 hard negative samples based on curriculum
        """
        self.current_epoch = epoch
        
        if self.curriculum_epochs:
            for e, ratio in self.curriculum_epochs:
                if epoch >= e:
                    self.hard_negative_type2_ratio = ratio
    
    def __getitem__(self, idx):
        # Get original sample
        sample = self.base_dataset[idx]
        gene_seq = sample['gene_seq']
        rna_seq = sample['rna_seq']
        target = sample['target']
        
        # Create positive sample by modifying low expression regions
        aug_gene_seq = self.augmenter.create_positive_pair(gene_seq, rna_seq)
        
        # Find out gene_seq dimension with size 5 (one-hot encoding dimension)
        dim_5 = None
        for i, dim_size in enumerate(gene_seq.shape):
            if dim_size == 5:
                dim_5 = i
                break
        
        # Create input pairs based on tensor shapes
        if dim_5 == 1:  # Shape like [seq_len, 5]
            original_input = torch.cat([gene_seq, rna_seq.unsqueeze(1)], dim=1)
            positive_input = torch.cat([aug_gene_seq, rna_seq.unsqueeze(1)], dim=1)
        elif dim_5 == 2:  # Shape like [batch, seq_len, 5]
            original_input = torch.cat([gene_seq, rna_seq.unsqueeze(2)], dim=2)
            positive_input = torch.cat([aug_gene_seq, rna_seq.unsqueeze(2)], dim=2)
        else:
            # Default case - assume the feature dimension is the last one
            rna_expanded = rna_seq.unsqueeze(-1)
            original_input = torch.cat([gene_seq, rna_expanded], dim=-1)
            positive_input = torch.cat([aug_gene_seq, rna_expanded], dim=-1)
        
        # Decide whether to create a negative sample
        if random.random() < self.negative_ratio:
            # Decide which type of hard negative to create
            if random.random() < self.hard_negative_type2_ratio:
                # Type 2: Modify high expression regions
                neg_gene_seq = self.augmenter.create_hard_negative_pair_type2(gene_seq, rna_seq)
                
                # Use same tensor connection logic
                if dim_5 == 1:
                    negative_input = torch.cat([neg_gene_seq, rna_seq.unsqueeze(1)], dim=1)
                elif dim_5 == 2:
                    negative_input = torch.cat([neg_gene_seq, rna_seq.unsqueeze(2)], dim=2)
                else:
                    negative_input = torch.cat([neg_gene_seq, rna_seq.unsqueeze(-1)], dim=-1)
                    
                is_positive = False
            else:
                # Type 1: Use RNA-seq from another sample
                if len(self.base_dataset) > 1:  # Ensure dataset has at least 2 samples
                    other_idx = random.randint(0, len(self.base_dataset) - 1)
                    attempts = 0
                    # Try to find a different sample, with max 10 attempts
                    while other_idx == idx and attempts < 10:
                        other_idx = random.randint(0, len(self.base_dataset) - 1)
                        attempts += 1
                    
                    other_sample = self.base_dataset[other_idx]
                    other_rna_seq = other_sample['rna_seq']
                    
                    # Use same tensor connection logic
                    if dim_5 == 1:
                        negative_input = torch.cat([gene_seq, other_rna_seq.unsqueeze(1)], dim=1)
                    elif dim_5 == 2:
                        negative_input = torch.cat([gene_seq, other_rna_seq.unsqueeze(2)], dim=2)
                    else:
                        negative_input = torch.cat([gene_seq, other_rna_seq.unsqueeze(-1)], dim=-1)
                        
                    is_positive = False
                else:
                    # If dataset is too small, fall back to type 2
                    neg_gene_seq = self.augmenter.create_hard_negative_pair_type2(gene_seq, rna_seq)
                    
                    if dim_5 == 1:
                        negative_input = torch.cat([neg_gene_seq, rna_seq.unsqueeze(1)], dim=1)
                    elif dim_5 == 2:
                        negative_input = torch.cat([neg_gene_seq, rna_seq.unsqueeze(2)], dim=2)
                    else:
                        negative_input = torch.cat([neg_gene_seq, rna_seq.unsqueeze(-1)], dim=-1)
                        
                    is_positive = False
        else:
            # Use positive sample if not creating a negative
            negative_input = positive_input
            is_positive = True
        
        return {
            'original': original_input,
            'positive': positive_input,
            'negative': negative_input,
            'is_positive': torch.tensor(float(is_positive), dtype=torch.float),
            'output_features': target
        }