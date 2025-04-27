import os
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob

class RiboSeqDataset(Dataset):
    """
    Dataset for contrastive learning with Ribosome Profiling data
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing the preprocessed data
        """
        self.data_dir = data_dir
        
        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Load gene sequences (single file containing all sequences)
        gene_seqs_path = os.path.join(data_dir, 'gene_seqs', 'gene_sequences.pt')
        self.gene_seqs = torch.load(gene_seqs_path)
        
        # Find all available RNA-seq and Ribo-seq data files
        rna_seq_files = glob.glob(os.path.join(data_dir, 'rna_seqs', '*.pt'))
        ribo_seq_files = glob.glob(os.path.join(data_dir, 'ribo_seqs', '*.pt'))
        
        print(f"Found {len(rna_seq_files)} RNA-seq files and {len(ribo_seq_files)} Ribo-seq files")
        
        # Load the first RNA-seq and Ribo-seq file
        # In a real-world scenario, you might want to handle multiple files differently
        if rna_seq_files and ribo_seq_files:
            self.rna_seqs = torch.load(rna_seq_files[0])
            self.ribo_seqs = torch.load(ribo_seq_files[0])
            print(f"Loaded RNA-seq file: {rna_seq_files[0]} with shape {self.rna_seqs.shape}")
            print(f"Loaded Ribo-seq file: {ribo_seq_files[0]} with shape {self.ribo_seqs.shape}")
        else:
            raise FileNotFoundError("No RNA-seq or Ribo-seq files found in the data directory")
        
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
    def __init__(self, base_dataset, augmenter, negative_ratio=0.3, 
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
        
        # Create positive input pair
        original_input = torch.cat([gene_seq, rna_seq.unsqueeze(1)], dim=1)
        positive_input = torch.cat([aug_gene_seq, rna_seq.unsqueeze(1)], dim=1)
        
        # Decide whether to create a negative sample
        if random.random() < self.negative_ratio:
            # Decide which type of hard negative to create
            if random.random() < self.hard_negative_type2_ratio:
                # Type 2: Modify high expression regions
                neg_gene_seq = self.augmenter.create_hard_negative_pair_type2(gene_seq, rna_seq)
                negative_input = torch.cat([neg_gene_seq, rna_seq.unsqueeze(1)], dim=1)
                is_positive = False
            else:
                # Type 1: Use RNA-seq from another sample
                other_idx = random.randint(0, len(self.base_dataset) - 1)
                while other_idx == idx:
                    other_idx = random.randint(0, len(self.base_dataset) - 1)
                other_sample = self.base_dataset[other_idx]
                other_rna_seq = other_sample['rna_seq']
                
                negative_input = torch.cat([gene_seq, other_rna_seq.unsqueeze(1)], dim=1)
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