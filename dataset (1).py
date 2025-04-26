import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm

class RiboSeqDataset(Dataset):
    """
    Dataset for Ribosome Profiling data
    """
    def __init__(self, data_dir, cell_types=None, transform=None):
        
        self.data_dir = data_dir
        self.transform = transform
        
        # Load metadata
        self.metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        
        # Filter by cell types if specified
        if cell_types:
            self.metadata = self.metadata[self.metadata['cell_type'].isin(cell_types)]
        
        # Load data
        print("Loading data...")
        self.gene_seqs = []
        self.rna_seqs = []
        self.ribo_seqs = []
        
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            gene_seq_file = os.path.join(data_dir, row['gene_seq_file'])
            rna_seq_file = os.path.join(data_dir, row['rna_seq_file'])
            ribo_seq_file = os.path.join(data_dir, row['ribo_seq_file'])
            
            gene_seq = torch.load(gene_seq_file)
            rna_seq = torch.load(rna_seq_file)
            ribo_seq = torch.load(ribo_seq_file)
            
            self.gene_seqs.append(gene_seq)
            self.rna_seqs.append(rna_seq)
            self.ribo_seqs.append(ribo_seq)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        gene_seq = self.gene_seqs[idx]
        rna_seq = self.rna_seqs[idx]
        ribo_seq = self.ribo_seqs[idx]
        
        # Concatenate gene sequence and RNA-seq as input
        input_data = torch.cat([gene_seq, rna_seq], dim=1)
        
        if self.transform:
            input_data = self.transform(input_data)
        
        return {
            'input': input_data,
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
        
        # Create positive sample by modifying low expression regions
        aug_gene_seq = self.augmenter.create_positive_pair(gene_seq, rna_seq)
        
        # Create positive input pair
        original_input = torch.cat([gene_seq, rna_seq], dim=1)
        positive_input = torch.cat([aug_gene_seq, rna_seq], dim=1)
        
        # Decide whether to create a negative sample
        if random.random() < self.negative_ratio:
            # Decide which type of hard negative to create
            if random.random() < self.hard_negative_type2_ratio:
                # Type 2: Modify high expression regions
                neg_gene_seq = self.augmenter.create_hard_negative_pair_type2(gene_seq, rna_seq)
                negative_input = torch.cat([neg_gene_seq, rna_seq], dim=1)
                is_positive = False
            else:
                # Type 1: Use RNA-seq from another sample
                rna_seq_pool = [self.base_dataset[i]['rna_seq'] for i in range(len(self.base_dataset)) if i != idx]
                if rna_seq_pool:
                    _, random_rna_seq = self.augmenter.create_hard_negative_pair_type1(
                        gene_seq, rna_seq_pool
                    )
                    negative_input = torch.cat([gene_seq, random_rna_seq], dim=1)
                    is_positive = False
                else:
                    # Fallback if no other samples available
                    negative_input = positive_input
                    is_positive = True
        else:
            # Use positive sample if not creating a negative
            negative_input = positive_input
            is_positive = True
        
        return {
            'original': original_input,
            'positive': positive_input,
            'negative': negative_input,
            'is_positive': torch.tensor(float(is_positive), dtype=torch.float)
        }
