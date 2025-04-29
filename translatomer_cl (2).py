import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np
import random
import model.blocks as blocks
from model.models import TransModel

class TranslatomerCL(nn.Module):
    """
    Translatomer-CL: A Contrastive Learning approach for Ribosome Profiling
    """
    def __init__(self, num_genomic_features, mid_hidden=512, record_attn=False):
        super(TranslatomerCL, self).__init__()
        print('Initializing TranslatomerCL')
        
        # Initialize the base Translatomer model
        self.encoder = TransModel(num_genomic_features, mid_hidden, record_attn)
        
        # Projection head for contrastive learning (SimCLR style)
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Fusion module for the second pre-training stage
        self.fusion_module = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """Forward pass through the encoder"""
        features = self.encoder(x)
        return features
    
    def get_embedding(self, x):
        """Get normalized embeddings for contrastive learning"""
        features = self.encoder(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=1)  # L2 normalization for cosine similarity
    
    def compute_matching_score(self, x1, x2):
        """Compute matching score between two samples for the second pre-training stage"""
        embed1 = self.get_embedding(x1)
        embed2 = self.get_embedding(x2)
        concat_embed = torch.cat([embed1, embed2], dim=1)
        matching_score = self.fusion_module(concat_embed)
        return matching_score.squeeze()


class EMA:
    """Exponential Moving Average for teacher model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = deepcopy(model)
        self.decay = decay
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)


def nt_xent_loss(z1, z2, temperature=0.1):
    """NT-Xent loss for contrastive learning (SimCLR style)"""
    batch_size = z1.size(0)
    
    # Compute cosine similarity between all pairs
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Mask out self-similarity
    sim_matrix.fill_diagonal_(-float('inf'))
    
    # Create positive pair mask
    pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=z1.device)
    pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z1.device)
    pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z1.device)
    
    # Extract positive pairs
    pos_pairs = sim_matrix[pos_mask.bool()].reshape(2 * batch_size, 1)
    
    # Compute log sum exp
    logits = torch.cat([pos_pairs, sim_matrix], dim=1)
    loss = -pos_pairs + torch.logsumexp(logits, dim=1)
    
    return loss.mean()


def matching_loss(scores, labels):
    """Matching loss for the second pre-training stage"""
    return F.binary_cross_entropy_with_logits(scores, labels)


class RiboSeqAugmenter:
    """
    Data augmenter for creating positive and hard-negative samples for Ribo-seq data
    """
    def __init__(self, low_expr_threshold=0.2, high_expr_threshold=0.8):
        """
        Args:
            low_expr_threshold: threshold to define low expression regions (below this percentile)
            high_expr_threshold: threshold to define high expression regions (above this percentile)
        """
        self.low_expr_threshold = low_expr_threshold
        self.high_expr_threshold = high_expr_threshold
    
    def create_positive_pair(self, gene_seq, rna_seq):
        """
        Create a positive sample pair by modifying low expression regions
        
        Args:
            gene_seq: original gene sequence [seq_len, 5] (one-hot encoded)
            rna_seq: original RNA-seq data [seq_len] or [seq_len/64]
        
        Returns:
            augmented gene sequence
        """
        # Create a copy of the original sequence
        aug_gene_seq = gene_seq.clone()
        
        # Ensure rna_seq has the right shape for masking
        if rna_seq.shape[0] < gene_seq.shape[0]:
            # If rna_seq is binned (e.g., 1024 bins), we need to expand it
            # to match the gene_seq length (e.g., 65536)
            expansion_factor = gene_seq.shape[0] // rna_seq.shape[0]
            expanded_rna_seq = rna_seq.unsqueeze(1).repeat(1, expansion_factor).flatten()
            low_expr_mask = expanded_rna_seq < torch.quantile(expanded_rna_seq, self.low_expr_threshold)
        else:
            # RNA-seq data is already at full resolution
            low_expr_mask = rna_seq < torch.quantile(rna_seq, self.low_expr_threshold)
        
        # Expand mask to match gene_seq dimensions
        low_expr_mask = low_expr_mask.unsqueeze(1).expand(-1, gene_seq.shape[1])
        
        # Randomly modify bases in low expression regions (with 20% probability)
        mod_prob = torch.rand(gene_seq.shape[0], 1).expand(-1, gene_seq.shape[1])
        mod_mask = (mod_prob < 0.5) & low_expr_mask
        
        # For nucleotide bases, we want to create valid one-hot encodings
        # Find positions to modify
        positions_to_modify = torch.any(mod_mask, dim=1)
        
        if positions_to_modify.sum() > 0:
            # For each position to modify, create a new random one-hot encoding
            new_bases = torch.zeros_like(aug_gene_seq[positions_to_modify])
            random_indices = torch.randint(0, 4, (positions_to_modify.sum(),))  # 0-3 for ATCG
            new_bases[torch.arange(positions_to_modify.sum()), random_indices] = 1.0
            
            # Update the augmented sequence
            aug_gene_seq[positions_to_modify] = new_bases
        
        return aug_gene_seq
    
    def create_hard_negative_pair_type2(self, gene_seq, rna_seq):
        """
        Create a hard-negative sample by modifying high expression regions
        
        Args:
            gene_seq: original gene sequence [seq_len, 5] (one-hot encoded)
            rna_seq: original RNA-seq data [seq_len] or [seq_len/64]
        
        Returns:
            augmented gene sequence
        """
        # Create a copy of the original sequence
        aug_gene_seq = gene_seq.clone()
        
        # Ensure rna_seq has the right shape for masking
        if rna_seq.shape[0] < gene_seq.shape[0]:
            # If rna_seq is binned (e.g., 1024 bins), we need to expand it
            # to match the gene_seq length (e.g., 65536)
            expansion_factor = gene_seq.shape[0] // rna_seq.shape[0]
            expanded_rna_seq = rna_seq.unsqueeze(1).repeat(1, expansion_factor).flatten()
            high_expr_mask = expanded_rna_seq > torch.quantile(expanded_rna_seq, self.high_expr_threshold)
        else:
            # RNA-seq data is already at full resolution
            high_expr_mask = rna_seq > torch.quantile(rna_seq, self.high_expr_threshold)
        
        # Expand mask to match gene_seq dimensions
        high_expr_mask = high_expr_mask.unsqueeze(1).expand(-1, gene_seq.shape[1])
        
        # Randomly modify bases in high expression regions (with lower probability - 5%)
        mod_prob = torch.rand(gene_seq.shape[0], 1).expand(-1, gene_seq.shape[1])
        mod_mask = (mod_prob < 0.5) & high_expr_mask
        
        # Find positions to modify
        positions_to_modify = torch.any(mod_mask, dim=1)
        
        if positions_to_modify.sum() > 0:
            # For each position to modify, create a new random one-hot encoding
            new_bases = torch.zeros_like(aug_gene_seq[positions_to_modify])
            random_indices = torch.randint(0, 4, (positions_to_modify.sum(),))  # 0-3 for ATCG
            new_bases[torch.arange(positions_to_modify.sum()), random_indices] = 1.0
            
            # Update the augmented sequence
            aug_gene_seq[positions_to_modify] = new_bases
        
        return aug_gene_seq