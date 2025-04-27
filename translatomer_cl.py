import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import numpy as np
import model.blocks as blocks
from model.models import TransModel

class TranslatomerCL(nn.Module):
    
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
        
        features = self.encoder(x)
        return features
    
    def get_embedding(self, x):
       
        features = self.encoder(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=1)  # L2 normalization for cosine similarity
    
    def compute_matching_score(self, x1, x2):
        
        embed1 = self.get_embedding(x1)
        embed2 = self.get_embedding(x2)
        concat_embed = torch.cat([embed1, embed2], dim=1)
        matching_score = self.fusion_module(concat_embed)
        return matching_score.squeeze()


class EMA:
    
    def __init__(self, model, decay=0.999):
        self.model = deepcopy(model)
        self.decay = decay
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)


def nt_xent_loss(z1, z2, temperature=0.1):
    
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
    
    return F.binary_cross_entropy_with_logits(scores, labels)


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
        mod_mask = (mod_prob < 0.2) & low_expr_mask
        
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
        mod_mask = (mod_prob < 0.20) & high_expr_mask
        
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