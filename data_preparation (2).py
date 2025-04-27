import os
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pyfaidx
from kipoiseq import Interval

class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = fasta_file
        self._chromosome_sizes = {k: len(v) for k, v in pyfaidx.Fasta(self.fasta).items()}
        self.seq_len = 65536

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(pyfaidx.Fasta(self.fasta).get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.end).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream
    
    def close(self):
        return pyfaidx.Fasta(self.fasta).close()

def one_hot_encode(sequence):
    """Convert DNA sequence to one-hot encoding"""
    en_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    en_seq = [en_dict[ch] for ch in sequence]
    np_seq = np.array(en_seq, dtype=int)
    seq_emb = np.zeros((len(np_seq), 5))
    seq_emb[np.arange(len(np_seq)), np_seq] = 1
    return seq_emb.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description='Prepare data for Translatomer-CL')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing data')
    parser.add_argument('--assembly', type=str, required=True, help='Genome assembly (hg38, mm10)')
    parser.add_argument('--train_data_file', type=str, required=True, help='File containing cell types and studies')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Read cell types and studies
    cell_studies = []
    with open(args.train_data_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cell_studies.append((parts[0], parts[1]))

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'gene_seqs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rna_seqs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ribo_seqs'), exist_ok=True)
    
    # Load region file
    region_file = os.path.join(args.data_root, args.assembly, 'gene_region_24chr.1.bed')
    if not os.path.exists(region_file):
        print(f"Error: Region file not found at {region_file}")
        return
    
    regions = pd.read_csv(region_file, sep='\t', names=['chr', 'start', 'end', 'strand'])
    print(f"Loaded region file with {len(regions)} regions")
    
    # Save metadata
    regions.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)
    
    # Initialize FASTA extractor
    fasta_path = os.path.join(args.data_root, args.assembly, f"{args.assembly}.fa")
    if not os.path.exists(fasta_path):
        print(f"Error: FASTA file not found at {fasta_path}")
        return
    
    fasta_extractor = FastaStringExtractor(fasta_path)
    
    # Extract gene sequences
    print("Extracting gene sequences...")
    gene_seqs = []
    for idx, row in tqdm(regions.iterrows(), total=len(regions), desc="Processing regions"):
        interval = Interval(row['chr'], row['start'], row['end']).resize(65536)
        sequence = fasta_extractor.extract(interval)
        gene_seq = one_hot_encode(sequence)
        gene_seqs.append(gene_seq)
    
    # Save gene sequences
    gene_seqs_tensor = torch.tensor(np.array(gene_seqs))
    torch.save(gene_seqs_tensor, os.path.join(args.output_dir, 'gene_seqs', 'gene_sequences.pt'))
    print(f"Saved gene sequences with shape {gene_seqs_tensor.shape}")
    
    # Process each cell type and study
    for cell_type, study in tqdm(cell_studies, desc="Processing datasets"):
        rna_seq_path = os.path.join(
            args.data_root, args.assembly, cell_type, study, 
            'input_features', 'tmp', f'{cell_type}_65536_log_24chr_rnaseq_final.pt'
        )
        
        ribo_seq_path = os.path.join(
            args.data_root, args.assembly, cell_type, study, 
            'output_features', 'tmp', f'{cell_type}_65536_1024_log_24chr_riboseq_final.pt'
        )
        
        if not os.path.exists(rna_seq_path):
            print(f"Warning: RNA-seq file not found at {rna_seq_path}")
            continue
            
        if not os.path.exists(ribo_seq_path):
            print(f"Warning: Ribo-seq file not found at {ribo_seq_path}")
            continue
        
        # Load data
        rna_seq = torch.load(rna_seq_path)
        ribo_seq = torch.load(ribo_seq_path)
        
        # Save processed data
        torch.save(rna_seq, os.path.join(args.output_dir, 'rna_seqs', f'{cell_type}_{study}.pt'))
        torch.save(ribo_seq, os.path.join(args.output_dir, 'ribo_seqs', f'{cell_type}_{study}.pt'))
    
    fasta_extractor.close()
    print("Data preparation completed!")

if __name__ == "__main__":
    main()
