import os
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pyfaidx
from kipoiseq import Interval
import sys

class TensorLoader:
    
    def __init__(self, region_count, seq_len=65536):
        self.region_count = region_count
        self.seq_len = seq_len
        
    def load(self, path):
        
        print(f"Loading tensor from {path}")
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return None
            
       
        try:
            if 'rnaseq_final.pt' in path:
                # RNA-seq数据使用np.fromfile加载
                print(f"Detected RNA-seq file, using np.fromfile")
                data = np.fromfile(path, dtype=np.float32)
                data = data.reshape(self.region_count, self.seq_len)
                tensor = torch.from_numpy(data)
                print(f"Successfully loaded RNA-seq tensor with shape {tensor.shape}")
                return tensor
            else:
                # Ribo-seq数据使用torch.load加载
                print(f"Detected Ribo-seq file, using torch.load")
                tensor = torch.load(path, map_location='cpu')
                print(f"Successfully loaded Ribo-seq tensor with shape {tensor.shape}")
                return tensor
        except Exception as e:
            print(f"Error loading tensor: {e}")
            print("Trying alternative loading methods...")
            
           
            try:
                
                tensor = torch.load(path, map_location='cpu')
                print(f"Successfully loaded with torch.load, shape: {tensor.shape}")
                return tensor
            except Exception as e1:
                print(f"Failed with torch.load: {e1}")
                
                try:
                    
                    data = np.fromfile(path, dtype=np.float32)
                    
                   
                    if 'rnaseq' in path:
                        data = data.reshape(self.region_count, self.seq_len)
                    elif 'riboseq' in path:
                        
                        data = data.reshape(self.region_count, 1024)
                    
                    tensor = torch.from_numpy(data)
                    print(f"Successfully loaded with np.fromfile, shape: {tensor.shape}")
                    return tensor
                except Exception as e2:
                    print(f"Failed with np.fromfile: {e2}")
                    
                    try:
                        
                        np_array = np.load(path, allow_pickle=True)
                        tensor = torch.from_numpy(np_array)
                        print(f"Successfully loaded with np.load, shape: {tensor.shape}")
                        return tensor
                    except Exception as e3:
                        print(f"Failed with np.load: {e3}")
                        print(f"All loading methods failed for {path}")
                        return None
        
    def close(self):
       
        pass

class FastaStringExtractor:
   
    def __init__(self, fasta_file):
        self.fasta = fasta_file
        self._chromosome_sizes = {k: len(v) for k, v in pyfaidx.Fasta(self.fasta).items()}
        self.seq_len = 65536

    def extract(self, interval: Interval, **kwargs) -> str:
        
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        
        sequence = str(pyfaidx.Fasta(self.fasta).get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.end).seq).upper()
        
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream
    
    def close(self):
        
        return pyfaidx.Fasta(self.fasta).close()

def one_hot_encode(sequence):
    
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
    
    
    cell_studies = []
    with open(args.train_data_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cell_studies.append((parts[0], parts[1]))
    
    print(f"Found {len(cell_studies)} cell types/studies in {args.train_data_file}")
    for cell, study in cell_studies:
        print(f"  - {cell} / {study}")

    
    os.makedirs(os.path.join(args.output_dir, 'gene_seqs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rna_seqs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ribo_seqs'), exist_ok=True)
    
   
    if args.assembly == "hg38":
        region_file = os.path.join(args.data_root, args.assembly, 'gene_region_24chr.1.bed')
    elif args.assembly == "mm10":
        region_file = os.path.join(args.data_root, args.assembly, 'gene_region_21chr.1.bed')
    else:
        print(f"Error: Unsupported assembly: {args.assembly}")
        return
        
    if not os.path.exists(region_file):
        print(f"Error: Region file not found at {region_file}")
        return
    
    
    with open(region_file, 'r') as file:
        region_count = sum(1 for line in file if line.strip())
    print(f"Region file contains {region_count} regions")
    
    
    tensor_loader = TensorLoader(region_count, seq_len=65536)
    
    
    regions = pd.read_csv(region_file, sep='\t', names=['chr', 'start', 'end', 'strand'])
    print(f"Loaded region file with {len(regions)} regions")
    
   
    regions.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)
    
    
    fasta_path = os.path.join(args.data_root, args.assembly, f"{args.assembly}.fa")
    if not os.path.exists(fasta_path):
        print(f"Error: FASTA file not found at {fasta_path}")
        return
    
    print("Extracting gene sequences...")
    fasta_extractor = FastaStringExtractor(fasta_path)
    
    
    gene_seqs_path = os.path.join(args.output_dir, 'gene_seqs', 'gene_sequences.pt')
    if os.path.exists(gene_seqs_path):
        print(f"Gene sequences already exist at {gene_seqs_path}, skipping extraction")
    else:
       
        gene_seqs = []
        for idx, row in tqdm(regions.iterrows(), total=len(regions), desc="Processing regions"):
            interval = Interval(row['chr'], row['start'], row['end']).resize(65536)
            sequence = fasta_extractor.extract(interval)
            gene_seq = one_hot_encode(sequence)
            gene_seqs.append(gene_seq)
        
       
        gene_seqs_tensor = torch.tensor(np.array(gene_seqs))
        torch.save(gene_seqs_tensor, gene_seqs_path)
        print(f"Saved gene sequences with shape {gene_seqs_tensor.shape}")
    
   
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
        
        
        rna_seq = tensor_loader.load(rna_seq_path)
        if rna_seq is None:
            print(f"Warning: Could not load RNA-seq data for {cell_type}/{study}")
            continue
            
        ribo_seq = tensor_loader.load(ribo_seq_path)
        if ribo_seq is None:
            print(f"Warning: Could not load Ribo-seq data for {cell_type}/{study}")
            continue
        
        
        output_rna_path = os.path.join(args.output_dir, 'rna_seqs', f'{cell_type}_{study}.pt')
        output_ribo_path = os.path.join(args.output_dir, 'ribo_seqs', f'{cell_type}_{study}.pt')
        
        torch.save(rna_seq, output_rna_path)
        torch.save(ribo_seq, output_ribo_path)
        print(f"Saved data for {cell_type}/{study}")
        print(f"  - RNA-seq shape: {rna_seq.shape}")
        print(f"  - Ribo-seq shape: {ribo_seq.shape}")
    
    fasta_extractor.close()
    tensor_loader.close()
    print("Data preparation completed!")

if __name__ == "__main__":
    main()
