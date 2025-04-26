#!/bin/bash
# Usage: ./process_srr_to_datafolder.sh <CELL_LINE> <STUDY_ID> <SRR_ID> <SEQ_TYPE>
set -e
set -o pipefail

CELL_LINE=$1
STUDY_ID=$2
SRR_ID=$3
SEQ_TYPE=$4 # "ribo-seq" or "rna-seq"
THREADS=70
BASE_DIR="/home/ubuntu/cs2952/555/hg38"
RRNA_INDEX="/home/ubuntu/cs2952/555/references/rrna/Homo_sapiens.rRNA"
TRNA_INDEX="/home/ubuntu/cs2952/555/references/trna/Homo_sapiens.tRNA"
GENOME_INDEX="/home/ubuntu/cs2952/555/references/genome/bowtie2/hg38"
TMP_DIR="/home/ubuntu/cs2952/555/tmp/${STUDY_ID}_${SRR_ID}_${SEQ_TYPE}"
FASTQ_DIR="${TMP_DIR}/fastq"
DERRNA_DIR="${TMP_DIR}/derRNA"
DETRNA_DIR="${TMP_DIR}/detRNA"
MAP_DIR="${TMP_DIR}/map"
MERGE_DIR="${MAP_DIR}/merge"
BIGWIG_DIR="${TMP_DIR}/bigwig"
LOG_DIR="${TMP_DIR}/logs"

if [ "$SEQ_TYPE" == "ribo-seq" ]; then
    TARGET_DIR="${BASE_DIR}/${CELL_LINE}/${STUDY_ID}/output_features"
elif [ "$SEQ_TYPE" == "rna-seq" ]; then
    TARGET_DIR="${BASE_DIR}/${CELL_LINE}/${STUDY_ID}/input_features"
else
    echo "Invalid SEQ_TYPE: must be 'ribo-seq' or 'rna-seq'"
    exit 1
fi

mkdir -p $FASTQ_DIR $DERRNA_DIR $DETRNA_DIR $MAP_DIR $MERGE_DIR $BIGWIG_DIR $LOG_DIR $TARGET_DIR
exec > >(tee -a "${LOG_DIR}/pipeline.log") 2>&1

echo "=== Prefetching ${SRR_ID} ==="
prefetch -O ${TMP_DIR} ${SRR_ID}

echo "=== Converting to FASTQ ==="
fasterq-dump --split-files -e ${THREADS} -O ${FASTQ_DIR} ${SRR_ID}
cd ${FASTQ_DIR}
for f in *.fastq; do gzip -f "$f"; done

echo "=== Filtering rRNA ==="
for fq in *.fastq.gz; do
    base=$(basename $fq .fastq.gz)
    #echo $(basename $fq .fastq.gz)
    bowtie2 -x ${RRNA_INDEX} --un-gz ${DERRNA_DIR}/${base}.derRNA.fq.gz -U $fq -p ${THREADS} \
    -S ${DERRNA_DIR}/${base}.rRNA.sam 2> ${LOG_DIR}/${base}_rrna.log
done


echo "=== Filtering tRNA ==="
for fq in ${DERRNA_DIR}/*.derRNA.fq.gz; do
    base=$(basename $fq .derRNA.fq.gz)
    bowtie2 -x ${TRNA_INDEX} --un-gz ${DETRNA_DIR}/${base}.detRNA.fq.gz -U $fq -p ${THREADS} \
    -S ${DETRNA_DIR}/${base}.tRNA.sam 2> ${LOG_DIR}/${base}_trna.log
done


echo "=== Mapping to genome and sorting ==="
for fq in ${DETRNA_DIR}/*.detRNA.fq.gz; do
    base=$(basename $fq .detRNA.fq.gz)
    bowtie2 --local -x ${GENOME_INDEX} -U $fq -p ${THREADS} 2> ${LOG_DIR}/${base}.map.log | \
    samtools view -@ ${THREADS} -bS - | \
    samtools sort -@ ${THREADS} -O bam -o ${MAP_DIR}/${base}.sorted.bam
    samtools index ${MAP_DIR}/${base}.sorted.bam
done

echo "=== Merging BAMs ==="
if [ $(ls ${MAP_DIR}/*.sorted.bam | wc -l) -gt 1 ]; then
    samtools merge -@ ${THREADS} ${MERGE_DIR}/merged.bam ${MAP_DIR}/*.sorted.bam
else
    ln -s $(readlink -f ${MAP_DIR}/*.sorted.bam) ${MERGE_DIR}/merged.bam
fi
samtools index ${MERGE_DIR}/merged.bam

echo "=== Generating bigWig ==="
if [ "$SEQ_TYPE" == "ribo-seq" ]; then
    OUT_PREFIX="ribo"
else
    OUT_PREFIX="rna"
fi

bamCoverage -p ${THREADS} -b ${MERGE_DIR}/merged.bam --normalizeUsing RPKM --binSize 1 \
-o ${TARGET_DIR}/${OUT_PREFIX}.bw > ${LOG_DIR}/${SRR_ID}.bw.log 2>&1

echo "=== Done. Output → ${TARGET_DIR}/${OUT_PREFIX}.bw"