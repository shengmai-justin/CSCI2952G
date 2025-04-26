#!/bin/bash

# This script runs the entire Translatomer-CL pipeline:
# 1. Prepare data from original Translatomer format
# 2. Stage 1 pretraining (SimCLR style)
# 3. Stage 2 pretraining (Teacher-Student self-distillation)
# 4. Final training with the pretrained model
# 5. Evaluation on test data

# Default parameters
DATA_ROOT="./data"                   
ASSEMBLY="hg38"                     
OUTPUT_DIR="./output/translatomer_cl" 
TRAIN_DATA_FILE="data_roots.txt"     
NUM_GENOMIC_FEATURES=6
HIDDEN_DIM=512
BATCH_SIZE=32
SEED=42
GPU=0

# Stage 1 pretraining parameters
STAGE1_EPOCHS=50
STAGE1_LR=1e-4
STAGE1_WEIGHT_DECAY=1e-6
STAGE1_TEMPERATURE=0.1
STAGE1_NEGATIVE_RATIO=0.3

# Stage 2 pretraining parameters
STAGE2_EPOCHS=30
STAGE2_LR=5e-5
STAGE2_WEIGHT_DECAY=1e-6
STAGE2_EMA_DECAY=0.999
STAGE2_NEGATIVE_RATIO=0.5
STAGE2_CURRICULUM="0:0.0,5:0.2,15:0.5,25:0.8"

# Training parameters
TRAIN_EPOCHS=100
TRAIN_LR=1e-4
TRAIN_WEIGHT_DECAY=1e-6
EARLY_STOPPING=10
VAL_SPLIT=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data_root)
      DATA_ROOT="$2"
      shift
      shift
      ;;
    --assembly)
      ASSEMBLY="$2"
      shift
      shift
      ;;
    --train_data_file)
      TRAIN_DATA_FILE="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --gpu)
      GPU="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=== Translatomer-CL Pipeline Configuration ==="
echo "Data root directory: $DATA_ROOT"
echo "Assembly: $ASSEMBLY"
echo "Train data file: $TRAIN_DATA_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "GPU ID: $GPU"
echo "================================================"

# Step 0: Prepare data from original Translatomer format
echo "[$(date)] Preparing data from original Translatomer format..."


if [ ! -f "prepare_data_revised.py" ]; then
    echo "Error: prepare_data_revised.py script not found!"
    exit 1
fi


python prepare_data_revised.py \
  --data_root "$DATA_ROOT" \
  --assembly "$ASSEMBLY" \
  --train_data_file "$TRAIN_DATA_FILE" \
  --output_dir "${OUTPUT_DIR}/data"


if [ $? -ne 0 ]; then
  echo "Data preparation failed. Exiting."
  exit 1
fi
echo "[$(date)] Data preparation completed!"

# Step 1: Stage 1 pretraining (SimCLR style)
echo "[$(date)] Starting Stage 1 pretraining (SimCLR style)..."
python pretrain_stage1.py \
  --data_dir "${OUTPUT_DIR}/data" \
  --output_dir "$OUTPUT_DIR" \
  --num_genomic_features "$NUM_GENOMIC_FEATURES" \
  --hidden_dim "$HIDDEN_DIM" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$STAGE1_EPOCHS" \
  --lr "$STAGE1_LR" \
  --weight_decay "$STAGE1_WEIGHT_DECAY" \
  --temperature "$STAGE1_TEMPERATURE" \
  --negative_ratio "$STAGE1_NEGATIVE_RATIO" \
  --seed "$SEED" \
  --gpu "$GPU"


if [ $? -ne 0 ]; then
  echo "Stage 1 pretraining failed. Exiting."
  exit 1
fi
echo "[$(date)] Stage 1 pretraining completed!"

# Step 2: Stage 2 pretraining (Teacher-Student self-distillation)
echo "[$(date)] Starting Stage 2 pretraining (Teacher-Student self-distillation)..."
python pretrain_stage2.py \
  --data_dir "${OUTPUT_DIR}/data" \
  --output_dir "$OUTPUT_DIR" \
  --stage1_model "$OUTPUT_DIR/translatomer_cl_stage1.pt" \
  --num_genomic_features "$NUM_GENOMIC_FEATURES" \
  --hidden_dim "$HIDDEN_DIM" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$STAGE2_EPOCHS" \
  --lr "$STAGE2_LR" \
  --weight_decay "$STAGE2_WEIGHT_DECAY" \
  --ema_decay "$STAGE2_EMA_DECAY" \
  --negative_ratio "$STAGE2_NEGATIVE_RATIO" \
  --curriculum_epochs "$STAGE2_CURRICULUM" \
  --seed "$SEED" \
  --gpu "$GPU"


if [ $? -ne 0 ]; then
  echo "Stage 2 pretraining failed. Exiting."
  exit 1
fi
echo "[$(date)] Stage 2 pretraining completed!"

# Step 3: Final training with the pretrained model
echo "[$(date)] Starting final training with the pretrained model..."
python train.py \
  --data_dir "${OUTPUT_DIR}/data" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model "$OUTPUT_DIR/translatomer_cl_teacher.pt" \
  --num_genomic_features "$NUM_GENOMIC_FEATURES" \
  --hidden_dim "$HIDDEN_DIM" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$TRAIN_EPOCHS" \
  --lr "$TRAIN_LR" \
  --weight_decay "$TRAIN_WEIGHT_DECAY" \
  --early_stopping "$EARLY_STOPPING" \
  --val_split "$VAL_SPLIT" \
  --seed "$SEED" \
  --gpu "$GPU"


if [ $? -ne 0 ]; then
  echo "Final training failed. Exiting."
  exit 1
fi
echo "[$(date)] Final training completed!"

# Step 4: Evaluation on test data
echo "[$(date)] Starting evaluation on test data..."
python evaluate.py \
  --data_dir "${OUTPUT_DIR}/data" \
  --output_dir "$OUTPUT_DIR/evaluation" \
  --model_path "$OUTPUT_DIR/translatomer_cl_best.pt" \
  --num_genomic_features "$NUM_GENOMIC_FEATURES" \
  --hidden_dim "$HIDDEN_DIM" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --gpu "$GPU"


if [ $? -ne 0 ]; then
  echo "Evaluation failed. Exiting."
  exit 1
fi
echo "[$(date)] Evaluation completed!"

echo "Translatomer-CL pipeline completed successfully!"
echo "Results are available in $OUTPUT_DIR"
