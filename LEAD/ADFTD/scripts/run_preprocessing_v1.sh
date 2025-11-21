#!/bin/bash
#SBATCH --job-name=ADFTD_v1
#SBATCH --partition=debug
#SBATCH --nodelist=node4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH -o /storage/connectome/bohee/DIVER_ADFTD/logs/v1_output_%j.log
#SBATCH -e /storage/connectome/bohee/DIVER_ADFTD/logs/v1_error_%j.log

echo "=========================================="
echo "ADFTD Preprocessing - Version 1 (v1)"
echo "Shape: (19, 30, 500) - 30sec segments at 500Hz"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Python environment
PYTHON_PATH=/storage/connectome/bohee/DIVER_ADFTD/conda_env/bin/python

# Verify Python environment
echo "Python: $PYTHON_PATH"
$PYTHON_PATH --version

# Move to working directory
cd /storage/connectome/bohee/DIVER_ADFTD/scripts

# Run preprocessing with version 1
# - Binary classification: HC vs AD only (FTD excluded)
# - Stratified random split with seed=42
# - 500Hz sampling rate maintained
# - 30-second segments
$PYTHON_PATH preprocessing_generalized_ADFTD.py \
    --dataset_name ADFTD \
    --data_path /storage/connectome/bohee/DIVER_ADFTD/data/raw \
    --save_path_parent /scratch/connectome/bohee/DIVER_ADFTD/data/processed_v1 \
    --coordinate_file_path /storage/connectome/bohee/DIVER_ADFTD/scripts/standard_1005.elc \
    --shape_version v1 \
    --resample_rate 500 \
    --highpass 0.5 \
    --lowpass 45.0 \
    --segment_len 30 \
    --percent 1.0 \
    --num_chunks 10

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
echo ""
echo "Expected output:"
echo "- Save path: /scratch/connectome/bohee/DIVER_ADFTD/data/processed_v1"
echo "- Shape: (19, 30, 500)"
echo "- Subjects: ~65 (HC: ~29, AD: ~36, FTD excluded)"
echo "- Split: train ~38, val ~13, test ~14"
echo "=========================================="
