#!/bin/bash
#SBATCH --job-name=ADFTD_v2
#SBATCH --partition=debug
#SBATCH --nodelist=node4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH -o /storage/connectome/bohee/DIVER_ADFTD/logs/v2_output_%j.log
#SBATCH -e /storage/connectome/bohee/DIVER_ADFTD/logs/v2_error_%j.log

echo "=========================================="
echo "ADFTD Preprocessing - Version 2 (v2)"
echo "Shape: Multi-scale LEAD-style"
echo "  - (19, 1, 500): 1sec at 500Hz"
echo "  - (19, 2, 250): 2sec at 250Hz"
echo "  - (19, 4, 125): 4sec at 125Hz"
echo "  - 50% overlap for each scale"
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

# Run preprocessing with version 2
# - Binary classification: HC vs AD only (FTD excluded)
# - Stratified random split with seed=42
# - 500Hz base sampling rate
# - Multi-scale segmentation (1s/2s/4s with 50% overlap)
$PYTHON_PATH preprocessing_generalized_ADFTD.py \
    --dataset_name ADFTD \
    --data_path /storage/connectome/bohee/DIVER_ADFTD/data/raw \
    --save_path_parent /scratch/connectome/bohee/DIVER_ADFTD/data/processed_v2 \
    --coordinate_file_path /storage/connectome/bohee/DIVER_ADFTD/scripts/standard_1005.elc \
    --shape_version v2 \
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
echo "- Save path: /scratch/connectome/bohee/DIVER_ADFTD/data/processed_v2"
echo "- Shapes: (19,1,500), (19,2,250), (19,4,125)"
echo "- Subjects: ~65 (HC: ~29, AD: ~36, FTD excluded)"
echo "- Split: train ~38, val ~13, test ~14"
echo "- More segments due to 50% overlap"
echo "=========================================="
