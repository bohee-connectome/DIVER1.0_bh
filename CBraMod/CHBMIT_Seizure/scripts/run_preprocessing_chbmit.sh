#!/bin/bash
#
# CHB-MIT Dataset Preprocessing Script for Perlmutter
#
# This script runs the CHB-MIT preprocessing on Perlmutter login node
# with sufficient memory allocation (200GB+)
#
# Usage:
#   bash run_preprocessing_chbmit.sh
#
# Author: Claude + User
# Date: 2025-01-21

set -e  # Exit on error
set -u  # Exit on undefined variable

# Initialize conda for non-interactive shell
module load python

# =============================================================================
# Configuration
# =============================================================================

# Python environment
CONDA_ENV="isruc_preprocess"  # Adjust if your environment has a different name

# Paths
DATA_PATH="/global/cfs/cdirs/m4750/DIVER/DOWNLOAD_DATASETS_MOVE_TO_M4750_LATER/CHB-MIT/physionet.org/files/chbmit/1.0.0"
LMDB_OUTPUT="/pscratch/sd/b/boheelee/DIVER/CHBMIT_preprocessing/lmdb_output/CHBMIT_Seizure"
ELC_FILE="/global/homes/b/boheelee/standard_1005.elc"
SCRIPT_DIR="/pscratch/sd/b/boheelee/DIVER/CHBMIT_preprocessing/scripts"
PREPROCESSING_SCRIPT="${SCRIPT_DIR}/preprocessing_chbmit.py"

# Parallelization
NUM_WORKERS=8  # Adjust based on available CPU cores

# Memory limit (ulimit in KB, 200GB = 200 * 1024 * 1024)
MEMORY_LIMIT_GB=200
MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))

# Logging
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/preprocessing_chbmit_${TIMESTAMP}.log"

# =============================================================================
# Setup
# =============================================================================

echo "=========================================="
echo "CHB-MIT Dataset Preprocessing"
echo "=========================================="
echo "Started at: $(date)"
echo ""

# Create log directory
mkdir -p "${LOG_DIR}"

# Create output directory
mkdir -p "${LMDB_OUTPUT}"

# Check if conda environment exists
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "ERROR: Conda environment '${CONDA_ENV}' not found"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Activate conda environment
echo "[INFO] Activating conda environment: ${CONDA_ENV}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Verify Python and required packages
echo "[INFO] Verifying Python environment..."
python --version
echo ""

# Check required packages
REQUIRED_PACKAGES="numpy scipy mne lmdb pyedflib"
for pkg in ${REQUIRED_PACKAGES}; do
    if ! python -c "import ${pkg}" 2>/dev/null; then
        echo "ERROR: Required package '${pkg}' not found"
        echo "Please install with: pip install ${pkg}"
        exit 1
    fi
done
echo "[INFO] All required packages found"
echo ""

# Set memory limit (disabled - conflicts with LMDB mmap)
# echo "[INFO] Setting memory limit to ${MEMORY_LIMIT_GB}GB"
# ulimit -v ${MEMORY_LIMIT_KB} || echo "[WARNING] Failed to set memory limit (may require sudo)"
echo "[INFO] Memory limit disabled to allow LMDB mmap allocation"
echo ""

# Display configuration
echo "Configuration:"
echo "  Data path:         ${DATA_PATH}"
echo "  LMDB output:       ${LMDB_OUTPUT}"
echo "  ELC file:          ${ELC_FILE}"
echo "  Preprocessing:     ${PREPROCESSING_SCRIPT}"
echo "  Number of workers: ${NUM_WORKERS}"
echo "  Memory limit:      ${MEMORY_LIMIT_GB}GB"
echo "  Log file:          ${LOG_FILE}"
echo ""

# Verify paths exist
if [ ! -d "${DATA_PATH}" ]; then
    echo "ERROR: Data path does not exist: ${DATA_PATH}"
    exit 1
fi

if [ ! -f "${ELC_FILE}" ]; then
    echo "ERROR: ELC file does not exist: ${ELC_FILE}"
    exit 1
fi

if [ ! -f "${PREPROCESSING_SCRIPT}" ]; then
    echo "ERROR: Preprocessing script does not exist: ${PREPROCESSING_SCRIPT}"
    exit 1
fi

# =============================================================================
# Run Preprocessing
# =============================================================================

echo "=========================================="
echo "Starting preprocessing..."
echo "=========================================="
echo ""

# Run with output to both console and log file
python "${PREPROCESSING_SCRIPT}" \
    --data_path "${DATA_PATH}" \
    --lmdb_path "${LMDB_OUTPUT}" \
    --elc_file "${ELC_FILE}" \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "${LOG_FILE}"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo "Output location: ${LMDB_OUTPUT}"
    echo "Log file:        ${LOG_FILE}"
    echo "Finished at:     $(date)"
    echo ""

    # Display output directory size
    echo "Output directory size:"
    du -sh "${LMDB_OUTPUT}"
    echo ""

    # Display LMDB split sizes
    if [ -d "${LMDB_OUTPUT}/train" ]; then
        echo "Train split size: $(du -sh ${LMDB_OUTPUT}/train | cut -f1)"
    fi
    if [ -d "${LMDB_OUTPUT}/val" ]; then
        echo "Val split size:   $(du -sh ${LMDB_OUTPUT}/val | cut -f1)"
    fi
    if [ -d "${LMDB_OUTPUT}/test" ]; then
        echo "Test split size:  $(du -sh ${LMDB_OUTPUT}/test | cut -f1)"
    fi
    echo ""

    exit 0
else
    echo ""
    echo "=========================================="
    echo "ERROR: Preprocessing failed!"
    echo "=========================================="
    echo "Check log file for details: ${LOG_FILE}"
    echo "Finished at: $(date)"
    echo ""
    exit 1
fi
