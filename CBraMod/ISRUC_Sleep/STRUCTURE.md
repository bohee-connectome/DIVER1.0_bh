# ISRUC-Sleep Data Structure

This document describes the complete directory structure for the ISRUC-Sleep preprocessing pipeline.

## Complete Directory Structure

```
ISRUC_Sleep/
├── README.md                       # Overview and usage guide
├── ISRUC_DATASET_INFO.md          # Detailed dataset documentation
├── STRUCTURE.md                    # This file - directory structure
│
├── scripts/                        # Preprocessing and validation scripts
│   ├── preprocessing_isruc-sleep.py    # Main preprocessing script
│   ├── check_lmdb_isruc.py            # LMDB validation script
│   └── standard_1005.elc              # Electrode location file (10-20 system)
│
├── logs/                          # Preprocessing logs and reports
│   └── preprocessing_v3_FULL.log  # Full preprocessing log
│
└── lmdb_output/                   # LMDB database output (NOT in repo)
    └── ISRUC_Sleep/
        ├── data.mdb               # LMDB data file
        └── lock.mdb               # LMDB lock file
```

## File Descriptions

### Documentation Files
- **README.md**: Quick start guide, task overview, usage instructions
- **ISRUC_DATASET_INFO.md**: Comprehensive dataset documentation including preprocessing pipeline details
- **STRUCTURE.md**: This file - complete directory structure

### Scripts Directory
Contains all preprocessing and validation scripts.

#### preprocessing_isruc-sleep.py
- **Purpose**: Main preprocessing script for ISRUC-Sleep dataset
- **Input**: Raw ISRUC-Sleep .rec files
- **Output**: LMDB database with preprocessed data
- **Key Functions**:
  - Load raw PSG recordings
  - Extract 6 EEG channels
  - Segment into 30-second epochs
  - Resample from 200 Hz to 500 Hz
  - Reshape to (6, 30, 500)
  - Store in LMDB with metadata

#### check_lmdb_isruc.py
- **Purpose**: Validate LMDB database integrity
- **Functions**:
  - Check LMDB file existence and accessibility
  - Verify data shapes and types
  - Validate label distributions
  - Generate validation report

#### standard_1005.elc
- **Purpose**: Electrode location file for 10-20 system
- **Format**: Standard ELC format
- **Usage**: Provides 3D coordinates for electrode positions
- **Electrodes**: F3, F4, C3, C4, O1, O2, A1, A2

### Logs Directory
Contains preprocessing logs and validation reports.

#### preprocessing_v3_FULL.log
- **Content**: Full log from preprocessing pipeline
- **Includes**:
  - Processing timestamps
  - Subject processing status
  - Data statistics
  - Error messages (if any)
  - Completion summary

### LMDB Output Directory
**Note**: This directory is NOT included in the repository due to large file size.

#### Structure
```
lmdb_output/ISRUC_Sleep/
├── data.mdb    # Main data file (~2-3 GB)
└── lock.mdb    # Lock file for concurrent access
```

#### LMDB Entry Format
Each entry in the LMDB database:
```python
Key: "{subject_id}_{epoch_index}"
# Example: "subject085_0", "subject085_1", ...

Value: pickle.dumps({
    "signal": np.ndarray,        # Shape: (6, 30, 500), dtype: float32
    "label": int,                # 0-4 (W, N1, N2, N3, REM)
    "elc_info": dict,            # Electrode information
    "metadata": {
        "subject_id": str,       # "subject085"
        "epoch_index": int,      # 0, 1, 2, ...
        "original_sampling_rate": int,  # 200
        "target_sampling_rate": int,    # 500
        "epoch_length_sec": int         # 30
    }
})
```

## Data Flow

```
Raw ISRUC Data (.rec files)
    ↓
[Load & Extract Channels] (6 EEG channels)
    ↓
[Segment] (30-second epochs)
    ↓
[Assign Labels] (from .txt files)
    ↓
[Resample] (200 Hz → 500 Hz)
    ↓
[Reshape] ((6, 6000) → (6, 15000) → (6, 30, 500))
    ↓
[Add ELC Info] (electrode positions)
    ↓
[Store in LMDB]
    ↓
lmdb_output/ISRUC_Sleep/
```

## Data Statistics

### Dataset Size
- **Total Subjects**: 100
- **Train Subjects**: 84 (subjects 1-84)
- **Validation Subjects**: 6 (subjects 85-90)
- **Test Subjects**: 10 (subjects 91-100)

### Epochs per Split
- **Train**: ~40,000 epochs
- **Validation**: ~3,000 epochs
- **Test**: ~5,000 epochs
- **Total**: ~48,000 epochs

### Storage Size
- **LMDB database**: ~2.4 GB (compressed)
- **Per epoch**: ~50 KB (signal + metadata)

## Files to Download from Server

The following files should be downloaded from the server:

**Server Path**: `/scratch/connectome/bohee/ISRUC_preprocessing/`

### Required Files
1. **scripts/preprocessing_isruc-sleep.py** - Main preprocessing script
2. **scripts/check_lmdb_isruc.py** - Validation script
3. **scripts/standard_1005.elc** - Electrode locations (may already exist)
4. **logs/preprocessing_v3_FULL.log** - Preprocessing log

### Optional Files
- Any additional utility scripts
- Additional validation reports

### Not Included in Repository
- **lmdb_output/** - Data files (too large, regenerate locally)
- **Raw ISRUC data** - Original .rec and .txt files

## Usage Notes

1. **Preprocessing**: Run `preprocessing_isruc-sleep.py` to generate LMDB
2. **Validation**: Run `check_lmdb_isruc.py` to verify data integrity
3. **Training**: Point your training script to `lmdb_output/ISRUC_Sleep/`
4. **Logs**: Check logs/ for preprocessing details and statistics

## Version Information

- **Structure Version**: 1.0
- **Last Updated**: 2025-11-22
- **Preprocessing Version**: v3 (FULL)
