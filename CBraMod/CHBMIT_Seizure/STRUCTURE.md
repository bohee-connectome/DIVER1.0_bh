# CHB-MIT Seizure Detection Data Structure

This document describes the complete directory structure for the CHB-MIT preprocessing pipeline.

## Complete Directory Structure

```
CHBMIT_Seizure/
├── README.md                       # Overview and usage guide
├── CHBMIT_DATASET_INFO.md         # Detailed dataset documentation
├── STRUCTURE.md                    # This file - directory structure
│
├── scripts/                        # Preprocessing and validation scripts
│   ├── preprocessing_chbmit.py         # Main preprocessing script
│   ├── check_lmdb_chbmit.py           # LMDB validation script
│   ├── run_preprocessing_chbmit.sh    # Preprocessing runner script
│   └── standard_1005.elc              # Electrode location file (10-20 system)
│
├── logs/                          # Preprocessing logs and reports
│   └── run_20251121_190529.log    # Preprocessing log
│
└── lmdb_output/                   # LMDB database output (NOT in repo)
    └── CHBMIT_Seizure/
        ├── data.mdb               # LMDB data file
        └── lock.mdb               # LMDB lock file
```

## File Descriptions

### Documentation Files
- **README.md**: Quick start guide, task overview, usage instructions
- **CHBMIT_DATASET_INFO.md**: Comprehensive dataset documentation including 2-stage preprocessing pipeline
- **STRUCTURE.md**: This file - complete directory structure

### Scripts Directory
Contains all preprocessing and validation scripts.

#### preprocessing_chbmit.py
- **Purpose**: Main preprocessing script for CHB-MIT dataset
- **Input**: Raw CHB-MIT .edf files and -summary.txt files
- **Output**: LMDB database with preprocessed data
- **Key Functions**:
  - Load raw EDF recordings
  - Parse summary files for seizure timestamps
  - Extract 16 bipolar channels
  - Segment into 10-second segments
  - Apply oversampling for seizure segments
  - Resample from 256 Hz to 500 Hz
  - Reshape to (16, 10, 500)
  - Store in LMDB with metadata

#### check_lmdb_chbmit.py
- **Purpose**: Validate LMDB database integrity
- **Functions**:
  - Check LMDB file existence and accessibility
  - Verify data shapes and types
  - Validate label distributions (especially class imbalance)
  - Count oversampled vs. regular segments
  - Generate validation report

#### run_preprocessing_chbmit.sh
- **Purpose**: Bash script to run preprocessing with proper parameters
- **Usage**: `bash run_preprocessing_chbmit.sh`
- **Features**:
  - Set environment variables
  - Run preprocessing with logging
  - Handle errors and cleanup

#### standard_1005.elc
- **Purpose**: Electrode location file for 10-20 system
- **Format**: Standard ELC format
- **Usage**: Provides 3D coordinates for electrode positions
- **Electrodes**: All electrodes used in 16 bipolar channels

### Logs Directory
Contains preprocessing logs and reports.

#### run_20251121_190529.log
- **Content**: Full log from preprocessing pipeline
- **Includes**:
  - Processing timestamps
  - Patient processing status
  - Segment counts (regular + oversampled)
  - Data statistics (seizure vs. non-seizure)
  - Error messages (if any)
  - Completion summary

### LMDB Output Directory
**Note**: This directory is NOT included in the repository due to large file size.

#### Structure
```
lmdb_output/CHBMIT_Seizure/
├── data.mdb    # Main data file (~10 GB)
└── lock.mdb    # Lock file for concurrent access
```

#### LMDB Entry Format
Each entry in the LMDB database:
```python
Key: "{patient}_{recording}_{segment_info}"
# Regular segment example: "chb01_01_0", "chb01_01_2560"
# Oversampled segment example: "chb01_03_s0_add_25344"

Value: pickle.dumps({
    "signal": np.ndarray,        # Shape: (16, 10, 500), dtype: float32
    "label": int,                # 0=non-seizure, 1=seizure
    "elc_info": dict,            # Electrode information (bipolar pairs)
    "metadata": {
        "patient_id": str,       # "chb01"
        "recording_id": str,     # "chb01_01"
        "segment_index": int,    # 0, 2560, 5120, ...
        "is_oversampled": bool,  # True for s-add segments
        "original_sampling_rate": int,  # 256
        "target_sampling_rate": int     # 500
    }
})
```

## Data Flow

```
Raw CHB-MIT Data (.edf + -summary.txt files)
    ↓
[Parse Summary Files] (extract seizure timestamps)
    ↓
[Load EDF & Extract Channels] (16 bipolar channels)
    ↓
[Segment - Regular] (10-second segments, 10-second step)
    ↓
[Segment - Oversampled] (seizure ±1s, 5-second step)
    ↓
[Merge Segments] (combine regular + oversampled)
    ↓
[Assign Labels] (0=non-seizure, 1=seizure)
    ↓
[Resample] (256 Hz → 500 Hz)
    ↓
[Reshape] ((16, 2560) → (16, 5000) → (16, 10, 500))
    ↓
[Add ELC Info] (bipolar electrode pair positions)
    ↓
[Store in LMDB]
    ↓
lmdb_output/CHBMIT_Seizure/
```

## Data Statistics

### Dataset Size
- **Total Patients**: 21
- **Train Patients**: 17 (chb01-20, excluding 12, 13, 17)
- **Validation Patients**: 2 (chb21-22)
- **Test Patients**: 2 (chb23-24)

### Segments per Split (Estimated)
- **Train**: ~204,000 segments
- **Validation**: ~24,000 segments
- **Test**: ~24,000 segments
- **Total**: ~252,000 segments

### Class Distribution
- **Before oversampling**: Non-seizure:Seizure ≈ 99:1
- **After oversampling**: Non-seizure:Seizure ≈ 30:1

### Storage Size
- **LMDB database**: ~10 GB (compressed)
- **Per segment**: ~40 KB (signal + metadata)

## Bipolar Channel Configuration

The 16 bipolar channels follow the Double Banana montage:

```
Left Lateral Chain:
├── FP1-F7  (Frontopolar to Frontal left)
├── F7-T7   (Frontal to Temporal left)
├── T7-P7   (Temporal to Parietal left)
└── P7-O1   (Parietal to Occipital left)

Right Lateral Chain:
├── FP2-F8  (Frontopolar to Frontal right)
├── F8-T8   (Frontal to Temporal right)
├── T8-P8   (Temporal to Parietal right)
└── P8-O2   (Parietal to Occipital right)

Left Parasagittal Chain:
├── FP1-F3  (Frontopolar to Frontal left)
├── F3-C3   (Frontal to Central left)
├── C3-P3   (Central to Parietal left)
└── P3-O1   (Parietal to Occipital left)

Right Parasagittal Chain:
├── FP2-F4  (Frontopolar to Frontal right)
├── F4-C4   (Frontal to Central right)
├── C4-P4   (Central to Parietal right)
└── P4-O2   (Parietal to Occipital right)
```

## Files to Download from Server

The following files should be downloaded from the server:

**Server Path**: `/scratch/connectome/bohee/CHBMIT_preprocessing/`

### Required Files
1. **scripts/preprocessing_chbmit.py** - Main preprocessing script
2. **scripts/check_lmdb_chbmit.py** - Validation script
3. **scripts/run_preprocessing_chbmit.sh** - Runner script
4. **scripts/standard_1005.elc** - Electrode locations (may already exist)
5. **logs/run_20251121_190529.log** - Preprocessing log

### Optional Files
- Any additional utility scripts
- Additional validation reports
- Intermediate processing results

### Not Included in Repository
- **lmdb_output/** - Data files (too large, regenerate locally)
- **Raw CHB-MIT data** - Original .edf and -summary.txt files

## Oversampling Strategy

### Regular Segments (Non-seizure)
- **Step size**: 10 seconds
- **Sampling rate**: 1 segment per 10 seconds
- **Purpose**: Sparse sampling of non-seizure periods

### Oversampled Segments (Seizure)
- **Base segments**: Same as regular (10-second step)
- **Additional segments**: 5-second step within seizure ±1 second
- **Effective rate**: ~3x oversampling for seizure periods
- **Purpose**: Balance class distribution

### Example
```
Seizure at 100-110 seconds:

Regular segments (10s step):
├── [90-100]: label=0 (non-seizure)
├── [100-110]: label=1 (seizure) ← Base seizure segment
└── [110-120]: label=0 (non-seizure)

Additional oversampled segments (5s step, 99-111s range):
├── [99-109]: label=1 (seizure)  ← s-0-add
├── [104-114]: label=1 (seizure) ← s-0-add
└── [109-119]: label=1 (seizure) ← s-0-add

Result: 1 base + 3 additional = 4 seizure segments total
```

## Usage Notes

1. **Preprocessing**: Run `bash run_preprocessing_chbmit.sh` to generate LMDB
2. **Validation**: Run `check_lmdb_chbmit.py` to verify data integrity
3. **Training**: Point your training script to `lmdb_output/CHBMIT_Seizure/`
4. **Class Weights**: Consider using class weights even after oversampling
5. **Logs**: Check logs/ for detailed statistics and potential issues

## Version Information

- **Structure Version**: 1.0
- **Last Updated**: 2025-11-22
- **Preprocessing Run**: 2025-11-21 19:05:29
