# ADFTD Data Structure

This document describes the complete directory structure for the ADFTD (Alzheimer's Disease & Frontotemporal Dementia) preprocessing pipeline.

## Complete Directory Structure

```
ADFTD/
├── README.md                       # Overview and usage guide
├── ADFTD_DATASET_INFO.md          # Detailed dataset documentation
├── STRUCTURE.md                    # This file - directory structure
│
├── scripts/                        # Preprocessing and validation scripts
│   ├── preprocessing_generalized_ADFTD.py           # Main preprocessing script
│   ├── preprocessing_generalized_datasetsetting_ADFTD.py  # Data split script
│   ├── clip_extraction_utils.py                     # Artifact removal utilities
│   ├── check_lmdb_adftd_v1.py                      # Validation script (v1)
│   ├── check_lmdb_adftd_v2.py                      # Validation script (v2)
│   ├── check_v2_shapes.py                          # Shape validation script
│   ├── run_preprocessing_v1.sh                     # Preprocessing runner (v1)
│   ├── run_preprocessing_v2.sh                     # Preprocessing runner (v2)
│   └── standard_1005.elc                           # Electrode location file
│
├── logs/                           # Preprocessing logs and reports
│   ├── v1_output_65051.log        # Version 1 output log
│   ├── v1_error_65051.log         # Version 1 error log
│   ├── v2_output_65052.log        # Version 2 output log
│   ├── v2_error_65052.log         # Version 2 error log
│   ├── validation_v1_report.txt   # Version 1 validation report
│   └── validation_v2_report.txt   # Version 2 validation report
│
└── data/                           # Data directory (NOT in repo)
    ├── processed_v1/               # Version 1 preprocessed data
    │   ├── 1.0_ADFTD/
    │   │   ├── train_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   ├── val_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   ├── test_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   ├── merged_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   └── ADFTD_1.0_complete.txt
    │   └── ADFTD_1.0_complete.txt
    ├── processed_v1.tar.gz         # Backup/archive of v1
    │
    ├── processed_v2/               # Version 2 preprocessed data (recommended)
    │   ├── 1.0_ADFTD/
    │   │   ├── train_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   ├── val_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   ├── test_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   ├── merged_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   │   ├── data.mdb
    │   │   │   └── lock.mdb
    │   │   └── ADFTD_1.0_complete.txt
    │   └── ADFTD_1.0_complete.txt
    └── processed_v2.tar.gz         # Backup/archive of v2
```

## Server Paths

### Scripts Location
```
Server: /storage/connectome/bohee/DIVER_ADFTD/scripts/
```

### Logs Location
```
Server: /storage/connectome/bohee/DIVER_ADFTD/logs/
```

### Data Location
```
Server: /scratch/connectome/bohee/DIVER_ADFTD/data/
```

## File Descriptions

### Documentation Files
- **README.md**: Quick start guide, task overview, usage instructions
- **ADFTD_DATASET_INFO.md**: Comprehensive dataset documentation including preprocessing pipeline details
- **STRUCTURE.md**: This file - complete directory structure

### Scripts Directory
Contains all preprocessing and validation scripts.

#### preprocessing_generalized_ADFTD.py
- **Purpose**: Main preprocessing script for ADFTD dataset
- **Input**: Raw ADFTD EDF files
- **Output**: LMDB databases (train/val/test/merged)
- **Key Functions**:
  - Load raw EDF recordings
  - Extract 19 standard channels
  - Apply artifact removal (clipping)
  - Segment into 10-second segments
  - Assign labels (CN=0, AD=1, FTD=2)
  - Resample from 256 Hz to 500 Hz
  - Reshape to (19, 10, 500)
  - Store in LMDB with metadata

#### preprocessing_generalized_datasetsetting_ADFTD.py
- **Purpose**: Data split configuration script
- **Functions**:
  - Define train/val/test splits (70/15/15)
  - Stratified split by diagnosis
  - Subject-level splitting
  - Generate split lists

#### clip_extraction_utils.py
- **Purpose**: Artifact detection and removal utilities
- **Functions**:
  - Amplitude clipping detection (|signal| > 100 μV)
  - Gradient clipping detection (rapid changes > 50 μV)
  - Flatline detection (std < 5 μV)
  - Extract clean continuous segments
  - Quality assessment

#### check_lmdb_adftd_v1.py
- **Purpose**: Validate v1 LMDB database integrity
- **Functions**:
  - Check LMDB file existence
  - Verify data shapes and types
  - Validate label distributions
  - Generate validation report

#### check_lmdb_adftd_v2.py
- **Purpose**: Validate v2 LMDB database integrity
- **Functions**: Same as v1 but for version 2 data

#### check_v2_shapes.py
- **Purpose**: Specifically check shape consistency in v2
- **Functions**:
  - Verify all signals have shape (19, 10, 500)
  - Check for shape anomalies
  - Report statistics

#### run_preprocessing_v1.sh / run_preprocessing_v2.sh
- **Purpose**: Bash scripts to run preprocessing with proper parameters
- **Features**:
  - Set environment variables
  - Run preprocessing with logging
  - Handle errors and cleanup
  - Submit to compute cluster (if applicable)

#### standard_1005.elc
- **Purpose**: Electrode location file for 10-20 system
- **Format**: Standard ELC format
- **Usage**: Provides 3D coordinates for electrode positions
- **Electrodes**: All 19 standard 10-20 system electrodes

### Logs Directory
Contains preprocessing logs and validation reports for both versions.

#### Version 1 Logs
- **v1_output_65051.log**: Standard output from v1 preprocessing
- **v1_error_65051.log**: Error messages from v1 preprocessing
- **validation_v1_report.txt**: Validation results for v1 data

#### Version 2 Logs (Recommended)
- **v2_output_65052.log**: Standard output from v2 preprocessing
- **v2_error_65052.log**: Error messages from v2 preprocessing
- **validation_v2_report.txt**: Validation results for v2 data

### Data Directory
**Note**: This directory is NOT included in the repository due to large file size and privacy concerns.

#### LMDB Entry Format
Each entry in the LMDB database:
```python
Key: "{subject_id}_{segment_index}"
# Example: "subject001_CN_0", "subject001_CN_1", ...

Value: pickle.dumps({
    "signal": np.ndarray,        # Shape: (19, 10, 500), dtype: float32
    "label": int,                # 0=CN, 1=AD, 2=FTD
    "elc_info": dict,            # Electrode information
    "metadata": {
        "subject_id": str,       # "subject001_CN"
        "segment_index": int,    # 0, 1, 2, ...
        "original_sampling_rate": int,  # 256
        "target_sampling_rate": int,    # 500
        "diagnosis": str         # "CN", "AD", or "FTD"
    }
})
```

#### LMDB Naming Convention
```
{split}_resample-{rate}_highpass-{hp}_lowpass-{lp}.lmdb
```
- **split**: train, val, test, or merged
- **rate**: 500 Hz
- **hp**: 0.5 Hz (high-pass filter)
- **lp**: 45.0 Hz (low-pass filter)

## Data Flow

```
Raw ADFTD Data (EDF files: CN/AD/FTD)
    ↓
[Load EDF & Extract Channels] (19 standard electrodes)
    ↓
[Clipping - Artifact Removal]
├── Amplitude clipping (|signal| > 100 μV)
├── Gradient clipping (rapid changes > 50 μV)
└── Flatline detection (std < 5 μV)
    ↓
[Extract Clean Segments]
    ↓
[Segment] (10-second segments, non-overlapping)
    ↓
[Assign Labels] (CN=0, AD=1, FTD=2)
    ↓
[Data Split] (Stratified 70/15/15, subject-level)
    ↓
[Resample] (256 Hz → 500 Hz)
    ↓
[Reshape] ((19, 2560) → (19, 5000) → (19, 10, 500))
    ↓
[Add ELC Info] (electrode positions)
    ↓
[Store in LMDB] (separate for train/val/test/merged)
    ↓
data/processed_v2/1.0_ADFTD/
```

## Data Statistics

### Dataset Size
- **Total Subjects**: 88 patients
  - CN: 30 subjects (~34%)
  - AD: 35 subjects (~40%)
  - FTD: 23 subjects (~26%)

### Data Split
- **Train**: ~62 subjects (70%, stratified)
- **Validation**: ~13 subjects (15%, stratified)
- **Test**: ~13 subjects (15%, stratified)

### Segments (Estimated, after clipping)
- **Train**: ~4,000 segments
- **Validation**: ~850 segments
- **Test**: ~850 segments
- **Total**: ~5,700 segments

### Storage Size
- **Per split LMDB**: ~75-100 MB
- **Merged LMDB**: ~300 MB
- **Total (all LMDBs)**: ~600 MB
- **Tar archives**: ~300 MB each (compressed)

## Preprocessing Versions

### Version 1 (v1)
- Initial implementation
- Basic artifact removal
- Standard preprocessing pipeline
- See logs/validation_v1_report.txt for results

### Version 2 (v2) - Recommended
- Improved artifact detection
- Enhanced quality control
- More robust clipping
- Better segment quality
- See logs/validation_v2_report.txt for results

## Channel Configuration

19 channels in standard 10-20 system:
```
Frontal Region:
├── FP1  (Frontopolar left)
├── FP2  (Frontopolar right)
├── F3   (Frontal left)
├── F4   (Frontal right)
├── F7   (Anterior temporal left)
├── F8   (Anterior temporal right)
└── FZ   (Frontal midline)

Central Region:
├── C3   (Central left)
├── C4   (Central right)
└── CZ   (Central midline)

Temporal Region:
├── T3   (Mid-temporal left)
├── T4   (Mid-temporal right)
├── T5   (Posterior temporal left)
└── T6   (Posterior temporal right)

Parietal Region:
├── P3   (Parietal left)
├── P4   (Parietal right)
└── PZ   (Parietal midline)

Occipital Region:
├── O1   (Occipital left)
└── O2   (Occipital right)
```

## Files to Download from Server

### Required Files

**Scripts** (from `/storage/connectome/bohee/DIVER_ADFTD/scripts/`):
1. preprocessing_generalized_ADFTD.py
2. preprocessing_generalized_datasetsetting_ADFTD.py
3. clip_extraction_utils.py
4. check_lmdb_adftd_v1.py
5. check_lmdb_adftd_v2.py
6. check_v2_shapes.py
7. run_preprocessing_v1.sh
8. run_preprocessing_v2.sh
9. standard_1005.elc

**Logs** (from `/storage/connectome/bohee/DIVER_ADFTD/logs/`):
1. v1_output_65051.log
2. v1_error_65051.log
3. v2_output_65052.log
4. v2_error_65052.log
5. validation_v1_report.txt
6. validation_v2_report.txt

### Not Included in Repository
- **data/** directory - All LMDB files (located at `/scratch/connectome/bohee/DIVER_ADFTD/data/`)
- **Raw EDF files** - Original patient data (confidential)
- **Intermediate processing files** - Temporary files from preprocessing

## Usage Notes

1. **Preprocessing**: Run `bash run_preprocessing_v2.sh` (recommended)
2. **Validation**: Run `python check_lmdb_adftd_v2.py`
3. **Training**: Use separate LMDB databases (train/val/test) or merged
4. **Class Balance**: Apply stratified sampling or class weights during training
5. **Logs**: Check logs/ for detailed preprocessing statistics

## Artifact Removal Details

### Amplitude Clipping
```python
threshold_high = 100  # μV
threshold_low = -100  # μV
# Remove segments where any sample exceeds thresholds
```

### Gradient Clipping
```python
threshold_gradient = 50  # μV
gradient = np.diff(signal)
# Remove segments with rapid changes exceeding threshold
```

### Flatline Detection
```python
threshold_std = 5  # μV
std_window = np.std(signal[window])
# Remove segments with insufficient variation
```

## Version Information

- **Structure Version**: 1.0
- **Last Updated**: 2025-11-22
- **Preprocessing Versions**: v1 (65051), v2 (65052)
- **Recommended Version**: v2

## Data Availability

**Important**: Due to patient privacy and data usage agreements:
- Raw EDF files are NOT publicly available
- LMDB files are NOT included in this repository
- Contact LEAD project team for data access
- See README.md for data access procedures
