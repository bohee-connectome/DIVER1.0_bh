# ADFTD: Alzheimer's Disease & Frontotemporal Dementia Classification

This directory contains preprocessing scripts and data for the ADFTD dataset applied to neurodegenerative disease classification task.

## Task Overview

- **Task**: 3-class classification
- **Classes**:
  - CN (Cognitively Normal) - Label 0
  - AD (Alzheimer's Disease) - Label 1
  - FTD (Frontotemporal Dementia) - Label 2
- **Dataset**: ADFTD EEG Dataset
- **Subjects**: 88 patients (30 CN, 35 AD, 23 FTD)
- **Original Paper**: Park et al. (2024), Journal of Neural Engineering

## Dataset Information

For detailed information about the ADFTD dataset, preprocessing pipeline, and data structure, see:
- **[ADFTD_DATASET_INFO.md](ADFTD_DATASET_INFO.md)** - Comprehensive dataset documentation

### Quick Summary
- **Channels**: 19 standard 10-20 system electrodes
- **Original Sampling Rate**: 256 Hz
- **Target Sampling Rate**: 500 Hz (resampled for DIVER)
- **Segment Length**: 10 seconds
- **Recording**: Eyes-closed resting state (5-30 minutes)
- **Final Shape**: (19, 10, 500) - 19 channels × 10 time segments × 500 samples/second

## Data Split

- **Train**: 70% (~62 subjects, stratified by class)
- **Validation**: 15% (~13 subjects, stratified by class)
- **Test**: 15% (~13 subjects, stratified by class)

Split is performed at the subject level to prevent data leakage.

## Directory Structure

```
ADFTD/
├── README.md                  # This file
├── ADFTD_DATASET_INFO.md      # Detailed dataset documentation
├── STRUCTURE.md               # Data structure documentation
├── scripts/                   # Preprocessing scripts
│   ├── preprocessing_generalized_ADFTD.py
│   ├── preprocessing_generalized_datasetsetting_ADFTD.py
│   ├── clip_extraction_utils.py
│   ├── check_lmdb_adftd_v1.py
│   ├── check_lmdb_adftd_v2.py
│   ├── check_v2_shapes.py
│   ├── run_preprocessing_v1.sh
│   ├── run_preprocessing_v2.sh
│   └── standard_1005.elc
├── logs/                      # Preprocessing logs
│   ├── v1_output_65051.log
│   ├── v1_error_65051.log
│   ├── v2_output_65052.log
│   ├── v2_error_65052.log
│   ├── validation_v1_report.txt
│   └── validation_v2_report.txt
└── data/                      # Data directory (not in repo)
    ├── processed_v1/          # Version 1 preprocessed data
    │   ├── 1.0_ADFTD/
    │   │   ├── train_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   ├── val_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   ├── test_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   ├── merged_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   └── ADFTD_1.0_complete.txt
    │   └── ADFTD_1.0_complete.txt
    ├── processed_v1.tar.gz    # Backup/archive
    ├── processed_v2/          # Version 2 preprocessed data (recommended)
    │   ├── 1.0_ADFTD/
    │   │   ├── train_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   ├── val_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   ├── test_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   ├── merged_resample-500_highpass-0.5_lowpass-45.0.lmdb/
    │   │   └── ADFTD_1.0_complete.txt
    │   └── ADFTD_1.0_complete.txt
    └── processed_v2.tar.gz    # Backup/archive
```

## Files Status

### Available Files
- ✅ `README.md` - This file
- ✅ `ADFTD_DATASET_INFO.md` - Dataset documentation
- ✅ `STRUCTURE.md` - Data structure documentation

### Files to Download from Server
The following files need to be downloaded from the server:

**Scripts** (from `/storage/connectome/bohee/DIVER_ADFTD/scripts/`):
- ⏳ `scripts/preprocessing_generalized_ADFTD.py`
- ⏳ `scripts/preprocessing_generalized_datasetsetting_ADFTD.py`
- ⏳ `scripts/clip_extraction_utils.py`
- ⏳ `scripts/check_lmdb_adftd_v1.py`
- ⏳ `scripts/check_lmdb_adftd_v2.py`
- ⏳ `scripts/check_v2_shapes.py`
- ⏳ `scripts/run_preprocessing_v1.sh`
- ⏳ `scripts/run_preprocessing_v2.sh`
- ⏳ `scripts/standard_1005.elc`

**Logs** (from `/storage/connectome/bohee/DIVER_ADFTD/logs/`):
- ⏳ `logs/v1_output_65051.log`
- ⏳ `logs/v1_error_65051.log`
- ⏳ `logs/v2_output_65052.log`
- ⏳ `logs/v2_error_65052.log`
- ⏳ `logs/validation_v1_report.txt`
- ⏳ `logs/validation_v2_report.txt`

### Data Files (Not in Repo)
Data files are located on the server at `/scratch/connectome/bohee/DIVER_ADFTD/data/` and are NOT included in this repository due to:
- Large file size (~300 MB for LMDB)
- Patient privacy concerns
- Data usage agreements

## Preprocessing

### Requirements
```bash
pip install numpy scipy mne lmdb
```

### Preprocessing Versions

#### Version 1 (v1)
```bash
cd scripts
bash run_preprocessing_v1.sh
```

#### Version 2 (v2) - Recommended
```bash
cd scripts
bash run_preprocessing_v2.sh
```

Version 2 includes improved artifact detection and quality control.

### Output
- LMDB files in `data/processed_v2/1.0_ADFTD/`
- Separate LMDB databases for train, val, test, and merged
- Each entry format:
```python
{
    "signal": np.array (19, 10, 500),  # float32
    "label": int (0, 1, or 2),  # CN=0, AD=1, FTD=2
    "elc_info": dict,
    "metadata": {
        "subject_id": "subject001_CN",
        "segment_index": 0,
        "original_sampling_rate": 256,
        "target_sampling_rate": 500,
        "diagnosis": "CN"  # or "AD", "FTD"
    }
}
```

## Artifact Removal (Clipping)

The preprocessing includes automatic artifact detection and removal:
- **Amplitude clipping**: Remove segments with |signal| > 100 μV
- **Gradient clipping**: Remove segments with rapid changes > 50 μV
- **Flatline detection**: Remove segments with std < 5 μV

Only clean, continuous segments are kept for analysis.

## Class Distribution

| Class | Label | Description | Subjects | Ratio |
|-------|-------|-------------|----------|-------|
| CN | 0 | Cognitively Normal | 30 | ~34% |
| AD | 1 | Alzheimer's Disease | 35 | ~40% |
| FTD | 2 | Frontotemporal Dementia | 23 | ~26% |

**Total**: 88 subjects

### Handling Class Imbalance
- **Stratified split**: Maintain class ratios in train/val/test
- **Balanced sampling**: Sample equally from each class during training
- **Class weights**: Apply weights inversely proportional to class frequencies

## Channel Configuration

19 channels in standard 10-20 system:
```
Frontal:  FP1, FP2, F3, F4, F7, F8, FZ
Central:  C3, C4, CZ
Temporal: T3, T4, T5, T6
Parietal: P3, P4, PZ
Occipital: O1, O2
```

## Validation

Check LMDB integrity:
```bash
# Check version 1
cd scripts
python check_lmdb_adftd_v1.py

# Check version 2 (recommended)
python check_lmdb_adftd_v2.py

# Check v2 shapes specifically
python check_v2_shapes.py
```

## Estimated Dataset Size

- **Average segments per subject**: 50-80 (after clipping)
- **Train**: ~4,000 segments (62 subjects)
- **Validation**: ~850 segments (13 subjects)
- **Test**: ~850 segments (13 subjects)
- **Total**: ~5,700 segments
- **LMDB Size**: ~300 MB (compressed)

## Data Availability

**Important**: The ADFTD dataset is NOT publicly available and is NOT included in this repository.

To obtain access:
1. Contact the LEAD project team
2. Sign data usage agreement
3. Receive raw EDF files
4. Run preprocessing scripts to generate LMDB

## Citation

```bibtex
@article{lead2024,
  title={LEAD: Learning EEG Analysis for neurodegenerative Diseases},
  author={Park, J. E., et al.},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

## License

- **Code**: See LEAD original repository for license
- **Data**: Research use only, redistribution not allowed
- **Citation Required**: Must cite when publishing results

## Contact

For questions about:
- **Dataset access**: Contact LEAD project team
- **Preprocessing**: See ADFTD_DATASET_INFO.md
- **Implementation**: Open an issue on GitHub
