# ISRUC-Sleep: Sleep Stage Classification

This directory contains preprocessing scripts and data for the ISRUC-Sleep dataset applied to sleep stage classification task.

## Task Overview

- **Task**: 5-class sleep stage classification
- **Classes**: W (Wake), N1, N2, N3, REM
- **Dataset**: ISRUC-Sleep Subgroup 1
- **Subjects**: 100 patients
- **Original Paper**: Khalighi et al. (2016), Computer Methods and Programs in Biomedicine

## Dataset Information

For detailed information about the ISRUC-Sleep dataset, preprocessing pipeline, and data structure, see:
- **[ISRUC_DATASET_INFO.md](ISRUC_DATASET_INFO.md)** - Comprehensive dataset documentation

### Quick Summary
- **Channels**: 6 EEG channels (F3-A2, C3-A2, F4-A1, C4-A1, O1-A2, O2-A1)
- **Original Sampling Rate**: 200 Hz
- **Target Sampling Rate**: 500 Hz (resampled for DIVER)
- **Epoch Length**: 30 seconds
- **Final Shape**: (6, 30, 500) - 6 channels × 30 time segments × 500 samples/second

## Data Split

- **Train**: Subjects 1-84 (84 subjects, ~40,000 epochs)
- **Validation**: Subjects 85-90 (6 subjects, ~3,000 epochs)
- **Test**: Subjects 91-100 (10 subjects, ~5,000 epochs)

## Directory Structure

```
ISRUC_Sleep/
├── README.md                  # This file
├── ISRUC_DATASET_INFO.md      # Detailed dataset documentation
├── STRUCTURE.md               # Data structure documentation
├── scripts/                   # Preprocessing scripts
│   ├── preprocessing_isruc-sleep.py      # Main preprocessing script
│   ├── check_lmdb_isruc.py               # LMDB validation script
│   └── [Other scripts - download from server]
├── logs/                      # Preprocessing logs
│   ├── preprocessing_v3_FULL.log         # Preprocessing log
│   └── [Other logs - download from server]
└── lmdb_output/               # LMDB output (not in repo)
    └── ISRUC_Sleep/
        ├── data.mdb
        └── lock.mdb
```

## Files Status

### Available Files
- ✅ `README.md` - This file
- ✅ `ISRUC_DATASET_INFO.md` - Dataset documentation
- ✅ `STRUCTURE.md` - Data structure documentation

### Files to Download from Server
The following files need to be downloaded from the server:
- ⏳ `scripts/preprocessing_isruc-sleep.py` - Main preprocessing script
- ⏳ `scripts/check_lmdb_isruc.py` - LMDB validation script
- ⏳ `logs/preprocessing_v3_FULL.log` - Preprocessing log

Server location: `/scratch/connectome/bohee/ISRUC_preprocessing/`

### Data Files (Not in Repo)
- ❌ `lmdb_output/ISRUC_Sleep/` - LMDB data files (too large, not stored in repo)

## Preprocessing

### Requirements
```bash
pip install numpy scipy mne lmdb
```

### Usage
```bash
cd scripts
python preprocessing_isruc-sleep.py
```

### Output
- LMDB files in `lmdb_output/ISRUC_Sleep/`
- Each entry format:
```python
{
    "signal": np.array (6, 30, 500),  # float32
    "label": int (0-4),  # W=0, N1=1, N2=2, N3=3, REM=4
    "elc_info": dict,
    "metadata": {
        "subject_id": "subject085",
        "epoch_index": 0,
        "original_sampling_rate": 200,
        "target_sampling_rate": 500,
        "epoch_length_sec": 30
    }
}
```

## Class Distribution

| Stage | Label | Description | Approximate Ratio |
|-------|-------|-------------|------------------|
| W | 0 | Wake | ~5% |
| N1 | 1 | Light Sleep | ~5% |
| N2 | 2 | Moderate Sleep | ~50% |
| N3 | 3 | Deep Sleep | ~20% |
| REM | 4 | REM Sleep | ~20% |

**Note**: Severe class imbalance with N2 dominating. Consider using class weights during training.

## Validation

Check LMDB integrity:
```bash
cd scripts
python check_lmdb_isruc.py
```

## Original Data Source

- **Dataset**: ISRUC-Sleep Database
- **Source**: https://sleeptight.isr.uc.pt/
- **PhysioNet**: https://physionet.org/content/isruc-sleep/1.0.0/
- **License**: Open access, research use
- **Citation Required**: Yes

## Citation

```bibtex
@article{khalighi2016isruc,
  title={ISRUC-Sleep: A comprehensive public dataset for sleep researchers},
  author={Khalighi, S. and Sousa, T. and Santos, J. M. and Nunes, U.},
  journal={Computer Methods and Programs in Biomedicine},
  volume={124},
  pages={180--192},
  year={2016},
  doi={10.1016/j.cmpb.2015.10.013}
}
```

## Contact

For questions about:
- **Dataset**: See ISRUC-Sleep official website
- **Preprocessing**: See ISRUC_DATASET_INFO.md
- **Implementation**: Open an issue on GitHub
