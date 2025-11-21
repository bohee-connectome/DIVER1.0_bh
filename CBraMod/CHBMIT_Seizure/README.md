# CHB-MIT: Seizure Detection

This directory contains preprocessing scripts and data for the CHB-MIT Scalp EEG Database applied to seizure detection task.

## Task Overview

- **Task**: Binary seizure detection
- **Classes**: Non-seizure (0), Seizure (1)
- **Dataset**: CHB-MIT Scalp EEG Database
- **Subjects**: 21 pediatric epilepsy patients
- **Original Paper**: Shoeb (2009), ICML

## Dataset Information

For detailed information about the CHB-MIT dataset, preprocessing pipeline, and data structure, see:
- **[CHBMIT_DATASET_INFO.md](CHBMIT_DATASET_INFO.md)** - Comprehensive dataset documentation

### Quick Summary
- **Channels**: 16 bipolar channels (Double Banana montage)
- **Original Sampling Rate**: 256 Hz
- **Target Sampling Rate**: 500 Hz (resampled for DIVER)
- **Segment Length**: 10 seconds
- **Final Shape**: (16, 10, 500) - 16 channels × 10 time segments × 500 samples/second

## Data Split

- **Train**: chb01-20 (17 patients, excluding 12, 13, 17)
- **Validation**: chb21-22 (2 patients)
- **Test**: chb23-24 (2 patients)

## Directory Structure

```
CHBMIT_Seizure/
├── README.md                  # This file
├── CHBMIT_DATASET_INFO.md     # Detailed dataset documentation
├── STRUCTURE.md               # Data structure documentation
├── scripts/                   # Preprocessing scripts
│   ├── preprocessing_chbmit.py           # Main preprocessing script
│   ├── check_lmdb_chbmit.py              # LMDB validation script
│   ├── run_preprocessing_chbmit.sh       # Preprocessing runner script
│   └── [Other scripts - download from server]
├── logs/                      # Preprocessing logs
│   ├── run_20251121_190529.log           # Preprocessing log
│   └── [Other logs - download from server]
└── lmdb_output/               # LMDB output (not in repo)
    └── CHBMIT_Seizure/
        ├── data.mdb
        └── lock.mdb
```

## Files Status

### Available Files
- ✅ `README.md` - This file
- ✅ `CHBMIT_DATASET_INFO.md` - Dataset documentation
- ✅ `STRUCTURE.md` - Data structure documentation

### Files to Download from Server
The following files need to be downloaded from the server:
- ⏳ `scripts/preprocessing_chbmit.py` - Main preprocessing script
- ⏳ `scripts/check_lmdb_chbmit.py` - LMDB validation script
- ⏳ `scripts/run_preprocessing_chbmit.sh` - Preprocessing runner script
- ⏳ `logs/run_20251121_190529.log` - Preprocessing log

Server location: `/scratch/connectome/bohee/CHBMIT_preprocessing/`

### Data Files (Not in Repo)
- ❌ `lmdb_output/CHBMIT_Seizure/` - LMDB data files (too large, not stored in repo)

## Preprocessing

### Requirements
```bash
pip install numpy scipy mne lmdb
```

### Usage
```bash
cd scripts
bash run_preprocessing_chbmit.sh
```

Or directly:
```bash
cd scripts
python preprocessing_chbmit.py
```

### Output
- LMDB files in `lmdb_output/CHBMIT_Seizure/`
- Each entry format:
```python
{
    "signal": np.array (16, 10, 500),  # float32
    "label": int (0 or 1),  # 0=non-seizure, 1=seizure
    "elc_info": dict,
    "metadata": {
        "patient_id": "chb01",
        "recording_id": "chb01_01",
        "segment_index": 0,
        "is_oversampled": False,  # True for s-add samples
        "original_sampling_rate": 256,
        "target_sampling_rate": 500
    }
}
```

## Class Imbalance

The CHB-MIT dataset has severe class imbalance:
- **Non-seizure segments**: ~99%
- **Seizure segments**: ~1%

### Solution: Oversampling
- **Non-seizure**: 10-second step (sparse sampling)
- **Seizure**: 5-second step (dense sampling, 2x oversampling)
- Additional seizure samples include ±1 second around seizure boundaries

After oversampling:
- Imbalance ratio improved from 99:1 to approximately 30:1

## Channel Configuration

16 bipolar channels in Double Banana montage:
```
Left Lateral:   FP1-F7, F7-T7, T7-P7, P7-O1
Right Lateral:  FP2-F8, F8-T8, T8-P8, P8-O2
Left Parasag:   FP1-F3, F3-C3, C3-P3, P3-O1
Right Parasag:  FP2-F4, F4-C4, C4-P4, P4-O2
```

## Validation

Check LMDB integrity:
```bash
cd scripts
python check_lmdb_chbmit.py
```

## Estimated Dataset Size

- **Train**: ~204,000 segments
- **Validation**: ~24,000 segments
- **Test**: ~24,000 segments
- **Total**: ~252,000 segments
- **LMDB Size**: ~10 GB (compressed)

## Original Data Source

- **Dataset**: CHB-MIT Scalp EEG Database
- **Source**: https://physionet.org/content/chbmit/1.0.0/
- **DOI**: 10.13026/C2K01R
- **License**: Open access, research use
- **Citation Required**: Yes

## Citation

```bibtex
@inproceedings{shoeb2009chbmit,
  title={Application of Machine Learning to Epileptic Seizure Detection},
  author={Shoeb, A.},
  booktitle={Proceedings of the 26th International Conference on Machine Learning},
  year={2009}
}

@misc{goldberger2000physionet,
  title={PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals},
  author={Goldberger, A. L. and Amaral, L. A. N. and Glass, L. and Hausdorff, J. M. and Ivanov, P. Ch. and Mark, R. G. and Mietus, J. E. and Moody, G. B. and Peng, C.-K. and Stanley, H. E.},
  journal={Circulation},
  volume={101},
  number={23},
  pages={e215--e220},
  year={2000}
}
```

## Contact

For questions about:
- **Dataset**: See PhysioNet CHB-MIT page
- **Preprocessing**: See CHBMIT_DATASET_INFO.md
- **Implementation**: Open an issue on GitHub
