# DIVER 1.0 - Downstream Tasks

Preprocessing pipelines for applying DIVER (Deep learning for Interictal EEG Variability Evaluation and Recognition) to downstream EEG analysis tasks.

## Overview

This repository contains preprocessing scripts and documentation for three downstream tasks:
- **CBraMod**: Clinical brain monitoring (sleep staging, seizure detection)
- **LEAD**: Neurodegenerative disease diagnosis (AD/FTD classification)

## Projects

### DIVER
- **Repository**: https://github.com/yourusername/DIVER
- **Paper**: [Link to DIVER paper]
- **Description**: Self-supervised learning framework for EEG representation learning

### CBraMod - Clinical Brain Monitoring
- 📂 **[CBraMod/](CBraMod/)** - Clinical monitoring tasks
  - 🛌 **[ISRUC-Sleep](CBraMod/ISRUC_Sleep/)** - Sleep stage classification (5-class)
  - 🧠 **[CHB-MIT](CBraMod/CHBMIT_Seizure/)** - Seizure detection (binary)

### LEAD - Neurodegenerative Diseases
- 📂 **[LEAD/](LEAD/)** - Disease diagnosis tasks
  - 🏥 **[ADFTD](LEAD/ADFTD/)** - AD/FTD classification (3-class)

## Repository Structure

```
DIVER1.0_bh/
├── CBraMod/
│   ├── ISRUC_Sleep/        # Sleep staging: 100 subjects, 6 channels
│   └── CHBMIT_Seizure/     # Seizure detection: 21 patients, 16 channels
└── LEAD/
    └── ADFTD/              # AD/FTD diagnosis: 88 subjects, 19 channels
```

Each task directory contains:
- `README.md` - Quick start guide and complete documentation
- `*_DATASET_INFO.md` - Comprehensive dataset documentation
- `scripts/` - Preprocessing and validation scripts
- `logs/` - Processing logs

---

## Preprocessing Pipeline

All tasks follow a unified preprocessing pipeline designed for DIVER:

### Common Steps

```
Raw EEG Data (EDF/REC format)
    ↓
[1. Load & Extract Channels]
    ├── ISRUC: 6 channels (F3, C3, O1, F4, C4, O2)
    ├── CHBMIT: 16 bipolar channels (Double Banana montage)
    └── ADFTD: 19 channels (standard 10-20 system)
    ↓
[2. Preprocessing & Artifact Removal]
    ├── ISRUC: 0.3-35 Hz bandpass, 50 Hz notch, average reference
    ├── CHBMIT: Channel extraction from raw EDF
    └── ADFTD: Clipping detection (amplitude, gradient, flatline) ✅
    ↓
[3. Segmentation]
    ├── ISRUC: 30-second epochs (aligned with annotations)
    ├── CHBMIT: 10-second segments (with seizure oversampling)
    └── ADFTD: 10-second segments (non-overlapping)
    ↓
[4. Label Assignment]
    ├── ISRUC: 5-class (W, N1, N2, N3, REM)
    ├── CHBMIT: Binary (non-seizure, seizure)
    └── ADFTD: 3-class (CN, AD, FTD)
    ↓
[5. Resampling to 500 Hz]  ← DIVER standard sampling rate
    ↓
[6. Reshape]
    ├── ISRUC: (6, 6000) → (6, 30, 500)
    ├── CHBMIT: (16, 2560) → (16, 10, 500)
    └── ADFTD: (19, 2560) → (19, 10, 500)
    ↓
[7. Add Metadata]  ← Dataset info, electrode positions, subject info
    ↓
[8. Store in LMDB]  ← Efficient batch loading for training
```

### Task-Specific Details

| Task | Channels | Segment | Original SR | Target SR | Output Shape |
|------|----------|---------|-------------|-----------|--------------|
| **ISRUC-Sleep** | 6 EEG | 30s epochs | 200 Hz | 500 Hz | (6, 30, 500) |
| **CHB-MIT** | 16 bipolar | 10s segments | 256 Hz | 500 Hz | (16, 10, 500) |
| **ADFTD** | 19 EEG | 10s segments | 256 Hz | 500 Hz | (19, 10, 500) |

---

## Data Format Comparison

### Current Standard: v2 (ISRUC-compatible) ✅

All datasets now use the unified format:

```python
{
    "sample": np.array,           # Signal data (channels, time_segments, samples)
    "label": int,                 # Task-specific label
    "data_info": {                # Unified metadata
        # Common fields (ISRUC-compatible)
        "Dataset": str,           # "ISRUC-Sleep", "CHBMIT-Seizure", "ADFTD"
        "modality": "EEG",
        "release": str,           # Dataset version
        "subject_id": str,
        "task": str,              # "sleep-staging", "seizure-detection", etc.
        "resampling_rate": 500,
        "original_sampling_rate": int,
        "segment_index": int,
        "start_time": float,
        "channel_names": list,
        # Task-specific fields
        ...
    }
}
```

### Format Evolution: v1 → v2

**CHBMIT and ADFTD** were updated from v1 to v2 for consistency:

| Field | v1 (Legacy) | v2 (Current) | Status |
|-------|-------------|--------------|--------|
| **Signal data** | `signal` | `sample` ✅ | Standardized |
| **Metadata** | `metadata` | `data_info` ✅ | Standardized |
| **Electrode info** | `elc_info` (separate) | Merged into `data_info` ✅ | Unified |
| **Common fields** | Missing | Added (`Dataset`, `modality`, `task`) ✅ | Complete |

**ISRUC** used the correct format from the start.

**Migration**: Use `migrate_chbmit_v1_to_v2.py` to convert existing v1 LMDB databases.

---

## Dataset Statistics

### ISRUC-Sleep
- **Subjects**: 100 healthy adults
- **Epochs**: ~89,283 (30-second epochs)
- **Split**: 80/10/10 (subjects 1-80 train, 81-90 val, 91-100 test)
- **Storage**: ~2.4 GB LMDB
- **Distribution**: N2 dominant (~40-50%), N1 rare (~5-10%)

### CHB-MIT Seizure
- **Patients**: 21 pediatric epilepsy patients
- **Segments**: 327,834 (10-second segments with oversampling)
- **Split**: chb01-20 train, chb21-22 val, chb23-24 test
- **Storage**: ~99 GB LMDB
- **Distribution**: Severe imbalance (30:1 non-seizure:seizure after oversampling)

### ADFTD
- **Subjects**: 88 patients (30 CN, 35 AD, 23 FTD)
- **Segments**: ~5,700 (10-second segments after artifact removal)
- **Split**: 70/15/15 (stratified by diagnosis)
- **Storage**: ~600 MB LMDB (all splits)
- **Distribution**: Slight imbalance (CN 34%, AD 40%, FTD 26%)

---

## Data Availability

- ✅ **ISRUC-Sleep**: Public (https://sleeptight.isr.uc.pt/)
- ✅ **CHB-MIT**: Public (https://physionet.org/content/chbmit/)
- ❌ **ADFTD**: Not public (contact LEAD team)

**Note**: This repository contains preprocessing scripts only. Raw data and LMDB files are not included.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/bohee-connectome/DIVER1.0_bh.git
cd DIVER1.0_bh

# Navigate to desired task
cd CBraMod/ISRUC_Sleep  # or CHBMIT_Seizure, or LEAD/ADFTD

# Install dependencies
pip install numpy scipy mne lmdb

# Run preprocessing (see task README for specific commands)
cd scripts
python preprocessing_*.py
```

---

## Citation

If you use this code, please cite:

### DIVER
```bibtex
@article{diver2024,
  title={DIVER: Deep learning for Interictal EEG Variability Evaluation and Recognition},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

### CBraMod (ISRUC-Sleep, CHB-MIT)
```bibtex
@inproceedings{cbramod2025,
  title={CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding},
  author={Lee, B. and Park, J. E. and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

### ISRUC-Sleep Dataset
```bibtex
@article{khalighi2016isruc,
  title={ISRUC-Sleep: A comprehensive public dataset for sleep researchers},
  author={Khalighi, S. and Sousa, T. and Santos, J. M. and Nunes, U.},
  journal={Computer Methods and Programs in Biomedicine},
  volume={124},
  pages={180--192},
  year={2016},
  publisher={Elsevier}
}
```

### CHB-MIT Dataset
```bibtex
@inproceedings{shoeb2009chbmit,
  title={Application of Machine Learning to Epileptic Seizure Detection},
  author={Shoeb, A.},
  booktitle={Proceedings of the 26th International Conference on Machine Learning},
  year={2009}
}
```

### LEAD (ADFTD)
```bibtex
@article{lead2024,
  title={LEAD: Learning EEG Analysis for neurodegenerative Diseases},
  author={Park, J. E. and Lee, B. and others},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

---

## Version Information

- **Data Format**: v2 (ISRUC-compatible, unified format)
- **Last Updated**: 2025-11-27
- **Migration Available**: v1 → v2 (for CHBMIT and ADFTD)

---

## License

MIT License - See original project repositories for dataset-specific licenses.

## Contact

For questions or issues, please open an issue on GitHub.
