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
- `README.md` - Quick start guide
- `*_DATASET_INFO.md` - Comprehensive dataset documentation
- `STRUCTURE.md` - Directory structure reference
- `scripts/` - Preprocessing and validation scripts
- `logs/` - Processing logs

## Common Preprocessing

All tasks follow a unified preprocessing pipeline:

1. **Load raw data** (EDF/REC format)
2. **Extract channels** (task-specific channel configurations)
3. **Segment** (10-30 second segments)
4. **Resample** to 500 Hz (DIVER standard)
5. **Reshape** to (channels, time_segments, samples_per_segment)
6. **Store in LMDB** for efficient batch loading

## Data Availability

- ✅ **ISRUC-Sleep**: Public (https://sleeptight.isr.uc.pt/)
- ✅ **CHB-MIT**: Public (https://physionet.org/content/chbmit/)
- ❌ **ADFTD**: Not public (contact LEAD team)

**Note**: This repository contains preprocessing scripts only. Raw data and LMDB files are not included.

## Quick Start

```bash
# Clone repository
git clone https://github.com/bohee-connectome/DIVER1.0_bh.git
cd DIVER1.0_bh

# Navigate to desired task
cd CBraMod/ISRUC_Sleep  # or CHBMIT_Seizure, or LEAD/ADFTD

# Install dependencies
pip install numpy scipy mne lmdb

# Run preprocessing
cd scripts
python preprocessing_*.py  # See task README for specific commands
```

## Citation

If you use this code, please cite:

```bibtex
@article{diver2024,
  title={DIVER: Deep learning for Interictal EEG Variability Evaluation and Recognition},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}

% For CBraMod tasks
@article{cbramod2024,
  title={CBraMod: Clinical Brain Monitoring with Deep Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}

% For LEAD tasks
@article{lead2024,
  title={LEAD: Learning EEG Analysis for neurodegenerative Diseases},
  author={Park, J. E., et al.},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

## License

MIT License - See original project repositories for dataset-specific licenses.

## Contact

For questions or issues, please open an issue on GitHub.
