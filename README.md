# DIVER 1.0 - Downstream Tasks Implementation

This repository contains implementations of DIVER (Deep learning for Interictal EEG Variability Evaluation and Recognition) model applied to various downstream tasks for EEG analysis.

## Overview

This repository demonstrates the application of DIVER pretrained models on three different downstream tasks:
- **CBraMod**: Clinical Brain Monitoring tasks (Sleep Staging, Seizure Detection)
- **LEAD**: Learning EEG Analysis for neurodegenerative Diseases

## Original Projects

### DIVER
- **Original Repository**: https://github.com/yourusername/DIVER
- **Paper**: [Link to DIVER paper]
- **Description**: DIVER is a self-supervised learning framework for EEG representation learning

### CBraMod
- **Original Repository**: https://github.com/yourusername/CBraMod
- **Paper**: [Link to CBraMod paper]
- **Description**: Clinical Brain Monitoring framework for sleep staging and seizure detection

### LEAD
- **Original Repository**: https://github.com/yourusername/LEAD
- **Paper**: Park, J. E., et al. (2024). LEAD: Learning EEG Analysis for neurodegenerative Diseases
- **Description**: Framework for diagnosing neurodegenerative diseases using EEG

## Repository Structure

```
DIVER1.0_bh/
├── DIVER/                      # DIVER pretrained model information
├── CBraMod/                    # Clinical Brain Monitoring tasks
│   ├── ISRUC_Sleep/           # Sleep staging on ISRUC dataset
│   └── CHBMIT_Seizure/        # Seizure detection on CHB-MIT dataset
└── LEAD/                       # Neurodegenerative disease diagnosis
    └── ADFTD/                 # AD/FTD classification
```

## Datasets

### ISRUC-Sleep (Sleep Staging)
- **Task**: 5-class sleep stage classification (W, N1, N2, N3, REM)
- **Subjects**: 100 patients
- **Data**: Polysomnography (PSG) with 6 EEG channels
- **Sampling Rate**: 200 Hz (resampled to 500 Hz)
- **Reference**: Khalighi et al. (2016), Computer Methods and Programs in Biomedicine

### CHB-MIT (Seizure Detection)
- **Task**: Binary classification (seizure vs. non-seizure)
- **Subjects**: 21 pediatric epilepsy patients
- **Data**: Scalp EEG with 16 bipolar channels
- **Sampling Rate**: 256 Hz (resampled to 500 Hz)
- **Reference**: Shoeb (2009), ICML

### ADFTD (Alzheimer's & Frontotemporal Dementia)
- **Task**: 3-class classification (CN vs. AD vs. FTD)
- **Subjects**: 88 patients (30 CN, 35 AD, 23 FTD)
- **Data**: Resting-state EEG with 19 channels
- **Sampling Rate**: 256 Hz (resampled to 500 Hz)
- **Reference**: Park et al. (2024), Journal of Neural Engineering

## Preprocessing

All datasets are preprocessed with the following pipeline:
1. **Channel extraction**: Extract specified EEG channels
2. **Artifact removal**: Remove noisy segments (for ADFTD and CHB-MIT)
3. **Segmentation**: Divide continuous signals into fixed-length segments
4. **Resampling**: Resample to 500 Hz for DIVER compatibility
5. **Reshape**: Convert to (channels, time_segments, samples_per_segment) format
6. **LMDB storage**: Store preprocessed data in LMDB format for efficient loading

## Data Availability

This repository contains only:
- Preprocessing scripts
- Dataset structure documentation
- README files with dataset information

**Note**: The actual EEG data is NOT included in this repository. Users must:
1. Obtain original datasets from their respective sources
2. Run preprocessing scripts to generate LMDB files
3. Follow data usage agreements and citations

## Installation

```bash
git clone https://github.com/bohee-connectome/DIVER1.0_bh.git
cd DIVER1.0_bh
```

## Usage

Refer to individual README files in each subdirectory:
- `CBraMod/ISRUC_Sleep/README.md`
- `CBraMod/CHBMIT_Seizure/README.md`
- `LEAD/ADFTD/README.md`

## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@article{diver2024,
  title={DIVER: Deep learning for Interictal EEG Variability Evaluation and Recognition},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}

@article{cbramod2024,
  title={CBraMod: Clinical Brain Monitoring with Deep Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}

@article{lead2024,
  title={LEAD: Learning EEG Analysis for neurodegenerative Diseases},
  author={Park, J. E., et al.},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

## License

This code is released under MIT License. However, please note:
- Each dataset has its own license and usage terms
- Original DIVER, CBraMod, and LEAD projects may have different licenses
- Refer to original repositories for licensing information

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: [your-email]

## Acknowledgments

- DIVER project team
- CBraMod project team
- LEAD project team
- Dataset contributors and maintainers
