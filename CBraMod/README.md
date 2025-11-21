# CBraMod - Clinical Brain Monitoring with DIVER

This directory contains implementations of DIVER model applied to Clinical Brain Monitoring (CBraMod) tasks.

## Original Project

- **Original Repository**: https://github.com/yourusername/CBraMod
- **Paper**: [Link to CBraMod paper]
- **Description**: Clinical Brain Monitoring framework for automated analysis of clinical EEG data

## Tasks

### 1. ISRUC-Sleep: Sleep Stage Classification
- **Location**: `ISRUC_Sleep/`
- **Task**: 5-class sleep stage classification
- **Classes**: Wake (W), N1, N2, N3, REM
- **Dataset**: ISRUC-Sleep Subgroup 1
- **Details**: See `ISRUC_Sleep/README.md`

### 2. CHB-MIT: Seizure Detection
- **Location**: `CHBMIT_Seizure/`
- **Task**: Binary seizure detection
- **Classes**: Non-seizure (0), Seizure (1)
- **Dataset**: CHB-MIT Scalp EEG Database
- **Details**: See `CHBMIT_Seizure/README.md`

## Repository Structure

```
CBraMod/
├── README.md                   # This file
├── ISRUC_Sleep/               # Sleep staging task
│   ├── README.md              # Task-specific documentation
│   ├── ISRUC_DATASET_INFO.md  # Detailed dataset information
│   ├── scripts/               # Preprocessing scripts
│   ├── logs/                  # Preprocessing logs
│   └── lmdb_output/           # LMDB data storage (not in repo)
└── CHBMIT_Seizure/            # Seizure detection task
    ├── README.md              # Task-specific documentation
    ├── CHBMIT_DATASET_INFO.md # Detailed dataset information
    ├── scripts/               # Preprocessing scripts
    ├── logs/                  # Preprocessing logs
    └── lmdb_output/           # LMDB data storage (not in repo)
```

## Common Preprocessing Pipeline

Both tasks follow a similar preprocessing pipeline:

1. **Load Raw Data**
   - ISRUC: `.rec` files (PSG data)
   - CHB-MIT: `.edf` files (EEG data)

2. **Channel Extraction**
   - ISRUC: 6 EEG channels (F3-A2, C3-A2, F4-A1, C4-A1, O1-A2, O2-A1)
   - CHB-MIT: 16 bipolar channels (Double Banana montage)

3. **Segmentation**
   - ISRUC: 30-second epochs (sleep study standard)
   - CHB-MIT: 10-second segments (clinical EEG standard)

4. **Label Assignment**
   - ISRUC: From `.txt` annotation files (5 sleep stages)
   - CHB-MIT: From `-summary.txt` files (seizure timestamps)

5. **Resampling**
   - Both: Resample to 500 Hz (DIVER standard)

6. **Reshaping**
   - ISRUC: (6, 6000) → (6, 15000) → (6, 30, 500)
   - CHB-MIT: (16, 2560) → (16, 5000) → (16, 10, 500)

7. **LMDB Storage**
   - Store in LMDB format for efficient batch loading

## Data Split

### ISRUC-Sleep
- **Train**: Subjects 1-84 (84 subjects)
- **Validation**: Subjects 85-90 (6 subjects)
- **Test**: Subjects 91-100 (10 subjects)

### CHB-MIT
- **Train**: chb01-20 (17 patients, excluding 12, 13, 17)
- **Validation**: chb21-22 (2 patients)
- **Test**: chb23-24 (2 patients)

## Class Imbalance

### ISRUC-Sleep
- **Problem**: N2 dominates (~50%), W and N1 are rare (~5% each)
- **Solution**: Class weights, weighted loss function

### CHB-MIT
- **Problem**: Non-seizure >> Seizure (99:1 ratio)
- **Solution**: Oversampling seizure segments (5-second step vs. 10-second step)

## ELC (Electrode Location) Files

Both tasks use `standard_1005.elc` file to store electrode positions:
- **ISRUC**: 6 reference electrodes (10-20 system)
- **CHB-MIT**: 16 bipolar pairs → 16 unique electrodes (10-20 system)

This spatial information is used by DIVER for position-aware learning.

## Usage

### ISRUC-Sleep
```bash
cd ISRUC_Sleep/scripts
python preprocessing_isruc-sleep.py
```

### CHB-MIT
```bash
cd CHBMIT_Seizure/scripts
bash run_preprocessing_chbmit.sh
```

## Citation

If you use CBraMod tasks in your research, please cite:

```bibtex
@article{cbramod2024,
  title={CBraMod: Clinical Brain Monitoring with Deep Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}

% ISRUC-Sleep dataset
@article{khalighi2016isruc,
  title={ISRUC-Sleep: A comprehensive public dataset for sleep researchers},
  author={Khalighi, S. and Sousa, T. and Santos, J. M. and Nunes, U.},
  journal={Computer Methods and Programs in Biomedicine},
  volume={124},
  pages={180--192},
  year={2016}
}

% CHB-MIT dataset
@inproceedings{shoeb2009chbmit,
  title={Application of Machine Learning to Epileptic Seizure Detection},
  author={Shoeb, A.},
  booktitle={Proceedings of the 26th International Conference on Machine Learning},
  year={2009}
}
```

## License

Please refer to:
- CBraMod original repository for code license
- ISRUC-Sleep dataset terms (open access, research use)
- CHB-MIT dataset terms (PhysioNet, open access)

## Contact

For questions about:
- **CBraMod framework**: See original repository
- **ISRUC preprocessing**: See `ISRUC_Sleep/README.md`
- **CHB-MIT preprocessing**: See `CHBMIT_Seizure/README.md`
