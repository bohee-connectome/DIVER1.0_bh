# LEAD - Learning EEG Analysis for neurodegenerative Diseases

This directory contains implementations of DIVER model applied to neurodegenerative disease diagnosis tasks.

## Original Project

- **Original Repository**: https://github.com/yourusername/LEAD
- **Paper**: Park, J. E., et al. (2024). LEAD: Learning EEG Analysis for neurodegenerative Diseases. Journal of Neural Engineering
- **Description**: Framework for diagnosing Alzheimer's Disease (AD) and Frontotemporal Dementia (FTD) using resting-state EEG

## Tasks

### ADFTD: Alzheimer's Disease & Frontotemporal Dementia Classification
- **Location**: `ADFTD/`
- **Task**: 3-class classification
- **Classes**:
  - CN (Cognitively Normal) - Label 0
  - AD (Alzheimer's Disease) - Label 1
  - FTD (Frontotemporal Dementia) - Label 2
- **Dataset**: ADFTD EEG Dataset
- **Details**: See `ADFTD/README.md`

## Repository Structure

```
LEAD/
├── README.md                   # This file
└── ADFTD/                     # AD/FTD classification task
    ├── README.md              # Task-specific documentation
    ├── ADFTD_DATASET_INFO.md  # Detailed dataset information
    ├── scripts/               # Preprocessing scripts
    │   ├── preprocessing_generalized_ADFTD.py
    │   ├── preprocessing_generalized_datasetsetting_ADFTD.py
    │   ├── clip_extraction_utils.py
    │   ├── check_lmdb_adftd_v1.py
    │   ├── check_lmdb_adftd_v2.py
    │   ├── run_preprocessing_v1.sh
    │   ├── run_preprocessing_v2.sh
    │   └── standard_1005.elc
    ├── logs/                  # Preprocessing logs
    │   ├── v1_output_65051.log
    │   ├── v1_error_65051.log
    │   ├── v2_output_65052.log
    │   ├── v2_error_65052.log
    │   ├── validation_v1_report.txt
    │   └── validation_v2_report.txt
    └── data/                  # Data directory (not in repo)
        ├── processed_v1/      # Version 1 preprocessed data
        │   └── 1.0_ADFTD/
        │       ├── train_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       ├── val_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       ├── test_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       ├── merged_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       └── ADFTD_1.0_complete.txt
        ├── processed_v1.tar.gz
        ├── processed_v2/      # Version 2 preprocessed data
        │   └── 1.0_ADFTD/
        │       ├── train_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       ├── val_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       ├── test_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       ├── merged_resample-500_highpass-0.5_lowpass-45.0.lmdb/
        │       └── ADFTD_1.0_complete.txt
        └── processed_v2.tar.gz
```

## Dataset Overview

### ADFTD Dataset
- **Total Subjects**: 88 patients
- **Class Distribution**:
  - CN (Cognitively Normal): 30 subjects (~34%)
  - AD (Alzheimer's Disease): 35 subjects (~40%)
  - FTD (Frontotemporal Dementia): 23 subjects (~26%)
- **Recording**: Eyes-closed resting state (5-30 minutes)
- **Channels**: 19 standard 10-20 system electrodes
- **Original Sampling Rate**: 256 Hz
- **Data Format**: EDF files

## Preprocessing Pipeline

1. **Load Raw EDF Files**
   - Load resting-state EEG data
   - Extract 19 channels (FP1, FP2, F3, F4, F7, F8, FZ, C3, C4, CZ, T3, T4, T5, T6, P3, P4, PZ, O1, O2)

2. **Clipping (Artifact Removal)**
   - Remove amplitude artifacts (|signal| > 100 μV)
   - Remove gradient artifacts (rapid changes > 50 μV)
   - Remove flatline segments (std < 5 μV)
   - Keep only clean continuous segments

3. **Segmentation**
   - Segment length: 10 seconds
   - Step size: 10 seconds (non-overlapping)
   - Samples per segment: 2560 (10s × 256 Hz)

4. **Resampling**
   - Resample from 256 Hz to 500 Hz
   - New samples per segment: 5000 (10s × 500 Hz)
   - Method: `scipy.signal.resample`

5. **Reshaping**
   - Original: (19 channels, 2560 samples)
   - After resampling: (19, 5000)
   - Final shape: (19, 10, 500) - 19 channels × 10 time segments × 500 samples/second

6. **Data Split**
   - Train: 70% (stratified by class)
   - Validation: 15% (stratified by class)
   - Test: 15% (stratified by class)
   - Split at subject level to prevent data leakage

7. **LMDB Storage**
   - Store as LMDB for efficient loading
   - Each entry: `{"signal": array, "label": int, "elc_info": dict, "metadata": dict}`

## Preprocessing Versions

### Version 1 (v1)
- Initial preprocessing implementation
- Basic artifact removal and segmentation
- See `logs/validation_v1_report.txt` for validation results

### Version 2 (v2)
- Improved artifact detection
- Enhanced quality control
- See `logs/validation_v2_report.txt` for validation results

## Class Imbalance Handling

The dataset has mild class imbalance:
- CN: ~34%, AD: ~40%, FTD: ~26%

Solutions applied:
1. **Stratified split**: Maintain class ratios in train/val/test
2. **Balanced sampling**: During training, sample equally from each class
3. **Class weights**: Apply weights inversely proportional to class frequencies

## ELC (Electrode Location) File

Uses `standard_1005.elc` to store 3D electrode positions:
- 19 monopolar electrodes
- Standard 10-20 system coordinates
- Used by DIVER for spatial encoding

## Usage

### Preprocessing
```bash
cd ADFTD/scripts

# Run version 1 preprocessing
bash run_preprocessing_v1.sh

# Run version 2 preprocessing
bash run_preprocessing_v2.sh
```

### Validation
```bash
# Check LMDB v1
python check_lmdb_adftd_v1.py

# Check LMDB v2
python check_lmdb_adftd_v2.py
```

## Data Availability

**Note**: The actual ADFTD dataset is NOT included in this repository due to:
- Patient privacy concerns
- Data usage agreements
- Large file size (~300 MB for LMDB)

To use this code:
1. Obtain ADFTD dataset from the LEAD project team
2. Place raw EDF files in appropriate directory
3. Run preprocessing scripts
4. LMDB files will be generated in `data/processed_v*/`

## Citation

If you use LEAD or ADFTD dataset in your research, please cite:

```bibtex
@article{lead2024,
  title={LEAD: Learning EEG Analysis for neurodegenerative Diseases},
  author={Park, J. E., et al.},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

## License

- **Code**: Refer to LEAD original repository for license
- **Data**: Research use only, redistribution not allowed
- **Citation required**: Must cite when publishing results

## Contact

For questions about:
- **LEAD framework**: See original LEAD repository
- **ADFTD dataset access**: Contact LEAD project team
- **Preprocessing**: See `ADFTD/README.md`

## Acknowledgments

- LEAD project team
- ADFTD dataset contributors
- Clinical collaborators and data providers
