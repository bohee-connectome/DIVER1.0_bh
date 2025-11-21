# ADFTD: Alzheimer's & Frontotemporal Dementia Classification

3-class classification (CN vs. AD vs. FTD) on ADFTD dataset using DIVER preprocessing pipeline.

## Quick Info

| Item | Value |
|------|-------|
| **Dataset** | ADFTD EEG Dataset |
| **Subjects** | 88 patients (30 CN, 35 AD, 23 FTD) |
| **Task** | 3-class classification |
| **Channels** | 19 standard 10-20 electrodes |
| **Sampling Rate** | 256 Hz → 500 Hz (resampled) |
| **Segment Length** | 10 seconds |
| **Output Shape** | (19, 10, 500) |
| **Recording** | Eyes-closed resting state |

## Documentation

- 📄 **[ADFTD_DATASET_INFO.md](ADFTD_DATASET_INFO.md)** - Complete dataset documentation
- 📁 **[STRUCTURE.md](STRUCTURE.md)** - Directory structure and file descriptions

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy mne lmdb
```

### 2. Run Preprocessing (Version 2 recommended)
```bash
cd scripts
bash run_preprocessing_v2.sh
```

### 3. Validate Output
```bash
python check_lmdb_adftd_v2.py
python check_v2_shapes.py
```

## Data Split

- **Train**: 70% (~62 subjects, stratified)
- **Validation**: 15% (~13 subjects, stratified)
- **Test**: 15% (~13 subjects, stratified)

## Output Format

```python
{
    "signal": np.array (19, 10, 500),  # 19 channels × 10 segments × 500 samples/sec
    "label": int (0, 1, or 2),         # CN=0, AD=1, FTD=2
    "elc_info": dict,
    "metadata": dict
}
```

## Preprocessing Versions

- **v1**: Initial implementation
- **v2**: Improved artifact detection (recommended)

## Citation

```bibtex
@article{lead2024,
  title={LEAD: Learning EEG Analysis for neurodegenerative Diseases},
  author={Park, J. E., et al.},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

## Data Availability

**Note**: ADFTD dataset is not publicly available. Contact LEAD project team for access.

## Links

- **LEAD Project**: [Contact for dataset access]
