# LEAD - Learning EEG Analysis for Neurodegenerative Diseases

DIVER preprocessing pipeline for diagnosing neurodegenerative diseases using resting-state EEG.

## Original Project

- **Repository**: https://github.com/yourusername/LEAD
- **Paper**: Park, J. E., et al. (2024). LEAD: Learning EEG Analysis for neurodegenerative Diseases. *Journal of Neural Engineering*

## Task

### ADFTD: Alzheimer's & Frontotemporal Dementia Classification
**3-class classification** (CN vs. AD vs. FTD) on resting-state EEG

- 📂 **[ADFTD/](ADFTD/)** - Task directory
- 📄 **[ADFTD/README.md](ADFTD/README.md)** - Quick start guide
- 📄 **[ADFTD/ADFTD_DATASET_INFO.md](ADFTD/ADFTD_DATASET_INFO.md)** - Complete documentation

**Quick stats**: 88 subjects (30 CN, 35 AD, 23 FTD), 19 channels, 10s segments, 256→500 Hz

---

## Preprocessing Pipeline

1. Load raw EDF files (eyes-closed resting state)
2. Extract 19 standard 10-20 electrodes
3. **Artifact removal** (amplitude, gradient, flatline detection)
4. Segmentation (10-second segments)
5. Resampling to 500 Hz (DIVER standard)
6. Reshape to (19, 10, 500)
7. Stratified train/val/test split (70/15/15)
8. LMDB storage

## Data Availability

**Note**: ADFTD dataset is not publicly available. Contact LEAD project team for access.

## Citation

```bibtex
@article{lead2024,
  title={LEAD: Learning EEG Analysis for neurodegenerative Diseases},
  author={Park, J. E., et al.},
  journal={Journal of Neural Engineering},
  year={2024}
}
```
