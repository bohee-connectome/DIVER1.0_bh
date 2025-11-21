# CBraMod - Clinical Brain Monitoring

DIVER preprocessing pipelines for clinical brain monitoring tasks: sleep stage classification and seizure detection.

## Original Project

- **Repository**: https://github.com/yourusername/CBraMod
- **Paper**: [Link to CBraMod paper]

## Tasks

### 1. ISRUC-Sleep: Sleep Stage Classification
**5-class classification** (W, N1, N2, N3, REM) on PSG data

- 📂 **[ISRUC_Sleep/](ISRUC_Sleep/)** - Task directory
- 📄 **[ISRUC_Sleep/README.md](ISRUC_Sleep/README.md)** - Quick start guide
- 📄 **[ISRUC_Sleep/ISRUC_DATASET_INFO.md](ISRUC_Sleep/ISRUC_DATASET_INFO.md)** - Complete documentation

**Quick stats**: 100 subjects, 6 EEG channels, 30s epochs, 200→500 Hz

---

### 2. CHB-MIT: Seizure Detection
**Binary classification** (non-seizure vs. seizure) on pediatric epilepsy data

- 📂 **[CHBMIT_Seizure/](CHBMIT_Seizure/)** - Task directory
- 📄 **[CHBMIT_Seizure/README.md](CHBMIT_Seizure/README.md)** - Quick start guide
- 📄 **[CHBMIT_Seizure/CHBMIT_DATASET_INFO.md](CHBMIT_Seizure/CHBMIT_DATASET_INFO.md)** - Complete documentation

**Quick stats**: 21 patients, 16 bipolar channels, 10s segments, 256→500 Hz

---

## Common Pipeline

Both tasks follow the same preprocessing approach:
1. Raw data loading (`.rec` or `.edf`)
2. Channel extraction
3. Segmentation (30s for sleep, 10s for seizure)
4. Resampling to 500 Hz (DIVER standard)
5. Reshape to (channels, time_segments, samples_per_segment)
6. LMDB storage

## Citation

```bibtex
@article{cbramod2024,
  title={CBraMod: Clinical Brain Monitoring with Deep Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```
