# CBraMod - Clinical Brain Monitoring

DIVER preprocessing pipelines for clinical brain monitoring tasks: sleep stage classification and seizure detection.

## Tasks

### 1. ISRUC-Sleep: Sleep Stage Classification
**5-class classification** (W, N1, N2, N3, REM) on PSG data

- 📂 **[ISRUC_Sleep/](ISRUC_Sleep/)** - Complete task documentation
- 📄 **[ISRUC_Sleep/README.md](ISRUC_Sleep/README.md)** - Quick start guide
- 📄 **[ISRUC_Sleep/ISRUC_DATASET_INFO.md](ISRUC_Sleep/ISRUC_DATASET_INFO.md)** - Dataset documentation

**Quick stats**: 100 subjects, 6 EEG channels, 30s epochs, 200→500 Hz

---

### 2. CHB-MIT: Seizure Detection
**Binary classification** (non-seizure vs. seizure) on pediatric epilepsy data

- 📂 **[CHBMIT_Seizure/](CHBMIT_Seizure/)** - Complete task documentation
- 📄 **[CHBMIT_Seizure/README.md](CHBMIT_Seizure/README.md)** - Quick start guide
- 📄 **[CHBMIT_Seizure/CHBMIT_DATASET_INFO.md](CHBMIT_Seizure/CHBMIT_DATASET_INFO.md)** - Dataset documentation

**Quick stats**: 21 patients, 16 bipolar channels, 10s segments, 256→500 Hz, v2 format (ISRUC-compatible)

---

## Documentation

For detailed preprocessing pipeline, data format comparison, and citations, see the **[main README](../README.md)**.

## Original Project

- **Paper**: Lee, B. and Park, J. E. et al. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. *ICLR 2025*
