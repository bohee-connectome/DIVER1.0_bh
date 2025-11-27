# ISRUC-Sleep: Sleep Stage Classification

Sleep stage classification (5-class: W, N1, N2, N3, REM) on ISRUC-Sleep dataset using DIVER preprocessing pipeline.

## Quick Info

| Item | Value |
|------|-------|
| **Dataset** | ISRUC-Sleep Subgroup 1 |
| **Subjects** | 100 healthy adults |
| **Task** | 5-class sleep stage classification |
| **Classes** | W (0), N1 (1), N2 (2), N3 (3), REM (4) |
| **Channels** | 6 EEG channels (10-20 system) |
| **Sampling Rate** | 200 Hz → 500 Hz (resampled) |
| **Epoch Length** | 30 seconds |
| **Output Shape** | (6, 30, 500) |

## Documentation

- 📄 **[ISRUC_DATASET_INFO.md](ISRUC_DATASET_INFO.md)** - Complete dataset documentation

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy mne lmdb
```

### 2. Run Preprocessing
```bash
cd scripts
python preprocessing_isruc-sleep.py
```

### 3. Validate Output
```bash
python check_lmdb_isruc.py
```

---

## 📦 Data Format

```python
{
    "sample": np.array,           # (6, 30, 500)
    "label": int,                 # 0=W, 1=N1, 2=N2, 3=N3, 4=REM
    "data_info": {                # Unified metadata format
        "Dataset": "ISRUC-Sleep",
        "modality": "EEG",
        "release": None,
        "subject_id": str,        # e.g., "Subgroup1_S001"
        "subgroup": str,          # "Subgroup1"
        "task": "sleep-staging",
        "resampling_rate": 500,
        "original_sampling_rate": 200,
        "segment_index": int,     # Epoch index
        "start_time": float,      # Seconds from recording start
        "channel_names": list,    # ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        "xyz_id": np.ndarray      # (6, 3) electrode 3D coordinates
    }
}
```

**Note:** ISRUC uses the correct unified format from the start. CHBMIT and LEAD were updated to match this standard.

---

## 📁 Directory Structure

```
ISRUC_Sleep/
├── README.md                       # This file - unified documentation
├── ISRUC_DATASET_INFO.md          # Detailed dataset info
│
├── scripts/                        # Preprocessing and validation
│   ├── preprocessing_isruc-sleep.py     # Main preprocessing ✅
│   ├── check_lmdb_isruc.py             # LMDB validator ✅
│   ├── test_isruc_format.py            # Format testing
│   └── standard_1005.elc               # Electrode locations
│
├── logs/                          # Processing logs
│   └── preprocessing_v3_FULL.log       # Full preprocessing log
│
└── lmdb_output/                   # LMDB database (not in repo)
    └── ISRUC_Sleep/
        ├── data.mdb               # ~2.4 GB
        └── lock.mdb
```

### Script Descriptions

| Script | Description |
|--------|-------------|
| **preprocessing_isruc-sleep.py** | Main preprocessing (generates LMDB) |
| **check_lmdb_isruc.py** | Validator (verifies data integrity, label distribution) |
| test_isruc_format.py | Format testing and debugging |
| standard_1005.elc | 10-20 electrode 3D coordinates |

---

## 🔄 Preprocessing Pipeline

```
Raw ISRUC Data (.rec files)
    ↓
[Load & Extract Channels]  ← 6 EEG channels (F3, C3, O1, F4, C4, O2)
    ↓
[Channel Matching]  ← Flexible matching (F3-A2, F3-A1, etc.)
    ↓
[Preprocessing]
├── Average reference
├── 0.3-35 Hz bandpass filter
└── 50 Hz notch filter
    ↓
[Segment]  ← 30-second epochs (aligned with annotations)
    ↓
[Assign Labels]  ← From expert annotations (_1.txt)
├── 0 → 0 (Wake)
├── 1 → 1 (N1)
├── 2 → 2 (N2)
├── 3 → 3 (N3)
└── 5 → 4 (REM)  ← Remap 5 to 4
    ↓
[Resample]  ← 200 Hz → 500 Hz
    ↓
[Reshape]  ← (6, 6000) → (6, 30, 500)
    ↓
[Add Metadata]  ← Dataset info, electrode positions
    ↓
[Store in LMDB]
    ↓
ISRUC_Sleep/
```

---

## 📊 Dataset Statistics

### Data Split (CBraMod Paper Setting)
Following ICLR 2025 CBraMod paper:
- **Train**: Subjects 1-80 (80 subjects, 80%)
- **Validation**: Subjects 81-90 (10 subjects, 10%)
- **Test**: Subjects 91-100 (10 subjects, 10%)

### Epochs per Split
```
Total: ~89,283 epochs
├── Train:  ~71,000 epochs (80%)
├── Val:     ~8,900 epochs (10%)
└── Test:    ~8,900 epochs (10%)
```

### Sleep Stage Distribution
Typical distribution across all subjects:
```
Wake (0):  ~15-20%
N1 (1):    ~5-10%
N2 (2):    ~40-50%  ← Most common
N3 (3):    ~15-20%
REM (4):   ~15-20%
```

### Storage Size
- **LMDB database**: ~2.4 GB
- **Per epoch**: ~50 KB (signal + metadata)

---

## 🧠 Channel Configuration

6 EEG channels from **10-20 system**:

```
Left Hemisphere:
├── F3  (Frontal left)
├── C3  (Central left)
└── O1  (Occipital left)

Right Hemisphere:
├── F4  (Frontal right)
├── C4  (Central right)
└── O2  (Occipital right)
```

**Reference:** Average reference (after extracting from original A1/A2 references)

**Flexible Channel Matching:**
The preprocessing automatically matches various naming conventions:
- `F3-A2`, `F3-A1`, `F3`, `EEG F3` → All mapped to `F3`

---

## 💡 Usage Notes

1. **Label Mapping**: Original label 5 (REM) is correctly remapped to 4
2. **Expert Annotations**: Uses expert 1 annotations (_1.txt)
3. **Missing Channels**: Handles flexible channel naming and missing data
4. **Preprocessing**: Follows CBraMod paper settings (0.3-35 Hz, 50 Hz notch)
5. **Data Split**: Subject-level split to avoid data leakage

---

## 📖 Citation

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

```bibtex
@inproceedings{cbramod2025,
  title={CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding},
  author={Lee, B. and Park, J. E. and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## 🔗 Links

- **Dataset**: https://sleeptight.isr.uc.pt/
- **PhysioNet**: https://physionet.org/content/isruc-sleep/1.0.0/
- **CBraMod Paper**: [ICLR 2025]

---

## 📌 Version Information

- **Data Format**: Unified format (sample, data_info)
- **Last Updated**: 2025-11-27
- **Preprocessing Version**: v3 FULL
- **Label Remapping**: 5 → 4 (REM) ✅ Verified
