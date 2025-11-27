# CHB-MIT: Seizure Detection

Binary seizure detection on CHB-MIT Scalp EEG Database using DIVER preprocessing pipeline.

## Quick Info

| Item | Value |
|------|-------|
| **Dataset** | CHB-MIT Scalp EEG Database |
| **Subjects** | 21 pediatric epilepsy patients |
| **Task** | Binary seizure detection |
| **Channels** | 16 bipolar channels (Double Banana montage) |
| **Sampling Rate** | 256 Hz → 500 Hz (resampled) |
| **Segment Length** | 10 seconds |
| **Output Shape** | (16, 10, 500) |
| **Data Format** | v2-keymodified (ISRUC-compatible) |

## Documentation

- 📄 **[CHBMIT_DATASET_INFO.md](CHBMIT_DATASET_INFO.md)** - Complete dataset documentation

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy mne lmdb pyedflib
```

### 2. Run Preprocessing (v2 Recommended)
```bash
cd scripts
python preprocessing_chbmit_v2-keymodified.py \
  --data_path /path/to/CHB-MIT/dataset \
  --lmdb_path /path/to/output/CHBMIT_Seizure_v2 \
  --elc_file standard_1005.elc
```

### 3. Migrate Existing v1 Data (Optional)
If you have existing v1 LMDB, convert to v2 format:
```bash
python migrate_chbmit_v1_to_v2.py \
  --input_lmdb /path/to/CHBMIT_Seizure \
  --output_lmdb /path/to/CHBMIT_Seizure_v2
```

### 4. Validate Output
```bash
python check_lmdb_chbmit_v2-compatible.py --lmdb_path /path/to/CHBMIT_Seizure_v2
```

---

## 📦 Data Format

### v2-keymodified (Current, ISRUC-Compatible) ✅

**File:** `preprocessing_chbmit_v2-keymodified.py`

```python
{
    "sample": np.array,           # (16, 10, 500) - Changed from 'signal'
    "label": int,                 # 0=non-seizure, 1=seizure
    "data_info": {                # Changed from 'metadata', unified format
        # ISRUC-compatible fields
        "Dataset": "CHBMIT-Seizure",
        "modality": "EEG",
        "release": "1.0.0",
        "subject_id": str,        # e.g., "chb01"
        "task": "seizure-detection",
        "resampling_rate": 500,
        "original_sampling_rate": 256,
        "segment_index": int,
        "start_time": float,
        "channel_names": list,    # 16 bipolar channel names
        "electrode_pairs": dict,  # Bipolar electrode pairs
        "electrode_positions": dict,  # 3D coordinates

        # CHBMIT-specific fields
        "segment_id": str,        # e.g., "chb01_01_0"
        "split": str,             # "train", "val", "test"
        "is_oversampled": bool    # Seizure oversampling flag
    }
}
```

**Why v2?**
- Unified data dictionary format across ISRUC, CHBMIT, ADFTD
- Electrode information merged into `data_info` (no separate `elc_info`)
- Consistent key names: `sample` (not `signal`), `data_info` (not `metadata`)
- Better dataset interoperability and model training compatibility

### v1 (Legacy, for Reference Only)

**File:** `preprocessing_chbmit.py` (kept for backward compatibility)

<details>
<summary>Click to see v1 format (not recommended for new projects)</summary>

```python
{
    "signal": np.array,           # (16, 10, 500)
    "label": int,
    "elc_info": dict,             # Separate electrode info
    "metadata": {
        "segment_id": str,
        "split": str,
        "is_oversampled": bool,
        "original_index": int,
        "original_sr": 256,
        "target_sr": 500
    }
}
```

**Issues with v1:**
- Inconsistent key names vs ISRUC format
- Electrode info separated from metadata
- Missing dataset-level metadata (Dataset, modality, task)
- Harder to maintain unified data loaders

</details>

---

## 📁 Directory Structure

```
CHBMIT_Seizure/
├── README.md                                    # This file - unified documentation
├── CHBMIT_DATASET_INFO.md                      # Detailed dataset info
│
├── scripts/                                     # Preprocessing and validation
│   ├── preprocessing_chbmit.py                      # v1 (legacy)
│   ├── preprocessing_chbmit_v2-keymodified.py       # v2 (recommended) ✅
│   ├── check_lmdb_chbmit.py                         # v1 validator
│   ├── check_lmdb_chbmit_v2-compatible.py           # v1/v2 validator ✅
│   ├── migrate_chbmit_v1_to_v2.py                   # v1→v2 migration ✅
│   ├── run_preprocessing_chbmit.sh                  # Runner script
│   └── standard_1005.elc                            # Electrode locations
│
├── logs/                                        # Processing logs
│   └── run_20251121_190529.log
│
└── lmdb_output/                                # LMDB databases (not in repo)
    ├── CHBMIT_Seizure/                              # v1 data
    └── CHBMIT_Seizure_v2/                           # v2 data ✅
```

### Script Descriptions

| Script | Version | Description |
|--------|---------|-------------|
| **preprocessing_chbmit_v2-keymodified.py** | v2 ✅ | Main preprocessing (ISRUC-compatible format) |
| **check_lmdb_chbmit_v2-compatible.py** | v2 ✅ | Validator (supports v1 & v2, shows format version) |
| **migrate_chbmit_v1_to_v2.py** | Migration ✅ | Convert existing v1 LMDB to v2 format |
| preprocessing_chbmit.py | v1 | Legacy preprocessing (kept for reference) |
| check_lmdb_chbmit.py | v1 | Legacy validator |

---

## 🔄 Preprocessing Pipeline

```
Raw CHB-MIT Data (.edf + -summary.txt)
    ↓
[Parse Summary Files]  ← Extract seizure timestamps
    ↓
[Load EDF & Extract Channels]  ← 16 bipolar channels
    ↓
[Segment - Regular]  ← 10-second segments, 10-second step
    ↓
[Segment - Oversampled]  ← Seizure ±1s, 5-second step
    ↓
[Merge & Label]  ← 0=non-seizure, 1=seizure
    ↓
[Resample]  ← 256 Hz → 500 Hz
    ↓
[Reshape]  ← (16, 2560) → (16, 10, 500)
    ↓
[Add Metadata]  ← Dataset info, electrode positions
    ↓
[Store in LMDB]  ← v2 format with data_info
    ↓
CHBMIT_Seizure_v2/
```

---

## 📊 Dataset Statistics

### Data Split
- **Train**: chb01-20 (17 patients, excluding chb12, chb13, chb17)
- **Validation**: chb21-22 (2 patients)
- **Test**: chb23-24 (2 patients)

### Segments per Split
```
Total Samples: 327,834
├── Train:  287,341 segments (87.6%)
├── Val:     23,065 segments  (7.0%)
└── Test:    17,428 segments  (5.3%)
```

### Class Distribution

**Before Oversampling:**
- Non-seizure : Seizure ≈ **99:1** (severe imbalance)

**After Oversampling:**
- Non-seizure : Seizure ≈ **30:1** (improved, but still use class weights!)
- Seizure segments: 5-second sliding window within seizure ±1s

### Storage Size
- **LMDB database**: ~99 GB
- **Per segment**: ~300 KB (signal + metadata)

---

## 🧠 Bipolar Channel Configuration

16 channels following **Double Banana montage** (10-20 system):

```
Left Lateral Chain:
├── FP1-F7  (Frontopolar to Frontal left)
├── F7-T7   (Frontal to Temporal left)
├── T7-P7   (Temporal to Parietal left)
└── P7-O1   (Parietal to Occipital left)

Right Lateral Chain:
├── FP2-F8  (Frontopolar to Frontal right)
├── F8-T8   (Frontal to Temporal right)
├── T8-P8   (Temporal to Parietal right)
└── P8-O2   (Parietal to Occipital right)

Left Parasagittal Chain:
├── FP1-F3  (Frontopolar to Frontal left)
├── F3-C3   (Frontal to Central left)
├── C3-P3   (Central to Parietal left)
└── P3-O1   (Parietal to Occipital left)

Right Parasagittal Chain:
├── FP2-F4  (Frontopolar to Frontal right)
├── F4-C4   (Frontal to Central right)
├── C4-P4   (Central to Parietal right)
└── P4-O2   (Parietal to Occipital right)
```

**Individual Electrodes Used:**
- FP1, FP2, F3, F4, F7, F8
- C3, C4, T7, T8, P3, P4
- P7, P8, O1, O2

---

## 🔁 Oversampling Strategy

### Regular Segments (All data)
- **Step**: 10 seconds (non-overlapping)
- **Purpose**: Standard sampling

### Oversampled Segments (Seizure only)
- **Base segments**: Same as regular
- **Additional segments**: 5-second step within **seizure ±1 second**
- **Effective rate**: ~3× oversampling for seizure periods
- **Flag**: `is_oversampled: True` in metadata

### Example
```
Seizure at 100-110 seconds:

Regular:
├── [90-100]:  label=0  (non-seizure)
├── [100-110]: label=1  (seizure) ← Base
└── [110-120]: label=0  (non-seizure)

Oversampled (99-111s range, 5s step):
├── [99-109]:  label=1, is_oversampled=True
├── [104-114]: label=1, is_oversampled=True
└── [109-119]: label=1, is_oversampled=True

→ Total: 4 seizure segments (1 base + 3 oversampled)
```

---

## 💡 Usage Notes

1. **Use v2 format** for new projects (ISRUC-compatible)
2. **Migrate v1 data** using `migrate_chbmit_v1_to_v2.py`
3. **Apply class weights** even after oversampling (still 30:1 imbalance)
4. **Consider patient-level splitting** to avoid data leakage
5. **Check logs/** for detailed preprocessing statistics

---

## 📖 Citation

```bibtex
@inproceedings{shoeb2009chbmit,
  title={Application of Machine Learning to Epileptic Seizure Detection},
  author={Shoeb, A.},
  booktitle={Proceedings of the 26th International Conference on Machine Learning},
  year={2009}
}
```

## 🔗 Links

- **Dataset**: https://physionet.org/content/chbmit/1.0.0/
- **DOI**: 10.13026/C2K01R
- **PhysioNet**: https://physionet.org/

---

## 📌 Version Information

- **Data Format**: v2-keymodified (ISRUC-compatible)
- **Last Updated**: 2025-11-27
- **Migration Available**: Yes (v1 → v2)
- **Recommended**: Use v2 for all new projects
