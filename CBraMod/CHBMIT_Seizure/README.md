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

## Documentation

- 📄 **[CHBMIT_DATASET_INFO.md](CHBMIT_DATASET_INFO.md)** - Complete dataset documentation
- 📁 **[STRUCTURE.md](STRUCTURE.md)** - Directory structure and file descriptions

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy mne lmdb
```

### 2. Run Preprocessing
```bash
cd scripts
bash run_preprocessing_chbmit.sh
```

### 3. Validate Output
```bash
python check_lmdb_chbmit.py
```

## Data Split

- **Train**: chb01-20 (17 patients, excluding 12, 13, 17)
- **Validation**: chb21-22 (2 patients)
- **Test**: chb23-24 (2 patients)

## Output Format

```python
{
    "signal": np.array (16, 10, 500),  # 16 channels × 10 segments × 500 samples/sec
    "label": int (0 or 1),             # 0=non-seizure, 1=seizure
    "elc_info": dict,
    "metadata": dict
}
```

## Class Imbalance

- **Before oversampling**: 99:1 (non-seizure:seizure)
- **After oversampling**: 30:1 (seizure segments sampled with 5s step)

## Citation

```bibtex
@inproceedings{shoeb2009chbmit,
  title={Application of Machine Learning to Epileptic Seizure Detection},
  author={Shoeb, A.},
  booktitle={Proceedings of the 26th International Conference on Machine Learning},
  year={2009}
}
```

## Links

- **Dataset**: https://physionet.org/content/chbmit/1.0.0/
- **DOI**: 10.13026/C2K01R
