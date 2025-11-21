# ISRUC-Sleep: Sleep Stage Classification

Sleep stage classification (5-class: W, N1, N2, N3, REM) on ISRUC-Sleep dataset using DIVER preprocessing pipeline.

## Quick Info

| Item | Value |
|------|-------|
| **Dataset** | ISRUC-Sleep Subgroup 1 |
| **Subjects** | 100 patients |
| **Task** | 5-class sleep stage classification |
| **Channels** | 6 EEG channels |
| **Sampling Rate** | 200 Hz → 500 Hz (resampled) |
| **Epoch Length** | 30 seconds |
| **Output Shape** | (6, 30, 500) |

## Documentation

- 📄 **[ISRUC_DATASET_INFO.md](ISRUC_DATASET_INFO.md)** - Complete dataset documentation
- 📁 **[STRUCTURE.md](STRUCTURE.md)** - Directory structure and file descriptions

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

## Data Split

- **Train**: Subjects 1-84 (84 subjects)
- **Validation**: Subjects 85-90 (6 subjects)
- **Test**: Subjects 91-100 (10 subjects)

## Output Format

```python
{
    "signal": np.array (6, 30, 500),  # 6 channels × 30 segments × 500 samples/sec
    "label": int (0-4),               # W=0, N1=1, N2=2, N3=3, REM=4
    "elc_info": dict,
    "metadata": dict
}
```

## Citation

```bibtex
@article{khalighi2016isruc,
  title={ISRUC-Sleep: A comprehensive public dataset for sleep researchers},
  author={Khalighi, S. and Sousa, T. and Santos, J. M. and Nunes, U.},
  journal={Computer Methods and Programs in Biomedicine},
  volume={124},
  pages={180--192},
  year={2016}
}
```

## Links

- **Dataset**: https://sleeptight.isr.uc.pt/
- **PhysioNet**: https://physionet.org/content/isruc-sleep/1.0.0/
