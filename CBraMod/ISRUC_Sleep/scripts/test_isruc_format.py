import mne
import numpy as np
import os
from collections import Counter

rec_file = '/pscratch/sd/b/boheelee/DIVER/ISRUC_test/1/1.rec'
annot_file = '/pscratch/sd/b/boheelee/DIVER/ISRUC_test/1/1_1.txt'

print("="*60)
print("ISRUC-Sleep Analysis")
print("="*60)

raw = mne.io.read_raw_edf(rec_file, preload=True, verbose=False)
print(f"\nDuration: {raw.times[-1]/3600:.2f} hours")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Channels: {len(raw.ch_names)}")
print(f"Data shape: {raw.get_data().shape}")

print(f"\nAll Channels:")
for i, ch in enumerate(raw.ch_names, 1):
    print(f"  {i:2d}. {ch}")

with open(annot_file, 'r') as f:
    labels = [int(line.strip()) for line in f if line.strip()]

print(f"\nAnnotations: {len(labels)} epochs")
print(f"Stage distribution: {Counter(labels)}")
print("="*60)
