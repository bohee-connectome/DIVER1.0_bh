import scipy
from scipy import signal
import os
import re
import lmdb
import pickle
import numpy as np
import mne
import shutil
import tempfile

def load_xyz_from_elc(elc_path: str,
                      want_channels: list[str]) -> np.ndarray:
    """
    Load electrode coordinates from .elc file
    """
    want_up = [ch.upper() for ch in want_channels]

    # Skip first 4 lines
    with open(elc_path, 'r') as f:
        lines = f.readlines()[4:]

    positions_start = labels_start = None
    for i, ln in enumerate(lines):
        ll = ln.strip().lower()
        if ll == "positions":
            positions_start = i + 1
        elif ll == "labels":
            labels_start = i + 1
            break
    if positions_start is None or labels_start is None:
        raise RuntimeError("ELC file missing Positions/Labels sections")

    # Read coordinates
    positions = []
    for ln in lines[positions_start:labels_start-1]:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        xyz = [float(p) for p in ln.split()[:3]]
        positions.append(np.array(xyz, dtype=float))

    # Read labels
    labels = [ln.strip().upper() for ln in lines[labels_start:]
              if ln.strip() and not ln.startswith('#')]

    if len(labels) != len(positions):
        raise RuntimeError("Labels count != Positions count")

    # Map coordinates (preserve want_channels order)
    xyz_list = []
    for ch in want_up:
        if ch in labels:
            idx = labels.index(ch)
            xyz_list.append(positions[idx])
        else:
            print(f"[ELC] Warning: {ch} not found; NaN inserted")
            xyz_list.append(np.full(3, np.nan))
    return np.vstack(xyz_list)  # shape = (len(want), 3)


# =============================================================================
# ISRUC-Sleep Dataset Configuration
# =============================================================================

# Data paths
root_dir = '/global/cfs/cdirs/m4750/DIVER/DOWNLOAD_DATASETS_MOVE_TO_M4750_LATER/ISRUC'
lmdb_path = '/pscratch/sd/b/boheelee/DIVER/ISRUC_preprocessing/lmdb_output/ISRUC_Sleep'
elc_file = '/global/homes/b/boheelee/standard_1005.elc'

# ISRUC channel selection (6 EEG channels from 10-20 system)
# Target channels: F3, C3, O1, F4, C4, O2 (with various reference notations)
target_channel_bases = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']

# Channel names for elc mapping (remove reference notation)
want_channels = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']

def find_matching_channels(available_channels, target_bases):
    """
    Find matching channels from available channels using flexible matching.
    Handles various naming conventions like:
    - F3-A2, F3-A1, F3, EEG F3, etc.
    """
    matched_channels = []
    matched_indices = []
    
    for target_base in target_bases:
        # Try multiple patterns
        patterns = [
            rf'^{re.escape(target_base)}-A[12]$',  # F3-A2, F3-A1
            rf'^{re.escape(target_base)}$',         # F3
            rf'^{re.escape(target_base)}\s',        # F3 (with space after)
            rf'\s{re.escape(target_base)}$',        # (space before) F3
            rf'EEG\s+{re.escape(target_base)}',     # EEG F3
            rf'{re.escape(target_base)}.*',         # F3-anything
        ]
        
        found = False
        for pattern in patterns:
            for idx, ch_name in enumerate(available_channels):
                # Case-insensitive matching
                if re.match(pattern, ch_name, re.IGNORECASE):
                    if idx not in matched_indices:  # Avoid duplicates
                        matched_channels.append(ch_name)
                        matched_indices.append(idx)
                        found = True
                        break
            if found:
                break
        
        if not found:
            # Try fuzzy matching: check if target_base is in channel name
            for idx, ch_name in enumerate(available_channels):
                ch_upper = ch_name.upper().replace('-', '').replace('_', '').replace(' ', '')
                target_upper = target_base.upper()
                if target_upper in ch_upper and idx not in matched_indices:
                    matched_channels.append(ch_name)
                    matched_indices.append(idx)
                    found = True
                    break
    
    return matched_channels, matched_indices

# Load xyz coordinates
xyz_array = load_xyz_from_elc(elc_file, want_channels)

# Data split: Following CBraMod paper (ICLR 2025)
# Reference: Khalighi et al., 2016 - ISRUC-Sleep dataset
# Paper setting: Use Subgroup1 only (100 healthy adults)
#   - Train: Subject 1-80 (80 subjects)
#   - Val: Subject 81-90 (10 subjects)
#   - Test: Subject 91-100 (10 subjects)
# Total samples: 89,240 30-second epochs across 100 subjects

subgroups = {
      'train': [('Subgroup1', i) for i in range(1, 81)],   # 1-80
      'val': [('Subgroup1', i) for i in range(81, 91)],    # 81-90
      'test': [('Subgroup1', i) for i in range(91, 101)],  # 91-100
  }

print("Dataset split (Starting from Subject 85):")
print(f"  Train: {len(subgroups['train'])} subjects (Skipped - already processed)")
print(f"  Val: {len(subgroups['val'])} subjects (Subject 85-90)")
print(f"  Test: {len(subgroups['test'])} subjects (Subject 91-100)")
print(f"  Total: {sum(len(v) for v in subgroups.values())} subjects to process")

# Initialize dataset tracking
dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

# Open LMDB
# Note: If LMDB already exists and map_size needs to be increased,
# you may need to delete the existing LMDB directory first
db = lmdb.open(lmdb_path, map_size=200 * 1024**3)  # 200GB (increased from 50GB)

# Load existing keys from LMDB to skip already processed subjects
existing_keys = set()
try:
    with db.begin() as txn:
        # First, try to get all keys directly from LMDB (more reliable)
        cursor = txn.cursor()
        all_lmdb_keys = [key.decode() for key, _ in cursor]
        
        # Filter out __keys__ and collect all sample keys
        for key in all_lmdb_keys:
            if key != '__keys__':
                existing_keys.add(key)
        
        print(f"Loaded {len(existing_keys)} existing sample keys from LMDB")
        
        # Also try to load __keys__ for dataset structure
        keys_data = txn.get('__keys__'.encode())
        if keys_data is not None:
            existing_dataset = pickle.loads(keys_data)
            # Also load existing dataset structure
            for split_key in existing_dataset:
                if split_key in dataset:
                    dataset[split_key] = existing_dataset[split_key]
            print(f"Loaded dataset structure from __keys__: {sum(len(v) for v in existing_dataset.values())} samples")
except Exception as e:
    print(f"Note: Could not load existing keys: {e}")
    print("Starting fresh or continuing...")

def is_subject_processed(subgroup_name, subject_id, existing_keys):
    """Check if a subject has already been processed by looking for sample keys."""
    subject_prefix = f'{subgroup_name}_S{subject_id:03d}_'
    # Check if any key with this prefix exists
    for key in existing_keys:
        if key.startswith(subject_prefix):
            return True
    return False

# =============================================================================
# Preprocessing Loop
# =============================================================================

for split_key in subgroups.keys():
    print(f"\n{'='*60}")
    print(f"Processing {split_key.upper()} split")
    print(f"{'='*60}")

    for subgroup_name, subject_id in subgroups[split_key]:
        # Check if subject has already been processed
        if is_subject_processed(subgroup_name, subject_id, existing_keys):
            print(f"⏭ Skipping {subgroup_name}/{subject_id}: Already processed")
            continue
        
        # Construct file paths
        rar_file = os.path.join(root_dir, subgroup_name, f'{subject_id}.rar')

        if not os.path.exists(rar_file):
            print(f"⚠ Skipping: {rar_file} not found")
            continue

        # Extract to temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Extract .rar file
            import subprocess
            extract_cmd = f'unrar x {rar_file} {temp_dir}/'
            subprocess.run(extract_cmd, shell=True, check=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Find extracted files
            subject_dir = os.path.join(temp_dir, str(subject_id))
            rec_file = os.path.join(subject_dir, f'{subject_id}.rec')
            annot_file_1 = os.path.join(subject_dir, f'{subject_id}_1.txt')
            annot_file_2 = os.path.join(subject_dir, f'{subject_id}_2.txt')

            # Check files exist
            if not os.path.exists(rec_file):
                print(f"⚠ Skipping {subgroup_name}/{subject_id}: .rec file not found")
                continue
            if not os.path.exists(annot_file_1):
                print(f"⚠ Skipping {subgroup_name}/{subject_id}: annotation file not found")
                continue

            # Copy .rec to .edf for MNE compatibility
            temp_edf = os.path.join(temp_dir, 'temp.edf')
            shutil.copy(rec_file, temp_edf)

            # Read EEG data
            raw = mne.io.read_raw_edf(temp_edf, preload=True, verbose=False)

            # Check available channels
            available_channels = raw.ch_names
            print(f"  Available channels in {subgroup_name}/{subject_id}: {available_channels}")
            
            # Find matching channels using flexible matching
            matched_channels, matched_indices = find_matching_channels(
                available_channels, target_channel_bases
            )
            
            # Check if we found all required channels
            if len(matched_channels) != len(target_channel_bases):
                missing_bases = []
                found_bases = []
                for i, base in enumerate(target_channel_bases):
                    if i >= len(matched_channels):
                        missing_bases.append(base)
                    else:
                        found_bases.append(f"{base} -> {matched_channels[i]}")
                
                print(f"  ✗ Error processing {subgroup_name}/{subject_id}: Missing channels")
                print(f"    Required channel bases: {target_channel_bases}")
                print(f"    Found matches: {found_bases}")
                print(f"    Missing: {missing_bases}")
                print(f"    Available channels: {available_channels}")
                continue
            
            # Reorder matched channels to match target_channel_bases order
            # Create a mapping to ensure correct order
            ordered_channels = []
            for base in target_channel_bases:
                for ch in matched_channels:
                    ch_upper = ch.upper().replace('-', '').replace('_', '').replace(' ', '')
                    base_upper = base.upper()
                    if base_upper in ch_upper or ch_upper.startswith(base_upper):
                        if ch not in ordered_channels:
                            ordered_channels.append(ch)
                            break
            
            if len(ordered_channels) != len(target_channel_bases):
                print(f"  ✗ Error processing {subgroup_name}/{subject_id}: Failed to order channels correctly")
                continue
            
            print(f"  Matched channels: {dict(zip(target_channel_bases, ordered_channels))}")
            
            # Select EEG channels
            raw.pick_channels(ordered_channels, ordered=True)

            # Interpolate bad channels if any
            if len(raw.info['bads']) > 0:
                print(f'  Interpolating bad channels: {raw.info["bads"]}')
                raw.interpolate_bads()

            # Store original sampling rate
            original_sampling_rate = int(raw.info['sfreq'])

            # Preprocessing pipeline
            raw.set_eeg_reference(ref_channels='average')
            raw.filter(l_freq=0.3, h_freq=35)  # 0.3-35 Hz bandpass (CBraMod setting)
            raw.notch_filter((50))  # 50 Hz notch filter (CBraMod setting)
            raw.resample(500)  # Resample to 500 Hz

            # Read annotations (use expert 1)
            with open(annot_file_1, 'r') as f:
                sleep_stages = [int(line.strip()) for line in f if line.strip()]

            # Get data array
            data = raw.get_data(units='uV')  # shape: (6, timepoints)

            # Segment into 30-second epochs
            epoch_duration = 30  # seconds
            sampling_rate = 500  # Hz after resampling
            samples_per_epoch = epoch_duration * sampling_rate  # 15000 samples

            num_epochs = len(sleep_stages)

            print(f"  Processing {subgroup_name}/{subject_id}: "
                  f"{num_epochs} epochs, {original_sampling_rate} Hz -> 500 Hz")

            # Process each epoch
            for epoch_idx in range(num_epochs):
                start_sample = epoch_idx * samples_per_epoch
                end_sample = start_sample + samples_per_epoch

                # Check if we have enough data
                if end_sample > data.shape[1]:
                    print(f"  ⚠ Epoch {epoch_idx} exceeds data length, skipping remaining epochs")
                    break

                # Extract epoch data
                epoch_data = data[:, start_sample:end_sample]  # (6, 15000)

                # Reshape to (channels, duration_sec, sampling_rate)
                # For 30-second epoch: (6, 30, 500)
                epoch_data = epoch_data.reshape(6, 30, 500)

                # Get sleep stage label
                # Label mapping (CBraMod setting): 0→0, 1→1, 2→2, 3→3, 5→4
                raw_label = sleep_stages[epoch_idx]
                label = 4 if raw_label == 5 else raw_label

                # Create sample key
                sample_key = f'{subgroup_name}_S{subject_id:03d}_E{epoch_idx:04d}'

                # Create data dictionary
                data_dict = {
                    'sample': epoch_data,
                    'label': label,
                    'data_info': {
                        'Dataset': 'ISRUC-Sleep',
                        'modality': 'EEG',
                        'release': None,
                        'subject_id': f'{subgroup_name}_S{subject_id:03d}',
                        'subgroup': subgroup_name,
                        'task': 'sleep-staging',
                        'resampling_rate': 500,
                        'original_sampling_rate': original_sampling_rate,
                        'segment_index': epoch_idx,
                        'start_time': epoch_idx * 30,
                        'channel_names': want_channels,
                        'xyz_id': xyz_array
                    }
                }

                # Write to LMDB
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()

                # Track in dataset and existing_keys
                dataset[split_key].append(sample_key)
                existing_keys.add(sample_key)

            print(f"  ✓ Completed {subgroup_name}/{subject_id}: {epoch_idx+1} epochs saved")

        except Exception as e:
            print(f"  ✗ Error processing {subgroup_name}/{subject_id}: {e}")

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

# Save dataset keys
txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()

# Print summary
print(f"\n{'='*60}")
print("Preprocessing Complete!")
print(f"{'='*60}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Val samples: {len(dataset['val'])}")
print(f"Test samples: {len(dataset['test'])}")
print(f"Total samples: {sum(len(v) for v in dataset.values())}")
print(f"\nLMDB saved to: {lmdb_path}")
print(f"{'='*60}")
