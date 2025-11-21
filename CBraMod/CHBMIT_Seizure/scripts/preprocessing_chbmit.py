#!/usr/bin/env python3
"""
CHB-MIT Seizure Detection Dataset Preprocessing for DIVER

This script processes CHB-MIT EEG data into LMDB format:
1. Load EDF files and metadata from summary files
2. Extract 16 bipolar channels (10-20 system)
3. Segment into 10-second windows
4. Resample from 256 Hz to 500 Hz
5. Reshape to (16, 10, 500) format
6. Label seizure vs. non-seizure segments
7. Apply oversampling for seizure segments
8. Save to LMDB with electrode position information

Author: Claude + User
Date: 2025-01-21
"""

import os
import re
import pickle
import numpy as np
import lmdb
import pyedflib
from scipy.signal import resample
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import argparse


# =============================================================================
# Configuration
# =============================================================================

# Paths (will be overridden by command-line arguments if provided)
DEFAULT_DATA_PATH = '/global/cfs/cdirs/m4750/DIVER/DOWNLOAD_DATASETS_MOVE_TO_M4750_LATER/CHB-MIT/physionet.org/files/chbmit/1.0.0'
DEFAULT_LMDB_PATH = '/pscratch/sd/b/boheelee/DIVER/CHBMIT_preprocessing/lmdb_output/CHBMIT_Seizure'
DEFAULT_ELC_FILE = '/global/homes/b/boheelee/standard_1005.elc'

# CHB-MIT 16 bipolar channels (10-20 system)
TARGET_CHANNELS = [
    # Left Lateral Chain
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    # Right Lateral Chain
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    # Left Parasagittal Chain
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    # Right Parasagittal Chain
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

# Individual electrodes used in bipolar channels
ELECTRODE_NAMES = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8',
                   'C3', 'C4', 'T7', 'T8', 'P3', 'P4',
                   'P7', 'P8', 'O1', 'O2']

# Sampling rates
ORIGINAL_SR = 256  # Hz
TARGET_SR = 500    # Hz

# Segment parameters
SEGMENT_LENGTH_SEC = 10  # seconds
SEGMENT_LENGTH_ORIGINAL = SEGMENT_LENGTH_SEC * ORIGINAL_SR  # 2560 samples
SEGMENT_LENGTH_TARGET = SEGMENT_LENGTH_SEC * TARGET_SR      # 5000 samples

# Oversampling parameters
NORMAL_STEP = SEGMENT_LENGTH_SEC * ORIGINAL_SR  # 10 sec step for normal
SEIZURE_STEP = 5 * ORIGINAL_SR                   # 5 sec step for seizure
SEIZURE_MARGIN = 1 * ORIGINAL_SR                 # ±1 sec around seizure

# Train/Val/Test split
TEST_PATIENTS = ["chb23", "chb24"]
VAL_PATIENTS = ["chb21", "chb22"]
# Train: all others (chb01-20, excluding chb12, chb13, chb17 which don't exist)

# Parameters for each patient: (patient_id, reference_file, start_file, end_file, summary_index)
PATIENT_PARAMS = [
    ("01", "01", 2, 46, 0),
    ("02", "01", 2, 35, 0),
    ("03", "01", 2, 38, 0),
    ("04", "07", 1, 43, 1),
    ("05", "01", 2, 39, 0),
    ("06", "01", 2, 24, 0),
    ("07", "01", 2, 19, 0),
    ("08", "02", 3, 29, 0),
    ("09", "02", 1, 19, 1),  # Fixed: incremental LMDB write prevents memory accumulation
    ("10", "01", 2, 89, 0),
    ("11", "01", 2, 99, 0),
    ("14", "01", 2, 42, 0),
    ("15", "02", 1, 63, 1),
    ("16", "01", 2, 19, 0),
    ("18", "02", 1, 36, 1),
    ("19", "02", 1, 30, 1),
    ("20", "01", 2, 68, 0),
    ("21", "01", 2, 33, 0),
    ("22", "01", 2, 77, 0),
    ("23", "06", 7, 20, 0),
    ("24", "01", 3, 21, 0),
]


# =============================================================================
# Utility Functions
# =============================================================================

def load_xyz_from_elc(elc_path: str, want_channels: list) -> np.ndarray:
    """
    Load electrode coordinates from .elc file

    Args:
        elc_path: Path to .elc file
        want_channels: List of electrode names (not bipolar pairs)

    Returns:
        np.ndarray: (N, 3) array of electrode positions
    """
    want_up = [ch.upper() for ch in want_channels]

    with open(elc_path, 'r') as f:
        lines = f.readlines()[4:]  # Skip first 4 lines

    # Find Positions and Labels sections
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
        raise RuntimeError(f"Labels count ({len(labels)}) != Positions count ({len(positions)})")

    # Map coordinates (preserve want_channels order)
    xyz_list = []
    for ch in want_up:
        if ch in labels:
            idx = labels.index(ch)
            xyz_list.append(positions[idx])
        else:
            print(f"[ELC] Warning: {ch} not found; using NaN")
            xyz_list.append(np.full(3, np.nan))

    return np.vstack(xyz_list)  # shape = (len(want_channels), 3)


def create_bipolar_elc_info(elc_path: str, bipolar_channels: list) -> dict:
    """
    Create electrode info dictionary for bipolar channels

    Args:
        elc_path: Path to .elc file
        bipolar_channels: List of bipolar channel names (e.g., "FP1-F7")

    Returns:
        dict: Contains channel_names, electrode_pairs, electrode_positions
    """
    # Extract individual electrode names from bipolar pairs
    all_electrodes = set()
    electrode_pairs = {}

    for bipolar_ch in bipolar_channels:
        # Split by '-' to get two electrodes
        parts = bipolar_ch.split('-')
        if len(parts) == 2:
            e1, e2 = parts[0].strip(), parts[1].strip()
            electrode_pairs[bipolar_ch] = [e1, e2]
            all_electrodes.add(e1)
            all_electrodes.add(e2)
        else:
            print(f"[WARNING] Invalid bipolar channel format: {bipolar_ch}")

    # Load electrode positions
    electrode_list = sorted(all_electrodes)
    try:
        positions = load_xyz_from_elc(elc_path, electrode_list)
        electrode_positions = {name: pos for name, pos in zip(electrode_list, positions)}
    except Exception as e:
        print(f"[WARNING] Failed to load ELC file: {e}")
        electrode_positions = {name: np.full(3, np.nan) for name in electrode_list}

    return {
        'channel_names': bipolar_channels,
        'electrode_pairs': electrode_pairs,
        'electrode_positions': electrode_positions
    }


def parse_summary_file(summary_path: str, filename: str) -> dict:
    """
    Parse CHB-MIT summary file to extract seizure metadata
    (Robust version that handles summary file errors)

    Args:
        summary_path: Path to summary.txt file
        filename: Target EDF filename (e.g., "chb01_01.edf")

    Returns:
        dict: {'seizures': int, 'times': [(start, end), ...]}
    """
    try:
        with open(summary_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[WARNING] Failed to read summary file {summary_path}: {e}")
        return {'seizures': 0, 'times': []}

    metadata = {'seizures': 0, 'times': []}

    for i in range(len(lines)):
        line = lines[i].split()

        # Find the target file
        if len(line) >= 3 and line[2] == filename:
            j = i + 1

            # Find number of seizures
            while j < len(lines):
                try:
                    if len(lines[j].split()) > 0 and lines[j].split()[0] == "Number":
                        seizures = int(lines[j].split()[-1])
                        metadata['seizures'] = seizures
                        break
                except (ValueError, IndexError):
                    pass
                j += 1

            # Extract seizure times if seizures > 0
            if metadata['seizures'] > 0:
                j = i + 1
                seizure_count = 0

                while j < len(lines) and seizure_count < metadata['seizures']:
                    try:
                        line_text = lines[j].strip()

                        # Look for "Seizure" + "Start" + "Time" (with or without number)
                        # Format 1: "Seizure 1 Start Time: 2996 seconds"
                        # Format 2: "Seizure Start Time: 2996 seconds"
                        if line_text.startswith("Seizure") and "Start" in line_text and "Time" in line_text:
                            parts = line_text.split()
                            # Find the time value (should be second-to-last word)
                            try:
                                start_time = int(parts[-2])
                            except (ValueError, IndexError):
                                j += 1
                                continue

                            # Look for corresponding End Time (next line or nearby)
                            end_time = None
                            for k in range(j + 1, min(j + 10, len(lines))):
                                end_line_text = lines[k].strip()
                                if end_line_text.startswith("Seizure") and "End" in end_line_text and "Time" in end_line_text:
                                    end_parts = end_line_text.split()
                                    try:
                                        end_time = int(end_parts[-2])
                                        j = k  # Move past this end time
                                        break
                                    except (ValueError, IndexError):
                                        pass

                            if end_time is not None:
                                # Convert time (seconds) to sample index
                                start = start_time * ORIGINAL_SR - 1
                                end = end_time * ORIGINAL_SR - 1
                                metadata['times'].append((start, end))
                                seizure_count += 1
                            else:
                                print(f"[WARNING] Could not find end time for seizure {seizure_count+1} in {filename} (start={start_time}s)")

                    except Exception as e:
                        print(f"[WARNING] Error parsing seizure time in {filename}: {e}")

                    j += 1
            break

    return metadata


def load_edf_channels(edf_path: str, target_channels: list) -> dict:
    """
    Load EDF file and extract target channels

    Args:
        edf_path: Path to EDF file
        target_channels: List of channel names to extract

    Returns:
        dict: {channel_name: np.array}
    """
    try:
        edf = pyedflib.EdfReader(edf_path)
    except Exception as e:
        print(f"[ERROR] Failed to open {edf_path}: {e}")
        return None

    # Get available channels
    available_channels = edf.getSignalLabels()
    n_samples = edf.getNSamples()[0] if len(edf.getNSamples()) > 0 else 0

    # Match target channels with available channels
    channel_data = {}
    for target_ch in target_channels:
        found = False

        # Try exact match first
        if target_ch in available_channels:
            idx = available_channels.index(target_ch)
            channel_data[target_ch] = edf.readSignal(idx)
            found = True
        else:
            # Try flexible matching (remove spaces, case-insensitive)
            target_normalized = target_ch.replace(' ', '').replace('-', '').upper()
            for idx, avail_ch in enumerate(available_channels):
                avail_normalized = avail_ch.replace(' ', '').replace('-', '').upper()
                if target_normalized == avail_normalized:
                    channel_data[target_ch] = edf.readSignal(idx)
                    found = True
                    break

        if not found:
            # Fill with zeros if channel not found
            print(f"[WARNING] Channel {target_ch} not found in {edf_path}, using zeros")
            channel_data[target_ch] = np.zeros(n_samples, dtype=float)

    edf.close()
    return channel_data


def segment_and_label(signal: np.ndarray, seizure_times: list,
                     recording_id: str, elc_info: dict) -> list:
    """
    Segment signal into 10-second windows and label seizure/non-seizure

    Args:
        signal: (16, N) array of EEG data at 256 Hz
        seizure_times: List of (start, end) seizure times in sample indices
        recording_id: Recording identifier (e.g., "chb01_01")
        elc_info: Electrode information dictionary

    Returns:
        list: List of segment dictionaries
    """
    segments = []
    signal_length = signal.shape[1]

    # 1) Basic segments (10-second step, non-overlapping)
    for i in range(0, signal_length, NORMAL_STEP):
        if i + SEGMENT_LENGTH_ORIGINAL > signal_length:
            break  # Skip incomplete segments

        segment = signal[:, i:i + SEGMENT_LENGTH_ORIGINAL]  # (16, 2560)

        # Check if segment contains seizure
        label = 0
        for seizure_start, seizure_end in seizure_times:
            if (i < seizure_start < i + SEGMENT_LENGTH_ORIGINAL or
                i < seizure_end < i + SEGMENT_LENGTH_ORIGINAL):
                label = 1
                break

        # Resample to 500 Hz
        segment_resampled = resample(segment, SEGMENT_LENGTH_TARGET, axis=1)  # (16, 5000)

        # Reshape to (16, 10, 500)
        segment_final = segment_resampled.reshape(16, 10, TARGET_SR)

        segments.append({
            'signal': segment_final.astype(np.float32),
            'label': label,
            'segment_id': f"{recording_id}_{i}",
            'is_oversampled': False,
            'original_index': i
        })

    # 2) Additional seizure samples (5-second step oversampling)
    for seizure_idx, (seizure_start, seizure_end) in enumerate(seizure_times):
        # Seizure ±1 second range
        start = max(0, seizure_start - SEIZURE_MARGIN)
        end = min(seizure_end + SEIZURE_MARGIN, signal_length)

        for i in range(start, end, SEIZURE_STEP):
            if i + SEGMENT_LENGTH_ORIGINAL > signal_length:
                break

            segment = signal[:, i:i + SEGMENT_LENGTH_ORIGINAL]

            # Resample to 500 Hz
            segment_resampled = resample(segment, SEGMENT_LENGTH_TARGET, axis=1)

            # Reshape to (16, 10, 500)
            segment_final = segment_resampled.reshape(16, 10, TARGET_SR)

            segments.append({
                'signal': segment_final.astype(np.float32),
                'label': 1,  # Always seizure
                'segment_id': f"{recording_id}_s{seizure_idx}_add_{i}",
                'is_oversampled': True,
                'original_index': i
            })

    return segments


def process_patient(patient_id: str, data_path: str, elc_info: dict) -> dict:
    """
    Process all recordings for a single patient

    Args:
        patient_id: Patient ID (e.g., "01")
        data_path: Root data directory
        elc_info: Electrode information dictionary

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    patient_name = f"chb{patient_id}"
    patient_dir = os.path.join(data_path, patient_name)
    summary_path = os.path.join(patient_dir, f"{patient_name}-summary.txt")

    if not os.path.exists(summary_path):
        print(f"[WARNING] Summary file not found for {patient_name}")
        return {'train': [], 'val': [], 'test': []}

    # Determine split
    if patient_name in TEST_PATIENTS:
        split = 'test'
    elif patient_name in VAL_PATIENTS:
        split = 'val'
    else:
        split = 'train'

    all_segments = []

    # Find all EDF files
    edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.edf')])

    for edf_file in tqdm(edf_files, desc=f"Processing {patient_name}", leave=False):
        try:
            edf_path = os.path.join(patient_dir, edf_file)

            # Parse metadata
            metadata = parse_summary_file(summary_path, edf_file)

            # Load EDF channels
            channel_data = load_edf_channels(edf_path, TARGET_CHANNELS)
            if channel_data is None:
                print(f"[WARNING] Skipping {edf_file}: Failed to load channels")
                continue

            # Convert to (16, N) array
            signal = np.array([channel_data[ch] for ch in TARGET_CHANNELS])

            # Segment and label
            recording_id = f"{patient_name}_{edf_file.replace('.edf', '')}"
            segments = segment_and_label(signal, metadata['times'], recording_id, elc_info)

            all_segments.extend(segments)

        except Exception as e:
            print(f"[ERROR] Failed to process {edf_file}: {e}")
            print(f"[INFO] Continuing with next file...")
            continue

    print(f"[INFO] {patient_name} ({split}): {len(all_segments)} segments "
          f"(Label 0: {sum(1 for s in all_segments if s['label'] == 0)}, "
          f"Label 1: {sum(1 for s in all_segments if s['label'] == 1)})")

    return {split: all_segments}


def save_to_lmdb(segments_dict: dict, lmdb_path: str, elc_info: dict):
    """
    Save segments to LMDB database

    Args:
        segments_dict: {'train': [...], 'val': [...], 'test': [...]}
        lmdb_path: Root LMDB directory
        elc_info: Electrode information dictionary
    """
    os.makedirs(lmdb_path, exist_ok=True)

    for split in ['train', 'val', 'test']:
        segments = segments_dict.get(split, [])
        if len(segments) == 0:
            print(f"[WARNING] No segments for {split}, skipping")
            continue

        split_lmdb_path = os.path.join(lmdb_path, split)
        os.makedirs(split_lmdb_path, exist_ok=True)

        # Calculate map_size (each segment ~40KB, add 50% buffer)
        map_size = max(len(segments) * 60 * 1024, 1024**3)  # At least 1GB

        env = lmdb.open(split_lmdb_path, map_size=map_size)

        with env.begin(write=True) as txn:
            for idx, seg in enumerate(tqdm(segments, desc=f"Saving {split} to LMDB")):
                key = seg['segment_id'].encode('utf-8')

                value = {
                    'signal': seg['signal'],  # (16, 10, 500)
                    'label': seg['label'],
                    'elc_info': elc_info,
                    'metadata': {
                        'segment_id': seg['segment_id'],
                        'is_oversampled': seg['is_oversampled'],
                        'original_index': seg['original_index'],
                        'original_sr': ORIGINAL_SR,
                        'target_sr': TARGET_SR
                    }
                }

                txn.put(key, pickle.dumps(value))

        env.close()
        print(f"[INFO] Saved {len(segments)} segments to {split_lmdb_path}")


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Preprocess CHB-MIT dataset for DIVER')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to CHB-MIT dataset')
    parser.add_argument('--lmdb_path', type=str, default=DEFAULT_LMDB_PATH,
                        help='Output LMDB path')
    parser.add_argument('--elc_file', type=str, default=DEFAULT_ELC_FILE,
                        help='Path to .elc electrode file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    args = parser.parse_args()

    print("=" * 80)
    print("CHB-MIT Seizure Detection Dataset Preprocessing")
    print("=" * 80)
    print(f"Data path:    {args.data_path}")
    print(f"LMDB path:    {args.lmdb_path}")
    print(f"ELC file:     {args.elc_file}")
    print(f"Workers:      {args.num_workers}")
    print(f"Original SR:  {ORIGINAL_SR} Hz")
    print(f"Target SR:    {TARGET_SR} Hz")
    print(f"Segment len:  {SEGMENT_LENGTH_SEC} sec")
    print(f"Output shape: (16, 10, {TARGET_SR})")
    print("=" * 80)

    # Load electrode information
    print("\n[1/4] Loading electrode information...")
    elc_info = create_bipolar_elc_info(args.elc_file, TARGET_CHANNELS)
    print(f"      Loaded positions for {len(elc_info['electrode_positions'])} electrodes")

    # Initialize single LMDB database (like ISRUC)
    print("\n[2/4] Initializing LMDB database...")
    os.makedirs(args.lmdb_path, exist_ok=True)
    db = lmdb.open(args.lmdb_path, map_size=200 * 1024**3)  # 200GB

    # Statistics counters
    stats = {'train': {'total': 0, 'normal': 0, 'seizure': 0},
             'val': {'total': 0, 'normal': 0, 'seizure': 0},
             'test': {'total': 0, 'normal': 0, 'seizure': 0}}

    # Dataset keys tracking (like ISRUC)
    dataset = {'train': [], 'val': [], 'test': []}

    # Process all patients and save incrementally
    print("\n[3/4] Processing patients and saving to LMDB...")
    for patient_id, _, _, _, _ in PATIENT_PARAMS:
        result = process_patient(patient_id, args.data_path, elc_info)

        # Save immediately to LMDB to avoid memory accumulation
        with db.begin(write=True) as txn:
            for split in ['train', 'val', 'test']:
                if split in result and len(result[split]) > 0:
                    segments = result[split]

                    for seg in segments:
                        segment_id = seg['segment_id']
                        key = segment_id.encode('utf-8')
                        value = {
                            'signal': seg['signal'],
                            'label': seg['label'],
                            'elc_info': elc_info,
                            'metadata': {
                                'segment_id': segment_id,
                                'split': split,
                                'is_oversampled': seg['is_oversampled'],
                                'original_index': seg['original_index'],
                                'original_sr': ORIGINAL_SR,
                                'target_sr': TARGET_SR
                            }
                        }
                        txn.put(key, pickle.dumps(value))

                        # Track dataset keys (like ISRUC)
                        dataset[split].append(segment_id)

                        # Update statistics
                        stats[split]['total'] += 1
                        if seg['label'] == 0:
                            stats[split]['normal'] += 1
                        else:
                            stats[split]['seizure'] += 1

                    # Free memory
                    del segments
                    import gc
                    gc.collect()

    # Save dataset keys (like ISRUC)
    print("\n[4/4] Saving dataset keys...")
    with db.begin(write=True) as txn:
        txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
    print(f"      Saved __keys__ with {sum(len(v) for v in dataset.values())} total samples")

    # Close LMDB database
    db.close()

    # Print statistics
    print("\n[4/4] Dataset statistics:")
    for split in ['train', 'val', 'test']:
        n_total = stats[split]['total']
        n_normal = stats[split]['normal']
        n_seizure = stats[split]['seizure']
        print(f"      {split.upper():5s}: {n_total:6d} segments "
              f"(Normal: {n_normal:5d}, Seizure: {n_seizure:5d}, "
              f"Ratio: {n_normal/max(n_seizure, 1):.1f}:1)")

    print("\n" + "=" * 80)
    print("Preprocessing completed successfully!")
    print(f"Output saved to: {args.lmdb_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
