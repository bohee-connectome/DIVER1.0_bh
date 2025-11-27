#!/usr/bin/env python3
"""
CHB-MIT LMDB Database Verification Script (v1 & v2 compatible)

This script checks the integrity and contents of CHB-MIT LMDB database:
- Verifies database structure (supports both v1 and v2-keymodified formats)
- Counts samples and patients per split
- Checks data shapes and types
- Analyzes label distribution
- Validates electrode information
- Reads __keys__ for train/val/test split info

Supported formats:
- v1: {'signal', 'label', 'metadata', 'elc_info'}
- v2-keymodified: {'sample', 'label', 'data_info'} (ISRUC-compatible)

Usage:
    python check_lmdb_chbmit.py --lmdb_path /path/to/CHBMIT_Seizure

Author: Claude + User
Date: 2025-01-21
Modified: 2025-01-27 (v2 compatibility)
"""

import lmdb
import pickle
import numpy as np
import argparse
from collections import defaultdict
import sys


def check_lmdb_database(lmdb_path):
    """
    Check single LMDB database with __keys__ for splits

    Args:
        lmdb_path: Path to LMDB directory

    Returns:
        dict: Statistics dictionary
    """
    print(f"\n{'='*80}")
    print(f"Checking CHB-MIT LMDB Database")
    print(f"{'='*80}")
    print(f"Path: {lmdb_path}")

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
    except Exception as e:
        print(f"[ERROR] Failed to open LMDB: {e}")
        return None

    stats = {
        'train': {'total': 0, 'patients': set(), 'recordings': set(),
                  'label_counts': defaultdict(int), 'oversampled': 0},
        'val': {'total': 0, 'patients': set(), 'recordings': set(),
                'label_counts': defaultdict(int), 'oversampled': 0},
        'test': {'total': 0, 'patients': set(), 'recordings': set(),
                 'label_counts': defaultdict(int), 'oversampled': 0}
    }

    shapes = defaultdict(int)
    dtypes = defaultdict(int)
    errors = []
    sample_keys = {'train': [], 'val': [], 'test': []}
    format_versions = {'v1': 0, 'v2': 0}  # Track format versions

    with env.begin() as txn:
        # Get total entries
        total_entries = txn.stat()['entries']
        print(f"\nTotal entries in database: {total_entries}")

        # Load __keys__ for split information
        keys_data = txn.get('__keys__'.encode())
        if keys_data is None:
            print("[ERROR] __keys__ not found in LMDB!")
            print("This database might not be in the correct format.")
            env.close()
            return None

        dataset = pickle.loads(keys_data)
        print(f"\n__keys__ loaded successfully:")
        print(f"  Train keys: {len(dataset.get('train', []))}")
        print(f"  Val keys:   {len(dataset.get('val', []))}")
        print(f"  Test keys:  {len(dataset.get('test', []))}")

        # Iterate through all splits
        for split in ['train', 'val', 'test']:
            keys_list = dataset.get(split, [])
            print(f"\n{'-'*80}")
            print(f"Processing {split.upper()} split ({len(keys_list)} keys)")
            print(f"{'-'*80}")

            for idx, key_str in enumerate(keys_list):
                if idx % 1000 == 0 and idx > 0:
                    print(f"  Processed {idx}/{len(keys_list)} samples...")

                try:
                    # Load data
                    key = key_str.encode('utf-8')
                    value = txn.get(key)

                    if value is None:
                        errors.append(f"{split}: Key in __keys__ but not in LMDB: {key_str}")
                        continue

                    data = pickle.loads(value)

                    # Check structure
                    if not isinstance(data, dict):
                        errors.append(f"{split}: Data is not dict: {key_str}")
                        continue

                    # Detect format version (v1 or v2)
                    is_v2 = 'sample' in data and 'data_info' in data
                    is_v1 = 'signal' in data and 'metadata' in data

                    if is_v2:
                        format_versions['v2'] += 1
                        data_key = 'sample'
                        meta_key = 'data_info'
                    elif is_v1:
                        format_versions['v1'] += 1
                        data_key = 'signal'
                        meta_key = 'metadata'
                    else:
                        errors.append(f"{split}: Unknown format (missing 'signal'/'sample' or 'metadata'/'data_info'): {key_str}")
                        continue

                    # Check required keys
                    if data_key not in data:
                        errors.append(f"{split}: Missing key '{data_key}': {key_str}")
                    if 'label' not in data:
                        errors.append(f"{split}: Missing key 'label': {key_str}")

                    # Check signal/sample shape
                    if data_key in data:
                        signal = data[data_key]
                        shape = signal.shape
                        dtype = signal.dtype
                        shapes[str(shape)] += 1
                        dtypes[str(dtype)] += 1

                        # Verify expected shape
                        if shape != (16, 10, 500):
                            errors.append(f"{split}: Unexpected shape {shape}: {key_str}")

                    # Check label
                    if 'label' in data:
                        label = data['label']
                        stats[split]['label_counts'][label] += 1

                        if label not in [0, 1]:
                            errors.append(f"{split}: Invalid label {label}: {key_str}")

                    # Check metadata/data_info
                    if meta_key in data:
                        metadata = data[meta_key]

                        # Verify split matches
                        if 'split' in metadata and metadata['split'] != split:
                            errors.append(f"{split}: Split mismatch - {meta_key} says {metadata['split']}: {key_str}")

                        # Track oversampling
                        if metadata.get('is_oversampled', False):
                            stats[split]['oversampled'] += 1

                        # For v2, verify ISRUC-compatible fields exist
                        if is_v2:
                            v2_required = ['Dataset', 'modality', 'task', 'subject_id',
                                          'resampling_rate', 'original_sampling_rate',
                                          'channel_names']
                            for field in v2_required:
                                if field not in metadata:
                                    errors.append(f"{split}: v2 missing field '{field}': {key_str}")

                    # Parse key for patient/recording info
                    # Format: "chb01_01_0" or "chb01_03_s0_add_25344"
                    parts = key_str.split('_')
                    if len(parts) >= 2:
                        patient_id = parts[0]  # e.g., "chb01"
                        recording_num = parts[1]  # e.g., "01"
                        recording_id = f"{patient_id}_{recording_num}"

                        stats[split]['patients'].add(patient_id)
                        stats[split]['recordings'].add(recording_id)

                    # Store sample keys for later inspection
                    if len(sample_keys[split]) < 3:
                        sample_keys[split].append(key_str)

                    stats[split]['total'] += 1

                except Exception as e:
                    errors.append(f"{split}: Error loading {key_str}: {e}")

            print(f"  Completed: {len(keys_list)} samples")

    env.close()

    # Print statistics
    print(f"\n{'='*80}")
    print("STATISTICS BY SPLIT")
    print(f"{'='*80}")

    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} Split:")
        print(f"{'-'*80}")
        print(f"  Total samples:        {stats[split]['total']}")
        print(f"  Unique patients:      {len(stats[split]['patients'])}")
        print(f"  Unique recordings:    {len(stats[split]['recordings'])}")
        print(f"  Oversampled samples:  {stats[split]['oversampled']} "
              f"({stats[split]['oversampled']/max(stats[split]['total'], 1)*100:.1f}%)")

        # Label distribution
        print(f"\n  Label distribution:")
        total_labeled = sum(stats[split]['label_counts'].values())
        for label in sorted(stats[split]['label_counts'].keys()):
            count = stats[split]['label_counts'][label]
            percentage = count / total_labeled * 100 if total_labeled > 0 else 0
            label_name = "Normal" if label == 0 else "Seizure"
            print(f"    Label {label} ({label_name}): {count:6d} ({percentage:5.1f}%)")

        if len(stats[split]['label_counts']) == 2:
            label_0 = stats[split]['label_counts'].get(0, 0)
            label_1 = stats[split]['label_counts'].get(1, 1)
            ratio = label_0 / label_1 if label_1 > 0 else float('inf')
            print(f"    Class imbalance ratio: {ratio:.1f}:1 (Normal:Seizure)")

        # Patients
        if len(stats[split]['patients']) > 0:
            patients_sorted = sorted(stats[split]['patients'])
            print(f"\n  Patients ({len(patients_sorted)}):")
            if len(patients_sorted) <= 20:
                print(f"    {', '.join(patients_sorted)}")
            else:
                print(f"    First 10: {', '.join(patients_sorted[:10])}")
                print(f"    Last 10:  {', '.join(patients_sorted[-10:])}")

        # Sample keys
        if len(sample_keys[split]) > 0:
            print(f"\n  Sample keys (first 3):")
            for key in sample_keys[split]:
                print(f"    {key}")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")

    print(f"\nFormat versions:")
    total_format_samples = sum(format_versions.values())
    for version, count in sorted(format_versions.items()):
        percentage = count / total_format_samples * 100 if total_format_samples > 0 else 0
        print(f"  {version}: {count} samples ({percentage:.1f}%)")
    if format_versions['v2'] > 0 and format_versions['v1'] > 0:
        print(f"  ⚠️  Mixed format detected! Consider migrating all to v2.")
    elif format_versions['v2'] > 0:
        print(f"  ✅ All samples in v2-keymodified format (ISRUC-compatible)")
    elif format_versions['v1'] > 0:
        print(f"  ℹ️  All samples in v1 format (consider migrating to v2)")

    print(f"\nData shapes:")
    for shape, count in sorted(shapes.items()):
        print(f"  {shape}: {count} samples")

    print(f"\nData types:")
    for dtype, count in sorted(dtypes.items()):
        print(f"  {dtype}: {count} samples")

    total_samples = sum(s['total'] for s in stats.values())
    total_patients = set()
    for s in stats.values():
        total_patients.update(s['patients'])

    print(f"\nTotal across all splits:")
    print(f"  Samples:  {total_samples}")
    print(f"  Patients: {len(total_patients)}")

    # Print errors
    if errors:
        print(f"\n{'!'*80}")
        print(f"ERRORS FOUND: {len(errors)}")
        print(f"{'!'*80}")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    else:
        print(f"\n{'✓'*80}")
        print("No errors found - data looks good!")
        print(f"{'✓'*80}")

    return stats, errors


def load_and_inspect_sample(lmdb_path, sample_key):
    """
    Load and inspect a specific sample in detail

    Args:
        lmdb_path: Path to LMDB directory
        sample_key: Key of sample to inspect
    """
    print(f"\n{'='*80}")
    print(f"Detailed inspection of sample: {sample_key}")
    print(f"{'='*80}")

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            value = txn.get(sample_key.encode('utf-8'))
            if value is None:
                print(f"[ERROR] Sample not found: {sample_key}")
                return

            data = pickle.loads(value)

            # Detect format version
            is_v2 = 'sample' in data and 'data_info' in data
            is_v1 = 'signal' in data and 'metadata' in data

            if is_v2:
                format_str = "v2-keymodified (ISRUC-compatible)"
                data_key = 'sample'
                meta_key = 'data_info'
            elif is_v1:
                format_str = "v1 (original)"
                data_key = 'signal'
                meta_key = 'metadata'
            else:
                format_str = "Unknown"
                data_key = 'signal' if 'signal' in data else 'sample'
                meta_key = 'metadata' if 'metadata' in data else 'data_info'

            print(f"\nData structure:")
            print(f"  Type: {type(data)}")
            print(f"  Format: {format_str}")
            print(f"  Keys: {list(data.keys())}")

            # Signal/Sample
            if data_key in data:
                signal = data[data_key]
                print(f"\n{data_key.capitalize()}:")
                print(f"  Shape:  {signal.shape}")
                print(f"  Dtype:  {signal.dtype}")
                print(f"  Min:    {signal.min():.3f}")
                print(f"  Max:    {signal.max():.3f}")
                print(f"  Mean:   {signal.mean():.3f}")
                print(f"  Std:    {signal.std():.3f}")

            # Label
            if 'label' in data:
                label = data['label']
                label_name = "Normal" if label == 0 else "Seizure"
                print(f"\nLabel: {label} ({label_name})")

            # ELC info (v1 only - in v2 it's merged into data_info)
            if 'elc_info' in data:
                elc_info = data['elc_info']
                print(f"\nElectrode information (v1 format):")
                print(f"  Channel names: {len(elc_info.get('channel_names', []))}")
                print(f"  Electrode pairs: {len(elc_info.get('electrode_pairs', {}))}")
                print(f"  Electrode positions: {len(elc_info.get('electrode_positions', {}))}")

                # Show first few channel pairs
                if 'electrode_pairs' in elc_info:
                    pairs = elc_info['electrode_pairs']
                    print(f"\n  First 3 channel pairs:")
                    for i, (ch_name, electrodes) in enumerate(list(pairs.items())[:3]):
                        print(f"    {ch_name}: {electrodes}")

            # Metadata/data_info
            if meta_key in data:
                metadata = data[meta_key]
                print(f"\n{meta_key.capitalize().replace('_', ' ')}:")

                # For v2, group fields by category
                if is_v2:
                    isruc_fields = ['Dataset', 'modality', 'release', 'subject_id', 'task',
                                   'resampling_rate', 'original_sampling_rate', 'segment_index', 'start_time']
                    electrode_fields = ['channel_names', 'electrode_pairs', 'electrode_positions']
                    chbmit_fields = ['segment_id', 'split', 'is_oversampled']

                    print(f"  ISRUC-compatible fields:")
                    for key in isruc_fields:
                        if key in metadata:
                            value = metadata[key]
                            if isinstance(value, (list, dict)):
                                print(f"    {key}: {type(value).__name__} (len={len(value)})")
                            else:
                                print(f"    {key}: {value}")

                    print(f"  Electrode information:")
                    for key in electrode_fields:
                        if key in metadata:
                            value = metadata[key]
                            if isinstance(value, (list, dict)):
                                print(f"    {key}: {type(value).__name__} (len={len(value)})")

                    print(f"  CHBMIT-specific fields:")
                    for key in chbmit_fields:
                        if key in metadata:
                            print(f"    {key}: {metadata[key]}")
                else:
                    # v1 format - just print all fields
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")

        env.close()

    except Exception as e:
        print(f"[ERROR] Failed to inspect sample: {e}")


def main():
    parser = argparse.ArgumentParser(description='Check CHB-MIT LMDB database')
    parser.add_argument('--lmdb_path', type=str,
                        default='/pscratch/sd/b/boheelee/DIVER/CHBMIT_preprocessing/lmdb_output/CHBMIT_Seizure',
                        help='Path to LMDB directory (single LMDB with __keys__)')
    parser.add_argument('--inspect_sample', type=str, default=None,
                        help='Inspect a specific sample key in detail')
    args = parser.parse_args()

    print("=" * 80)
    print("CHB-MIT LMDB Database Verification (Single LMDB)")
    print("=" * 80)
    print(f"LMDB path: {args.lmdb_path}")
    print("=" * 80)

    # Check database
    result = check_lmdb_database(args.lmdb_path)

    if result is None:
        sys.exit(1)

    stats, errors = result

    # Inspect specific sample if requested
    if args.inspect_sample:
        load_and_inspect_sample(args.lmdb_path, args.inspect_sample)

    print(f"\n{'='*80}")
    print("Verification complete!")
    print(f"{'='*80}")

    # Exit with error code if errors found
    if len(errors) > 0:
        print(f"\n[WARNING] Found {len(errors)} errors - please review!")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] All checks passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
