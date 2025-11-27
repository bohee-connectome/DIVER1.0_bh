#!/usr/bin/env python3
"""
CHB-MIT LMDB Migration Script: v1 -> v2-keymodified

This script migrates existing CHBMIT LMDB data from v1 format to v2-keymodified format
to match ISRUC-Sleep data dictionary structure.

Changes:
- 'signal' -> 'sample'
- 'metadata' -> 'data_info'
- 'elc_info' merged into 'data_info'
- Added ISRUC-compatible fields (Dataset, modality, task, etc.)
- Preserved CHBMIT-specific fields (segment_id, is_oversampled, split)

Usage:
    python migrate_chbmit_v1_to_v2.py --input_lmdb /path/to/old/CHBMIT_Seizure \
                                       --output_lmdb /path/to/new/CHBMIT_Seizure_v2

Author: Claude + User
Date: 2025-01-27
"""

import os
import lmdb
import pickle
import argparse
from tqdm import tqdm


def migrate_sample(old_data, segment_id):
    """
    Convert v1 sample format to v2-keymodified format

    V1 format:
    {
        'signal': array,
        'label': int,
        'elc_info': {...},
        'metadata': {
            'segment_id': str,
            'split': str,
            'is_oversampled': bool,
            'original_index': int,
            'original_sr': int,
            'target_sr': int
        }
    }

    V2 format (ISRUC-compatible):
    {
        'sample': array,
        'label': int,
        'data_info': {
            'Dataset': 'CHBMIT-Seizure',
            'modality': 'EEG',
            'release': '1.0.0',
            'subject_id': str,
            'task': 'seizure-detection',
            'resampling_rate': int,
            'original_sampling_rate': int,
            'segment_index': int,
            'start_time': float,
            'channel_names': list,
            'electrode_pairs': dict,
            'electrode_positions': dict,
            'segment_id': str,
            'split': str,
            'is_oversampled': bool
        }
    }
    """
    # Extract patient info from segment_id (e.g., 'chb01' from 'chb01_01_0')
    patient_name = segment_id.split('_')[0]

    # Get old metadata
    old_metadata = old_data.get('metadata', {})
    old_elc_info = old_data.get('elc_info', {})

    # Create new v2 format
    new_data = {
        'sample': old_data['signal'],  # Changed key name
        'label': old_data['label'],
        'data_info': {  # Changed from 'metadata'
            # ISRUC-compatible fields
            'Dataset': 'CHBMIT-Seizure',
            'modality': 'EEG',
            'release': '1.0.0',
            'subject_id': patient_name,
            'task': 'seizure-detection',
            'resampling_rate': old_metadata.get('target_sr', 500),
            'original_sampling_rate': old_metadata.get('original_sr', 256),
            'segment_index': old_metadata.get('original_index', 0),
            'start_time': old_metadata.get('original_index', 0) / old_metadata.get('original_sr', 256),
            'channel_names': old_elc_info.get('channel_names', []),
            'electrode_pairs': old_elc_info.get('electrode_pairs', {}),
            'electrode_positions': old_elc_info.get('electrode_positions', {}),
            # CHBMIT-specific fields (preserved)
            'segment_id': old_metadata.get('segment_id', segment_id),
            'split': old_metadata.get('split', 'unknown'),
            'is_oversampled': old_metadata.get('is_oversampled', False),
        }
    }

    return new_data


def migrate_lmdb(input_path, output_path, dry_run=False):
    """
    Migrate entire LMDB database from v1 to v2 format

    Args:
        input_path: Path to v1 LMDB database
        output_path: Path to save v2 LMDB database
        dry_run: If True, only show statistics without writing
    """
    print("=" * 80)
    print("CHB-MIT LMDB Migration: v1 -> v2-keymodified")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Mode:   {'DRY RUN (no write)' if dry_run else 'WRITE'}")
    print("=" * 80)

    # Open input LMDB
    print("\n[1/4] Opening input LMDB...")
    try:
        input_env = lmdb.open(input_path, readonly=True, lock=False)
    except Exception as e:
        print(f"[ERROR] Failed to open input LMDB: {e}")
        return False

    # Get total entries and __keys__
    print("\n[2/4] Reading database structure...")
    with input_env.begin() as txn:
        total_entries = txn.stat()['entries']
        print(f"  Total entries: {total_entries}")

        # Load __keys__
        keys_data = txn.get('__keys__'.encode())
        if keys_data is None:
            print("[ERROR] __keys__ not found in input LMDB!")
            input_env.close()
            return False

        dataset = pickle.loads(keys_data)
        print(f"  Train samples: {len(dataset.get('train', []))}")
        print(f"  Val samples:   {len(dataset.get('val', []))}")
        print(f"  Test samples:  {len(dataset.get('test', []))}")

    if dry_run:
        print("\n[DRY RUN] Would migrate the following:")
        print(f"  Total samples to migrate: {sum(len(v) for v in dataset.values())}")
        input_env.close()
        return True

    # Create output LMDB
    print("\n[3/4] Creating output LMDB...")
    os.makedirs(output_path, exist_ok=True)
    output_env = lmdb.open(output_path, map_size=200 * 1024**3)  # 200GB

    # Migrate all samples
    print("\n[4/4] Migrating samples...")
    migration_stats = {
        'train': {'success': 0, 'error': 0},
        'val': {'success': 0, 'error': 0},
        'test': {'success': 0, 'error': 0}
    }

    with input_env.begin() as input_txn:
        with output_env.begin(write=True) as output_txn:
            for split in ['train', 'val', 'test']:
                keys_list = dataset.get(split, [])
                print(f"\n  Migrating {split.upper()} split ({len(keys_list)} samples)...")

                for key_str in tqdm(keys_list, desc=f"  {split}"):
                    try:
                        # Read old format
                        key = key_str.encode('utf-8')
                        old_value = input_txn.get(key)

                        if old_value is None:
                            print(f"    [WARNING] Key not found: {key_str}")
                            migration_stats[split]['error'] += 1
                            continue

                        old_data = pickle.loads(old_value)

                        # Convert to new format
                        new_data = migrate_sample(old_data, key_str)

                        # Write to output LMDB
                        output_txn.put(key, pickle.dumps(new_data))

                        migration_stats[split]['success'] += 1

                    except Exception as e:
                        print(f"    [ERROR] Failed to migrate {key_str}: {e}")
                        migration_stats[split]['error'] += 1

            # Write __keys__ to output LMDB
            print("\n  Writing __keys__ to output LMDB...")
            output_txn.put('__keys__'.encode(), pickle.dumps(dataset))

    # Close databases
    input_env.close()
    output_env.close()

    # Print migration summary
    print("\n" + "=" * 80)
    print("MIGRATION SUMMARY")
    print("=" * 80)

    total_success = 0
    total_error = 0

    for split in ['train', 'val', 'test']:
        success = migration_stats[split]['success']
        error = migration_stats[split]['error']
        total_success += success
        total_error += error

        print(f"\n{split.upper()} split:")
        print(f"  Success: {success}")
        print(f"  Errors:  {error}")

    print(f"\nTOTAL:")
    print(f"  Success: {total_success}")
    print(f"  Errors:  {total_error}")

    if total_error == 0:
        print("\n✅ Migration completed successfully!")
        print(f"   Output saved to: {output_path}")
    else:
        print("\n⚠️  Migration completed with errors!")
        print(f"   Please review the errors above.")

    print("=" * 80)

    return total_error == 0


def main():
    parser = argparse.ArgumentParser(description='Migrate CHB-MIT LMDB from v1 to v2-keymodified')
    parser.add_argument('--input_lmdb', type=str, required=True,
                        help='Path to input LMDB (v1 format)')
    parser.add_argument('--output_lmdb', type=str, required=True,
                        help='Path to output LMDB (v2 format)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be migrated without writing')
    args = parser.parse_args()

    success = migrate_lmdb(args.input_lmdb, args.output_lmdb, args.dry_run)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
