#!/usr/bin/env python3
"""
Add xyz_id to CHBMIT v2 LMDB

This script adds a new 'xyz_id' field to data_info that contains
electrode positions ordered by channel_names (first electrode of each bipolar pair).

Usage:
    python add_xyz_id_to_lmdb.py \
        --input_lmdb /path/to/CHBMIT_Seizure_v2 \
        --output_lmdb /path/to/CHBMIT_Seizure_v2_xyz

Author: Claude + User
Date: 2025-01-27
"""

import os
import lmdb
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def extract_first_electrode(channel_name):
    """
    Extract first electrode from bipolar channel name

    Args:
        channel_name: Bipolar channel (e.g., "FP1-F7")

    Returns:
        str: First electrode name (e.g., "FP1")
    """
    parts = channel_name.split('-')
    if len(parts) >= 1:
        return parts[0].strip()
    else:
        return channel_name


def create_xyz_id(channel_names, electrode_positions):
    """
    Create xyz_id array from channel_names and electrode_positions

    Args:
        channel_names: List of bipolar channel names
        electrode_positions: Dict of {electrode_name: position_array}

    Returns:
        np.ndarray: (N_channels, 3) array of positions
    """
    xyz_list = []

    for ch_name in channel_names:
        # Get first electrode from bipolar pair
        first_elec = extract_first_electrode(ch_name)

        # Look up position
        if first_elec in electrode_positions:
            pos = electrode_positions[first_elec]
            xyz_list.append(pos)
        else:
            # If not found, use NaN
            print(f"[WARNING] Electrode {first_elec} not found in electrode_positions")
            xyz_list.append(np.full(3, np.nan))

    return np.array(xyz_list)  # shape (N_channels, 3)


def add_xyz_id_to_lmdb(input_path, output_path):
    """
    Add xyz_id field to all samples in LMDB

    Args:
        input_path: Input LMDB path
        output_path: Output LMDB path
    """
    print("=" * 80)
    print("Add xyz_id to CHBMIT v2 LMDB")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Open input LMDB
    print("\n[1/3] Opening input LMDB...")
    input_env = lmdb.open(input_path, readonly=True, lock=False)

    with input_env.begin() as txn:
        # Load __keys__
        keys_data = txn.get('__keys__'.encode())
        if keys_data is None:
            print("[ERROR] __keys__ not found!")
            input_env.close()
            return False

        dataset = pickle.loads(keys_data)
        total_samples = sum(len(v) for v in dataset.values())
        print(f"  Total samples: {total_samples}")

    # Create output LMDB
    print("\n[2/3] Creating output LMDB...")
    os.makedirs(output_path, exist_ok=True)
    output_env = lmdb.open(output_path, map_size=200 * 1024**3)  # 200GB

    # Process all samples
    print("\n[3/3] Processing samples and adding xyz_id...")

    stats = {
        'success': 0,
        'error': 0,
        'xyz_id_shapes': []
    }

    # Use batch commits to reduce memory
    BATCH_SIZE = 1000

    with input_env.begin() as input_txn:
        for split in ['train', 'val', 'test']:
            keys_list = dataset.get(split, [])
            if len(keys_list) == 0:
                continue

            print(f"\n  Processing {split.upper()} split ({len(keys_list)} samples)...")

            # Process in batches
            for batch_start in range(0, len(keys_list), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(keys_list))
                batch_keys = keys_list[batch_start:batch_end]

                with output_env.begin(write=True) as output_txn:
                    for key_str in tqdm(batch_keys, desc=f"  {split} [{batch_start:>6d}-{batch_end:>6d}]", leave=False):
                        try:
                            # Read old data
                            key = key_str.encode('utf-8')
                            old_value = input_txn.get(key)

                            if old_value is None:
                                print(f"    [WARNING] Key not found: {key_str}")
                                stats['error'] += 1
                                continue

                            data = pickle.loads(old_value)

                            # Extract channel_names and electrode_positions
                            data_info = data['data_info']
                            channel_names = data_info.get('channel_names', [])
                            electrode_positions = data_info.get('electrode_positions', {})

                            # Create xyz_id
                            xyz_id = create_xyz_id(channel_names, electrode_positions)

                            # Add xyz_id to data_info
                            data_info['xyz_id'] = xyz_id

                            # Write to output
                            output_txn.put(key, pickle.dumps(data))

                            stats['success'] += 1
                            stats['xyz_id_shapes'].append(xyz_id.shape)

                        except Exception as e:
                            print(f"    [ERROR] Failed to process {key_str}: {e}")
                            stats['error'] += 1

                # Free memory
                import gc
                gc.collect()

    # Write __keys__ to output LMDB
    print("\n  Writing __keys__ to output LMDB...")
    with output_env.begin(write=True) as output_txn:
        output_txn.put('__keys__'.encode(), pickle.dumps(dataset))

    # Close databases
    input_env.close()
    output_env.close()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Success: {stats['success']}")
    print(f"Errors:  {stats['error']}")

    if len(stats['xyz_id_shapes']) > 0:
        unique_shapes = set(stats['xyz_id_shapes'])
        print(f"\nxyz_id shapes found: {unique_shapes}")

        # Verify all are (16, 3)
        if unique_shapes == {(16, 3)}:
            print("✅ All xyz_id arrays have correct shape (16, 3)")
        else:
            print("⚠️  Some xyz_id arrays have unexpected shapes!")

    if stats['error'] == 0:
        print(f"\n✅ Processing completed successfully!")
        print(f"   Output saved to: {output_path}")
    else:
        print(f"\n⚠️  Processing completed with {stats['error']} errors!")

    print("=" * 80)

    return stats['error'] == 0


def main():
    parser = argparse.ArgumentParser(description='Add xyz_id to CHBMIT v2 LMDB')
    parser.add_argument('--input_lmdb', type=str, required=True,
                        help='Input LMDB path (v2 format)')
    parser.add_argument('--output_lmdb', type=str, required=True,
                        help='Output LMDB path (with xyz_id)')
    args = parser.parse_args()

    success = add_xyz_id_to_lmdb(args.input_lmdb, args.output_lmdb)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
