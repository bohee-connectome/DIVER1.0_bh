#!/usr/bin/env python3
"""
Validate xyz_id in CHBMIT LMDB

Check if xyz_id is correctly aligned with channel_names.

Usage:
    python validate_xyz_id.py --lmdb_path /path/to/CHBMIT_Seizure_v2_xyz
"""

import lmdb
import pickle
import argparse
import numpy as np


def validate_xyz_id(lmdb_path):
    """
    Validate xyz_id alignment with channel_names
    """
    print("=" * 80)
    print("Validate xyz_id in CHBMIT LMDB")
    print("=" * 80)
    print(f"LMDB path: {lmdb_path}\n")

    # Open LMDB
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        # Load __keys__
        keys_data = txn.get('__keys__'.encode())
        if keys_data is None:
            print("[ERROR] __keys__ not found!")
            env.close()
            return

        dataset = pickle.loads(keys_data)

        # Get first train sample
        first_key = dataset['train'][0]
        print(f"Inspecting sample: {first_key}\n")

        # Load sample
        sample_data = txn.get(first_key.encode())
        data = pickle.loads(sample_data)

        # Check data structure
        data_info = data['data_info']

        # Get fields
        channel_names = data_info.get('channel_names', [])
        electrode_positions = data_info.get('electrode_positions', {})
        xyz_id = data_info.get('xyz_id', None)

        print("=" * 80)
        print("CHANNEL NAMES")
        print("=" * 80)
        print(f"Total: {len(channel_names)} channels\n")
        for i, ch in enumerate(channel_names):
            # Extract first electrode
            first_elec = ch.split('-')[0].strip()
            print(f"  {i:2d}. {ch:15s} → First electrode: {first_elec}")

        print("\n" + "=" * 80)
        print("XYZ_ID ARRAY")
        print("=" * 80)

        if xyz_id is None:
            print("❌ xyz_id field NOT FOUND in data_info!")
            env.close()
            return

        print(f"Shape: {xyz_id.shape}")
        print(f"Dtype: {xyz_id.dtype}\n")

        if xyz_id.shape[0] != len(channel_names):
            print(f"⚠️  WARNING: xyz_id has {xyz_id.shape[0]} rows but channel_names has {len(channel_names)} channels!")

        print("xyz_id contents:")
        for i, pos in enumerate(xyz_id):
            if i < len(channel_names):
                ch_name = channel_names[i]
                first_elec = ch_name.split('-')[0].strip()
                print(f"  {i:2d}. {ch_name:15s} ({first_elec:4s}) → {pos}")

        print("\n" + "=" * 80)
        print("VALIDATION CHECK")
        print("=" * 80)

        all_match = True
        for i, ch_name in enumerate(channel_names):
            first_elec = ch_name.split('-')[0].strip()

            # Get position from xyz_id
            xyz_pos = xyz_id[i]

            # Get position from electrode_positions
            if first_elec in electrode_positions:
                elec_pos = electrode_positions[first_elec]

                # Check if they match
                if np.allclose(xyz_pos, elec_pos, atol=1e-6):
                    status = "✓"
                else:
                    status = "✗ MISMATCH"
                    all_match = False

                print(f"  {i:2d}. {ch_name:15s} ({first_elec:4s}): {status}")
                if status == "✗ MISMATCH":
                    print(f"      xyz_id:             {xyz_pos}")
                    print(f"      electrode_positions: {elec_pos}")
            else:
                print(f"  {i:2d}. {ch_name:15s} ({first_elec:4s}): ✗ NOT FOUND in electrode_positions")
                all_match = False

        print("\n" + "=" * 80)
        if all_match and xyz_id.shape == (len(channel_names), 3):
            print("✅ xyz_id is correctly aligned with channel_names!")
        else:
            print("⚠️  xyz_id has alignment issues!")
        print("=" * 80)

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Validate xyz_id in CHBMIT LMDB')
    parser.add_argument('--lmdb_path', type=str, required=True,
                        help='Path to CHBMIT LMDB with xyz_id')
    args = parser.parse_args()

    validate_xyz_id(args.lmdb_path)


if __name__ == '__main__':
    main()
