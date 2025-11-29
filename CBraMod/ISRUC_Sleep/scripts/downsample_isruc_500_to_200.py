#!/usr/bin/env python3
"""
Downsample ISRUC LMDB from 500Hz to 200Hz

Reads existing 500Hz LMDB and creates new 200Hz version.
Only changes: sample shape (6,30,500)→(6,30,200) and resampling_rate 500→200
Everything else preserved.

Created for DIVER 1.0 project
Author: Bohee Lee
Date: 2025-01-29
"""

import lmdb
import pickle
import numpy as np
from scipy.signal import resample
from pathlib import Path
import argparse
from tqdm import tqdm


def downsample_signal(signal_500hz, target_samples=200):
    """Downsample from (6, 30, 500) to (6, 30, 200)"""
    n_channels, n_segments, _ = signal_500hz.shape
    signal_200hz = np.zeros((n_channels, n_segments, target_samples), dtype=np.float32)

    for ch in range(n_channels):
        for seg in range(n_segments):
            signal_200hz[ch, seg, :] = resample(signal_500hz[ch, seg, :], target_samples)

    return signal_200hz


def check_lmdb_structure(lmdb_path, num_samples=3):
    """Dry run: Check LMDB structure before downsampling"""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)

    print("\n" + "=" * 70)
    print("DRY RUN: Checking LMDB structure")
    print("=" * 70)

    with env.begin() as txn:
        total = txn.stat()['entries']
        print(f"Total entries: {total:,}")

        cursor = txn.cursor()
        checked = 0

        for key, value in cursor:
            if key == b'__keys__':
                print(f"\n✓ Found __keys__ (will be copied as-is, no changes)")
                continue

            if checked >= num_samples:
                break

            data = pickle.loads(value)

            print(f"\n--- Sample {checked + 1}: {key.decode()} ---")
            print(f"Top-level keys: {list(data.keys())}")
            print(f"")
            print(f"WILL CHANGE:")
            print(f"  sample.shape:  {data['sample'].shape} → (6, 30, 200)")
            print(f"  resampling_rate: {data['data_info']['resampling_rate']} → 200")
            print(f"")
            print(f"WILL PRESERVE:")
            print(f"  label: {data['label']}")
            print(f"  data_info keys: {list(data['data_info'].keys())}")
            for key_name in data['data_info']:
                if key_name != 'resampling_rate':
                    print(f"    {key_name}: {data['data_info'][key_name]}")

            checked += 1

    env.close()
    print("\n" + "=" * 70)
    print("✓ Only 2 things will change:")
    print("  1. sample shape: (6, 30, 500) → (6, 30, 200)")
    print("  2. resampling_rate: 500 → 200")
    print("✓ Everything else preserved exactly")
    print("=" * 70)


def downsample_lmdb(input_lmdb_path, output_lmdb_path, batch_size=1000):
    """Downsample entire LMDB from 500Hz to 200Hz"""
    input_path = Path(input_lmdb_path)
    output_path = Path(output_lmdb_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input LMDB not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Downsampling: 500Hz → 200Hz")
    print(f"Shape: (6, 30, 500) → (6, 30, 200)\n")

    # Open input LMDB
    input_env = lmdb.open(str(input_path), readonly=True, lock=False)

    with input_env.begin() as txn:
        total_samples = txn.stat()['entries']

    print(f"Total entries: {total_samples:,}\n")

    # Calculate output size
    input_size = sum(f.stat().st_size for f in input_path.glob('*'))
    estimated_output_size = int(input_size * 0.4 * 1.2)  # 40% + 20% buffer

    # Open output LMDB
    output_env = lmdb.open(str(output_path), map_size=estimated_output_size)

    processed = 0
    dataset_keys = None

    with input_env.begin() as input_txn:
        cursor = input_txn.cursor()
        output_txn = output_env.begin(write=True)

        try:
            for key, value in tqdm(cursor, total=total_samples, desc="Downsampling"):
                # Copy __keys__ as-is
                if key == b'__keys__':
                    dataset_keys = value
                    continue

                # Load sample
                data = pickle.loads(value)

                # Downsample signal
                signal_500 = data['sample']  # (6, 30, 500)
                signal_200 = downsample_signal(signal_500, target_samples=200)

                # Update
                data['sample'] = signal_200
                data['data_info']['resampling_rate'] = 200

                # Save
                output_txn.put(key, pickle.dumps(data))
                processed += 1

                # Batch commit
                if processed % batch_size == 0:
                    output_txn.commit()
                    output_txn = output_env.begin(write=True)

            # Save __keys__
            if dataset_keys is not None:
                output_txn.put(b'__keys__', dataset_keys)

            output_txn.commit()

        except Exception as e:
            output_txn.abort()
            raise e

    input_env.close()
    output_env.close()

    print("\n" + "=" * 70)
    print("✅ Downsampling complete!")
    print(f"Processed: {processed:,} samples")
    print(f"Output: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Downsample ISRUC LMDB 500Hz → 200Hz")
    parser.add_argument('--input_lmdb', type=str, required=True, help='Input 500Hz LMDB path')
    parser.add_argument('--output_lmdb', type=str, help='Output 200Hz LMDB path (required for actual run)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size (default: 1000)')
    parser.add_argument('--dry_run', action='store_true', help='Check structure only, do not downsample')
    parser.add_argument('--check_samples', type=int, default=3, help='Number of samples to check in dry run')

    args = parser.parse_args()

    if args.dry_run:
        # Dry run: just check structure
        check_lmdb_structure(args.input_lmdb, args.check_samples)
    else:
        # Real run: downsample
        if not args.output_lmdb:
            parser.error("--output_lmdb is required when not using --dry_run")
        downsample_lmdb(args.input_lmdb, args.output_lmdb, args.batch_size)


if __name__ == "__main__":
    main()
