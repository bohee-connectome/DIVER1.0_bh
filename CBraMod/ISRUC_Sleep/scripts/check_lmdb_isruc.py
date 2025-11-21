import lmdb
import pickle
import numpy as np
from collections import Counter

lmdb_path = '/pscratch/sd/b/boheelee/DIVER/ISRUC_preprocessing/lmdb_output/ISRUC_Sleep'

print("="*70)
print("ISRUC-Sleep LMDB Complete Verification")
print("="*70)

env = lmdb.open(lmdb_path, readonly=True)
with env.begin() as txn:
    # Get dataset keys
    keys_data = txn.get('__keys__'.encode())
    if not keys_data:
        print("\n❌ No dataset keys found in LMDB!")
        env.close()
        exit(1)

    dataset = pickle.loads(keys_data)

    # ============================================================
    # 1. Dataset Split Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("[1] Dataset Split Summary")
    print(f"{'='*70}")
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Val samples:   {len(dataset['val']):,}")
    print(f"  Test samples:  {len(dataset['test']):,}")
    print(f"  Total:         {sum(len(v) for v in dataset.values()):,}")

    # Expected values
    expected_total = 89283
    actual_total = sum(len(v) for v in dataset.values())
    if actual_total == expected_total:
        print(f"  ✅ Total matches expected: {expected_total:,}")
    else:
        print(f"  ⚠️  Total: {actual_total:,}, Expected: {expected_total:,}")

    # ============================================================
    # 2. Sample Structure Verification
    # ============================================================
    print(f"\n{'='*70}")
    print("[2] Sample Structure Verification")
    print(f"{'='*70}")

    for split in ['train', 'val', 'test']:
        if dataset[split]:
            first_key = dataset[split][0]
            sample_data = pickle.loads(txn.get(first_key.encode()))

            print(f"\n{split.upper()} - First Sample:")
            print(f"  Key: {first_key}")
            print(f"  Sample shape: {sample_data['sample'].shape}")
            print(f"  Label: {sample_data['label']}")
            print(f"  Subject: {sample_data['data_info']['subject_id']}")
            print(f"  Task: {sample_data['data_info']['task']}")
            print(f"  Channels: {sample_data['data_info']['channel_names']}")
            print(f"  Resampling rate: {sample_data['data_info']['resampling_rate']} Hz")
            print(f"  Original SR: {sample_data['data_info']['original_sampling_rate']} Hz")

            # Shape validation
            expected_shape = (6, 30, 500)
            if sample_data['sample'].shape == expected_shape:
                print(f"  ✅ Shape correct: {expected_shape}")
            else:
                print(f"  ❌ Shape mismatch! Expected {expected_shape}, got {sample_data['sample'].shape}")

    # ============================================================
    # 3. Complete Label Distribution (ALL samples)
    # ============================================================
    print(f"\n{'='*70}")
    print("[3] Complete Label Distribution (ALL Samples)")
    print(f"{'='*70}")

    all_labels_combined = []

    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} split:")
        labels = []

        # Check ALL samples in this split
        print(f"  Checking {len(dataset[split]):,} samples...", end=' ')
        for key in dataset[split]:
            data = pickle.loads(txn.get(key.encode()))
            labels.append(data['label'])
        print("Done!")

        all_labels_combined.extend(labels)
        label_counts = Counter(labels)
        total = len(labels)

        print(f"  Total samples: {total:,}")
        print(f"  Label distribution:")

        stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'REM(old)'}
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percent = count / total * 100
            stage_name = stage_names.get(label, 'Unknown')
            print(f"    {label} ({stage_name:8s}): {count:7,} ({percent:5.2f}%)")

        # Split-specific validation
        if 5 in label_counts:
            print(f"  ❌ ERROR: Label 5 found! REM not mapped to 4")
        elif 4 in label_counts:
            print(f"  ✅ Label 4 (REM) found - correctly mapped")
        else:
            print(f"  ⚠️  WARNING: No REM (label 4) in this split")

    # ============================================================
    # 4. Overall Label Validation
    # ============================================================
    print(f"\n{'='*70}")
    print("[4] Overall Label Validation")
    print(f"{'='*70}")

    all_labels_set = set(all_labels_combined)
    all_label_counts = Counter(all_labels_combined)

    print(f"\nAll unique labels found: {sorted(all_labels_set)}")

    expected_labels = {0, 1, 2, 3, 4}

    # Check for label 5 (should NOT exist)
    if 5 in all_labels_set:
        print(f"❌ CRITICAL ERROR: Label 5 found in dataset!")
        print(f"   REM was not properly mapped from 5 to 4")
    else:
        print(f"✅ Label 5 not found (REM correctly mapped)")

    # Check for label 4 (should exist)
    if 4 in all_labels_set:
        print(f"✅ Label 4 (REM) exists in dataset")
    else:
        print(f"❌ ERROR: Label 4 (REM) not found!")

    # Check if all expected labels exist
    if all_labels_set == expected_labels:
        print(f"✅ All expected labels present: {sorted(expected_labels)}")
    else:
        missing = expected_labels - all_labels_set
        extra = all_labels_set - expected_labels
        if missing:
            print(f"⚠️  Missing labels: {sorted(missing)}")
        if extra:
            print(f"⚠️  Unexpected labels: {sorted(extra)}")

    # ============================================================
    # 5. Processing Parameters Verification
    # ============================================================
    print(f"\n{'='*70}")
    print("[5] Processing Parameters Verification")
    print(f"{'='*70}")

    # Check a sample for processing parameters
    sample_key = dataset['train'][0] if dataset['train'] else dataset['val'][0]
    sample = pickle.loads(txn.get(sample_key.encode()))

    print(f"\nFilter & Processing Settings:")
    print(f"  Channels: {sample['data_info']['channel_names']}")
    print(f"  Expected: ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']")
    if sample['data_info']['channel_names'] == ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']:
        print(f"  ✅ Channels correct")

    print(f"\n  Original SR: {sample['data_info']['original_sampling_rate']} Hz")
    print(f"  Resampling SR: {sample['data_info']['resampling_rate']} Hz")
    print(f"  Expected: 200 Hz → 500 Hz")
    if sample['data_info']['original_sampling_rate'] == 200 and sample['data_info']['resampling_rate'] == 500:
        print(f"  ✅ Sampling rates correct")

    print(f"\n  Filter: 0.3-35 Hz bandpass, 50 Hz notch (from processing log)")
    print(f"  Output shape: {sample['sample'].shape}")
    print(f"  Expected: (6, 30, 500)")
    if sample['sample'].shape == (6, 30, 500):
        print(f"  ✅ Output shape correct")

    # ============================================================
    # 6. Final Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("[6] Final Summary")
    print(f"{'='*70}")

    total_samples = sum(len(v) for v in dataset.values())
    all_checks_passed = True

    checks = {
        "Total samples": total_samples == 89283,
        "Train samples": len(dataset['train']) > 70000,
        "Val samples": len(dataset['val']) > 9000,
        "Test samples": len(dataset['test']) > 8000,
        "Sample shape": sample['sample'].shape == (6, 30, 500),
        "Labels (0-4)": all_labels_set == {0, 1, 2, 3, 4},
        "No label 5": 5 not in all_labels_set,
        "REM mapped to 4": 4 in all_labels_set,
        "Sampling rate": sample['data_info']['resampling_rate'] == 500,
    }

    print(f"\nValidation Results:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_checks_passed = False

    print(f"\n{'='*70}")
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED! Dataset is ready for training.")
    else:
        print("⚠️  SOME CHECKS FAILED! Please review the results above.")
    print(f"{'='*70}")

env.close()
