import lmdb
import pickle
from collections import Counter

lmdb_path = '/scratch/connectome/bohee/DIVER_ADFTD/data/processed_v2/1.0_ADFTD/merged_resample-500_highpass-0.5_lowpass-45.0.lmdb'
env = lmdb.open(lmdb_path, readonly=True, lock=False)

shape_counter = Counter()
marker_counter = Counter()
total = 0

print("Scanning v2 LMDB for shapes and markers...")
print("=" * 60)

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        data = pickle.loads(value)
        
        if 'sample' in data:
            shape = tuple(data['sample'].shape)
            shape_counter[shape] += 1
        
        # v2는 data_info에 marker가 없을 수도 있으므로 직접 shape로 판단
        total += 1
        
        if total % 50000 == 0:
            print(f"Processed {total} samples...")

env.close()

print("=" * 60)
print(f"\nTotal samples: {total}")
print(f"\nShape distribution:")
for shape, count in sorted(shape_counter.items()):
    percentage = count / total * 100
    print(f"  {shape}: {count:6d} samples ({percentage:5.2f}%)")

print(f"\nExpected v2 shapes:")
print(f"  (19, 1, 500): 500Hz - 1sec windows")
print(f"  (19, 2, 250): 250Hz - 2sec windows")
print(f"  (19, 4, 125): 125Hz - 4sec windows")

# Verify distribution
expected = {(19, 1, 500), (19, 2, 250), (19, 4, 125)}
actual = set(shape_counter.keys())

if actual == expected:
    print(f"\n[SUCCESS] All three shapes present!")
else:
    print(f"\n[WARNING] Shape mismatch!")
    print(f"  Expected: {expected}")
    print(f"  Actual: {actual}")
