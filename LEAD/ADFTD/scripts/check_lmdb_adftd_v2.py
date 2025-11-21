import lmdb
import pickle

lmdb_path = '/scratch/connectome/bohee/DIVER_ADFTD/data/processed_v2/1.0_ADFTD/merged_resample-500_highpass-0.5_lowpass-45.0.lmdb'
env = lmdb.open(lmdb_path, readonly=True, lock=False)

with env.begin() as txn:
    cursor = txn.cursor()
    key, value = next(cursor.iternext())
    data = pickle.loads(value)
    
    print(f"First key: {key.decode('utf-8')}")
    print(f"Data type: {type(data)}")
    print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    print("")
    
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape = {v.shape}, dtype = {v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")

env.close()
