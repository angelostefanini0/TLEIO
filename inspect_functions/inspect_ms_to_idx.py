import h5py
import numpy as np

def inspect_ms_to_idx(path='data/eds/ms_to_idx.h5'):
    with h5py.File(path, 'r') as f:
        print(f"=== Datasets in {path} ===")
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\n  [{name}]")
                print(f"    shape : {obj.shape}")
                print(f"    dtype : {obj.dtype}")
                print(f"    first 10 values: {obj[:10].tolist()}")
                print(f"    last  10 values: {obj[-10:].tolist()}")
        f.visititems(visitor)

if __name__ == "__main__":
    inspect_ms_to_idx()
