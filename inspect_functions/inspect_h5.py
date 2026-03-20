import h5py

def inspect_h5(path):
    with h5py.File(path, 'r') as f:
        print(f"=== Structure of {path} ===")
        f.visititems(lambda name, obj: print(f"  {'[Dataset]' if isinstance(obj, h5py.Dataset) else '[Group]  '} {name} {getattr(obj, 'shape', '')} {getattr(obj, 'dtype', '')}"))

inspect_h5('data/eds/events.h5')
inspect_h5('data/eds/ms_to_idx.h5')
