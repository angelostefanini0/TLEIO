import h5py
import numpy as np

path = "data/eds/processed/peanuts/events.h5"
with h5py.File(path, "r") as f:
    t = f["events/t"][...]
    t=np.asarray(t, dtype=np.int64)
    print(t[0])