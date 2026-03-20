import h5py
import numpy as np
from pathlib import Path

def verify_ms_to_idx(
    events_h5: Path = Path('data/eds/events.h5'),
    ms_to_idx_h5: Path = Path('data/eds/ms_to_idx.h5'),
    t_key: str = 't',
    idx_key: str = 'ms_to_idx'
) -> None:
    with h5py.File(events_h5, 'r') as f:
        if t_key not in f:
            raise KeyError(f"Timestamp key '{t_key}' not found in {events_h5}")
        t = f[t_key][:]

    with h5py.File(ms_to_idx_h5, 'r') as f:
        available_keys = list(f.keys())
        if idx_key not in f:
            raise KeyError(
                f"Dataset '{idx_key}' not found in {ms_to_idx_h5}. "
                f"Available keys: {available_keys}"
            )
        ms_to_idx = f[idx_key][:]

    print(f"Loaded {len(t)} events from:       {events_h5}")
    print(f"Loaded ms_to_idx (len={len(ms_to_idx)}) from: {ms_to_idx_h5}")
    print(f"Time range: {t[0]} µs → {t[-1]} µs ({(t[-1]-t[0])/1e6:.3f} s)")
    print()

    # Match processing.py: work in relative time
    t_relative = t - t[0]

    # Property (3)
    assert ms_to_idx[0] == 0, \
        f"FAIL property (3): ms_to_idx[0] = {ms_to_idx[0]}, expected 0"
    print("PASS property (3): ms_to_idx[0] == 0")

    # Properties (1) and (2)
    sample_ms = np.linspace(1, len(ms_to_idx) - 1, min(1000, len(ms_to_idx) - 1), dtype=int)
    for ms in sample_ms:
        idx = ms_to_idx[ms]
        boundary_us = ms * 1000  # relative boundary

        assert t_relative[idx] >= boundary_us, (
            f"FAIL property (1) at ms={ms}: "
            f"t_rel[{idx}]={t_relative[idx]} µs < {boundary_us} µs"
        )
        if idx > 0:
            assert t_relative[idx - 1] < boundary_us, (
                f"FAIL property (2) at ms={ms}: "
                f"t_rel[{idx-1}]={t_relative[idx-1]} µs >= {boundary_us} µs (not tight)"
            )

    print("PASS property (1): t_rel[ms_to_idx[ms]] >= ms * 1000 for all sampled ms")
    print("PASS property (2): t_rel[ms_to_idx[ms] - 1] < ms * 1000 for all sampled ms")

    # Sanity: non-decreasing
    assert np.all(np.diff(ms_to_idx) >= 0), \
        "FAIL sanity check: ms_to_idx is not non-decreasing!"
    print("PASS sanity check: ms_to_idx is non-decreasing")

    # Sanity: indices within bounds
    assert ms_to_idx[-1] <= len(t), \
        f"FAIL sanity check: ms_to_idx[-1]={ms_to_idx[-1]} > n_events={len(t)}"
    print("PASS sanity check: all indices within valid event range")

    print()
    print("✅ All checks passed. ms_to_idx is correct.")


if __name__ == "__main__":
    verify_ms_to_idx()
