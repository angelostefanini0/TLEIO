# local data folder

import os
import json
import glob
import numpy as np
import pandas as pd
import argparse
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

NPY_TS_OFFSET_US = 0.0

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_eds_imu_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()

    skip = 1 if ("timestamp" in first.lower() and "," not in first) else 0

    df = pd.read_csv(
        path,
        sep=",",
        engine="python",
        skipinitialspace=True,
        header=None,
        skiprows=skip,
        names=["ts_ns", "gx", "gy", "gz", "ax", "ay", "az"],
    )

    for c in ["ts_ns", "gx", "gy", "gz", "ax", "ay", "az"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts_ns", "gx", "gy", "gz", "ax", "ay", "az"]).copy()
    df["ts_ns"] = df["ts_ns"].astype(np.int64)
    df = df.sort_values("ts_ns").drop_duplicates("ts_ns").reset_index(drop=True)

    return df

def read_eds_gt_txt(path: str) -> np.ndarray:
    gt = np.loadtxt(path)

    if gt.ndim != 2 or gt.shape[1] < 8:
        raise ValueError("GT file must have at least 8 columns: t_s, p(3), q(4)")

    gt = gt[:, :8].astype(np.float64)

    t = gt[:, 0]
    keep = np.concatenate([[True], np.diff(t) > 0])
    gt = gt[keep]

    return gt

def make_uniform_timebase(t_start_s: float, t_end_s: float, freq_hz: float) -> np.ndarray:
    dt = 1.0 / float(freq_hz)
    n = int(np.floor((t_end_s - t_start_s) / dt)) + 1
    return t_start_s + np.arange(n, dtype=np.float64) * dt

def compute_velocity(pos: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(pos, dtype=np.float64)

    if len(t_s) < 3:
        if len(t_s) >= 2:
            vel[1:] = np.diff(pos, axis=0) / np.diff(t_s)[:, None]
            vel[0] = vel[1]
        return vel

    vel[1:-1] = (pos[2:] - pos[:-2]) / (t_s[2:] - t_s[:-2])[:, None]
    vel[0] = vel[1]
    vel[-1] = vel[-2]

    return vel

def write_calibration_json(path: str) -> None:
    calib = {
        "Calibrated": False,
        "Label": "eds_placeholder_imu_0",
        "SerialNumber": "eds://unknown",
        "Accelerometer": {
            "Bias": {"Name": "Constant", "Offset": [0.0, 0.0, 0.0]},
            "Model": {
                "Name": "Linear",
                "RectificationMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "TimeOffsetSec_Device_Accel": 0.0,
        },
        "Gyroscope": {
            "Bias": {"Name": "Constant", "Offset": [0.0, 0.0, 0.0]},
            "Model": {
                "Name": "Linear",
                "RectificationMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "TimeOffsetSec_Device_Gyro": 0.0,
        },
        "T_Device_Imu": {
            "Translation": [0.0, 0.0, 0.0],
            "UnitQuaternion": [1.0, [0.0, 0.0, 0.0]],
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

def write_description_json(path: str, nrows: int, freq_hz: float, t_first_us: float, t_last_us: float) -> None:
    desc = {
        "columns_name(width)": [
            "ts_us(1)", "gyr(3)", "acc(3)", "q(4)", "pos(3)", "vel(3)"
        ],
        "num_rows": int(nrows),
        "approximate_frequency_hz": float(freq_hz),
        "t_start_us": float(t_first_us),
        "t_end_us": float(t_last_us),
        "time_to_seconds_scale": 1e-6,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=2)

def convert_eds_to_tlio(imu_csv: str, gt_txt: str, out_dir: str, freq_hz: float) -> None:
    ensure_dir(out_dir)

    imu = read_eds_imu_csv(imu_csv)
    gt = read_eds_gt_txt(gt_txt)

    t_imu_s = (imu["ts_ns"].to_numpy(dtype=np.int64)) / 1e9
    t_gt_s = gt[:, 0].astype(np.float64)

    t_start = max(float(t_imu_s.min()), float(t_gt_s.min()))
    t_end = min(float(t_imu_s.max()), float(t_gt_s.max()))

    if not (t_end > t_start):
        raise RuntimeError(
            f"No overlap.\n"
            f"IMU: [{t_imu_s.min():.6f},{t_imu_s.max():.6f}] s\n"
            f"GT : [{t_gt_s.min():.6f},{t_gt_s.max():.6f}] s"
        )

    t_grid = make_uniform_timebase(t_start, t_end, freq_hz)

    # Resample IMU
    gyr = np.zeros((len(t_grid), 3), dtype=np.float64)
    acc = np.zeros((len(t_grid), 3), dtype=np.float64)

    for i, c in enumerate(["gx", "gy", "gz"]):
        f = interp1d(t_imu_s, imu[c].to_numpy(dtype=np.float64), kind="linear", bounds_error=False, fill_value="extrapolate")
        gyr[:, i] = f(t_grid)

    for i, c in enumerate(["ax", "ay", "az"]):
        f = interp1d(t_imu_s, imu[c].to_numpy(dtype=np.float64), kind="linear", bounds_error=False, fill_value="extrapolate")
        acc[:, i] = f(t_grid)

    # Resample Position
    pos_f = interp1d(t_gt_s, gt[:, 1:4], axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    pos = pos_f(t_grid)

    # Resample Orientation
    rots = R.from_quat(gt[:, 4:8]) 
    slerp = Slerp(t_gt_s.astype(np.float64), rots)
    rot_grid = slerp(t_grid)
    quat = rot_grid.as_quat()

    # Apply Rotation to IMU data (to EDS world frame)
    gyr = rot_grid.apply(gyr)
    acc = rot_grid.apply(acc) - np.array([0,0,-9.81])

    # Apply Rotation to IMU data (to TLIO world frame) and add back gravity (match TLIO golden dataset)
    eds2tlio = R.from_euler('x', np.pi)
    gyr = eds2tlio.apply(gyr)
    acc = eds2tlio.apply(acc)
    acc = acc + np.array([0,0,9.81])

    # Enforce continuous quaternion mapping
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] = -quat[i]

    vel = compute_velocity(pos, t_grid)

    # Generate Timestamps
    dt_us = 1e6 / float(freq_hz)
    ts_us = NPY_TS_OFFSET_US + np.arange(len(t_grid), dtype=np.float64) * dt_us

    # Save NPY
    data_npy = np.hstack([ts_us[:, None], gyr, acc, quat, pos, vel]).astype(np.float64)
    np.save(os.path.join(out_dir, "imu0_resampled.npy"), data_npy)

    # Save JSON Description
    write_description_json(
        os.path.join(out_dir, "imu0_resampled_description.json"),
        nrows=data_npy.shape[0],
        freq_hz=freq_hz,
        t_first_us=float(ts_us[0]),
        t_last_us=float(ts_us[-1]),
    )

    # Save RAW IMU CSV
    imu_mask = (t_imu_s >= t_start) & (t_imu_s <= t_end)
    raw_ts_ns = imu["ts_ns"].to_numpy(dtype=np.int64)[imu_mask]
    raw_gx = imu["gx"].to_numpy(dtype=np.float64)[imu_mask]
    raw_gy = imu["gy"].to_numpy(dtype=np.float64)[imu_mask]
    raw_gz = imu["gz"].to_numpy(dtype=np.float64)[imu_mask]
    raw_ax = imu["ax"].to_numpy(dtype=np.float64)[imu_mask]
    raw_ay = imu["ay"].to_numpy(dtype=np.float64)[imu_mask]
    raw_az = imu["az"].to_numpy(dtype=np.float64)[imu_mask]

    raw_ts_ns = raw_ts_ns - raw_ts_ns[0]

    df_samples = pd.DataFrame({
        "#timestamp [ns]": raw_ts_ns,
        "temperature [degC]": np.zeros(len(raw_ts_ns), dtype=np.float64),
        "w_RS_S_x [rad s^-1]": raw_gx,
        "w_RS_S_y [rad s^-1]": raw_gy,
        "w_RS_S_z [rad s^-1]": raw_gz,
        "a_RS_S_x [m s^-2]": raw_ax,
        "a_RS_S_y [m s^-2]": raw_ay,
        "a_RS_S_z [m s^-2]": raw_az,
    })
    df_samples.to_csv(os.path.join(out_dir, "imu_samples_0.csv"), index=False)

    # Save Calibration
    write_calibration_json(os.path.join(out_dir, "calibration.json"))

    print(f"  [+] Saved {data_npy.shape[0]} rows to {out_dir}")

def process_all_datasets(root_dir: str, output_base: str, freq_hz: float = 200.0):
    folders = glob.glob(os.path.join(root_dir, "*/"))
    
    if not folders:
        print(f"No folders found in {root_dir}")
        return

    for folder_path in sorted(folders):
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
        # Extract the sequence number for the output (e.g., 'seq01')
        seq_num = folder_name.split('_')[0]
        output_name = f"seq{seq_num}"
        
        imu_path = os.path.join(folder_path, "imu.csv")
        gt_path = os.path.join(folder_path, "stamped_groundtruth.txt")
        out_dir = os.path.join(output_base, output_name)
        
        print(f"Processing: {folder_name} -> {output_name}")
        
        # Check if the required files actually exist before calling the converter
        if not os.path.exists(imu_path) or not os.path.exists(gt_path):
            print(f"  [-] Skipped: Missing imu.csv or stamped_groundtruth.txt in {folder_name}")
            continue

        try:
            convert_eds_to_tlio(imu_csv=imu_path, gt_txt=gt_path, out_dir=out_dir, freq_hz=freq_hz)
        except Exception as e:
            print(f"  [-] Failed to process {folder_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert EDS sequences to TLIO format.")
    
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing the raw sequence folders (e.g., path/to/EDS)"
    )
    
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        required=True, 
        help="Base directory where the output sequence folders will be saved"
    )
    
    parser.add_argument(
        "--freq_hz", 
        type=float, 
        default=200.0, 
        help="Target frequency in Hz for the output data (default: 200.0)"
    )

    args = parser.parse_args()

    # Ensure the base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting batch conversion to new TLIO format...")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Frequency: {args.freq_hz} Hz\n")

    # Call the batch processing function with the provided arguments
    process_all_datasets(args.input_dir, args.output_dir, args.freq_hz)
    
    print("\nBatch conversion complete.")