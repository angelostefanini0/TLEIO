from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

import hdf5plugin
import h5py



import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gt_training import *

"""This script processes event-based datasets stored in HDF5 files and augments it with a temporal lookup table called `ms_to_idx`.

The input folder is expected to contain raw event data (timestamps, pixel coordinates, and polarity), either in the root or under an `events/` group. 
The script reads the event timestamps, verifies that they are sorted, and computes a mapping from each millisecond to the index of the first 
event occurring at or after that time.

The resulting `ms_to_idx` array enables fast temporal slicing of events without repeatedly searching through the full timestamp array.

A new HDF5 file is created as output for each input sequence. This file contains:
- An `events/` group with the datasets `p`, `t`, `x`, and `y`
- The computed `ms_to_idx` dataset stored at the root level

Command to run:
python scripts/processing_tartan.py data/tartan \
--save-path data/tartan/processed_train \
--save_path_validation data/tartan/processed_validation \
--validation-seq 3 \
--save_path_testing data/eds/processed_testing \
--test-seq 0,6 \
--overwrite \
--timestamps-key events/t \
--process_gt pose_lcam_front.txt \
--generate_imu_csv true
--delta_t_ms 50 \
--anchor_hz 20


python scripts/processing.py data/eds/raw --save-path data/eds/processed_train --save_path_validation data/eds/processed_validation --validation-seq 3 --save_path_testing data/eds/processed_testing --test-seq 0,6 --overwrite --timestamps-key t --process_gt imu.csv stamped_groundtruth.txt --delta_t_ms 50 --anchor_hz 20

"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ms_to_idx lookup table from event timestamps in an HDF5 file."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the folder with the raw dataset",
    )
    parser.add_argument(
        "--timestamps-key",
        type=str,
        default="events/t",
        help="Internal HDF5 path to the timestamps dataset. Default: events/t",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Path where the output files will be saved.",
    )
    parser.add_argument(
        "--save_path_testing",
        type=Path,
        default=None,
        help="Path where the testing sequence output files will be saved.",
    )
    parser.add_argument(
        "--save_path_validation",
        type=Path,
        default=None,
        help="Path where the validation sequence output files will be saved.",
    )
    parser.add_argument(
        "--test-seq",
        type=str,
        default="",
        help=(
            "Comma-separated list of sequences to save under --save_path_testing. "
            "You can use indices like '0,6' or sequence names like "
            "'00_peanuts_dark,06_ziggy_and_fuzz'."
        ),
    )
    parser.add_argument(
        "--validation-seq",
        type=str,
        default="",
        help=(
            "Comma-separated list of sequences to save under --save_path_validation. "
            "You can use indices like '1,3' or sequence names like "
            "'01_peanuts_light,03_rocket_earth_dark'."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ms_to_idx",
        help="Name of the dataset in the output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--remove-raw",
        action="store_true",
        help="Remove raw input files after processing.",
    )
    parser.add_argument(
        "--process_gt",
        nargs="*",
        default=[],
        help=(
            "Optional list of supplementary files to process from the source file directory"
            "to the output file directory. Example: --process_gt pose_lcam_front.txt"
        ),
    )
    parser.add_argument(
        "--generate_imu_csv", 
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True, 
        help="Synthetize a csv file from the available imu data"
    )
    parser.add_argument(
        "--delta_t_ms", 
        type=float, default=50.0, 
        help="Voxel duration in ms"
    )
    
    parser.add_argument(
        "--anchor_t_ms", 
        type=float, 
        default=50.0, 
        help="Anchor step duration in ms"
    )

    return parser.parse_args()

def ensure_file_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path

def parse_sequence_selection(value: str, available_names: list[str]) -> set[str]:
    if not value.strip():
        return set()

    available_set = set(available_names)
    index_to_name = {str(idx): name for idx, name in enumerate(available_names)}
    suffix_to_name = {}
    for name in available_names:
        suffix = name.split("_")[-1]
        suffix_to_name.setdefault(suffix, []).append(name)

    selected = set()
    items = [item.strip() for item in value.split(",") if item.strip()]
    for item in items:
        if item in available_set:
            selected.add(item)
            continue

        if item in index_to_name:
            selected.add(index_to_name[item])
            continue

        if item in suffix_to_name:
            matches = suffix_to_name[item]
            if len(matches) == 1:
                selected.add(matches[0])
                continue

            raise ValueError(
                f"Sequence selector '{item}' is ambiguous. Matches: {', '.join(sorted(matches))}. "
                "Use the full sequence name instead."
            )

        if item.isdigit():
            normalized = str(int(item))
            if normalized in index_to_name:
                selected.add(index_to_name[normalized])
                continue

        raise ValueError(
            f"Unknown sequence '{item}'. "
            f"Use indices like '0,6' or full names from: {', '.join(available_names)}"
        )

    return selected


def iter_tartan_sequences(input_path: Path) -> list[tuple[str, Path]]:
    sequences = []
    for env in sorted(input_path.iterdir()):
        if not env.is_dir():
            continue

        env_name = env.name
        for diff in sorted(env.iterdir()):
            if not diff.is_dir():
                continue

            diff_name = diff.name
            for seq in sorted(diff.iterdir()):
                if not seq.is_dir():
                    continue
                if not (seq / "events.h5").exists():
                    continue

                full_name = f"{env_name}_{diff_name}_{seq.name}"
                sequences.append((full_name, seq))

    return sequences


def get_missing_gt_files(
    seq_dir: Path,
    files_to_copy: list[str],
    generate_imu_csv: bool,
) -> list[str]:
    missing = []

    for filename in files_to_copy:
        if not (seq_dir / filename).exists():
            missing.append(filename)

    if files_to_copy and not (seq_dir / "imu" / "cam_time.txt").exists():
        missing.append("imu/cam_time.txt")

    if generate_imu_csv:
        for rel_path in ["imu/acc.txt", "imu/gyro.txt", "imu/imu_time.txt"]:
            if not (seq_dir / rel_path).exists():
                missing.append(rel_path)

    return missing



def load_timestamps(h5_path: Path, timestamps_key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        candidates = [timestamps_key]
        if "/" not in timestamps_key:
            candidates.append(f"events/{timestamps_key}")

        for key in candidates:
            if key in f:
                t = f[key][...]
                return np.asarray(t, dtype=np.int64)

        raise KeyError(
            f"Dataset '{timestamps_key}' not found. "
            f"Tried: {', '.join(candidates)}. Root keys: {', '.join(f.keys())}"
        )


def validate_sorted_non_decreasing(t: np.ndarray) -> None:
    if np.any(t[1:] < t[:-1]):
        raise ValueError("Timestamps must be sorted in non-decreasing order.")


def build_ms_to_idx(t_us: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Builds the mapping from milliseconds to indices using the following rule: 
    # This is the mapping from milliseconds to event index:
    It is defined such that
    (1) t[ms_to_idx[ms]] >= ms*1000, for ms > 0
    (2) t[ms_to_idx[ms] - 1] < ms*1000, for ms > 0
    (3) ms_to_idx[0] == 0
    , where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
    
    As an example, given 't' and 'ms':
    t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
    ms:   0       1       2       3       4       5       6       7       8       9
    
    we get
    
    ms_to_idx:
          0       2       2       3       3       3       5       5       8       9
    """
    validate_sorted_non_decreasing(t_us)
    
    t0 = t_us[0]
    print(f"Initial time:{t0}")
    t_relative = t_us - t0
    
    max_ms = int(np.ceil(t_relative[-1] / 1000.0))
    ms_grid_us = np.arange(max_ms + 1, dtype=np.int64) * 1000
    
    ms_to_idx = np.searchsorted(t_relative, ms_grid_us, side="left").astype(np.int64)
    return ms_to_idx, int(t0)

def write_to_new_file(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    data: np.ndarray,
    overwrite: bool,
    t_us: np.ndarray,
    is_tartan: bool = False,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "w-"
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, mode) as f_out:
        events_out = f_out.create_group("events")

        for key in ["p", "t", "x", "y"]:
            if key == "t":
                src_data = (t_us - t_us[0]).astype(t_us.dtype)
                src_dtype = src_data.dtype
            else:
                if f"events/{key}" in f_in:
                    src = f_in[f"events/{key}"]
                elif key in f_in:
                    src = f_in[key]
                else:
                    raise KeyError(f"Dataset '{key}' not found in root or in events/")

                src_data = src[...]

                if key == "p":
                    src_data = ((np.asarray(src_data, dtype=np.int8) + 1) // 2).astype(np.uint8)

                src_dtype = src_data.dtype

            events_out.create_dataset(key, data=src_data, dtype=src_dtype)

        f_out.create_dataset(dataset_name, data=data, dtype=data.dtype)

def copy_calibration_if_present(source_h5: Path, dest_h5: Path) -> None:
    source_dir = source_h5.parent
    dest_dir = dest_h5.parent
    calibration_path = source_dir / "K.yaml"
    if not calibration_path.exists():
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(calibration_path, dest_dir / calibration_path.name)
    print(f"Copied calibration: {calibration_path.name} -> {dest_dir}")

def process_gt(
    source_h5: Path,
    dest_h5: Path,
    files_to_copy: list[str],
    t0: int, 
    t_end_us: int,
    delta_t_ms: float, 
    anchor_t_ms: float, 
    generate_imu_csv: bool
) -> None:
    
    source_dir = source_h5.parent
    dest_dir = dest_h5.parent

    for filename in files_to_copy:
        #OPEN THE SOURCE POSE FILE
        s_pose_file = source_dir / filename
        if not s_pose_file.exists():
            raise FileNotFoundError(f"Supplementary file not found: {s_pose_file}")
        
        #OPEN THE SOURCE TIMESTAMP FILE 
        s_time_file = source_dir / "imu" / "cam_time.txt"
        d_file = dest_dir / "stamped_groundtruth.txt"

        source_pose_data = np.loadtxt(s_pose_file, delimiter=None, ndmin=2)
        source_time_data = np.loadtxt(s_time_file)

        #MERGE TIME AND POSE INTO SINGLE DATA
        stamped_gt = np.column_stack((source_time_data, source_pose_data))
    
        print(f"Processing: {filename} -> {dest_dir}")
        offset_timestamps(stamped_gt, dest_dir, d_file, t0, t_end_us, delta_t_ms, anchor_t_ms)
    
    if generate_imu_csv: 
        imu_acc_file = source_dir / "imu" / "acc.txt"
        imu_gyro_file = source_dir / "imu" / "gyro.txt"
        imu_time_file = source_dir / "imu" / "imu_time.txt"
        
        imu_acc_data = np.loadtxt(imu_acc_file, delimiter=None)
        imu_gyro_data = np.loadtxt(imu_gyro_file, delimiter=None)
        imu_time_data = np.round(np.loadtxt(imu_time_file) * 1e6).astype(np.int64)
        imu_time_rel = imu_time_data - t0
        keep_mask = (imu_time_rel >= 0) & (imu_time_rel <= t_end_us)
        imu_data_full = np.column_stack(
            (
                imu_time_rel[keep_mask],
                imu_gyro_data[keep_mask],
                imu_acc_data[keep_mask],
            )
        )
        fmt = ["%d"] + ["%.10f"] * (imu_data_full.shape[1] - 1)
        d_file_imu = dest_dir / "imu.csv"
        np.savetxt(d_file_imu, imu_data_full, delimiter=",", fmt=fmt)

def offset_timestamps(
    data: np.ndarray,
    d_folder: Path,
    d_file: Path,
    t0: int,
    t_end_us: int,
    delta_t_ms: float,
    anchor_hz: float,
) -> np.ndarray:
    
    if data.shape[1] < 1:
        raise ValueError("Input file must contain at least one column.")

    # Convert timestamps to us and shift them so events start at 0
    converted_first_col = data[:, 0] * 1e6 - t0

    # Keep only samples inside the event time span [0, t_end_us]
    keep_mask = (converted_first_col >= 0) & (converted_first_col <= t_end_us)

    filtered = data[keep_mask].copy()
    filtered[:, 0] = converted_first_col[keep_mask].astype(np.int64)

    if filtered.shape[0] == 0:
        raise ValueError(f"No samples left after cropping gt to event range.")

    fmt = ["%d"] + ["%.10f"] * (filtered.shape[1] - 1)
    np.savetxt(d_file, filtered, delimiter=" ", fmt=fmt)
    
    generate_supervision(delta_t_ms, anchor_hz, filtered, d_folder)

    return filtered


def generate_supervision(
        delta_t_ms: float = 50.0,
        anchor_t_ms: float = 50.0, 
        gt_data: np.ndarray = None, 
        out_dir : Path = None,
        ): 
    delta_t_us = int(round(delta_t_ms * 1000.0))
    anchor_step_us = int(round(anchor_t_ms * 1000.0))
    
    ts_gt, pos_gt, quat_gt = load_gt(gt_data)

    anchors_us = get_anchor_grid(
        gt_timestamps_us=ts_gt,
        delta_t_us=delta_t_us,
        anchor_step_us=anchor_step_us,
    )

    anchor_pos, anchor_quat = interpolate_gt_to_anchors(
        gt_timestamps_us=ts_gt,
        gt_pos=pos_gt,
        gt_quat=quat_gt,
        anchors_us=anchors_us,
    )

    rel = compute_relative_motions(
        anchor_ts=anchors_us,
        anchor_pos=anchor_pos,
        anchor_quat=anchor_quat,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_anchor_poses(out_dir / "anchor_poses.txt", anchors_us, anchor_pos, anchor_quat)
    save_relative_motions(out_dir / "relative_motions.txt", rel)

    print(f"GT poses:           {len(ts_gt)}")
    print(f"Anchors generated:  {len(anchors_us)}")
    print(f"Relative motions:   {len(rel)}")
    if len(anchors_us) > 0:
        print(f"First anchor [us]:  {anchors_us[0]}")
        print(f"Last anchor  [us]:  {anchors_us[-1]}")
        print(f"Step         [us]:  {anchor_step_us}")


def convert_tartan_timestamps_ns_to_us(t_ns: np.ndarray) -> np.ndarray:
    """Convert TartanEvent timestamps from nanoseconds to microseconds."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    t_us = t_ns // 1000
    if np.any(t_us[1:] < t_us[:-1]):
        t_us = np.maximum.accumulate(t_us)
    return t_us


def count_tartan_timestamp_reversals(t_ns: np.ndarray) -> tuple[int, float]:
    """Count backward timestamp steps after ns -> us conversion."""
    t_us = np.asarray(t_ns, dtype=np.int64) // 1000
    num_pairs = max(0, len(t_us) - 1)
    if num_pairs == 0:
        return 0, 0.0

    num_reversals = int(np.count_nonzero(t_us[1:] < t_us[:-1]))
    pct_reversals = 100.0 * num_reversals / num_pairs
    return num_reversals, pct_reversals



def sanity_check_processed_sequence(seq_dir: Path) -> bool:
    """Check processed data consistency and print whether conversion is safe."""
    events_h5 = seq_dir / "events.h5"
    anchor_file = seq_dir / "anchor_poses.txt"
    ok = True

    anchor_data = np.loadtxt(anchor_file, skiprows=1, ndmin=2)
    if anchor_data.shape[0] == 0:
        print(f"[SANITY][FAIL] No anchors found in: {anchor_file}")
        return False

    last_anchor_us = int(anchor_data[-1, 0])

    with h5py.File(events_h5, "r") as f:
        t = np.asarray(f["events/t"], dtype=np.int64)
        p = np.asarray(f["events/p"])

    

    last_event_us = int(t[-1])
    polarity_ok = np.all(np.isin(p, [0, 1]))
    anchor_ok = last_anchor_us <= last_event_us

    if not anchor_ok:
        ok = False
        print(
            f"[SANITY][FAIL] Last anchor time ({last_anchor_us} us) "
            f"> last event time ({last_event_us} us)"
        )
    else:
        print(
            f"[SANITY][OK] Last anchor time ({last_anchor_us} us) "
            f"<= last event time ({last_event_us} us)"
        )

    if not polarity_ok:
        ok = False
        unique_p = np.unique(p)
        print(f"[SANITY][FAIL] Invalid event polarity values found: {unique_p}")
    else:
        print("[SANITY][OK] Event polarity is binary {0,1}")

    if ok:
        print("[SANITY] Conversion safe")
    else:
        print("[SANITY] Conversion NOT safe")

    return ok



def main() -> None:
    """
    Parse command-line arguments for building an ms_to_idx lookup table
    from event timestamps stored in an HDF5 file.
    """
    #PARSE ARGUMENTS
    args = parse_args()
    input_path = ensure_file_exists(args.file)
    
    #BUILD PROCESSED SEQUENCE TREE CLEANLY
    sequence_dirs = []
    sequence_names = []
    testing_sequences = []
    validation_sequences = []

    #GET ALL SEQUENCES
    sequences = iter_tartan_sequences(input_path)
    sequence_names = [name for name, _ in sequences]
    sequence_dirs = [seq_dir for _, seq_dir in sequences]
    
    #GET TESTING AND VALIDATION SEQUENCES OUT OF ALL SEQUENCES   
    testing_sequences = parse_sequence_selection(args.test_seq, sequence_names)
    validation_sequences = parse_sequence_selection(args.validation_seq, sequence_names)

    #MAIN PROCESSING LOOP 
    for seq, seq_name in zip(sequence_dirs, sequence_names): 
        if args.process_gt:
            missing_gt_files = get_missing_gt_files(
                seq_dir=seq,
                files_to_copy=args.process_gt,
                generate_imu_csv=args.generate_imu_csv,
            )
            if missing_gt_files:
                print(
                    f"Skipping {seq_name}: missing GT/IMU files: "
                    f"{', '.join(missing_gt_files)}"
                )
                continue

        event_path = seq / "events.h5"
        t_raw = load_timestamps(event_path, args.timestamps_key)
        num_reversals, pct_reversals = count_tartan_timestamp_reversals(t_raw)
        t_us = convert_tartan_timestamps_ns_to_us(t_raw)


        print(f"Loaded timestamps from: {event_path}")
        print(
            f"Timestamp reversals after ns->us conversion: "
            f"{num_reversals}/{max(0, len(t_raw) - 1)} "
            f"({pct_reversals:.6f}%)"
        )
        
        ms_to_idx, t0 = build_ms_to_idx(t_us)
        t_end_us = int(t_us[-1] - t_us[0])
        print(f"Recording duration: {len(ms_to_idx) - 1} ms")

        if args.save_path or args.save_path_testing:
            base_save_path = args.save_path
            if seq_name in testing_sequences:
                base_save_path = args.save_path_testing
            elif seq_name in validation_sequences:
                base_save_path = args.save_path_validation

            output_path = base_save_path / seq_name / "events.h5"
            write_to_new_file(
                input_path=event_path,
                output_path=output_path,
                dataset_name=args.dataset_name,
                data=ms_to_idx,
                overwrite=args.overwrite,
                t_us=t_us
            )

            print(f"Wrote dataset '{args.dataset_name}' into: {output_path}")
            
            if args.process_gt:
                
                process_gt( event_path,
                            output_path,
                            args.process_gt,
                            t0, 
                            t_end_us,
                            args.delta_t_ms, 
                            args.anchor_t_ms, 
                            args.generate_imu_csv,
                )

                sanity_check_processed_sequence(output_path.parent)
            
            copy_calibration_if_present(event_path, output_path)
        
        # DELETE RECURSIVELY THE RAW INPUT FILES IF SPECIFIED
        # if args.remove_raw:
        #     if seq.exists() and seq.is_dir():
        #         shutil.rmtree(seq)
        #         print(f"Successfully removed: {seq}", flush=True)


if __name__ == "__main__":
    main()
