from __future__ import annotations
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys

import hdf5plugin
import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.gt_training import *
from scripts.utils.parsing_utils import *
from scripts.utils.config import default_config_path, parse_args_with_config

"""This script processes event-based datasets stored in HDF5 files and augments it with a temporal lookup table called `ms_to_idx`.

The input folder is expected to contain raw event data (timestamps, pixel coordinates, and polarity), either in the root or under an `events/` group. 
The script reads the event timestamps, verifies that they are sorted, and computes a mapping from each millisecond to the index of the first 
event occurring at or after that time.

The resulting `ms_to_idx` array enables fast temporal slicing of events without repeatedly searching through the full timestamp array.

For each processed Tartan sequence, the script writes:
- `events_meta.h5` with corrected `events/t` and the computed `ms_to_idx`
- `events.h5` either as a symlink to the original raw event file or as a
  streamed byte-for-byte copy when `--materialize-events-file` is enabled
- the processed GT text files used downstream

Command to run:
python scripts/processing/processing_tartan.py data/tartan \
--save-path data/tartan/processed_train \
--overwrite \
--timestamps-key events/t \
--process_gt pose_lcam_front.txt \
--delta_t_ms 50 \
--anchor_t_ms 50

"""

DEFAULT_EVENT_CHUNK_SIZE = 2_000_000
DEFAULT_STREAM_COPY_BUFFER_BYTES = 16 * 1024 * 1024
METADATA_COMPRESSION = "gzip"
METADATA_COMPRESSION_LEVEL = 4


@dataclass(frozen=True)
class TartanProcessingSummary:
    """Compact summary returned by the streamed Tartan timestamp preprocessing pass."""
    t0_us: int
    t_end_us: int
    num_events: int
    num_reversals: int
    pct_reversals: float
    duration_ms: int

def ensure_file_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path

def parse_args() -> argparse.Namespace:
    """
    Argument parser for the processing script. Arguments include: save paths for training, validation and testing,
    voxel duration, time in between gt anchors and more
    """
    parser = argparse.ArgumentParser(
        description="Build ms_to_idx lookup table from event timestamps in an HDF5 file."
    )
    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
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
        "--start-seq",
        type=str,
        default="",
        help=(
            "Start processing from this sequence (inclusive), skipping all earlier ones. "
            "You can use an index like '3' or a full sequence name."
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
        "--process_gt",
        nargs="*",
        default=[],
        help=(
            "Optional list of supplementary files to process from the source file directory"
            "to the output file directory. Example: --process_gt pose_lcam_front.txt"
        ),
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
    parser.add_argument(
        "--event-chunk-size",
        type=int,
        default=DEFAULT_EVENT_CHUNK_SIZE,
        help=(
            "Number of events to process at a time when building the Tartan sidecar "
            f"metadata. Default: {DEFAULT_EVENT_CHUNK_SIZE}"
        ),
    )
    parser.add_argument(
        "--materialize-events-file",
        action="store_true",
        help=(
            "Write a real `events.h5` file into each processed Tartan sequence "
            "by streaming a byte-for-byte copy of the raw file. "
            "Default behavior is to create a symlink instead."
        ),
    )
    parser.add_argument(
        "--remove-raw-after-materialize",
        action="store_true",
        help=(
            "After successfully materializing a real processed `events.h5`, "
            "delete the corresponding raw Tartan sequence directory and prune "
            "now-empty parent folders under the input root."
        ),
    )

    return parse_args_with_config(
        parser,
        default_config_path("processing_tartan"),
        required=("file",),
    )


def resolve_h5_dataset(
    h5_file: h5py.File,
    key: str,
    *,
    allow_event_prefix: bool = False,
) -> h5py.Dataset:
    """
    Return a dataset by key, optionally also searching under the `events/` group.
    """
    
    candidates = [key]
    if allow_event_prefix and "/" not in key:
        candidates.append(f"events/{key}")

    for candidate in candidates:
        if candidate in h5_file:
            return h5_file[candidate]

    raise KeyError(
        f"Dataset '{key}' not found. "
        f"Tried: {', '.join(candidates)}. Root keys: {', '.join(h5_file.keys())}"
    )

def create_or_replace_symlink(target_path: Path, link_path: Path, overwrite: bool) -> None:
    """
    Create the processed `events.h5` symlink that points back to the raw event file.
    """
    
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {link_path}. Use --overwrite to replace it."
            )
        if link_path.is_dir() and not link_path.is_symlink():
            raise IsADirectoryError(f"Cannot replace directory with symlink: {link_path}")
        link_path.unlink()

    relative_target = Path(
        os.path.relpath(target_path.resolve(), start=link_path.parent.resolve())
    )
    link_path.symlink_to(relative_target)


def stream_copy_file(
    source_path: Path,
    dest_path: Path,
    overwrite: bool,
    buffer_bytes: int = DEFAULT_STREAM_COPY_BUFFER_BYTES,
) -> None:
    """
    Copy a file in fixed-size chunks so large HDF5 files never need to be held in RAM.
    """
    
    if buffer_bytes <= 0:
        raise ValueError("stream copy buffer must be a positive integer.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() or dest_path.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {dest_path}. Use --overwrite to replace it."
            )
        if dest_path.is_dir() and not dest_path.is_symlink():
            raise IsADirectoryError(f"Cannot replace directory with file: {dest_path}")
        dest_path.unlink()

    with source_path.open("rb") as src, dest_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=buffer_bytes)


def create_processed_events_file(
    source_path: Path,
    dest_path: Path,
    overwrite: bool,
    materialize_events_file: bool,
) -> str:
    """
    Create the processed `events.h5` as either a symlink or a streamed real copy.
    """
    
    if materialize_events_file:
        stream_copy_file(source_path, dest_path, overwrite=overwrite)
        return "copied"

    create_or_replace_symlink(source_path, dest_path, overwrite=overwrite)
    return "linked"


def remove_raw_sequence_tree(seq_dir: Path, root_dir: Path, protected_paths: tuple[Path, ...] = ()) -> None:
    """
    Remove a raw sequence directory and then prune empty parents up to `root_dir`.
    """
    
    seq_resolved = seq_dir.resolve()
    root_resolved = root_dir.resolve()
    protected_resolved = tuple(path.resolve() for path in protected_paths)

    if seq_resolved == root_resolved:
        raise ValueError(f"Refusing to remove the dataset root itself: {root_dir}")

    for protected in protected_resolved:
        if protected == seq_resolved or protected.is_relative_to(seq_resolved):
            raise ValueError(
                f"Refusing to remove raw sequence {seq_dir} because it would also remove "
                f"protected output path {protected}"
            )

    shutil.rmtree(seq_resolved)

    parent = seq_resolved.parent
    while parent != root_resolved and parent.is_relative_to(root_resolved):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def iter_event_slices(length: int, chunk_size: int):
    """
    Yield contiguous 1D slices so large event arrays can be streamed safely.
    """
    
    if chunk_size <= 0:
        raise ValueError("--event-chunk-size must be a positive integer.")

    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def build_tartan_sidecar(
    input_path: Path,
    metadata_path: Path,
    timestamps_key: str,
    dataset_name: str,
    overwrite: bool,
    chunk_size: int,
) -> TartanProcessingSummary:
    """
    Stream Tartan timestamps into a compact metadata sidecar.

    The sidecar stores corrected relative timestamps under `events/t`, the
    `ms_to_idx` lookup table, and small attributes used later by the loader.
    This keeps preprocessing memory bounded by `chunk_size` instead of the full
    event count.
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "w-"
    with h5py.File(input_path, "r") as f_in, h5py.File(metadata_path, mode) as f_out:
        timestamps_ds = resolve_h5_dataset(
            f_in,
            timestamps_key,
            allow_event_prefix=True,
        )
        num_events = int(timestamps_ds.shape[0])
        if num_events == 0:
            raise ValueError(f"No timestamps found in dataset '{timestamps_key}'.")

        output_chunks = timestamps_ds.chunks
        if output_chunks is None:
            output_chunks = (min(chunk_size, num_events),)

        events_out = f_out.create_group("events")
        corrected_t_ds = events_out.create_dataset(
            "t",
            shape=timestamps_ds.shape,
            dtype=np.int64,
            chunks=output_chunks,
            compression=METADATA_COMPRESSION,
            compression_opts=METADATA_COMPRESSION_LEVEL,
            shuffle=True,
        )

        num_reversals = 0
        last_raw_us: int | None = None
        last_corrected_us: int | None = None
        t0_us: int | None = None
        last_rel_us = 0
        next_ms = 1
        ms_to_idx_parts = [np.array([0], dtype=np.int64)]

        for sl in iter_event_slices(num_events, chunk_size):
            chunk_us = np.asarray(timestamps_ds[sl], dtype=np.int64)
            chunk_us //= 1000

            if last_raw_us is not None and chunk_us[0] < last_raw_us:
                num_reversals += 1
            if chunk_us.size > 1:
                num_reversals += int(np.count_nonzero(chunk_us[1:] < chunk_us[:-1]))
            last_raw_us = int(chunk_us[-1])

            if last_corrected_us is not None:
                np.maximum(chunk_us, last_corrected_us, out=chunk_us)
            np.maximum.accumulate(chunk_us, out=chunk_us)
            last_corrected_us = int(chunk_us[-1])

            if t0_us is None:
                t0_us = int(chunk_us[0])
                print(f"Initial time:{t0_us}")

            rel_chunk_us = chunk_us - t0_us
            corrected_t_ds[sl] = rel_chunk_us
            last_rel_us = int(rel_chunk_us[-1])
            chunk_start_idx = int(sl.start)

            gap_end_ms = int(rel_chunk_us[0] // 1000)
            if next_ms <= gap_end_ms:
                ms_to_idx_parts.append(
                    np.full(gap_end_ms - next_ms + 1, chunk_start_idx, dtype=np.int64)
                )
                next_ms = gap_end_ms + 1

            local_max_ms = int(rel_chunk_us[-1] // 1000)
            if next_ms <= local_max_ms:
                ms_values = np.arange(next_ms, local_max_ms + 1, dtype=np.int64)
                local_idx = np.searchsorted(
                    rel_chunk_us,
                    ms_values * 1000,
                    side="left",
                ).astype(np.int64)
                ms_to_idx_parts.append(local_idx + chunk_start_idx)
                next_ms = local_max_ms + 1

        max_ms = int(np.ceil(last_rel_us / 1000.0))
        if next_ms <= max_ms:
            ms_to_idx_parts.append(
                np.full(max_ms - next_ms + 1, num_events, dtype=np.int64)
            )

        ms_to_idx = np.concatenate(ms_to_idx_parts)
        f_out.create_dataset(
            dataset_name,
            data=ms_to_idx,
            dtype=np.int64,
            compression=METADATA_COMPRESSION,
            compression_opts=METADATA_COMPRESSION_LEVEL,
            shuffle=True,
        )
        f_out.create_dataset("t_offset", data=np.array(0, dtype=np.int64))
        f_out.create_dataset("raw_t0_us", data=np.array(t0_us, dtype=np.int64))
        f_out.attrs["normalize_polarity_to_binary"] = 1
        f_out.attrs["metadata_format"] = "tartan_sidecar_v1"

    num_pairs = max(0, num_events - 1)
    pct_reversals = 100.0 * num_reversals / num_pairs if num_pairs else 0.0
    duration_ms = int(ms_to_idx.shape[0] - 1)
    return TartanProcessingSummary(
        t0_us=int(t0_us),
        t_end_us=int(last_rel_us),
        num_events=num_events,
        num_reversals=num_reversals,
        pct_reversals=pct_reversals,
        duration_ms=duration_ms,
    )

def offset_timestamps(
    data: np.ndarray,
    d_folder: Path,
    d_file: Path,
    t0: int,
    t_end_us: int,
    delta_t_ms: float,
    anchor_t_ms: float,
):
    """
    
    """    
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

    #SAVE THE GROUND TRUTH WITH COMMON TIME-FRAME FOR FILTER OPERATION
    fmt = ["%d"] + ["%.10f"] * (filtered.shape[1] - 1)
    np.savetxt(d_file, filtered, delimiter=" ", fmt=fmt)
    

    delta_t_us = int(round(delta_t_ms * 1000.0))
    anchor_step_us = int(round(anchor_t_ms * 1000.0))
    
    ts_gt, pos_gt, quat_gt = load_gt(filtered)

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

    #SAVE EVERYTHING TO THE SAME FOLDER 
    d_folder.mkdir(parents=True, exist_ok=True)
    save_anchor_poses(d_folder / "anchor_poses.txt", anchors_us, anchor_pos, anchor_quat)
    save_relative_motions(d_folder / "relative_motions.txt", rel)

    print(f"GT poses:           {len(ts_gt)}")
    print(f"Anchors generated:  {len(anchors_us)}")
    print(f"Relative motions:   {len(rel)}")
    if len(anchors_us) > 0:
        print(f"First anchor [us]:  {anchors_us[0]}")
        print(f"Last anchor  [us]:  {anchors_us[-1]}")
        print(f"Step         [us]:  {anchor_step_us}")
    

def process_gt(
    source_h5: Path,
    dest_h5: Path,
    files_to_copy: list[str],
    t0: int, 
    t_end_us: int,
    delta_t_ms: float, 
    anchor_t_ms: float,
) -> None:
    
    source_dir = source_h5.parent
    dest_dir = dest_h5.parent

    #COMBINE POSE AND TIME INTO A SINGLE DATA FRAME 
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

        #MERGE TIME AND POSE INTO SINGLE DATAFRAME
        stamped_gt = np.column_stack((source_time_data, source_pose_data))
    
        print(f"Processing: {filename} -> {dest_dir}")
        
        offset_timestamps(stamped_gt, dest_dir, d_file, t0, t_end_us, delta_t_ms, anchor_t_ms)


def sanity_check_processed_sequence(seq_dir: Path, chunk_size: int) -> bool:
    """
    Validate the processed Tartan sequence without reading full event arrays.

    The check verifies that anchor timestamps remain within the event time span
    and that event polarity values are compatible with the chosen lazy
    normalization scheme.
    """
    events_h5 = seq_dir / "events.h5"
    metadata_h5 = seq_dir / "events_meta.h5"
    anchor_file = seq_dir / "anchor_poses.txt"
    ok = True

    anchor_data = np.loadtxt(anchor_file, skiprows=1, ndmin=2)
    if anchor_data.shape[0] == 0:
        print(f"[SANITY][FAIL] No anchors found in: {anchor_file}")
        return False

    last_anchor_us = int(anchor_data[-1, 0])

    with h5py.File(events_h5, "r") as raw_f:
        if metadata_h5.exists():
            with h5py.File(metadata_h5, "r") as meta_f:
                t_ds = meta_f["events/t"]
                normalize_polarity = bool(meta_f.attrs.get("normalize_polarity_to_binary", 0))
                last_event_us = int(t_ds[-1])
        else:
            normalize_polarity = False
            last_event_us = int(raw_f["events/t"][-1])

        p_ds = raw_f["events/p"]
        polarity_ok = True
        invalid_p = None
        for sl in iter_event_slices(int(p_ds.shape[0]), chunk_size):
            p_chunk = np.asarray(p_ds[sl])
            if normalize_polarity:
                invalid_mask = (p_chunk != -1) & (p_chunk != 1)
            else:
                invalid_mask = (p_chunk != 0) & (p_chunk != 1)

            if np.any(invalid_mask):
                polarity_ok = False
                invalid_p = np.unique(p_chunk[invalid_mask])
                break

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
        print(f"[SANITY][FAIL] Invalid event polarity values found: {invalid_p}")
    else:
        if normalize_polarity:
            print("[SANITY][OK] Event polarity can be converted lazily from {-1,1} to {0,1}")
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

    if args.remove_raw_after_materialize and not args.materialize_events_file:
        raise ValueError(
            "--remove-raw-after-materialize requires --materialize-events-file. "
            "Removing raw files would break the default symlink-based layout."
        )

    #BUILD PROCESSED SEQUENCE TREE CLEANLY
    sequence_dirs = []
    sequence_names = []
    testing_sequences = []
    validation_sequences = []

    #GET ALL SEQUENCES
    sequences = iter_tartan_sequences(input_path)
    all_sequence_names = [name for name, _ in sequences]
    all_sequence_dirs = [seq_dir for _, seq_dir in sequences]
    
    #GET TESTING AND VALIDATION SEQUENCES OUT OF ALL SEQUENCES   
    testing_sequences = parse_sequence_selection(args.test_seq, all_sequence_names)
    validation_sequences = parse_sequence_selection(args.validation_seq, all_sequence_names)

    #OPTIONALLY START FROM A SEQUENCE, INSTEAD OF PROCESSING EVERYTHING FROM SCRATCH
    start_sequence = parse_single_sequence_selector(
        args.start_seq,
        all_sequence_names,
        "--start-seq",
    )

    sequence_names = all_sequence_names
    sequence_dirs = all_sequence_dirs
    if start_sequence is not None:
        start_idx = all_sequence_names.index(start_sequence)
        sequence_names = all_sequence_names[start_idx:]
        sequence_dirs = all_sequence_dirs[start_idx:]
        print(
            f"Starting from sequence {start_sequence} "
            f"(index {start_idx}); skipped {start_idx} earlier sequence(s)."
        )

    #MAIN PROCESSING LOOP 
    for seq, seq_name in zip(sequence_dirs, sequence_names): 
        #SOME TARTAN SEQUENCES DO NOT HAVE GT DATA, SO THESE GET SKIPPED
        if args.process_gt:
            missing_gt_files = get_missing_gt_files(
                seq_dir=seq,
                files_to_copy=args.process_gt,
            )
            if missing_gt_files:
                print(
                    f"Skipping {seq_name}: missing GT files: "
                    f"{', '.join(missing_gt_files)}"
                )
                continue
        
        
        if args.save_path or args.save_path_testing:
            base_save_path = args.save_path
            if seq_name in testing_sequences:
                base_save_path = args.save_path_testing
            elif seq_name in validation_sequences:
                base_save_path = args.save_path_validation

            if base_save_path is None:
                raise ValueError(
                    f"No output path configured for sequence '{seq_name}'. "
                    "Set --save-path for training sequences."
                )

            event_path = seq / "events.h5"
            output_dir = base_save_path / seq_name
            output_events_path = output_dir / "events.h5"
            output_metadata_path = output_dir / "events_meta.h5"

            summary = build_tartan_sidecar(
                input_path=event_path,
                metadata_path=output_metadata_path,
                timestamps_key=args.timestamps_key,
                dataset_name=args.dataset_name,
                overwrite=args.overwrite,
                chunk_size=args.event_chunk_size,
            )
            events_file_mode = create_processed_events_file(
                event_path,
                output_events_path,
                overwrite=args.overwrite,
                materialize_events_file=args.materialize_events_file,
            )

            # LOG INFO ABOUT EVENT DATA
            print(f"Loaded timestamps from: {event_path}")
            print(
                f"Timestamp reversals after ns->us conversion: "
                f"{summary.num_reversals}/{max(0, summary.num_events - 1)} "
                f"({summary.pct_reversals:.6f}%)"
            )
            print(f"Recording duration: {summary.duration_ms} ms")

            print(
                f"Wrote dataset '{args.dataset_name}' into: {output_metadata_path}"
            )
            if events_file_mode == "copied":
                print(f"Copied raw events into: {output_events_path}")
            else:
                print(f"Linked raw events into: {output_events_path}")
            
            # GROUND-TRUTH PROCESSING
            if args.process_gt:
                
                process_gt( event_path,
                            output_events_path,
                            args.process_gt,
                            summary.t0_us, 
                            summary.t_end_us,
                            args.delta_t_ms, 
                            args.anchor_t_ms,
                )

                sanity_check_processed_sequence(
                    output_dir,
                    chunk_size=args.event_chunk_size,
                )

            if args.remove_raw_after_materialize:
                if events_file_mode != "copied":
                    raise RuntimeError(
                        "Raw removal is only allowed after materializing a real events.h5 copy."
                    )
                remove_raw_sequence_tree(
                    seq,
                    input_path,
                    protected_paths=(output_dir,),
                )
                print(f"Removed raw sequence tree: {seq}")
            
        
if __name__ == "__main__":
    main()
