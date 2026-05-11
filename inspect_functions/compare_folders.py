#!/usr/bin/env python3
"""
Deep-compare two dataset folders.

The script compares the recursive folder tree, symlinks, regular file bytes,
and HDF5 internals for .h5/.hdf5 files. It exits with:
  0: folders are identical under the selected checks
  1: differences were found
  2: usage error or an unreadable/corrupted file was encountered
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import stat
import sys
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any


H5_SUFFIXES = {".h5", ".hdf5", ".hdf"}


@dataclass(frozen=True)
class Entry:
    kind: str
    mode: int
    size: int | None = None
    link_target: str | None = None
    mtime_ns: int | None = None


class Reporter:
    def __init__(self, max_diffs: int) -> None:
        self.max_diffs = max_diffs
        self.count = 0
        self.truncated = False
        self.lines: list[str] = []

    def add(self, rel_path: str, message: str) -> bool:
        self.count += 1
        if self.max_diffs > 0 and len(self.lines) >= self.max_diffs:
            self.truncated = True
            return False
        self.lines.append(f"{rel_path}: {message}")
        return True

    def can_continue(self) -> bool:
        return self.max_diffs == 0 or len(self.lines) < self.max_diffs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep-compare two folders, including HDF5 contents."
    )
    parser.add_argument("left", type=Path, help="First folder to compare")
    parser.add_argument("right", type=Path, help="Second folder to compare")
    parser.add_argument(
        "--h5-chunk-mb",
        type=int,
        default=64,
        help="Chunk size used while comparing HDF5 datasets. Default: 64",
    )
    parser.add_argument(
        "--hash-chunk-mb",
        type=int,
        default=64,
        help="Chunk size used while hashing regular files. Default: 64",
    )
    parser.add_argument(
        "--byte-exact-h5",
        action="store_true",
        help="Also require .h5/.hdf5 raw file bytes to be identical.",
    )
    parser.add_argument(
        "--check-permissions",
        action="store_true",
        help="Also compare permission bits.",
    )
    parser.add_argument(
        "--check-mtime",
        action="store_true",
        help="Also compare modification timestamps in nanoseconds.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="GLOB",
        help="Ignore a relative path glob. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=100,
        help="Maximum differences to print. Use 0 for unlimited. Default: 100",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr while comparing files.",
    )
    return parser.parse_args()


def rel_posix(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return "." if rel == "." else rel


def ignored(rel_path: str, patterns: list[str]) -> bool:
    if rel_path == ".":
        return False
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def entry_kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def build_index(root: Path, ignore_patterns: list[str]) -> dict[str, Entry]:
    entries: dict[str, Entry] = {}
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"not a directory: {root}")

    entries["."] = Entry(
        kind="dir",
        mode=root_stat.st_mode,
        mtime_ns=root_stat.st_mtime_ns,
    )

    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)

        kept_dirs = []
        for name in sorted(dirnames):
            path = current_path / name
            rel = rel_posix(path, root)
            if ignored(rel, ignore_patterns):
                continue

            st = path.lstat()
            kind = entry_kind(st.st_mode)
            entries[rel] = Entry(
                kind=kind,
                mode=st.st_mode,
                size=st.st_size if kind == "file" else None,
                link_target=os.readlink(path) if kind == "symlink" else None,
                mtime_ns=st.st_mtime_ns,
            )
            if kind == "dir":
                kept_dirs.append(name)

        dirnames[:] = kept_dirs

        for name in sorted(filenames):
            path = current_path / name
            rel = rel_posix(path, root)
            if ignored(rel, ignore_patterns):
                continue

            st = path.lstat()
            kind = entry_kind(st.st_mode)
            entries[rel] = Entry(
                kind=kind,
                mode=st.st_mode,
                size=st.st_size if kind == "file" else None,
                link_target=os.readlink(path) if kind == "symlink" else None,
                mtime_ns=st.st_mtime_ns,
            )

    return entries


def is_h5_path(path: Path) -> bool:
    return path.suffix.lower() in H5_SUFFIXES


def sha256_file(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compare_raw_file(
    left: Path,
    right: Path,
    rel_path: str,
    reporter: Reporter,
    chunk_bytes: int,
) -> None:
    left_size = left.stat().st_size
    right_size = right.stat().st_size
    if left_size != right_size:
        reporter.add(rel_path, f"file size differs ({left_size} != {right_size})")
        return

    left_hash = sha256_file(left, chunk_bytes)
    right_hash = sha256_file(right, chunk_bytes)
    if left_hash != right_hash:
        reporter.add(rel_path, f"SHA-256 differs ({left_hash} != {right_hash})")


def load_h5_modules() -> tuple[Any, Any]:
    try:
        import hdf5plugin  # noqa: F401
    except Exception:
        pass

    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "HDF5 comparison requires h5py and numpy. Install the project environment first."
        ) from exc

    return h5py, np


def as_array(value: Any, np: Any) -> Any:
    return np.asarray(value)


def arrays_equal(left: Any, right: Any, np: Any) -> bool:
    left_arr = as_array(left, np)
    right_arr = as_array(right, np)
    if left_arr.shape != right_arr.shape or left_arr.dtype != right_arr.dtype:
        return False

    try:
        if left_arr.dtype.kind in {"f", "c"}:
            return bool(np.array_equal(left_arr, right_arr, equal_nan=True))
        return bool(np.array_equal(left_arr, right_arr))
    except TypeError:
        return repr(left) == repr(right)


def value_desc(value: Any, np: Any) -> str:
    arr = as_array(value, np)
    return f"shape={arr.shape}, dtype={arr.dtype}, value={repr(value)[:200]}"


def compare_attrs(
    left_obj: Any,
    right_obj: Any,
    rel_path: str,
    h5_path: str,
    reporter: Reporter,
    np: Any,
) -> None:
    left_attrs = set(left_obj.attrs.keys())
    right_attrs = set(right_obj.attrs.keys())

    for name in sorted(left_attrs - right_attrs):
        if not reporter.add(rel_path, f"HDF5 {h5_path} missing attr on right: {name}"):
            return
    for name in sorted(right_attrs - left_attrs):
        if not reporter.add(rel_path, f"HDF5 {h5_path} missing attr on left: {name}"):
            return

    for name in sorted(left_attrs & right_attrs):
        left_value = left_obj.attrs[name]
        right_value = right_obj.attrs[name]
        if not arrays_equal(left_value, right_value, np):
            reporter.add(
                rel_path,
                (
                    f"HDF5 {h5_path} attr {name!r} differs "
                    f"({value_desc(left_value, np)} != {value_desc(right_value, np)})"
                ),
            )
            if not reporter.can_continue():
                return


def h5_link_desc(link: Any, h5py: Any) -> tuple[str, tuple[Any, ...]]:
    if isinstance(link, h5py.HardLink):
        return ("hard", ())
    if isinstance(link, h5py.SoftLink):
        return ("soft", (link.path,))
    if isinstance(link, h5py.ExternalLink):
        return ("external", (link.filename, link.path))
    return (type(link).__name__, ())


def h5_object_kind(obj: Any, h5py: Any) -> str:
    if isinstance(obj, h5py.Group):
        return "group"
    if isinstance(obj, h5py.Dataset):
        return "dataset"
    if isinstance(obj, h5py.Datatype):
        return "datatype"
    return type(obj).__name__


def compare_dataset_properties(
    left_ds: Any,
    right_ds: Any,
    rel_path: str,
    h5_path: str,
    reporter: Reporter,
    np: Any,
) -> bool:
    props = [
        ("shape", left_ds.shape, right_ds.shape),
        ("dtype", str(left_ds.dtype), str(right_ds.dtype)),
        ("maxshape", left_ds.maxshape, right_ds.maxshape),
        ("chunks", left_ds.chunks, right_ds.chunks),
        ("compression", left_ds.compression, right_ds.compression),
        ("compression_opts", repr(left_ds.compression_opts), repr(right_ds.compression_opts)),
        ("shuffle", left_ds.shuffle, right_ds.shuffle),
        ("fletcher32", left_ds.fletcher32, right_ds.fletcher32),
        ("scaleoffset", left_ds.scaleoffset, right_ds.scaleoffset),
    ]

    same = True
    for name, left_value, right_value in props:
        if left_value != right_value:
            same = False
            reporter.add(
                rel_path,
                f"HDF5 {h5_path} dataset {name} differs ({left_value!r} != {right_value!r})",
            )
            if not reporter.can_continue():
                return False

    try:
        if not arrays_equal(left_ds.fillvalue, right_ds.fillvalue, np):
            same = False
            reporter.add(rel_path, f"HDF5 {h5_path} dataset fillvalue differs")
    except Exception as exc:
        same = False
        reporter.add(rel_path, f"HDF5 {h5_path} could not compare fillvalue: {exc}")

    return same


def dataset_rows_per_chunk(shape: tuple[int, ...], dtype: Any, chunk_bytes: int) -> int:
    if len(shape) == 0:
        return 1
    if dtype.hasobject:
        return 1024

    tail_items = prod(shape[1:]) if len(shape) > 1 else 1
    bytes_per_row = max(1, tail_items * max(1, dtype.itemsize))
    return max(1, chunk_bytes // bytes_per_row)


def dataset_slice(shape: tuple[int, ...], start: int, end: int) -> tuple[slice, ...]:
    return (slice(start, end),) + tuple(slice(None) for _ in shape[1:])


def compare_dataset_values(
    left_ds: Any,
    right_ds: Any,
    rel_path: str,
    h5_path: str,
    reporter: Reporter,
    np: Any,
    chunk_bytes: int,
) -> None:
    if left_ds.shape != right_ds.shape or str(left_ds.dtype) != str(right_ds.dtype):
        return

    if left_ds.shape is None:
        if not arrays_equal(left_ds[()], right_ds[()], np):
            reporter.add(rel_path, f"HDF5 {h5_path} null dataset value differs")
        return

    if left_ds.shape == ():
        if not arrays_equal(left_ds[()], right_ds[()], np):
            reporter.add(rel_path, f"HDF5 {h5_path} scalar dataset value differs")
        return

    if 0 in left_ds.shape:
        return

    rows = dataset_rows_per_chunk(left_ds.shape, left_ds.dtype, chunk_bytes)
    for start in range(0, left_ds.shape[0], rows):
        end = min(start + rows, left_ds.shape[0])
        slc = dataset_slice(left_ds.shape, start, end)
        try:
            left_chunk = left_ds[slc]
            right_chunk = right_ds[slc]
        except Exception as exc:
            raise RuntimeError(f"HDF5 {rel_path}:{h5_path} failed reading slice {slc}: {exc}") from exc

        if not arrays_equal(left_chunk, right_chunk, np):
            reporter.add(rel_path, f"HDF5 {h5_path} dataset values differ in slice {slc}")
            return


def compare_h5_group(
    left_group: Any,
    right_group: Any,
    rel_path: str,
    h5_path: str,
    reporter: Reporter,
    h5py: Any,
    np: Any,
    chunk_bytes: int,
) -> None:
    compare_attrs(left_group, right_group, rel_path, h5_path, reporter, np)
    if not reporter.can_continue():
        return

    left_names = set(left_group.keys())
    right_names = set(right_group.keys())

    for name in sorted(left_names - right_names):
        if not reporter.add(rel_path, f"HDF5 {h5_path} missing object on right: {name}"):
            return
    for name in sorted(right_names - left_names):
        if not reporter.add(rel_path, f"HDF5 {h5_path} missing object on left: {name}"):
            return

    for name in sorted(left_names & right_names):
        child_path = f"{h5_path.rstrip('/')}/{name}" if h5_path != "/" else f"/{name}"
        left_link = left_group.get(name, getlink=True)
        right_link = right_group.get(name, getlink=True)
        left_link_desc = h5_link_desc(left_link, h5py)
        right_link_desc = h5_link_desc(right_link, h5py)
        if left_link_desc != right_link_desc:
            reporter.add(
                rel_path,
                f"HDF5 {child_path} link differs ({left_link_desc!r} != {right_link_desc!r})",
            )
            if not reporter.can_continue():
                return
            continue

        if left_link_desc[0] != "hard":
            continue

        try:
            left_obj = left_group[name]
            right_obj = right_group[name]
        except Exception as exc:
            reporter.add(rel_path, f"HDF5 {child_path} failed opening object: {exc}")
            if not reporter.can_continue():
                return
            continue

        left_kind = h5_object_kind(left_obj, h5py)
        right_kind = h5_object_kind(right_obj, h5py)
        if left_kind != right_kind:
            reporter.add(
                rel_path,
                f"HDF5 {child_path} object kind differs ({left_kind} != {right_kind})",
            )
            if not reporter.can_continue():
                return
            continue

        if left_kind == "group":
            compare_h5_group(
                left_obj,
                right_obj,
                rel_path,
                child_path,
                reporter,
                h5py,
                np,
                chunk_bytes,
            )
        elif left_kind == "dataset":
            compare_attrs(left_obj, right_obj, rel_path, child_path, reporter, np)
            if not reporter.can_continue():
                return
            same_props = compare_dataset_properties(
                left_obj,
                right_obj,
                rel_path,
                child_path,
                reporter,
                np,
            )
            if same_props and reporter.can_continue():
                compare_dataset_values(
                    left_obj,
                    right_obj,
                    rel_path,
                    child_path,
                    reporter,
                    np,
                    chunk_bytes,
                )

        if not reporter.can_continue():
            return


def compare_h5_file(
    left: Path,
    right: Path,
    rel_path: str,
    reporter: Reporter,
    chunk_bytes: int,
) -> None:
    h5py, np = load_h5_modules()
    try:
        with h5py.File(left, "r") as left_h5, h5py.File(right, "r") as right_h5:
            compare_h5_group(left_h5, right_h5, rel_path, "/", reporter, h5py, np, chunk_bytes)
    except OSError as exc:
        raise RuntimeError(f"failed to open HDF5 file pair {rel_path}: {exc}") from exc


def compare_entry_metadata(
    rel_path: str,
    left_entry: Entry,
    right_entry: Entry,
    reporter: Reporter,
    check_permissions: bool,
    check_mtime: bool,
) -> None:
    if left_entry.kind != right_entry.kind:
        reporter.add(rel_path, f"entry kind differs ({left_entry.kind} != {right_entry.kind})")
        return

    if left_entry.kind == "symlink" and left_entry.link_target != right_entry.link_target:
        reporter.add(
            rel_path,
            f"symlink target differs ({left_entry.link_target!r} != {right_entry.link_target!r})",
        )

    if check_permissions:
        left_mode = stat.S_IMODE(left_entry.mode)
        right_mode = stat.S_IMODE(right_entry.mode)
        if left_mode != right_mode:
            reporter.add(rel_path, f"permissions differ ({oct(left_mode)} != {oct(right_mode)})")

    if check_mtime and left_entry.mtime_ns != right_entry.mtime_ns:
        reporter.add(rel_path, f"mtime differs ({left_entry.mtime_ns} != {right_entry.mtime_ns})")


def compare_folders(args: argparse.Namespace) -> int:
    left_root = args.left.resolve()
    right_root = args.right.resolve()
    if not left_root.is_dir():
        print(f"ERROR: left path is not a directory: {left_root}", file=sys.stderr)
        return 2
    if not right_root.is_dir():
        print(f"ERROR: right path is not a directory: {right_root}", file=sys.stderr)
        return 2

    reporter = Reporter(max_diffs=args.max_diffs)
    hash_chunk_bytes = max(1, args.hash_chunk_mb) * 1024 * 1024
    h5_chunk_bytes = max(1, args.h5_chunk_mb) * 1024 * 1024

    if args.verbose:
        print(f"Indexing {left_root}", file=sys.stderr)
    left_entries = build_index(left_root, args.ignore)
    if args.verbose:
        print(f"Indexing {right_root}", file=sys.stderr)
    right_entries = build_index(right_root, args.ignore)

    left_paths = set(left_entries)
    right_paths = set(right_entries)

    for rel_path in sorted(left_paths - right_paths):
        if not reporter.add(rel_path, "exists only on left"):
            break
    for rel_path in sorted(right_paths - left_paths):
        if not reporter.add(rel_path, "exists only on right"):
            break

    for rel_path in sorted(left_paths & right_paths):
        if not reporter.can_continue():
            break

        left_entry = left_entries[rel_path]
        right_entry = right_entries[rel_path]
        compare_entry_metadata(
            rel_path,
            left_entry,
            right_entry,
            reporter,
            check_permissions=args.check_permissions,
            check_mtime=args.check_mtime,
        )
        if not reporter.can_continue() or left_entry.kind != right_entry.kind:
            continue

        if left_entry.kind != "file":
            continue

        left_file = left_root / rel_path
        right_file = right_root / rel_path
        if args.verbose:
            print(f"Comparing {rel_path}", file=sys.stderr)
        if is_h5_path(left_file):
            compare_h5_file(left_file, right_file, rel_path, reporter, h5_chunk_bytes)
            if args.byte_exact_h5 and reporter.can_continue():
                compare_raw_file(left_file, right_file, rel_path, reporter, hash_chunk_bytes)
        else:
            compare_raw_file(left_file, right_file, rel_path, reporter, hash_chunk_bytes)

    if reporter.lines:
        print("DIFFERENCES FOUND")
        for line in reporter.lines:
            print(f"  - {line}")
        if reporter.truncated:
            print(f"  - Output truncated after {args.max_diffs} differences.")
        print(f"\nTotal differences counted: {reporter.count}")
        return 1

    print("OK: folders are identical under the selected checks.")
    print(f"Compared: {left_root}")
    print(f"Against:  {right_root}")
    if args.ignore:
        print(f"Ignored patterns: {', '.join(args.ignore)}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return compare_folders(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
