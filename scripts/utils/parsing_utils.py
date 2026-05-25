from __future__ import annotations
from pathlib import Path




def parse_sequence_selection(value: str, available_names: list[str]) -> set[str]:
    """
    This method takes numbers or names of sequences to have as validation or testing sequences
    """
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


def parse_single_sequence_selector(
    value: str,
    available_names: list[str],
    arg_name: str,
) -> str | None:
    """Resolve an optional single sequence selector passed as index or full name."""
    if not value.strip():
        return None

    selected = parse_sequence_selection(value, available_names)
    if len(selected) != 1:
        raise ValueError(
            f"{arg_name} accepts exactly one sequence, got: '{value}'."
        )

    return next(iter(selected))


def iter_tartan_sequences(input_path: Path) -> list[tuple[str, Path]]:
    """Enumerate Tartan sequences as `(flattened_name, sequence_dir)` pairs."""
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
) -> list[str]:
    """List GT files required for processing that are missing in a sequence."""
    missing = []

    for filename in files_to_copy:
        if not (seq_dir / filename).exists():
            missing.append(filename)

    if files_to_copy and not (seq_dir / "imu" / "cam_time.txt").exists():
        missing.append("imu/cam_time.txt")

    return missing
