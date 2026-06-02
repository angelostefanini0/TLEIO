import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.config import default_config_path, parse_known_args_with_config


REQUIRED_SEQUENCE_FILES = ("derotated_voxels.npy", "relative_motions.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_root", type=str, default=None)
    parser.add_argument("--checkpoint_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--average_overlaps", action="store_true")
    parser.add_argument(
        "--raw_output",
        action="store_true",
        help="Save one raw model-output file per sequence.",
    )
    args, extra_args = parse_known_args_with_config(
        parser,
        default_config_path("batch_test"),
        required=("batch_root", "checkpoint_file", "output_dir"),
    )

    batch_root = Path(args.batch_root)
    checkpoint_file = Path(args.checkpoint_file)
    output_dir = Path(args.output_dir)
    test_script = Path(__file__).resolve().with_name("test.py")

    if not batch_root.is_dir():
        raise SystemExit(f"Batch root does not exist or is not a directory: {batch_root}")
    if not test_script.is_file():
        raise SystemExit(f"Could not find test.py next to batch_test.py: {test_script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = sorted(path for path in batch_root.iterdir() if path.is_dir())
    valid_sequences = []
    for sequence_dir in sequence_dirs:
        missing = [
            filename
            for filename in REQUIRED_SEQUENCE_FILES
            if not (sequence_dir / filename).exists()
        ]
        if missing:
            print(
                f"Warning: skipping {sequence_dir.name}; missing {', '.join(missing)}",
                file=sys.stderr,
            )
            continue
        valid_sequences.append(sequence_dir)

    if not valid_sequences:
        raise SystemExit(
            f"No valid sequences found in {batch_root}. "
            f"Expected files: {', '.join(REQUIRED_SEQUENCE_FILES)}"
        )

    for sequence_dir in valid_sequences:
        output_file = output_dir / f"{sequence_dir.name}.txt"

        print(f"Testing {sequence_dir.name} -> {output_file}")
        command = [
            sys.executable,
            str(test_script),
            "--sequence_dir",
            str(sequence_dir),
            "--checkpoint_file",
            str(checkpoint_file),
            "--output_file",
            str(output_file),
        ]
        if args.raw_output:
            command.extend(
                [
                    "--save_raw_outputs",
                    "--raw_model_output_file",
                    str(output_dir / f"{sequence_dir.name}_raw.txt"),
                ]
            )
        if args.average_overlaps:
            command.append("--average_overlaps")
        command.extend(extra_args)

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error: sequence failed: {sequence_dir.name}", file=sys.stderr)
            raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
