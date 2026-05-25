from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_ROOT = REPO_ROOT / "cfg"


def default_config_path(name: str) -> Path:
    return CFG_ROOT / f"{name}.yaml"


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    default_config: str | Path,
    required: tuple[str, ...] = (),
) -> argparse.Namespace:
    """Parse YAML defaults first, then let CLI arguments override them."""
    default_config = Path(default_config)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="YAML file with parser defaults. CLI arguments override this file.",
    )

    _disable_argparse_required(parser)
    config_probe, _ = parser.parse_known_args()
    config_values = _load_config(config_probe.config)
    _apply_config_defaults(parser, config_values)

    args = parser.parse_args()
    _validate_required(args, required)
    return args


def parse_known_args_with_config(
    parser: argparse.ArgumentParser,
    default_config: str | Path,
    required: tuple[str, ...] = (),
) -> tuple[argparse.Namespace, list[str]]:
    """Like parse_args_with_config, preserving unknown CLI args."""
    default_config = Path(default_config)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="YAML file with parser defaults. CLI arguments override this file.",
    )

    _disable_argparse_required(parser)
    config_probe, _ = parser.parse_known_args()
    config_values = _load_config(config_probe.config)
    _apply_config_defaults(parser, config_values)

    args, unknown = parser.parse_known_args()
    _validate_required(args, required)
    return args, unknown


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        if path == default_config_path(path.stem):
            return {}
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r") as fh:
        loaded = yaml.safe_load(fh) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping of argument names to values.")
    return loaded


def _apply_config_defaults(
    parser: argparse.ArgumentParser,
    config_values: dict[str, Any],
) -> None:
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(key for key in config_values if key not in actions)
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")

    coerced = {
        key: _coerce_value(actions[key], value)
        for key, value in config_values.items()
    }
    parser.set_defaults(**coerced)

    for action in parser._actions:
        if getattr(action, "required", False):
            action.required = False


def _disable_argparse_required(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if getattr(action, "required", False):
            action.required = False


def _coerce_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        return None

    if isinstance(action, argparse._StoreTrueAction | argparse._StoreFalseAction):
        return bool(value)

    converter = getattr(action, "type", None)
    if converter is None:
        return value

    if isinstance(value, list):
        return [converter(item) for item in value]

    return converter(value)


def _validate_required(args: argparse.Namespace, required: tuple[str, ...]) -> None:
    missing = [
        name
        for name in required
        if getattr(args, name, None) in (None, "")
    ]
    if missing:
        raise SystemExit(
            "Missing required argument(s): "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            + ". Provide them on the CLI or in the YAML config."
        )
