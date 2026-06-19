"""Export individual X/Y/Z/XY panels from existing filter result PNGs.

This script does not replot trajectories. It crops the already-generated
diagnostic PNGs and optionally applies a small raster ink boost so text and
lines read better in slides/papers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def crop_white_border(image: Image.Image, threshold: int = 251, pad: int = 8) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    nonwhite = np.any(array < threshold, axis=2)
    rows = np.where(np.any(nonwhite, axis=1))[0]
    cols = np.where(np.any(nonwhite, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return image
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, image.height)
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, image.width)
    return image.crop((c0, r0, c1, r1))


def rel_crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x0, y0, x1, y1 = box
    return image.crop(
        (
            round(x0 * image.width),
            round(y0 * image.height),
            round(x1 * image.width),
            round(y1 * image.height),
        )
    )


def boost_ink(image: Image.Image, passes: int) -> Image.Image:
    if passes <= 0:
        return image

    out = np.asarray(image.convert("RGBA")).copy()
    for _ in range(passes):
        rgb = out[..., :3].astype(np.int16)
        channel_range = rgb.max(axis=2) - rgb.min(axis=2)
        dark = rgb.min(axis=2) < 90
        saturated = (channel_range > 55) & (rgb.min(axis=2) < 190)
        ink = dark | saturated

        thickened = out.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted_pixels = np.roll(out, shift=(dy, dx), axis=(0, 1))
            shifted_ink = np.roll(ink, shift=(dy, dx), axis=(0, 1))
            if dy < 0:
                shifted_ink[dy:, :] = False
            elif dy > 0:
                shifted_ink[:dy, :] = False
            if dx < 0:
                shifted_ink[:, dx:] = False
            elif dx > 0:
                shifted_ink[:, :dx] = False
            target = shifted_ink & ~ink
            thickened[target] = shifted_pixels[target]
        out = thickened
    return Image.fromarray(out, mode="RGBA")


def upscale(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rel_box(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        round(x0 * image.width),
        round(y0 * image.height),
        round(x1 * image.width),
        round(y1 * image.height),
    )


def remove_top_legend(image: Image.Image) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    # Remove only the centered Ground Truth / TLEIO legend from the original PNG.
    draw.rectangle(rel_box(image, (0.36, 0.000, 0.64, 0.145)), fill=(255, 255, 255, 255))
    return image


def enlarge_time_label(image: Image.Image, scale: float) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = load_font(max(20, round(image.height * 0.072 * scale)), bold=True)
    # Cover the old bottom x-axis label and redraw only that label larger.
    draw.rectangle(rel_box(image, (0.43, 0.900, 0.58, 1.000)), fill=(255, 255, 255, 255))
    text = "Time [s]"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = image.width // 2 - text_w // 2
    y = round(image.height * 0.955) - text_h // 2
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)
    return image


def export_sequence(seq_dir: Path, out_root: Path, dpi: int, ink_boost: int, scale: float) -> None:
    sequence = seq_dir.name
    trajectory_path = seq_dir / f"{sequence}_trajectory_comparison.png"
    projections_path = seq_dir / f"{sequence}_projections.png"
    if not trajectory_path.exists() or not projections_path.exists():
        print(f"skip {sequence}: missing existing trajectory/projection PNG")
        return

    trajectory = crop_white_border(Image.open(trajectory_path).convert("RGBA"), pad=8)
    projections = crop_white_border(Image.open(projections_path).convert("RGBA"), pad=8)

    # Existing trajectory figure is X/Y/Z stacked vertically.
    panels = {
        "x": rel_crop(trajectory, (0.000, 0.000, 1.000, 0.350)),
        "y": rel_crop(trajectory, (0.000, 0.335, 1.000, 0.675)),
        "z": rel_crop(trajectory, (0.000, 0.650, 1.000, 1.000)),
        # Existing projection figure is [XY | XZ | YZ]; export only XY.
        "xy_2d": rel_crop(projections, (0.000, 0.000, 0.333, 1.000)),
    }

    seq_out = out_root / sequence
    seq_out.mkdir(parents=True, exist_ok=True)
    for name, panel in panels.items():
        panel = crop_white_border(panel, pad=6)
        if name == "x":
            panel = remove_top_legend(panel)
        if name == "z":
            panel = enlarge_time_label(panel, scale)
        panel = boost_ink(panel, ink_boost)
        panel = upscale(panel, scale)
        out_path = seq_out / f"{sequence}_{name}.png"
        panel.convert("RGB").save(out_path, dpi=(dpi, dpi))
        print(f"saved {out_path}")


def parse_sequence_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Folder with existing result PNGs in <sequence> subfolders.")
    parser.add_argument("--out-root", type=Path, required=True, help="Separate output folder for exported panels.")
    parser.add_argument("--sequence", type=str, default=None, help="Optional comma-separated sequence list.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ink-boost", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.25, help="Raster scale-up for larger readable text.")
    args = parser.parse_args()

    keep = parse_sequence_arg(args.sequence)
    for seq_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        if keep is not None and seq_dir.name not in keep:
            continue
        export_sequence(seq_dir, args.out_root, args.dpi, args.ink_boost, args.scale)


if __name__ == "__main__":
    main()
