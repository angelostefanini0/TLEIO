"""Compose existing filter PNGs without replotting trajectory data."""

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


def xy_projection_from_existing(projections: Image.Image) -> Image.Image:
    # Existing projection figures are laid out as [XY | XZ | YZ].
    return projections.crop((0, 0, projections.width // 3, projections.height))


def boost_ink(image: Image.Image, passes: int) -> Image.Image:
    """Thicken dark/saturated pixels slightly without replotting the source data."""

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


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


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


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(0, 0, 0, 255)) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width // 2, xy[1] - height // 2), text, font=font, fill=fill)


def draw_rotated_centered(image: Image.Image, center: tuple[int, int], text: str, font) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    label = Image.new("RGBA", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((6 - bbox[0], 6 - bbox[1]), text, font=font, fill=(0, 0, 0, 255))
    rotated = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2))


def rel_box(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        round(x0 * image.width),
        round(y0 * image.height),
        round(x1 * image.width),
        round(y1 * image.height),
    )


def rel_point(image: Image.Image, point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0] * image.width), round(point[1] * image.height))


def adjust_trajectory_panel_text(image: Image.Image, label_scale: float) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    white = (255, 255, 255, 255)

    # Remove the top legend only. Keep subplot titles, ticks, grids, and curves.
    draw.rectangle(rel_box(image, (0.35, 0.000, 0.66, 0.055)), fill=white)

    label_font = load_font(max(18, round(image.height * 0.021 * label_scale)), bold=True)
    time_font = load_font(max(18, round(image.height * 0.019 * label_scale)), bold=True)

    # Enlarge y-axis labels without touching tick labels or plotted curves.
    for box in (
        (0.000, 0.150, 0.045, 0.305),
        (0.000, 0.470, 0.045, 0.625),
        (0.000, 0.795, 0.045, 0.950),
    ):
        draw.rectangle(rel_box(image, box), fill=white)

    draw_rotated_centered(image, rel_point(image, (0.021, 0.228)), "X [m]", label_font)
    draw_rotated_centered(image, rel_point(image, (0.021, 0.548)), "Y [m]", label_font)
    draw_rotated_centered(image, rel_point(image, (0.021, 0.870)), "Z [m]", label_font)

    # Enlarge only the bottom time label.
    draw.rectangle(rel_box(image, (0.46, 0.960, 0.56, 1.000)), fill=white)
    text = "Time [s]"
    draw_centered(draw, (image.width // 2, round(image.height * 0.982)), text, time_font)
    return image


def compose_sequence(
    seq_dir: Path,
    out_suffix: str,
    dpi: int,
    gap: int,
    ink_boost: int,
    label_scale: float,
) -> Path | None:
    sequence = seq_dir.name
    trajectory_path = seq_dir / f"{sequence}_trajectory_comparison.png"
    projections_path = seq_dir / f"{sequence}_projections.png"
    if not trajectory_path.exists() or not projections_path.exists():
        print(f"skip {sequence}: missing existing trajectory/projection PNG")
        return None

    trajectory = crop_white_border(Image.open(trajectory_path), pad=10).convert("RGBA")
    xy_projection = crop_white_border(xy_projection_from_existing(Image.open(projections_path)), pad=10).convert("RGBA")
    xy_projection = resize_to_height(xy_projection, trajectory.height)

    trajectory = adjust_trajectory_panel_text(trajectory, label_scale)
    trajectory = boost_ink(trajectory, ink_boost)
    xy_projection = boost_ink(xy_projection, ink_boost)

    canvas = Image.new(
        "RGBA",
        (trajectory.width + gap + xy_projection.width, trajectory.height),
        (255, 255, 255, 255),
    )
    canvas.alpha_composite(trajectory, (0, 0))
    canvas.alpha_composite(xy_projection, (trajectory.width + gap, 0))
    out_path = seq_dir / f"{sequence}{out_suffix}.png"
    canvas.convert("RGB").save(out_path, dpi=(dpi, dpi))
    print(f"saved {out_path}")
    return out_path


def parse_sequence_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Folder with <sequence> subfolders containing existing PNGs.")
    parser.add_argument("--sequence", type=str, default=None, help="Optional comma-separated sequence list.")
    parser.add_argument("--out-suffix", default="_wacv_layout", help="Suffix for composed output PNGs.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--gap", type=int, default=24, help="Pixel gap between trajectory and XY panels.")
    parser.add_argument(
        "--label-scale",
        type=float,
        default=1.0,
        help="Scale for the replacement Time [s] label in the composed trajectory panel.",
    )
    parser.add_argument(
        "--ink-boost",
        type=int,
        default=1,
        help="Pixel-thickening passes for text and colored lines in the composed output only.",
    )
    args = parser.parse_args()

    keep = parse_sequence_arg(args.sequence)
    for seq_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        if keep is not None and seq_dir.name not in keep:
            continue
        compose_sequence(seq_dir, args.out_suffix, args.dpi, args.gap, args.ink_boost, args.label_scale)


if __name__ == "__main__":
    main()
