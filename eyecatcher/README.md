# TartanAir Office Eye-Catcher

This folder contains a standalone script to create a paper eye-catcher from a
TartanAir v1 `office` sequence. It fuses RGB-D frames into a colored point cloud
using the provided camera poses, then overlays the camera trajectory.

The script expects a TartanAir v1 sequence with:

```text
image_left/
depth_left/
pose_left.txt
```

By default it looks for:

```text
data/tartanair/office/Easy/P000
```

Run from the repository root:

```bash
python eyecatcher/create_tartanair_office_eyecatcher.py
```

If the required Office sequence is not present yet, the script can download the
needed TartanAir v1 archives from Hugging Face and extract the selected
trajectory:

```bash
python eyecatcher/create_tartanair_office_eyecatcher.py --download-missing
```

This downloads `image_left.zip`, `depth_left.zip`, and `flow_mask.zip` for the
Office environment and selected difficulty. TartanAir packages these archives at
environment/difficulty level, so the download is larger than a single sequence,
but the script extracts only the requested trajectory when the archive layout
allows it.

Outputs are written to `eyecatcher/output`:

```text
office_scene.ply        # colored fused point cloud
office_trajectory.ply   # trajectory as a 3D line set
office_eyecatcher.png   # quick preview render
```

Useful overrides:

```bash
python eyecatcher/create_tartanair_office_eyecatcher.py \
  --sequence-root data/tartanair/office/Easy/P001 \
  --frame-stride 8 \
  --pixel-stride 3 \
  --max-depth 50 \
  --voxel-size 0.06
```

If the point cloud is too sparse, reduce `--pixel-stride` or `--frame-stride`.
If it is too large, increase `--voxel-size`, `--pixel-stride`, or
`--frame-stride`.

To make the trajectory stand out while preserving the same metric path, keep
`--trajectory-lift 0.0` and use the halo/opacity controls. If the path still
looks like sparse dots in the preview, disable the sparse markers and render
extra samples along the same centerline:

```bash
python eyecatcher/create_tartanair_office_eyecatcher.py \
  --trajectory-lift 0.0 \
  --trajectory-linewidth 7.0 \
  --trajectory-halo-linewidth 14.0 \
  --trajectory-marker-step 0 \
  --trajectory-sample-spacing 0.04 \
  --trajectory-sample-size 10 \
  --point-alpha 0.34
```

The preview viewpoint can be changed with `--view-elev` and `--view-azim`.
For example:

```bash
python eyecatcher/create_tartanair_office_eyecatcher.py \
  --view-elev 22 \
  --view-azim -105
```

To create an eye-catcher directly from an RGB frame, with the camera path up to
that point reprojected into the image, run:

```bash
python eyecatcher/create_reprojected_trajectory_eyecatcher.py
```

The script automatically chooses a sufficiently late frame that keeps a large
fraction of its trajectory history in view. It checks the frame's depth image
so trajectory sections hidden by scene geometry do not count as visible and are
not drawn. Pass `--frame N` to override the selection. The result is written to
`eyecatcher/output/office_reprojected_trajectory.png`.

To browse every frame interactively before choosing one, run:

```bash
python eyecatcher/play_reprojected_trajectory.py
```

Press space to pause, use the left/right arrows (or `A`/`D`) to step through
frames, press `S` to save the current candidate, and press `Q` to quit. The
current frame number is shown in the upper-left corner.

For the preview image, the script crops the point cloud around the trajectory
and trims isolated outliers by default. The current eye-catcher was generated
with:

```bash
python eyecatcher/create_tartanair_office_eyecatcher.py \
  --max-depth 18 \
  --trajectory-margin-xy 4 \
  --trajectory-margin-z 2 \
  --voxel-size 0.04 \
  --frame-stride 5 \
  --pixel-stride 2 \
  --max-frames 280 \
  --max-render-points 450000 \
  --trajectory-lift 0.0 \
  --trajectory-linewidth 4.0 \
  --trajectory-halo-linewidth 9.0 \
  --trajectory-marker-step 20 \
  --trajectory-marker-size 16 \
  --point-size 0.06 \
  --point-alpha 0.34
```
