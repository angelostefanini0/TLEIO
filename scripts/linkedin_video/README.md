# LinkedIn video

`make_video.py` renders the TLEIO LinkedIn video frame by frame: title card, the demo (RGB and events side by side, then events and the live trajectory), the demo panels flying into the method diagram, and a closing card.

```bash
.venv/bin/python scripts/linkedin_video/make_video.py           # render
.venv/bin/python scripts/linkedin_video/make_video.py --final   # render and copy the full-res video to outputs/linkedin/
```

Full-resolution renders and the render cache go to `~/tleio_linkedin_renders/`. A 480p preview is written to `outputs/linkedin/tleio_linkedin_preview.mp4`. The timeline is cut at section boundaries and each piece is cached under a hash of the code it depends on, so editing one section re-renders only that section.

## Inputs

| Input | Source |
| --- | --- |
| Demo (events and trajectory) | `website/assets/tleio-demo.mp4` |
| Method diagram | `assets/model_arch.pdf` (the paper figure), rasterized with `pdftocairo` (Poppler) |
| ETH Zürich and Google logos | `assets/`, from Wikimedia Commons |
| RGB frames of the demo sequence | TartanAir competition `MH001`, not in git (see below) |

The demo shows TartanAir competition sequence `MH001`. Its RGB frames (357 PNGs, about 316 MB) are expected in `data/tartanair/competition/rgb/mono/MH001/`. They come from the TartanAir competition archive (8.2 GB). You can stream it and extract only this sequence:

```bash
mkdir -p data/tartanair/competition/rgb
curl -sSL "https://drive.usercontent.google.com/download?id=1N9BkpQuibIyIBkLxVPUuoB-eDOMFqY8D&export=download&confirm=t" \
  | tar -xz -C data/tartanair/competition/rgb ./mono/MH001/
```

Requirements: the repository environment (NumPy, Pillow, imageio with imageio-ffmpeg) and Noto Sans in `/usr/share/fonts/truetype/noto`.
