# TLEIO project page

Static GitHub Pages site for *Tight Learned Event-Inertial Odometry (TLEIO)*, [arXiv:2610.33356](https://arxiv.org/abs/2610.33356). GitHub Actions deploys `website/` when changes reach `main`; the page is served at `https://angelostefanini0.github.io/TLEIO/`.

Preview locally (the sequence explorer loads JSON, so open it over HTTP, not `file://`):

```bash
python -m http.server 8000 -d website
```

## What is on the page

| Section | Source |
| --- | --- |
| Teaser video, method figures | Repository demo and `figures/method/` |
| Qualitative results | Local runs of the released checkpoint, rendered by `scripts/project_page/` |
| Benchmark (TartanAir and runtime plots, tables) | Values published in the paper, `assets/benchmarks/reported-results.json` |

## Reproduce the local runs

The runs cover all 15 sequences of the TartanEvent `office` environment (Easy P000–P006, Hard P000–P007). The raw office event archive is about 21 GB. With the `tleio` environment from `../environment.yaml` active:

```bash
git submodule update --init --recursive
python scripts/project_page/reproduce.py --repo .               # all office sequences
python scripts/project_page/reproduce.py --repo . --sequences Hard/P003   # or a subset
python scripts/project_page/render_sequences.py                 # videos and figures
python scripts/project_page/render_paper_results.py             # paper plots
```

`reproduce.py` downloads the released checkpoint and `args.txt` when absent, processes the sequences, synthesizes IMU measurements, precomputes derotated event voxels, runs EventsFormer and the SC-EKF, and writes poses, metrics and a provenance manifest to `website/results/`. `render_sequences.py` renders, per sequence, a synchronized events / 3D-trajectory video (latest 60k raw events, 2× speed) and an aligned comparison figure. Its summary, `assets/figures/sequences.json`, drives the sequence player. `tune_filter.py` calibrates the two filter scales once for all sequences. `style.py` holds the shared palette and typeface.

## Protocol of the local runs

The office environment is part of the TartanEvent training data. Event derotation and IMU synthesis (200 Hz) use ground truth, and the filter starts from the ground-truth pose. The network displacement scale and measurement-covariance scale are calibrated once for all sequences (0.85, 4.0); nothing is tuned per sequence. Trajectories are rigidly SE(3)-aligned to ground truth with scale fixed, and ATE is independently checked with `evo`. These runs show how the pipeline behaves; they are not held-out results. The held-out benchmark numbers on the page are the published ones.
