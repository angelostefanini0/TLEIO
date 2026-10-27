"""Global calibration of two SC-EKF parameters on the TartanEvent office sequences.

One (network_scale, meas_cov_scale) pair is chosen for all sequences together, by the
mean of ATE / path length; nothing is tuned per sequence. Requires the network
predictions written by reproduce.py. Writes website/assets/figures/filter-calibration.{json,svg,png}.

  python scripts/project_page/tune_filter.py [--jobs 16]
"""
from pathlib import Path
from multiprocessing import Pool
from dataclasses import replace
import argparse, contextlib, io, itertools, json, os, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(Path(__file__).parent)]
PROC = ROOT/'data/tartanair/project_page_processed_train'; FIGS = ROOT/'website/assets/figures'
NETWORK_SCALES = [.58, .7, .8, .85, .9, .95, 1.0, 1.1]   # .58 / .33: repository defaults
COV_SCALES = [.33, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def ate(est, gt):
    p, q = est[:, 1:4], gt[:, 1:4]
    u, _, vt = np.linalg.svd((p - p.mean(0)).T @ (q - q.mean(0)))
    c = np.eye(3); c[-1, -1] = np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum(((p - p.mean(0)) @ (u @ c @ vt) + q.mean(0) - q)**2, axis=1))))


def evaluate(job):
    seq, ns, cs = job
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
    from main_filter import CONFIG, run_filter
    cfg = replace(CONFIG, dataset='tartanair', sequence=seq, processed_dir=PROC, output_root=ROOT/'outputs/project_page_tuning',
                  network_scale=ns, meas_cov_scale=cs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try: r = run_filter(cfg)
        except Exception: return seq, ns, cs, None, None
    est, gt = r['trajectory'], r['ground_truth']
    if not np.isfinite(est).all(): return seq, ns, cs, None, None
    return seq, ns, cs, ate(est, gt), float(np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=16); a = ap.parse_args()
    seqs = sorted(p.name for p in PROC.iterdir() if (p/f'{p.name}.txt').exists())
    jobs = list(itertools.product(seqs, NETWORK_SCALES, COV_SCALES))
    with Pool(a.jobs) as pool: rows = pool.map(evaluate, jobs, chunksize=1)
    grid = np.full((len(NETWORK_SCALES), len(COV_SCALES)), np.nan)
    for i, ns in enumerate(NETWORK_SCALES):
        for j, cs in enumerate(COV_SCALES):
            r = [x for x in rows if x[1] == ns and x[2] == cs]
            if all(x[3] is not None for x in r): grid[i, j] = 100*np.mean([x[3]/x[4] for x in r])
    i, j = np.unravel_index(np.nanargmin(grid), grid.shape); best = dict(network_scale=NETWORK_SCALES[i], meas_cov_scale=COV_SCALES[j])
    from main_filter import CONFIG
    out = dict(objective='mean over sequences of ATE / path length [%] (SE(3) alignment, scale fixed)', sequences=seqs,
               network_scales=NETWORK_SCALES, meas_cov_scales=COV_SCALES, grid_percent=np.where(np.isfinite(grid), grid, None).tolist(),
               best=best, best_percent=float(grid[i, j]),
               repository_default=dict(network_scale=CONFIG.network_scale, meas_cov_scale=CONFIG.meas_cov_scale),
               per_run=[dict(sequence=s, network_scale=ns, meas_cov_scale=cs, ate_m=e, path_m=l) for s, ns, cs, e, l in rows])
    FIGS.mkdir(parents=True, exist_ok=True); (FIGS/'filter-calibration.json').write_text(json.dumps(out, indent=1) + '\n')

    from style import plt, save, INK
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
    im = ax.imshow(grid, cmap='Blues_r', aspect='auto', vmin=np.nanmin(grid), vmax=np.nanmin(grid)*2.5)
    ax.set_xticks(range(len(COV_SCALES)), [f'{c:g}' for c in COV_SCALES]); ax.set_yticks(range(len(NETWORK_SCALES)), [f'{n:g}' for n in NETWORK_SCALES])
    ax.set(xlabel='Measurement covariance scale', ylabel='Network displacement scale')
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, '–' if np.isnan(v) else f'{v:.1f}', ha='center', va='center', fontsize=8,
                color='white' if np.isfinite(v) and v < np.nanmin(grid)*1.7 else INK, fontweight='bold' if (r, c) == (i, j) else 'normal')
    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='#c0392b', lw=1.6))
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); fig.colorbar(im, ax=ax, shrink=.85, label='Mean ATE / path length [%] ↓').outline.set_visible(False)
    save(fig, FIGS/'filter-calibration')
    print(json.dumps(dict(best=best, best_percent=out['best_percent']), indent=1))


if __name__ == '__main__': main()
