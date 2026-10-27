"""Figures and data for results reported in the paper (arXiv:2610.33356, Tables 1-2, Sec. 4.3).
These are published values, not rerun here; local runs are rendered by render_sequences.py.
"""
from pathlib import Path
import json, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from style import BLUE, GRID, BASELINES, plt, save

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT/'website/assets/benchmarks'; OUT.mkdir(parents=True, exist_ok=True)

TARTANAIR = dict(
    sequences=[f'ME{i:02}' for i in range(8)] + [f'MH{i:02}' for i in range(8)],
    methods={'EKLT-VIO': ('E+F+I', [12.80, 9.10, 16.23, 18.08, 16.96, 6.84, 21.44, 9.06, 14.68, 4.35, 10.20, 2.13, 2.72, 4.48, 6.95, 19.72], 10.98),
             'ORB-SLAM3': ('F', [13.61, 16.86, 20.57, 16.00, 22.27, 9.28, 21.61, 7.74, 15.44, 2.92, 13.51, 8.18, 2.59, 21.91, 11.70, 25.88], 14.38),
             'COLMAP': ('F', [15.20, 5.58, 10.86, 3.93, 2.62, 14.78, 7.00, 18.47, 12.26, 13.45, 13.45, 20.95, 24.97, 16.79, 7.01, 7.97], 12.50),
             'DROID-VO': ('F', [.22, .15, .24, 1.27, 1.04, .14, 1.32, .77, .32, .13, .08, .09, 1.52, .69, .39, .97], .58),
             'DPVO': ('F', [.16, .11, .11, .66, .31, .14, .30, .13, .21, .04, .04, .08, .58, .17, .11, .15], .21),
             'RAMP-VO': ('E+F', [.20, .04, .10, .46, .16, .13, .12, .12, .36, .06, .04, .04, .41, .25, .11, .07], .17),
             'TLIO': ('I', [1.45, 2.30, 1.42, 7.31, 1.14, 1.85, 2.47, 1.37, 2.39, .94, 1.61, 3.72, 1.87, 1.35, 1.25, 1.66], 2.26),
             'TLEIO': ('E+I', [2.13, 1.20, 1.45, 1.37, 1.37, 1.19, .48, .72, 2.65, .14, 1.16, .22, .60, .48, 1.15, 1.44], 1.10)})
DAVIS = dict(
    sequences=['boxes_translation', 'hdr_boxes', 'boxes_6dof', 'poster_translation', 'hdr_poster', 'poster_6dof'],
    methods={'EKLT-VIO': ('E+F+I', [.48, .46, .84, .35, .65, .35], .52), 'Lee et al.': ('E+F+I', [.74, .69, .77, .28, .52, .59], .60),
             'Rebecq et al.': ('E+I', [.57, .92, .69, .89, .59, .82], .75), 'HASTE-VIO': ('E+I', [2.55, 1.75, 2.03, 1.34, .57, 1.50], 1.62),
             'Zhu et al.': ('E+I', [2.69, 1.23, 3.61, .94, 2.63, 3.56], 2.44), 'Ultimate-SLAM': ('E+I', [.76, .67, .44, .15, .49, .30], .47),
             'DEIO': ('E+I', [.07, .09, .05, .04, .06, .08], .07), 'DEVO': ('E', [.06, .06, .71, .06, .14, .44], .25),
             'TLEIO': ('E+I', [.31, .55, .49, .43, .50, .43], .45)})
RUNTIME = dict(methods=['TLEIO', 'RAMP-VO', 'DEVO', 'DEIO'], runs=5, warmup_runs=2,
               hardware='NVIDIA RTX 4000 Ada 20 GB · AMD Threadripper 7960X',
               events_millions={'EDS 00': 229.8, 'EDS 01': 340.8, 'EDS 09': 120.1},
               mean_seconds={'EDS 00': [47.847, 67.0, 70.2, 71.4], 'EDS 01': [75.205, 98.1, 133.8, 134.0], 'EDS 09': [48.863, 39.454, 88.1, 95.7]},
               tleio_std_seconds={'EDS 00': .196, 'EDS 01': .260, 'EDS 09': .332}, ramp_vo_std_seconds={'EDS 09': .575},
               totals_seconds=[171.915, 204.555, 292.130, 301.051],
               precision='Baseline per-sequence means as labelled in paper Fig. 8; totals as reported in Sec. 4.3')

# TartanAir: TLEIO against its IMU-only counterpart TLIO, per sequence.
seqs = TARTANAIR['sequences']; tlio = TARTANAIR['methods']['TLIO'][1]; tleio = TARTANAIR['methods']['TLEIO'][1]
fig, ax = plt.subplots(figsize=(11, 3.4), layout='constrained'); x = np.arange(len(seqs)); w = .38
ax.bar(x - w/2, tlio, w, color=BASELINES[0], label='TLIO  (IMU)  ·  avg 2.26 m')
ax.bar(x + w/2, tleio, w, color=BLUE, label='TLEIO  (events + IMU)  ·  avg 1.10 m')
ax.set_xticks(x, seqs, fontsize=8.5); ax.set_ylabel('ATE [m] ↓'); ax.set_ylim(0, 7.8); ax.set_xlim(-.6, len(seqs) - .4)
ax.grid(axis='y', color=GRID, lw=.7); ax.set_axisbelow(True); ax.legend(loc='upper right', fontsize=9)
ax.axvline(7.5, color='#c8c8c8', lw=.8, ls=':')
save(fig, OUT/'tartanair')

# Runtime: per recording, end-to-end, mean of five runs.
fig, axs = plt.subplots(1, 3, figsize=(11, 3.1), layout='constrained', sharey=True)
colors = [BLUE] + BASELINES
for ax, (name, vals) in zip(axs, RUNTIME['mean_seconds'].items()):
    ax.bar(range(4), vals, width=.64, color=colors); ax.set_xticks(range(4), RUNTIME['methods'], fontsize=8.5)
    ax.set_title(f'{name}  ·  {RUNTIME["events_millions"][name]:.1f}M events', fontsize=10)
    ax.grid(axis='y', color=GRID, lw=.7); ax.set_axisbelow(True); ax.set_ylim(0, 155)
    for i, v in enumerate(vals): ax.text(i, v + 3, f'{v:.1f}', ha='center', fontsize=8.5, fontweight='bold' if v == min(vals) else 'normal')
axs[0].set_ylabel('Runtime [s] ↓')
save(fig, OUT/'runtime')

(OUT/'reported-results.json').write_text(json.dumps(dict(
    source='arXiv:2610.33356v1, Tables 1-2 and Sec. 4.3', status='Published values; not rerun for this page',
    tartanair_ate_m=TARTANAIR, davis240c_mpe_percent=DAVIS, runtime=RUNTIME), indent=1, ensure_ascii=False) + '\n')
print('Rendered paper result figures.')
