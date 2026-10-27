"""Render project-page media for every locally processed TartanEvent sequence.

Inputs are the saved outputs of reproduce.py (website/results/<seq>) and the
processed raw event streams. Nothing is smoothed or re-estimated: trajectories
are the exported filter estimates, rigidly SE(3)-aligned (scale fixed) to
ground truth. Event frames are raw: the most recent 60k events (at most 10 ms).

  python scripts/project_page/render_sequences.py [--jobs 6] [--no-video]
"""
from pathlib import Path
from contextlib import closing
from multiprocessing import Pool
import argparse, json, sys
import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from style import BLUE, GT, RED, INK, MUTED, GRID, POS, NEG, AXIS_RGB, save as save_fig
RESULTS = ROOT/'website/results'; PROC = ROOT/'data/tartanair/project_page_processed_train'
VIDEOS = ROOT/'website/assets/videos'; FIGS = ROOT/'website/assets/figures'
FPS, WINDOW_US = 15, 1000   # TartanEvent is dense: 1 ms already holds 10^5 events
TARGET_EVENTS, MAX_WINDOW_US = 60000, 10000   # videos: adaptive window, latest 60k events
SPEED = 2                    # playback speed; keeps the fifteen videos around 60 MB


def label(seq):
    _, diff, name = seq.split('_'); return f'Office · {diff} · {name}'


def load(seq):
    """Estimate as produced by the filter (raw, starts at the ground-truth pose), ground truth,
    and the rigid SE(3) alignment of the estimate used for ATE."""
    d = RESULTS/seq; e = np.loadtxt(d/'estimate.tum'); g = np.loadtxt(d/'groundtruth.tum')
    p, q = e[:, 1:4], g[:, 1:4]
    u, _, vt = np.linalg.svd((p - p.mean(0)).T @ (q - q.mean(0)))
    c = np.eye(3); c[-1, -1] = np.linalg.det(u @ vt); r = u @ c @ vt
    aligned = (p - p.mean(0)) @ r + q.mean(0)
    rot = r.T @ Rotation.from_quat(e[:, 4:8]).as_matrix()   # aligned body-to-world rotations
    return dict(t=e[:, 0] - e[0, 0], t0=e[0, 0], p=aligned, q=q, rot=rot,
                p_raw=p, rot_raw=Rotation.from_quat(e[:, 4:8]).as_matrix(),
                metrics=json.loads((d/'metrics.json').read_text()))


def event_image(reader, t_us, target=None):
    """Raw events before t_us: a fixed WINDOW_US window, or (target) the latest `target` events within 10 ms."""
    im = np.full((480, 640, 3), 255, np.uint8)
    ev = reader.get_events(max(0, t_us - (MAX_WINDOW_US if target else WINDOW_US)), t_us)
    if ev is None or not len(ev['x']): return (im, 0, 0.) if target else (im, 0)
    x, y, pol, t = ev['x'].astype(int), ev['y'].astype(int), ev['p'], ev['t']
    if target: x, y, pol, t = x[-target:], y[-target:], pol[-target:], t[-target:]
    ok = (x >= 0) & (x < 640) & (y >= 0) & (y < 480); x, y, pol = x[ok], y[ok], pol[ok]
    im[y[pol == 0], x[pol == 0]] = NEG; im[y[pol > 0], x[pol > 0]] = POS   # positive drawn last, as in the repository demo
    return (im, int(ok.sum()), (t_us - float(t[0]))/1000) if target else (im, int(ok.sum()))


def thicken(im):
    """Draw each event as a 2×2 dot, as bold as in the repository demo."""
    out = im.copy(); ink = im.min(2) < 255
    for dy, dx in ((0, 1), (1, 0), (1, 1)):
        m = np.zeros_like(ink); m[dy:, dx:] = ink[:ink.shape[0]-dy, :ink.shape[1]-dx]
        src = np.zeros_like(im); src[dy:, dx:] = im[:im.shape[0]-dy, :im.shape[1]-dx]
        fill = m & ~ink; out[fill] = src[fill]
    return out


def cube(ax, pts, pad=1.12):
    """Equal metric scale on all axes; the box is only as tall as the motion needs."""
    c = (pts.min(0) + pts.max(0))/2; s = max(np.ptp(pts[:, :2], axis=0).max()*pad/2, 1.0)
    zr = min(max(np.ptp(pts[:, 2])*pad/(2*s), .3), 1.0)
    ax.set_xlim(c[0]-s, c[0]+s); ax.set_ylim(c[1]-s, c[1]+s); ax.set_zlim(c[2]-s*zr, c[2]+s*zr); ax.set_box_aspect((1, 1, zr))
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3 if zr < .6 else 5))
    return s


def style3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0)); a._axinfo['grid'].update(color=GRID, linewidth=.6); a.line.set_color('#bbbbbb')
    ax.tick_params(labelsize=8, colors='#666', pad=0)
    ax.set_xlabel('x [m]', labelpad=-2, fontsize=9); ax.set_ylabel('y [m]', labelpad=-2, fontsize=9); ax.set_zlabel('z [m]', labelpad=-1, fontsize=9)


def render_video(seq):
    """Synchronized raw events and 3D trajectory with the estimated body frame, after the repository demo.
    The estimate is shown as the filter produces it, without alignment: it starts at the ground-truth pose
    and the gap that opens up is the accumulated drift."""
    import imageio.v2 as imageio
    from PIL import Image
    from src.learning.dataloader.events_to_voxel.reader import EDSReader
    s = load(seq); t, p, q, rot = s['t'], s['p_raw'], s['q'], s['rot_raw']; dur = t[-1]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.text(.5, .925, f'TLEIO  ·  {label(seq)}', ha='center', fontsize=16, fontweight='bold', color=INK)
    ev_ax = fig.add_axes([.025, .13, .5, .69]); ev_ax.axis('off')
    img = ev_ax.imshow(np.full((480, 640, 3), 255, np.uint8)); ev_title = ev_ax.set_title('', fontsize=11, color=INK, pad=8)
    ax = fig.add_axes([.5, .03, .46, .84], projection='3d'); style3d(ax)
    ax.plot(*q.T, ls=(0, (4, 3)), color='#b9b9b9', lw=1.0)
    gt_line, = ax.plot([], [], [], ls=(0, (4, 3)), color=GT, lw=1.3, label='Ground truth')
    est_line, = ax.plot([], [], [], color=BLUE, lw=1.9, label='TLEIO')
    ax.scatter(*q[0], color='black', s=18, depthshade=False, label='Start')
    span = cube(ax, np.vstack([p, q])); L = span*.16
    axes = [ax.plot([], [], [], color=c, lw=2.2, solid_capstyle='round')[0] for c in AXIS_RGB]
    dot, = ax.plot([], [], [], 'o', color=BLUE, ms=6, mec='white', mew=1.5)
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.0, .98))
    info = fig.text(.56, .83, '', fontsize=9, color='#444', family='Noto Sans Mono', va='top', linespacing=1.5)
    fig.text(.03, .045, 'TartanEvent office · estimate as produced by the filter, no alignment · body frame: x red, y green, z blue', fontsize=9, color=MUTED)
    fig.text(.97, .045, f'{SPEED}× speed  ·  arXiv:2610.33356', ha='right', fontsize=9, color=MUTED)
    out = VIDEOS/f'{seq}.mp4'; n = int(np.ceil(dur*FPS/SPEED)) + 1
    writer = imageio.get_writer(str(out), fps=FPS, codec='libx264', quality=None, macro_block_size=None, pixelformat='yuv420p',
                                ffmpeg_params=['-crf', '29', '-preset', 'slow', '-movflags', '+faststart'])
    with closing(EDSReader(PROC/seq/'events.h5', PROC/seq/'events_meta.h5')) as reader, writer:
        for k in range(n):
            now = min(k*SPEED/FPS, dur); i = max(1, np.searchsorted(t, now, side='right'))
            im, count, win = event_image(reader, int(round((s['t0'] + now)*1e6)), TARGET_EVENTS)
            img.set_data(thicken(im)); ev_title.set_text(f'Raw events  ·  {win:.1f} ms  ·  {count:,} events')
            gt_line.set_data_3d(*q[:i].T); est_line.set_data_3d(*p[:i].T)
            for j, a in enumerate(axes): a.set_data_3d(*np.stack([p[i-1], p[i-1] + L*rot[i-1][:, j]]).T)
            dot.set_data_3d([p[i-1, 0]], [p[i-1, 1]], [p[i-1, 2]])
            info.set_text(f'time   {now:5.1f} / {dur:.1f} s\ndrift  {np.linalg.norm(p[i-1] - q[i-1]):5.2f} m')
            ax.view_init(elev=24, azim=-60 + 50*now/dur)
            fig.canvas.draw(); frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); writer.append_data(frame)
        for _ in range(FPS): writer.append_data(frame)
        Image.fromarray(frame).save(out.with_suffix('.jpg'), quality=90)   # poster: the completed trajectory
    plt.close(fig); print(seq, 'video', n, 'frames', flush=True)
    return dict(file=out.name, sequence=seq, frames=n + FPS, fps=FPS, playback_speed=SPEED, event_window=f'latest {TARGET_EVENTS} events, at most {MAX_WINDOW_US/1000:g} ms')



def profiles_figure(seq, s):
    t, p, q = s['t'], s['p'], s['q']
    fig = plt.figure(figsize=(10, 4.2), layout='constrained'); g = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])
    for j in range(3):
        ax = fig.add_subplot(g[j, 0]); ax.plot(t, q[:, j], color=GT, ls=(0, (4, 2.5)), lw=1.2); ax.plot(t, p[:, j], color=BLUE, lw=1.3)
        ax.set_ylabel('xyz'[j] + ' [m]'); ax.grid(color=GRID, lw=.6); ax.set_xlim(0, t[-1])
        if j < 2: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel('Time [s]')
    ax = fig.add_subplot(g[:, 1]); ax.plot(q[:, 0], q[:, 1], color=GT, ls=(0, (4, 2.5)), lw=1.3, label='Ground truth')
    ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.4, label='TLEIO'); ax.scatter(q[0, 0], q[0, 1], color='black', s=16, zorder=5, label='Start')
    ax.scatter(p[-1, 0], p[-1, 1], marker='x', color=RED, s=28, zorder=5, label='End'); ax.set(xlabel='x [m]', ylabel='y [m]')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(color=GRID, lw=.6); ax.legend(fontsize=8, loc='best')
    fig.savefig(RESULTS/seq/'aligned-comparison.png', dpi=170); plt.close(fig)



def save(fig, name, svg=True): save_fig(fig, FIGS/name, svg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--no-video', action='store_true'); a = ap.parse_args()
    for d in (FIGS, VIDEOS): d.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in RESULTS.iterdir() if (p/'estimate.tum').exists())
    data = {s: load(s) for s in seqs}
    for s in seqs: profiles_figure(s, data[s])
    summary = [dict(sequence=s, label=label(s), **{k: data[s]['metrics'][k] for k in ('ate_se3_rmse_m', 'rotation_rmse_deg', 'duration_s', 'path_length_m')}) for s in seqs]
    media = []
    if not a.no_video:
        with Pool(a.jobs) as pool: media = pool.map(render_video, seqs)
    out = dict(source='website/results/<seq> exported by reproduce.py; raw TartanEvent event streams',
               alignment='Rigid SE(3) fit over all matched samples; scale fixed',
               sequences=summary, videos=media)
    (FIGS/'sequences.json').write_text(json.dumps(out, indent=2) + '\n'); print('done', flush=True)


if __name__ == '__main__': main()
