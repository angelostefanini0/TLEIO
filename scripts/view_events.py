import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_events(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        if "events" in f:
            x = np.asarray(f["events/x"]).reshape(-1)
            y = np.asarray(f["events/y"]).reshape(-1)
            t = np.asarray(f["events/t"]).reshape(-1)
            p = np.asarray(f["events/p"]).reshape(-1)
        else:
            x = np.asarray(f["x"]).reshape(-1)
            y = np.asarray(f["y"]).reshape(-1)
            t = np.asarray(f["t"]).reshape(-1)
            p = np.asarray(f["p"]).reshape(-1)

    if not (len(x) == len(y) == len(t) == len(p)):
        raise ValueError("x, y, t, p hanno dimensioni incoerenti")

    return x, y, t, p


def convert_t_to_seconds_relative(t: np.ndarray) -> np.ndarray:
    t = t.astype(np.float64)

    t = t - t[0]
    t_max = t[-1]

    
    if t_max > 1e5:
        t = t * 1e-6
    if t_max > 1e3:
        pass


    if t[-1] > 1e4:
        t = (t / 1e9) if t[-1] > 1e7 else (t / 1e6)

    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path a events.h5")
    parser.add_argument("--dt", type=float, default=0.01, help="Finestra temporale in secondi")
    parser.add_argument("--playback", type=float, default=1.0, help="Velocità playback")
    parser.add_argument("--interval", type=int, default=30, help="Refresh finestra in ms")
    parser.add_argument("--max-events", type=int, default=30000, help="Max eventi per frame")
    args = parser.parse_args()

    x, y, t, p = load_events(args.h5)
    t = convert_t_to_seconds_relative(t)

    print(f"Loaded {len(t):,} events")
    print(f"Duration: {t[0]:.6f}s -> {t[-1]:.6f}s")
    print(f"Resolution: {int(x.max())+1} x {int(y.max())+1}")

    width = int(x.max()) + 1
    height = int(y.max()) + 1

    fig, ax = plt.subplots(figsize=(8, 6))
    scat = ax.scatter([], [], s=1)
    title = ax.set_title("EDS Event Viewer")

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    t0 = 0.0
    t1 = float(t[-1])
    current_time = t0

    def update(_frame_idx):
        nonlocal current_time

        start_t = current_time
        end_t = current_time + args.dt

        idx = np.flatnonzero((t >= start_t) & (t < end_t))

        if len(idx) > args.max_events:
            idx = idx[:args.max_events]

        if len(idx) == 0:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_color([])
        else:
            pts = np.column_stack((x[idx], y[idx]))
            colors = np.where(p[idx] > 0, "red", "blue")
            scat.set_offsets(pts)
            scat.set_color(colors)

        title.set_text(
            f"EDS Event Viewer | t = {start_t:.3f}s -> {end_t:.3f}s | events = {len(idx)}"
        )

        current_time += args.dt * args.playback
        if current_time >= t1:
            current_time = t0

        return scat, title

    anim = FuncAnimation(
        fig,
        update,
        interval=args.interval,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()