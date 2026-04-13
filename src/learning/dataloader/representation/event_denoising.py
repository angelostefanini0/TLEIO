from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def _background_activity_keep_numba(
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    height: int,
    width: int,
    dt_us: int,
    radius: int,
    min_supporters: int,
) -> np.ndarray:
    keep = np.zeros(x.shape[0], dtype=np.bool_)
    neg_inf = np.int64(-(1 << 60))
    last_ts = np.full((height, width), neg_inf, dtype=np.int64)

    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        ti = t_us[i]

        x0 = 0 if xi < radius else xi - radius
        x1 = width if xi + radius + 1 > width else xi + radius + 1
        y0 = 0 if yi < radius else yi - radius
        y1 = height if yi + radius + 1 > height else yi + radius + 1

        supporters = 0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if ti - last_ts[yy, xx] <= dt_us:
                    supporters += 1
                    if supporters >= min_supporters:
                        break
            if supporters >= min_supporters:
                break

        if supporters >= min_supporters:
            keep[i] = True

        last_ts[yi, xi] = ti

    return keep


@njit(cache=True)
def _background_activity_keep_numba_same_polarity(
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    pol_idx: np.ndarray,
    height: int,
    width: int,
    dt_us: int,
    radius: int,
    min_supporters: int,
) -> np.ndarray:
    keep = np.zeros(x.shape[0], dtype=np.bool_)
    neg_inf = np.int64(-(1 << 60))
    last_ts = np.full((2, height, width), neg_inf, dtype=np.int64)

    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        ti = t_us[i]
        pi = pol_idx[i]

        x0 = 0 if xi < radius else xi - radius
        x1 = width if xi + radius + 1 > width else xi + radius + 1
        y0 = 0 if yi < radius else yi - radius
        y1 = height if yi + radius + 1 > height else yi + radius + 1

        supporters = 0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if ti - last_ts[pi, yy, xx] <= dt_us:
                    supporters += 1
                    if supporters >= min_supporters:
                        break
            if supporters >= min_supporters:
                break

        if supporters >= min_supporters:
            keep[i] = True

        last_ts[pi, yi, xi] = ti

    return keep


def background_activity_filter_raw(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t_us: np.ndarray,
    height: int,
    width: int,
    dt_us: int = 1000,
    radius: int = 1,
    min_supporters: int = 1,
    same_polarity_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (len(x) == len(y) == len(p) == len(t_us)):
        raise ValueError("x, y, p, and t_us must have the same length.")
    if dt_us <= 0:
        raise ValueError("dt_us must be > 0.")
    if radius < 0:
        raise ValueError("radius must be >= 0.")
    if min_supporters < 1:
        raise ValueError("min_supporters must be >= 1.")

    x = np.asarray(x)
    y = np.asarray(y)
    p = np.asarray(p)
    t_us = np.asarray(t_us, dtype=np.int64)

    if len(x) == 0:
        return x, y, p, t_us, np.zeros(0, dtype=bool)

    keep = np.zeros(len(x), dtype=bool)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return x[valid], y[valid], p[valid], t_us[valid], keep

    valid_indices = np.flatnonzero(valid)
    x_valid = x[valid].astype(np.int64, copy=False)
    y_valid = y[valid].astype(np.int64, copy=False)
    p_valid = p[valid]
    t_valid = t_us[valid]

    order = np.argsort(t_valid, kind="stable")
    valid_indices = valid_indices[order]
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    p_valid = p_valid[order]
    t_valid = t_valid[order]

    if same_polarity_only:
        pol_idx = (p_valid > 0).astype(np.int64)
        keep_valid = _background_activity_keep_numba_same_polarity(
            x=x_valid,
            y=y_valid,
            t_us=t_valid,
            pol_idx=pol_idx,
            height=height,
            width=width,
            dt_us=dt_us,
            radius=radius,
            min_supporters=min_supporters,
        )
    else:
        keep_valid = _background_activity_keep_numba(
            x=x_valid,
            y=y_valid,
            t_us=t_valid,
            height=height,
            width=width,
            dt_us=dt_us,
            radius=radius,
            min_supporters=min_supporters,
        )

    keep[valid_indices[keep_valid]] = True
    return (
        x_valid[keep_valid],
        y_valid[keep_valid],
        p_valid[keep_valid],
        t_valid[keep_valid],
        keep,
    )


def background_activity_filter_events(
    events: np.ndarray,
    height: int,
    width: int,
    dt_us: int = 1000,
    radius: int = 1,
    min_supporters: int = 1,
    same_polarity_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Expected events with shape [N, 4] in [y, x, t_sec, p] format.")

    if len(events) == 0:
        return events, np.zeros(0, dtype=bool)

    filtered_x, filtered_y, filtered_p, filtered_t_us, keep = background_activity_filter_raw(
        x=events[:, 1],
        y=events[:, 0],
        p=events[:, 3],
        t_us=np.rint(events[:, 2] * 1e6).astype(np.int64, copy=False),
        height=height,
        width=width,
        dt_us=dt_us,
        radius=radius,
        min_supporters=min_supporters,
        same_polarity_only=same_polarity_only,
    )

    filtered_events = np.empty((len(filtered_x), 4), dtype=np.float64)
    filtered_events[:, 0] = filtered_y
    filtered_events[:, 1] = filtered_x
    filtered_events[:, 2] = filtered_t_us.astype(np.float64) / 1e6
    filtered_events[:, 3] = filtered_p
    return filtered_events, keep
