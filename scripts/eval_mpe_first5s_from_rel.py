import numpy as np
from pathlib import Path

GT_PATH = "data/davis240c/processed/boxes_translation/stamped_groundtruth.txt"
REL_PATH = "data/davis240c/predicted_relative_motions/rescue_v6_e70_flip_p/raw_feature_affine_calib.txt"
GT_REL_PATH = "data/davis240c/processed/boxes_translation/relative_motions.txt"
SAVE_DIR = Path("plots/davis240c/MPE_first5s_raw_feature_affine_calib")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

gt = np.loadtxt(GT_PATH, skiprows=1)
rel = np.loadtxt(REL_PATH, skiprows=1)
gt_rel = np.loadtxt(GT_REL_PATH, skiprows=1)

gt_ts = gt[:,0].astype(np.int64)
gt_pos_all = gt[:,1:4]

rel_ts = rel[:,:2].astype(np.int64)
pred_rel = rel[:,2:5]
gt_rel_vec = gt_rel[:,2:5]

anchor_ts = np.r_[rel_ts[0,0], rel_ts[:,1]]

idx = np.searchsorted(gt_ts, anchor_ts)
idx = np.clip(idx, 1, len(gt_ts)-1)
left = idx - 1
right = idx
choose_right = np.abs(gt_ts[right] - anchor_ts) < np.abs(gt_ts[left] - anchor_ts)
idx = np.where(choose_right, right, left)

gt_pos = gt_pos_all[idx]
gt_dp_world = gt_pos[1:] - gt_pos[:-1]

def quat_to_R_xyzw(q):
    x,y,z,w = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x,y,z,w = q / n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])

def quat_to_R_wxyz(q):
    w,x,y,z = q
    return quat_to_R_xyzw(np.array([x,y,z,w]))

def integrate(rel_vec, mode):
    pred_pos = np.zeros((len(rel_vec)+1, 3))
    if mode == "world":
        inc = rel_vec
    else:
        if gt.shape[1] < 8:
            raise RuntimeError("GT file has no quaternion columns, cannot use rotation modes.")
        q_all = gt[idx[:-1],4:8]
        inc = np.zeros_like(rel_vec)

        for i, q in enumerate(q_all):
            if mode == "R_xyzw":
                R = quat_to_R_xyzw(q)
                inc[i] = R @ rel_vec[i]
            elif mode == "Rt_xyzw":
                R = quat_to_R_xyzw(q)
                inc[i] = R.T @ rel_vec[i]
            elif mode == "R_wxyz":
                R = quat_to_R_wxyz(q)
                inc[i] = R @ rel_vec[i]
            elif mode == "Rt_wxyz":
                R = quat_to_R_wxyz(q)
                inc[i] = R.T @ rel_vec[i]
            else:
                raise ValueError(mode)

    pred_pos[1:] = np.cumsum(inc, axis=0)
    return pred_pos, inc

# scegli automaticamente la convenzione che ricostruisce meglio le GT relative
modes = ["world"]
if gt.shape[1] >= 8:
    modes += ["R_xyzw", "Rt_xyzw", "R_wxyz", "Rt_wxyz"]

best_mode = None
best_err = 1e18

for mode in modes:
    _, inc_gt = integrate(gt_rel_vec, mode)
    e = np.sqrt(np.mean(np.sum((inc_gt - gt_dp_world)**2, axis=1)))
    print(f"convention check {mode:8s}: gt-step RMSE = {e:.6e}")
    if e < best_err:
        best_err = e
        best_mode = mode

print("\nselected integration mode:", best_mode)

pred_pos, _ = integrate(pred_rel, best_mode)

# porta entrambe le traiettorie a origine zero prima dell'allineamento
pred_pos = pred_pos - pred_pos[0]
gt_pos0 = gt_pos - gt_pos[0]

t_start = anchor_ts[0]
first5 = anchor_ts <= t_start + 5_000_000
after5 = ~first5

def umeyama(X, Y, with_scale=True):
    # trova s,R,t tali che Y ~= s * X @ R.T + t
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    C = (Yc.T @ Xc) / len(X)
    U, D, Vt = np.linalg.svd(C)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1,-1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_x = np.mean(np.sum(Xc**2, axis=1))
        scale = np.trace(np.diag(D) @ S) / var_x
    else:
        scale = 1.0

    t = mu_y - scale * (R @ mu_x)
    X_aligned = scale * (X @ R.T) + t
    return X_aligned, scale, R, t

def traj_length(P):
    return np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1))

def report(name, P):
    err = np.linalg.norm(P - gt_pos0, axis=1)
    L = traj_length(gt_pos0)

    print(f"\n=== {name} ===")
    print("ATE mean [m] all:      ", np.mean(err))
    print("ATE RMSE [m] all:      ", np.sqrt(np.mean(err**2)))
    print("MPE mean [%] all:      ", 100 * np.mean(err) / L)
    print("MPE RMSE [%] all:      ", 100 * np.sqrt(np.mean(err**2)) / L)

    if after5.sum() > 0:
        err2 = err[after5]
        print("ATE mean [m] after5s:  ", np.mean(err2))
        print("ATE RMSE [m] after5s:  ", np.sqrt(np.mean(err2**2)))
        print("MPE mean [%] after5s:  ", 100 * np.mean(err2) / L)
        print("MPE RMSE [%] after5s:  ", 100 * np.sqrt(np.mean(err2**2)) / L)

    print("GT traj length [m]:    ", L)

pred_se3, s_se3, R_se3, t_se3 = umeyama(pred_pos[first5], gt_pos0[first5], with_scale=False)
pred_se3_all = pred_pos @ R_se3.T + t_se3

pred_sim3, s_sim3, R_sim3, t_sim3 = umeyama(pred_pos[first5], gt_pos0[first5], with_scale=True)
pred_sim3_all = s_sim3 * (pred_pos @ R_sim3.T) + t_sim3

print("\nfirst5 samples:", first5.sum())
print("SE3 scale:", s_se3)
print("Sim3 scale:", s_sim3)

report("SE3 alignment first 5s", pred_se3_all)
report("Sim3 alignment first 5s", pred_sim3_all)

np.savetxt(
    SAVE_DIR / "trajectory_aligned_se3.txt",
    np.column_stack([anchor_ts, pred_se3_all, gt_pos0]),
    fmt=["%d","%.10f","%.10f","%.10f","%.10f","%.10f","%.10f"],
    header="t_us pred_x pred_y pred_z gt_x gt_y gt_z",
    comments=""
)

np.savetxt(
    SAVE_DIR / "trajectory_aligned_sim3.txt",
    np.column_stack([anchor_ts, pred_sim3_all, gt_pos0]),
    fmt=["%d","%.10f","%.10f","%.10f","%.10f","%.10f","%.10f"],
    header="t_us pred_x pred_y pred_z gt_x gt_y gt_z",
    comments=""
)

print("\nsaved to", SAVE_DIR)
