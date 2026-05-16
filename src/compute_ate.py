import numpy as np
import argparse
from pathlib import Path
import subprocess
import sys
import re


def read_trajectory_table(filename, time_scale=1.0, skip_header=False):
    rows = []
    with open(filename, 'r') as f:
        if skip_header:
            next(f)

        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            rows.append([
                float(parts[0]) * time_scale,
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            ])

    if not rows:
        raise ValueError("Empty trajectory file.")
    return np.asarray(rows, dtype=np.float64)


def read_trajectory_file(filename, time_scale=1.0, skip_header=False):
    trajectory = {}
    with open(filename, 'r') as f:
        if skip_header:
            next(f)  
            
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            timestamp = float(parts[0]) * time_scale
            position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            trajectory[timestamp] = position
    return trajectory


def trajectory_dict_from_table(trajectory_table, time_scale=1.0):
    trajectory_np = np.asarray(trajectory_table, dtype=np.float64)
    if trajectory_np.ndim != 2 or trajectory_np.shape[1] < 4:
        raise ValueError("Trajectory table must have shape [N, >=4].")

    trajectory = {}
    for row in trajectory_np:
        timestamp = float(row[0]) * time_scale
        trajectory[timestamp] = row[1:4].copy()
    return trajectory


def associate_trajectories(groundtruth, estimate, max_time_diff=0.02):
    gt_keys = list(groundtruth.keys())
    est_keys = list(estimate.keys())
    
    matches = []
    for t_est in est_keys:
        diffs = np.abs(np.array(gt_keys) - t_est)
        idx_min = np.argmin(diffs)
        
        if diffs[idx_min] < max_time_diff:
            matches.append((gt_keys[idx_min], t_est))
            
    gt_points = np.array([groundtruth[m[0]] for m in matches]).T
    est_points = np.array([estimate[m[1]] for m in matches]).T
    
    return gt_points, est_points


def interpolate_positions(gt_times, gt_positions, query_times):
    gt_times = np.asarray(gt_times, dtype=np.float64)
    gt_positions = np.asarray(gt_positions, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)

    if gt_times.ndim != 1 or gt_positions.ndim != 2 or gt_positions.shape[1] != 3:
        raise ValueError("Ground-truth trajectory must have shape [N] and [N, 3].")
    if len(gt_times) != len(gt_positions):
        raise ValueError("Ground-truth timestamps and positions must have the same length.")
    if len(gt_times) < 2:
        raise ValueError("Need at least two ground-truth samples for interpolation.")
    if np.any(np.diff(gt_times) <= 0.0):
        raise ValueError("Ground-truth timestamps must be strictly increasing.")
    
    interpolated = np.empty((len(query_times), 3), dtype=np.float64)
    for axis in range(3):
        interpolated[:, axis] = np.interp(query_times, gt_times, gt_positions[:, axis])
    return interpolated


def align_umeyama(model, data):
    N = model.shape[1]
    
    mu_M = model.mean(axis=1, keepdims=True)
    mu_D = data.mean(axis=1, keepdims=True)
    
    M_centered = model - mu_M
    D_centered = data - mu_D
    
    sigma_sq_D = np.mean(np.sum(D_centered**2, axis=0))
    H = (D_centered @ M_centered.T) / N
    
    U, S_diag, Vt = np.linalg.svd(H)
    V = Vt.T
    
    d = np.sign(np.linalg.det(V @ U.T))
    S_matrix = np.diag([1, 1, d])
    
    R = V @ S_matrix @ U.T
    s = (1.0 / sigma_sq_D) * np.trace(np.diag(S_diag) @ S_matrix)
    T = mu_M - s * (R @ mu_D)
    
    return s, R, T


def compute_rmse(gt_points, est_points, s, R, T):
    aligned_est_points = s * (R @ est_points) + T
    errors = gt_points - aligned_est_points
    squared_errors = np.sum(errors**2, axis=0)
    rmse = np.sqrt(np.mean(squared_errors))
    
    return rmse


def compute_ate_from_tables(
    groundtruth_table,
    estimate_table,
    groundtruth_time_scale=1.0,
    estimate_time_scale=1.0,
    max_time_diff=0.02,
):
    del max_time_diff

    gt_np = np.asarray(groundtruth_table, dtype=np.float64)
    est_np = np.asarray(estimate_table, dtype=np.float64)
    if gt_np.ndim != 2 or gt_np.shape[1] < 4:
        raise ValueError("Ground-truth trajectory table must have shape [N, >=4].")
    if est_np.ndim != 2 or est_np.shape[1] < 4:
        raise ValueError("Estimated trajectory table must have shape [N, >=4].")

    gt_times = gt_np[:, 0] * groundtruth_time_scale
    gt_positions = gt_np[:, 1:4]
    est_times = est_np[:, 0] * estimate_time_scale
    est_positions = est_np[:, 1:4]

    aligned_gt_positions = interpolate_positions(gt_times, gt_positions, est_times)
    gt_points = aligned_gt_positions.T
    est_points = est_positions.T

    s, R, T = align_umeyama(gt_points, est_points)
    ate_rmse = compute_rmse(gt_points, est_points, s, R, T)
    
    return ate_rmse, s, R, T


def process_sequence(dataset, sequence, root):
    sequence_dir = root / "data" / dataset / "processed" / sequence
    anchor_groundtruth_file = sequence_dir / "anchor_poses.txt"
    dense_groundtruth_file = sequence_dir / "stamped_groundtruth.txt"
    estimate_file = root / "outputs" / "main_filter" / dataset / sequence / "stamped_traj_estimate.txt"
    
    print(f"\nElaborating sequence: {sequence}")
    
    # 1. Eseguiamo main_filter.py per ottenere pos_rmse e rot_rmse
    main_filter_script = Path(__file__).resolve().parent / "main_filter.py"
    
    if not main_filter_script.exists():
        print(f"  -> ERRORE: Script {main_filter_script} non trovato.")
        return None
        
    cmd = [sys.executable, str(main_filter_script), "--dataset", dataset, "--sequence", sequence]
    
    # Esegue main_filter.py catturandone lo standard output
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if proc.returncode != 0:
        print(f"  -> ERRORE durante l'esecuzione di main_filter.py:\n{proc.stderr}")
        return None

    pos_rmse, rot_rmse = None, None
    
    # Lettore linea per linea per un'estrazione robusta
    for line in proc.stdout.split('\n'):
        line_lower = line.lower()
        
        # Estrae i numeri in virgola mobile e scarta le linee senza numeri
        nums = re.findall(r"([0-9]+\.[0-9]+(?:[eE][-+]?[0-9]+)?)", line)
        if not nums:
            continue
            
        val = float(nums[-1]) # In genere l'ultimo numero sulla riga è il valore cercato
        
        # Posizione: se la riga contiene 'pos' o 'trans' ed è riferita all'RMSE
        if re.search(r'(pos|trans)', line_lower) and 'rmse' in line_lower:
            pos_rmse = val
            
        # Rotazione: se la riga contiene 'rot', 'ang' o 'att' ed è riferita all'RMSE
        if re.search(r'(rot|ang|att)', line_lower) and 'rmse' in line_lower:
            # Assicuriamoci che non stia matchando per via di "Root Mean Square"
            clean_line = line_lower.replace('root', '')
            if re.search(r'(rot|ang|att)', clean_line):
                rot_rmse = val
        
    # Avviso di sicurezza in caso di mancato riconoscimento
    if pos_rmse is None or rot_rmse is None:
        print("  -> ATTENZIONE: POS_RMSE o ROT_RMSE non trovati! Righe disponibili contenenti 'rmse':")
        for line in proc.stdout.split('\n'):
            if 'rmse' in line.lower():
                print(f"       {line.strip()}")
        pos_rmse = pos_rmse or 0.0
        rot_rmse = rot_rmse or 0.0

    # 2. Ora che l'estimate file è stato rigenerato dal filtro, calcoliamo l'ATE
    try:
        if anchor_groundtruth_file.exists():
            gt_table = read_trajectory_table(anchor_groundtruth_file, time_scale=1e-6, skip_header=True)
        else:
            gt_table = read_trajectory_table(dense_groundtruth_file, time_scale=1e-6, skip_header=False)
        est_table = read_trajectory_table(estimate_file, time_scale=1.0, skip_header=True)
    except FileNotFoundError as e:
        print(f"  -> Missing file: {e.filename}")
        return None
    except (StopIteration, ValueError) as e:
        print(f"  -> {e}")
        return None

    try:
        ate_rmse, s, _, _ = compute_ate_from_tables(
            gt_table,
            est_table,
            groundtruth_time_scale=1.0,
            estimate_time_scale=1.0,
        )
    except ValueError as e:
        print(f"  -> ERROR: {e}")
        return None
    
    print(f"  -> ATE: {ate_rmse:.6f} | POS RMSE: {pos_rmse:.6f} | ROT RMSE: {rot_rmse:.6f}")
    return ate_rmse, pos_rmse, rot_rmse


def main():
    parser = argparse.ArgumentParser(description="Evaluate ATE between groundtruth and estimate.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (es. eds)")
    parser.add_argument("--sequence", type=str, default=None, help="Sequence (optional. If absent, evaluates the entire dataset).")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    processed_dir = ROOT / "data" / args.dataset / "processed"

    if not processed_dir.exists():
        print(f"Error: Directory not found: {processed_dir}")
        return

    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = [d.name for d in processed_dir.iterdir() if d.is_dir()]

    sequences.sort()
    
    print(f"Found {len(sequences)} sequences in dataset '{args.dataset}'.")
    
    results = {}
    for seq in sequences:
        metrics = process_sequence(args.dataset, seq, ROOT)
        if metrics is not None:
            results[seq] = metrics

    if len(results) > 1:
        print("\n" + "=" * 75)
        print(f"{'FINAL SUMMARY':<30} | {'ATE':<10} | {'POS RMSE':<10} | {'ROT RMSE':<10}")
        print("=" * 75)
        for seq, (ate, pos, rot) in results.items():
            print(f"{seq:<30} | {ate:<10.6f} | {pos:<10.6f} | {rot:<10.6f}")
        
        avg_ate = sum(r[0] for r in results.values()) / len(results)
        avg_pos = sum(r[1] for r in results.values()) / len(results)
        avg_rot = sum(r[2] for r in results.values()) / len(results)
        
        print("-" * 75)
        print(f"{'AVG':<30} | {avg_ate:<10.6f} | {avg_pos:<10.6f} | {avg_rot:<10.6f}")
        print("=" * 75)

if __name__ == "__main__":
    main()