import numpy as np
import argparse
from pathlib import Path

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

def process_sequence(dataset, sequence, root):
    groundtruth_file = root / "data" / dataset / "processed" / sequence / "stamped_groundtruth.txt"
    estimate_file = root / "outputs" / "main_filter" / sequence / "stamped_traj_estimate.txt"
    
    print(f"\nElaborating sequence: {sequence}")
    
    try:
        gt_dict = read_trajectory_file(groundtruth_file, time_scale=1e-6, skip_header=False)
        est_dict = read_trajectory_file(estimate_file, time_scale=1.0, skip_header=True)
    except FileNotFoundError as e:
        print(f"  -> Missing file: {e.filename}")
        return None
    except StopIteration:
        print("  -> Empty file")
        return None

    gt_points, est_points = associate_trajectories(gt_dict, est_dict, max_time_diff=0.02)
    
    if gt_points.shape[1] == 0:
        print("  -> ERROR: No point found.")
        return None
        
    s, R, T = align_umeyama(gt_points, est_points)
    ate_rmse = compute_rmse(gt_points, est_points, s, R, T)
    
    print(f"  -> ATE (RMSE): {ate_rmse:.6f}  |  Scale (s): {s:.4f}")
    return ate_rmse

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
        ate = process_sequence(args.dataset, seq, ROOT)
        if ate is not None:
            results[seq] = ate

    if len(results) > 1:
        print("\n" + "="*40)
        print("FINAL SUMMARY ATE")
        print("="*40)
        for seq, ate in results.items():
            print(f"{seq:<30} : {ate:.6f}")
        
        avg_ate = sum(results.values()) / len(results)
        print("-" * 40)
        print(f"{'AVG':<30} : {avg_ate:.6f}")
        print("="*40)

if __name__ == "__main__":
    main()