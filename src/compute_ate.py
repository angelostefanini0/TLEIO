import numpy as np
import argparse
from pathlib import Path

def read_trajectory_file(filename, time_scale=1.0, skip_header=False):
    """
    Legge un file di traiettoria.
    Permette di scalare il tempo (es. 1e-6 per convertire microsecondi in secondi).
    Se skip_header=True, scarta la prima riga del file.
    """
    trajectory = {}
    with open(filename, 'r') as f:
        if skip_header:
            next(f)  # Scarta la prima riga
            
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            # Estraiamo il timestamp e lo scaliamo
            timestamp = float(parts[0]) * time_scale
            position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            trajectory[timestamp] = position
    return trajectory

def associate_trajectories(groundtruth, estimate, max_time_diff=0.02):
    """
    Associa i punti delle due traiettorie in base al timestamp più vicino.
    """
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
    """
    Passaggio 1: Algoritmo di Umeyama per trovare s, R, T.
    """
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
    """
    Passaggio 2: Calcolo dell'RMSE dopo l'allineamento.
    """
    aligned_est_points = s * (R @ est_points) + T
    errors = gt_points - aligned_est_points
    squared_errors = np.sum(errors**2, axis=0)
    rmse = np.sqrt(np.mean(squared_errors))
    
    return rmse

def main():
    # Impostazione del parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Calcola l'ATE tra groundtruth e stima.")
    parser.add_argument("--dataset", type=str, required=True, help="Nome del dataset (es. eds)")
    parser.add_argument("--sequence", type=str, required=True, help="Nome della sequenza (es. 00_peanuts_dark)")
    args = parser.parse_args()

    # ROOT è la cartella genitore di 'src' (ovvero la root del progetto)
    ROOT = Path(__file__).resolve().parent.parent

    # Costruzione dinamica dei percorsi
    groundtruth_file = ROOT / "data" / args.dataset / "processed" / args.sequence / "stamped_groundtruth.txt"
    estimate_file = ROOT / "outputs" / "main_filter" / args.sequence / "stamped_traj_estimate.txt"
    
    print("Percorsi dei file:")
    print(f" - Groundtruth: {groundtruth_file}")
    print(f" - Estimate:    {estimate_file}\n")
    
    print("Lettura dei file in corso...")
    try:
        # Moltiplichiamo per 10^-6 per portare la groundtruth da microsecondi a secondi (senza scartare righe)
        gt_dict = read_trajectory_file(groundtruth_file, time_scale=1e-6, skip_header=False)
        
        # La traiettoria stimata è in secondi (time_scale=1.0) e scartiamo la prima riga (skip_header=True)
        est_dict = read_trajectory_file(estimate_file, time_scale=1.0, skip_header=True)
    except FileNotFoundError as e:
        print(f"Errore: Il file non esiste. \n{e}")
        return
    except StopIteration:
        print("Errore: Il file di stima sembra essere vuoto o contiene solo una riga.")
        return

    print("Associazione dei timestamp...")
    gt_points, est_points = associate_trajectories(gt_dict, est_dict, max_time_diff=0.02)
    
    if gt_points.shape[1] == 0:
        print("Errore: Nessun punto associato trovato. Controlla i timestamp dei file.")
        return
        
    print(f"Trovati {gt_points.shape[1]} punti validi per il calcolo.")

    # --- STEP 1: Allineamento ---
    print("Calcolo della trasformazione di allineamento (s, R, T)...")
    s, R, T = align_umeyama(gt_points, est_points)
    
    # --- STEP 2: Calcolo RMSE ---
    print("Calcolo dell'RMSE (ATE)...")
    ate_rmse = compute_rmse(gt_points, est_points, s, R, T)
    
    print("\n" + "-"*30)
    print(f"RISULTATI PER {args.sequence}:")
    print(f"Scale (s): {s:.4f}")
    print(f"Translation (T):\n{T}")
    print(f"Rotation (R):\n{R}")
    print(f"-> ATE (RMSE): {ate_rmse:.6f}")
    print("-"*30)

if __name__ == "__main__":
    main()