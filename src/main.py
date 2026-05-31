"""Online Event-Visual Odometry pipeline integrating Transformer and EKF.

1. Initializes the ViT model and the MSCKF.
2. Runs a calibration window to estimate the translation scale factor by 
   aligning network outputs with IMU pre-integration.
3. Runs the main filtering loop, derotating events using EKF's gyro bias 
   estimates before passing them to the network.
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

# EKF Imports
from filter.scekf import ImuMSCKF
from filter.imu_buffer import ImuMeasurement
from filter.measurement_triplet import make_default_joint_covariance
from main_filter import RunnerConfig, _apply_initial_offsets, _load_anchor_poses, _load_sequence_imu

# Network Imports
from learning.network.build_model import build_model
from learning.dataloader.events_to_voxel.reader import EDSReader
from learning.dataloader.representation.voxel_grid import VoxelGrid


def perform_linear_alignment(accumulated_p_net: np.ndarray, 
                             accumulated_p_ekf: np.ndarray, 
                             timestamps_s: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Solves the visual-inertial linear system: s * p_net - delta_v0 * t = delta_p_ekf
    
    Args:
        accumulated_p_net: (N, 3) array of accumulated network translations.
        accumulated_p_ekf: (N, 3) array of EKF metric displacements (p - p0) 
                           propagated with initial v0 guess and gravity.
        timestamps_s: (N,) array of elapsed time (t) since initialization.
        
    Returns:
        scale_factor (float): The recovered metric scale factor.
        delta_v0 (np.ndarray): 3D velocity correction vector to refine v0.
    """
    N = len(timestamps_s)
    
    # Linear system formulation: A * x = b
    # x = [scale_factor, delta_v0_x, delta_v0_y, delta_v0_z]^T
    A = np.zeros((3 * N, 4))
    b = np.zeros((3 * N, 1))
    
    for i in range(N):
        t = timestamps_s[i]
        p_net = accumulated_p_net[i]
        delta_p_ekf = accumulated_p_ekf[i]
        
        # X-axis row
        A[3*i, 0] = p_net[0]
        A[3*i, 1] = -t
        b[3*i, 0] = delta_p_ekf[0]
        
        # Y-axis row
        A[3*i+1, 0] = p_net[1]
        A[3*i+1, 2] = -t
        b[3*i+1, 0] = delta_p_ekf[1]
        
        # Z-axis row
        A[3*i+2, 0] = p_net[2]
        A[3*i+2, 3] = -t
        b[3*i+2, 0] = delta_p_ekf[2]
        
    # Solve via Ordinary Least Squares
    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    
    scale_factor = float(x[0, 0])
    delta_v0 = x[1:4, 0]
    
    return scale_factor, delta_v0


def derotate_events(events_dict: dict, bg_ekf: np.ndarray, latest_gyro: np.ndarray, 
                    K: np.ndarray, dist_coeffs: np.ndarray, R_cam_imu: np.ndarray, 
                    t_start_us: float) -> dict:
    """
    Applies mathematically rigorous and vectorized motion compensation 
    to raw events using exact camera geometry and EKF gyro bias.
    """
    x = events_dict['x'].astype(np.float32)
    y = events_dict['y'].astype(np.float32)
    t_us = events_dict['t'] # Original microsecond timestamps
    p = events_dict['p']
    
    if len(x) == 0:
        return events_dict

    # 1. Transform Angular Velocity
    # Correct IMU gyro with EKF bias, then rotate into Camera frame
    omega_imu = latest_gyro - bg_ekf
    omega_cam = R_cam_imu @ omega_imu  # Shape: (3,)
    
    # 2. Undistort points (Applies K_inv and distortion model)
    # Output is normalized rays on the Z=1 plane: (N, 2) -> [X/Z, Y/Z]
    pts = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    normalized_rays_2d = cv2.undistortPoints(pts, K, dist_coeffs).reshape(-1, 2)
    
    # Convert to 3D homogeneous rays: [X, Y, 1]
    rays_3d = np.pad(normalized_rays_2d, ((0, 0), (0, 1)), constant_values=1.0) # Shape: (N, 3)
    
    # 3. Vectorized Rotation
    # Calculate elapsed time (in seconds) for each event relative to the window start
    dt_sec = (t_us - t_start_us).astype(np.float32) * 1e-6
    
    # Create the rotation vector for each event: theta = omega * dt
    # Shape: (N, 3)
    theta_array = dt_sec[:, np.newaxis] * omega_cam[np.newaxis, :]
    
    # Generate all rotation matrices efficiently and apply to rays
    rot_objs = Rotation.from_rotvec(theta_array)
    rays_derotated = rot_objs.apply(rays_3d) # Shape: (N, 3)
    
    # 4. Reproject to the perfect (undistorted) pixel plane using K
    rays_pixel = (K @ rays_derotated.T).T # Shape: (N, 3)
    
    x_new = rays_pixel[:, 0] / rays_pixel[:, 2]
    y_new = rays_pixel[:, 1] / rays_pixel[:, 2]
    
    # Filter out events that fall outside the image bounds after derotation
    h, w = 480, 640
    valid_mask = (x_new >= 0) & (x_new < w) & (y_new >= 0) & (y_new < h)
    
    # Normalize the time to [0, 1] for the VoxelGrid
    t_normalized = (t_us - t_us[0]).astype(np.float32)
    if t_normalized[-1] > 0:
        t_normalized = t_normalized / t_normalized[-1]

    derotated_events = {
        'x': x_new[valid_mask],
        'y': y_new[valid_mask],
        'p': p[valid_mask],
        't': t_normalized[valid_mask]
    }
    
    return derotated_events


def main():
    parser = argparse.ArgumentParser(description="Online EVO Pipeline with Linear Alignment")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to sequence directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to network checkpoint (.pth)")
    parser.add_argument("--dataset", type=str, default="eds", choices=["eds", "tartanair"], help="Dataset name to set camera intrinsics")
    parser.add_argument("--calibration_window", type=int, default=15, help="Number of clips for visual-inertial initialization")
    parser.add_argument("--clip_len", type=int, default=3, help="Number of frames per window for the network")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of bins in VoxelGrid")
    args = parser.parse_args()

    seq_path = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 0. SETUP CAMERA INTRINSICS & EXTRINSICS
    # ==========================================
    if args.dataset == "eds":
        fx, fy = 560.8520948927032, 560.6295819972383
        cx, cy = 313.00733235019237, 217.32858679842997
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
        
        dist_coeffs = np.array([-0.09776467241921379, 0.2143738428636279, 
                                -0.004710710105172864, -0.004215916089401789], dtype=np.float64)
        
        # Extrinsic Rotation from IMU to Camera
        R_cam_imu = np.array([
            [0.99989644, -0.00203358, -0.01424667],
            [0.00170302,  0.99972994, -0.02317612],
            [0.01428995,  0.02314946,  0.99962988]
        ], dtype=np.float64)
        
    elif args.dataset == "tartanair":
        fx, fy = 320, 320
        cx, cy = 320, 240
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros(4, dtype=np.float64)
        R_cam_imu = np.eye(3, dtype=np.float64)
    else:
        K = np.eye(3, dtype=np.float64)
        dist_coeffs = np.zeros(4, dtype=np.float64)
        R_cam_imu = np.eye(3, dtype=np.float64)

    print(f"[Init] Dataset set to '{args.dataset}'.")

    # ==========================================
    # 1. INITIALIZE TRANSFORMER NETWORK
    # ==========================================
    net_args = {
        "checkpoint_path": os.path.dirname(args.checkpoint),
        "checkpoint": os.path.basename(args.checkpoint),
        "num_bins": args.num_bins,
        "clip_len": args.clip_len
    }
    
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    model_params = checkpoint_data.get("model_params", {
        "embed_dim": 384, "patch_size": 16, "attention_type": "divided_space_time",
        "depth": 6, "heads": 6, "dim_head": 64, "attn_dropout": 0.1, "ff_dropout": 0.1, "time_only": False
    })

    print("[Init] Building Vision Transformer...")
    model, _ = build_model(net_args, model_params)
    model.eval()

    voxel_grid = VoxelGrid(args.num_bins, height=480, width=640, normalize=True)
    event_reader = EDSReader(seq_path / "events.h5")

    # ==========================================
    # 2. ONLINE STREAM LOADERS & HELPERS
    # ==========================================
    imu_table = _load_sequence_imu(seq_path)

    def get_imu_chunk(t_start_s, t_end_s):
        mask = (imu_table[:, 0] > t_start_s) & (imu_table[:, 0] <= t_end_s)
        chunk = imu_table[mask]
        measurements = []
        prev_t = t_start_s
        for row in chunk:
            t = row[0]
            measurements.append(ImuMeasurement(
                timestamp=t, dt=t - prev_t, gyro=row[1:4], accel=row[4:7]
            ))
            prev_t = t
        return measurements

    def extract_and_derotate_events(t_start_us, t_end_us, bg_ekf, latest_gyro):
        event_data = event_reader.get_events(int(t_start_us), int(t_end_us))
        if event_data is None:
            return torch.zeros((args.num_bins, 480, 640), dtype=torch.float32)
            
        derotated_events = derotate_events(
            event_data, 
            bg_ekf,
            latest_gyro,
            K,
            dist_coeffs,
            R_cam_imu,
            t_start_us=float(t_start_us)
        )
        
        voxel = voxel_grid.convert_events(derotated_events)
        return voxel.float()

    # ==========================================
    # 3. KINEMATIC PROPAGATION & LINEAR ALIGNMENT
    # ==========================================
    print(f"[Calibration] Aligning network scale using {args.calibration_window} clips...")
    
    config = RunnerConfig()
    g_world = np.asarray(config.gravity_world_mps2, dtype=np.float64)
    timestamps_us, positions, quaternions = _load_anchor_poses(seq_path)
    anchor_times_s = timestamps_us.astype(np.float64) * 1e-6
    
    # Initialize from Ground Truth
    p0 = positions[0].astype(np.float64)
    R0 = Rotation.from_quat(quaternions[0]).as_matrix()
    dt0 = max(anchor_times_s[1] - anchor_times_s[0], 1e-9)
    v0_gt = (positions[1] - positions[0]) / dt0
    R0, v0_gt, p0 = _apply_initial_offsets(R0, v0_gt.astype(np.float64), p0, config)
    
    # Run calibration EKF with full gravity and ground-truth velocity
    ekf_calib = ImuMSCKF(config)
    ekf_calib.g = g_world
    ekf_calib.initialize_with_state(anchor_times_s[0], R0, v0_gt, p0, np.zeros(3), np.zeros(3))
    
    accumulated_p_net = []
    accumulated_p_ekf = []
    elapsed_times = []
    
    current_net_p = np.zeros(3)
    current_R = R0.copy()
    
    for idx in range(1, args.calibration_window + 1):
        if idx >= len(anchor_times_s): break
            
        t_start_s = anchor_times_s[idx - 1]
        t_end_s = anchor_times_s[idx]
        
        # Propagate IMU
        imu_chunk = get_imu_chunk(t_start_s, t_end_s)
        ekf_calib.propagate(imu_chunk)
        
        # Get latest gyro reading for derotation
        latest_gyro = imu_chunk[-1].gyro if len(imu_chunk) > 0 else np.zeros(3)
        
        # Process visual voxels
        voxels = []
        for j in range(args.clip_len):
            voxel_idx = idx - args.clip_len + 1 + j
            if voxel_idx < 1: voxel_idx = 1
            
            voxel = extract_and_derotate_events(
                timestamps_us[voxel_idx - 1], 
                timestamps_us[voxel_idx], 
                ekf_calib.state.bg,
                latest_gyro
            )
            voxels.append(voxel)
            
        clip = torch.stack(voxels, dim=1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(clip).view(1, args.clip_len - 1, 7)
            rel_t_body = output[0, -1, :3].cpu().numpy() 
            rel_R_body = Rotation.from_quat(output[0, -1, 3:]).as_matrix()
            
        # Accumulate unscaled visual trajectory
        current_net_p = current_net_p + current_R @ rel_t_body
        current_R = current_R @ rel_R_body
        
        # Save records for alignment optimization
        accumulated_p_net.append(current_net_p.copy())
        accumulated_p_ekf.append(ekf_calib.state.p - p0) 
        elapsed_times.append(t_end_s - anchor_times_s[0])

    # Execute linear batch alignment
    scale_factor, delta_v0 = perform_linear_alignment(
        np.array(accumulated_p_net), 
        np.array(accumulated_p_ekf), 
        np.array(elapsed_times)
    )
    
    v0_refined = v0_gt + delta_v0
    
    print(f"[Calibration] Alignment Success!")
    print(f"[Calibration] Calculated Scale Factor: {scale_factor:.4f}")
    print(f"[Calibration] Ground Truth v0: {v0_gt}")
    print(f"[Calibration] Refined Initial Velocity (v0): {v0_refined}")

    # ==========================================
    # 4. RESET RUNTIME EKF & ONLINE PROCESSING LOOP
    # ==========================================
    print("[Online] Resetting EKF with optimized initial states and beginning tracking loop...")
    
    ekf_main = ImuMSCKF(config)
    ekf_main.g = g_world
    # Initialize the main loop EKF with the refined velocity
    ekf_main.initialize_with_state(anchor_times_s[0], R0, v0_refined, p0, np.zeros(3), np.zeros(3))
    ekf_main.augment_clone() 
    
    triplet_buffer = []

    # Stream filtering over the entire trajectory sequence
    for anchor_idx in range(1, len(anchor_times_s)):
        t_start_s = anchor_times_s[anchor_idx - 1]
        t_end_s = anchor_times_s[anchor_idx]
        
        # 1. Continuous IMU Integration
        imu_chunk = get_imu_chunk(t_start_s, t_end_s)
        ekf_main.propagate(imu_chunk)
        ekf_main.augment_clone()
        
        latest_gyro = imu_chunk[-1].gyro if len(imu_chunk) > 0 else np.zeros(3)

        # 2. Live Event Processing and Network Inference
        voxels = []
        for j in range(args.clip_len):
            idx = anchor_idx - args.clip_len + 1 + j
            if idx < 1: idx = 1
            
            voxel = extract_and_derotate_events(
                timestamps_us[idx - 1], 
                timestamps_us[idx], 
                ekf_main.state.bg,
                latest_gyro
            )
            voxels.append(voxel)
            
        clip = torch.stack(voxels, dim=1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(clip).view(1, args.clip_len - 1, 7)
            net_prediction = output[0, -1, :].cpu().numpy()
            
        # 3. Enforce Recovered Metric Scale on Network Translation
        net_prediction[:3] *= scale_factor
        
        # 4. MSCKF Triplet Measurement Update (Requires 4 edges)
        triplet_buffer.append(net_prediction[:3])
        
        if len(triplet_buffer) >= 4:
            measurement_4x3 = np.array(triplet_buffer[-4:])
            update_data = {
                "relative_pose": measurement_4x3,
                "joint_covariance": None 
            }
            
            ekf_main.update(update_data)
            ekf_main.marginalize_oldest_clone()
            triplet_buffer.pop(0) 

        if anchor_idx % 50 == 0:
            print(f"Processed {anchor_idx}/{len(anchor_times_s)} steps. Estimated Position: {ekf_main.state.p}")

    print("[Online] Pipeline Processing Complete.")

if __name__ == "__main__":
    main()