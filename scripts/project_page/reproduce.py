#!/usr/bin/env python3
"""Reproduce the project-page training-split demonstration, without tuning on results.

Run with the TLEIO environment activated:
  python reproduce.py --repo /path/to/TLEIO --site /path/to/tleio-project/dist
Dependencies: repository environment.yaml plus torch, huggingface_hub, hdf5plugin,
gdown. Requires the released checkpoint and its original args.txt. If absent,
this script downloads the official Google Drive folder. Data downloads the office
training environment (21.6 GB event archive). The script selects complete
sequences (all 15 office sequences by default, or --sequences Easy/P000 ...),
preserving all downloaded raw data.

Ground-truth pose information is used for derotation, synthetic IMU generation,
filter initialization, and evaluation. This is NOT held-out evaluation.
"""
from pathlib import Path
import argparse, contextlib, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone

OFFICE = [('Easy', f'P{i:03}') for i in range(7)] + [('Hard', f'P{i:03}') for i in range(8)]
DRIVE = 'https://drive.google.com/drive/folders/1RnAKGuD_6BHSWama648qtUKyGo524Bw3'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--site',type=Path,default=None)
    parser.add_argument('--skip-download',action='store_true')
    parser.add_argument('--skip-preprocessing',action='store_true',help='Reuse processed sequences and voxels from a previous run')
    parser.add_argument('--network-scale',type=float,default=None,help='Override the filter default (see tune_filter.py)')
    parser.add_argument('--meas-cov-scale',type=float,default=None,help='Override the filter default (see tune_filter.py)')
    parser.add_argument('--sequences',nargs='+',default=None,help='Difficulty/Pxxx pairs, e.g. Easy/P000 Hard/P003')
    a=parser.parse_args()
    SELECTION=[tuple(s.split('/')) for s in a.sequences] if a.sequences else OFFICE; repo=a.repo.resolve(); os.chdir(repo)
    site=a.site.resolve() if a.site else repo/'website'
    out=site/'results'; out.mkdir(parents=True,exist_ok=True)
    logroot=repo/'logs'/'project_page'; logroot.mkdir(parents=True,exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',MPLBACKEND='Agg')
    sys.path[:0]=[str(repo),str(repo/'src')]
    timings={}
    def run(label, args):
        print(label,flush=True); t=time.monotonic()
        with (logroot/(label+'.log')).open('w') as log:
            subprocess.run([sys.executable,*map(str,args)],check=True,stdout=log,stderr=subprocess.STDOUT)
        timings[label]=round(time.monotonic()-t,3)
    ckpt=repo/'checkpoints/tleio_additional/checkpoint_last.pth.zip'
    if not ckpt.exists(): run('checkpoint_download',['-m','gdown','--folder',DRIVE,'-O','checkpoints/'])
    if not a.skip_download:
        if not all((repo/'data/tartanair/office'/d/p/'events.h5').exists() for d,p in SELECTION):
            run('dataset_download',['scripts/download/download_tartanair.py','--env','office','--difficulty','easy','hard','--keep-zip'])
    selected=repo/'data/tartanair/project_page_raw'
    for diff,name in SELECTION:
        raw=repo/'data/tartanair/office'/diff/name
        if not (raw/'events.h5').exists(): raise FileNotFoundError(raw)
        link=selected/'office'/diff/name; link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists(): link.symlink_to(raw,target_is_directory=True)
    proc=repo/'data/tartanair/project_page_processed_train'
    vox=repo/'data/tartanair/project_page_precomputed_train'
    cfg=json.loads(ckpt.with_name('args.txt').read_text())
    if not a.skip_preprocessing: run('processing',['scripts/processing/processing_tartan.py',selected,'--save-path',proc,'--overwrite'])
    for diff,name in SELECTION if not a.skip_preprocessing else []:
        seq=f'office_{diff}_{name}'; seqdir=proc/seq
        # Standard TartanAir pinhole intrinsics, also used by RAMP-VO's
        # TartanEvent loader: github.com/uzh-rpg/rampvo/blob/master/evaluate.py
        (seqdir/'K.yaml').write_text('cam1:\n  intrinsics: [320.0, 320.0, 320.0, 240.0]\n  resolution: [640, 480]\n')
        run(seq+'_imu',['scripts/processing/imu_synthesizer.py','--sequence_dir',seqdir,'--overwrite'])
    pre=['scripts/processing/precompute_derotated_voxels.py','--root_dir',proc,'--output_dir',vox,'--overwrite']
    for k in ['delta_t_ms','num_bins','downsampling_factor','denoising','denoise_dt_us','denoise_radius','denoise_min_supporters','denoise_same_polarity_only','derotate','derotation_slices']:
        pre += ['--'+k,str(cfg[k]).lower() if isinstance(cfg[k],bool) else str(cfg[k])]
    pre+=['--patch_size',str(cfg['model_params']['patch_size'])]
    if not a.skip_preprocessing: run('voxelization',pre)
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from dataclasses import replace
    from main_filter import CONFIG, run_filter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    results=[]
    for diff,name in SELECTION:
        seq=f'office_{diff}_{name}'; dest=out/seq; dest.mkdir(exist_ok=True)
        run(seq+'_inference',['scripts/testing/test.py','--sequence_dir',vox/seq,'--checkpoint_file',ckpt,'--output_file',proc/seq/(seq+'.txt'),'--average_overlaps','--save_covariance'])
        print(seq+'_filter',flush=True); t=time.monotonic()
        config=replace(CONFIG,dataset='tartanair',sequence=seq,processed_dir=proc,output_root=repo/'outputs/project_page',plot_transformer=True,plot_projections=True)
        if a.network_scale is not None: config=replace(config,network_scale=a.network_scale)
        if a.meas_cov_scale is not None: config=replace(config,meas_cov_scale=a.meas_cov_scale)
        with (logroot/(seq+'_filter.log')).open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            r=run_filter(config)
        timings[seq+'_filter']=round(time.monotonic()-t,3)
        est=r['trajectory']; gt=r['ground_truth']
        if est.shape != gt.shape or not np.allclose(est[:,0],gt[:,0],atol=1e-9): raise ValueError('Timestamp/shape mismatch')
        if not np.isfinite(est).all(): raise ValueError('Non-finite filter trajectory')
        p=est[:,1:4]; q=gt[:,1:4]; err=np.linalg.norm(p-q,axis=1)
        # Rigid SE(3) Kabsch alignment: scale fixed at 1. All samples.
        pc=p-p.mean(0); qc=q-q.mean(0)
        u,s,vt=np.linalg.svd(pc.T@qc)
        correction=np.eye(3); correction[-1,-1]=np.linalg.det(u@vt)
        rot=u@correction@vt
        aligned=pc@rot+q.mean(0)
        ate=np.linalg.norm(aligned-q,axis=1)
        angle=(Rotation.from_quat(gt[:,4:8]).inv()*Rotation.from_quat(est[:,4:8])).magnitude()
        metrics=dict(position_rmse_m=float(np.sqrt(np.mean(err**2))),ate_se3_rmse_m=float(np.sqrt(np.mean(ate**2))),rotation_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angle**2)))),duration_s=float(est[-1,0]-est[0,0]),path_length_m=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),samples=len(est),updates_attempted=r['num_updates_attempted'],updates_rejected=r['num_updates_rejected'])
        np.savetxt(dest/'estimate.tum',est,fmt='%.9f',header='timestamp_s px py pz qx qy qz qw')
        np.savetxt(dest/'groundtruth.tum',gt,fmt='%.9f',header='timestamp_s px py pz qx qy qz qw')
        shutil.copy2(proc/seq/(seq+'.txt'),dest/'relative_predictions.txt')
        (dest/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
        # Keep every recorded anchor in the interactive result, no fabricated curves.
        data=dict(sequence=seq,label=f'Office / {diff} / {name}',split='training',metrics=metrics,times=(est[:,0]-est[0,0]).round(6).tolist(),groundtruth=q.round(6).tolist(),estimate=p.round(6).tolist())
        (dest/'trajectory.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
        for ax,(i,j) in zip(axs,[(0,1),(0,2),(1,2)]):
            ax.plot(q[:,i],q[:,j],c='#6391bd',lw=1.5,label='Ground truth')
            ax.plot(p[:,i],p[:,j],c='#287957',lw=1.2,label='TLEIO')
            ax.set(xlabel='XYZ'[i]+' [m]',ylabel='XYZ'[j]+' [m]'); ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.15)
        axs[0].legend(fontsize=8); fig.suptitle(f'Office {diff} {name} · training split · original world frame',fontsize=12)
        fig.savefig(dest/'projections.png',dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained'); ax.plot(est[:,0]-est[0,0],err,c='#287957',lw=1.3); ax.set(xlabel='Time [s]',ylabel='Position error [m]',title=f'Office {diff} {name} · unaligned translation error'); ax.grid(alpha=.15); fig.savefig(dest/'error.png',dpi=160); plt.close(fig)
        # Independent evaluator, runs on the exact exported TUM files.
        run(seq+'_evo_se3',['-c','try:\n from evo.cli.entry_points import ape\nexcept ImportError:\n from evo.entry_points import ape\nape()','tum',dest/'groundtruth.tum',dest/'estimate.tum','--align','--no_warnings','--save_results',dest/'evo_se3.zip'])
        results.append(dict(sequence=seq,label=data['label'],metrics=metrics,data=f'{seq}/trajectory.json',projection=f'{seq}/projections.png',error=f'{seq}/error.png'))
        print(json.dumps(results[-1],indent=2),flush=True)
    cpuhash=hashlib.sha256()
    with ckpt.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): cpuhash.update(chunk)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),repository='https://github.com/angelostefanini0/TLEIO',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint=dict(source=DRIVE,file=str(ckpt.relative_to(repo)),sha256=cpuhash.hexdigest(),args=cfg),hardware=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.version),protocol=dict(split='TartanEvent training: office '+', '.join(f'{d}/{n}' for d,n in SELECTION)+', complete sequences',imu='Ground-truth synthesized, 200 Hz; original synthesizer noise/bias defaults, random seed 7',derotation='Ground-truth rotations; 100 slices per 50 ms event window',intrinsics=[320,320,320,240],intrinsics_source='https://github.com/uzh-rpg/rampvo/blob/master/evaluate.py',initialization='Ground-truth initial pose and initial velocity from first two anchor poses',network_scale=config.network_scale,meas_cov_scale=config.meas_cov_scale,covariance='Predicted per-axis sigmas (--save_covariance); overlap averaging',normalization='Checkpoint normalize_voxel_nonzero=true; target stats from checkpoint',position_rmse='Euclidean translation RMSE in original world frame, no alignment',ate='Rigid SE(3) alignment across all matching timestamps, no scale correction; independently checked with evo',filter_tuning='Repository defaults, no per-sequence tuning' if a.network_scale is None and a.meas_cov_scale is None else 'network_scale and meas_cov_scale calibrated once for all office sequences by tune_filter.py; no per-sequence tuning',limitations='Training demonstration with GT-assisted preprocessing, simulated IMU and GT initialization; not held-out or real-sensor benchmark'),timings_s=timings,runs=results)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,out/'reproduce.py')
    print('Completed. Manifest:',out/'manifest.json',flush=True)

if __name__=='__main__': main()
