import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

def load_nifti(filepath):
    print(f"loading file: {filepath}")
    img = nib.load(filepath)
    data = img.get_fdata()
    print(f"  shape: {data.shape}  (x, y, z, time)")
    return data, img.affine, img


def get_middle_slice(volume_3d):
    z_mid = volume_3d.shape[2] // 2
    return volume_3d[:, :, z_mid]


def compute_frame_diff(data_4d):
    n_vols = data_4d.shape[-1]
    diffs = []
    
    for t in range(1, n_vols):
        prev_vol = data_4d[:, :, :, t - 1]
        curr_vol = data_4d[:, :, :, t]
        
        # mean abs diff normalized by mean signal (so it's relative, not absolute)
        mean_signal = np.mean(np.abs(prev_vol)) 
        diff = np.mean(np.abs(curr_vol - prev_vol)) / mean_signal
        diffs.append(diff)
    
    return np.array(diffs)


def compute_ref_correlation(data_4d, ref_vol_idx=0):
    
    ref_vol = get_middle_slice(data_4d[:, :, :, ref_vol_idx])
    ref_flat = ref_vol.flatten()
    
    correlations = []
    
    for t in range(data_4d.shape[-1]):
        curr_vol = get_middle_slice(data_4d[:, :, :, t])
        curr_flat = curr_vol.flatten()
        
        # pearson correlation 
        corr = np.corrcoef(ref_flat, curr_flat)[0, 1]
        correlations.append(corr)
    
    return np.array(correlations)


def compute_mean_signal_per_vol(data_4d):
    """
    just the average signal intensity in each volume
    big sudden drops or jumps can indicate motion or RF issues
    """
    n_vols = data_4d.shape[-1]
    means = [np.mean(data_4d[:, :, :, t]) for t in range(n_vols)]
    return np.array(means)


def flag_bad_volumes(frame_diffs, corr_values, diff_threshold=0.15, corr_threshold=0.95):
    bad_by_diff = np.where(frame_diffs > diff_threshold)[0] + 1  
    bad_by_corr = np.where(corr_values < corr_threshold)[0]
    
    
    bad_vols = list(set(bad_by_diff.tolist() + bad_by_corr.tolist()))
    bad_vols.sort()
    
    return bad_vols


#report

def motion_summary(data_4d, bad_vols, frame_diffs, corr_values):
    n_vols = data_4d.shape[-1]
    pct_bad = len(bad_vols) / n_vols * 100
    
    print("\n.           MOTION QC REPORT          ")
    print(f"  Total volumes      : {n_vols}")
    print(f"  Bad volumes flagged: {len(bad_vols)} ({pct_bad:.1f}%)")
    
    if bad_vols:
        print(f"  Bad volume indices : {bad_vols}")
    
    print(f"  Max frame diff     : {frame_diffs.max():.4f}")
    print(f"  Mean frame diff    : {frame_diffs.mean():.4f}")
    print(f"  Min correlation    : {corr_values.min():.4f}")
    print(f"  Mean correlation   : {corr_values.mean():.4f}")
    
    # overall result
    if pct_bad > 20:
        verdict = "FAIL (>20% volumes affected, consider excluding scan)"
    elif pct_bad > 5:
        verdict = "WARN (5-20% volumes affected, review manually)"
    else:
        verdict = "PASS (<5% volumes affected)"
    
    print(f"\n  Overall verdict    : {verdict}")
    print("--\n")
    
    return {"n_vols": n_vols, "n_bad": len(bad_vols), "pct_bad": pct_bad, "verdict": verdict}


# plotting

def plot_motion_metrics(frame_diffs, corr_values, mean_signals, bad_vols, 
                         diff_threshold=0.15, corr_threshold=0.95,
                         save_path=None):
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle("ASL Motion QC - Volume Metrics", fontsize=14)
    
    n_vols = len(corr_values)
    
    # plot 1: frame diffs
    ax = axes[0]
    t_diff = np.arange(1, len(frame_diffs) + 1)
    ax.plot(t_diff, frame_diffs, color='steelblue', linewidth=1.2, label='frame diff')
    ax.axhline(y=diff_threshold, color='red', linestyle='--', alpha=0.7, label=f'threshold ({diff_threshold})')
    
    # highlight bad vols
    bad_in_range = [v for v in bad_vols if 1 <= v <= len(frame_diffs)]
    if bad_in_range:
        ax.scatter(bad_in_range, frame_diffs[[v-1 for v in bad_in_range]], 
                   color='red', zorder=5, label='flagged', s=50)
    
    ax.set_ylabel("Frame Diff (normalized)")
    ax.set_title("Frame-to-Frame Difference")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # plot 2: correlation
    ax = axes[1]
    t_corr = np.arange(n_vols)
    ax.plot(t_corr, corr_values, color='darkorange', linewidth=1.2)
    ax.axhline(y=corr_threshold, color='red', linestyle='--', alpha=0.7, label=f'threshold ({corr_threshold})')
    
    bad_corr = [v for v in bad_vols if 0 <= v < n_vols]
    if bad_corr:
        ax.scatter(bad_corr, corr_values[bad_corr], color='red', zorder=5, s=50, label='flagged')
    
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation with Reference Volume")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.8, 1.02)
    
    # plot 3: mean signal
    ax = axes[2]
    ax.plot(t_corr, mean_signals, color='seagreen', linewidth=1.2)
    ax.set_ylabel("Mean Intensity")
    ax.set_title("Mean Signal per Volume (spikes = possible motion/RF artifact)")
    ax.set_xlabel("Volume index (time)")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()




def run_motion_qc(nifti_path, diff_threshold=0.15, corr_threshold=0.95, output_dir=None):
    
    data, affine, img = load_nifti(nifti_path)
    
    if data.ndim != 4:
        raise ValueError(f"expected 4D data, got shape {data.shape}")
    
    if data.shape[-1] < 5:
        raise ValueError("need at least 5 timepoints for motion QC to make sense")
    
    
    print("computing motion metrics...")
    frame_diffs = compute_frame_diff(data)
    corr_values = compute_ref_correlation(data, ref_vol_idx=0)
    mean_signals = compute_mean_signal_per_vol(data)
    
    bad_vols = flag_bad_volumes(frame_diffs, corr_values, diff_threshold, corr_threshold)
    
    summary = motion_summary(data, bad_vols, frame_diffs, corr_values)
    
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.basename(nifti_path).replace(".nii.gz", "").replace(".nii", "")
        save_path = os.path.join(output_dir, f"{fname}_motion_qc.png")
    else:
        save_path = None
    
    plot_motion_metrics(frame_diffs, corr_values, mean_signals, bad_vols,
                        diff_threshold, corr_threshold, save_path=save_path)
    
    # return everything in case someone wants to do more stuff with it
    return {
        **summary,
        "bad_vols": bad_vols,
        "frame_diffs": frame_diffs,
        "corr_values": corr_values,
        "mean_signals": mean_signals
    }




if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        test_file = "test_asl.nii.gz" 
        print(f"no file given, using: {test_file}")
    else:
        test_file = sys.argv[1]
    
    results = run_motion_qc(
        nifti_path=test_file,
        diff_threshold=0.15,
        corr_threshold=0.95,
        output_dir="outputs"   
    )
