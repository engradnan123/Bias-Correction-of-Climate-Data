# ============================================================
# BIAS CORRECTION OF REANALYSIS RAINFALL
# ============================================================

from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from scipy.stats import gamma, ks_2samp
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# USER SETTINGS (PATH FROM YOUR DRIVE).
# SET THE THRESHOLD TO DETECT RAINFALL EVENTS IN YOUR DATA
# ============================================================
input_file = r"E:\Data\Corrected_Data\Bias_Correc_Rainfall.csv"

obs_col = "Observed_Rainfall"
sim_col = "Reanalyis_Rainfall"

rain_threshold = 1.0  # mm/day

output_dir = os.path.dirname(input_file)
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# READ DATA
# ============================================================
df = pd.read_csv(input_file)
print(f"✅ File loaded: {input_file}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

obs = pd.to_numeric(df[obs_col], errors="coerce")
sim = pd.to_numeric(df[sim_col], errors="coerce")

print(f"\nData Summary:")
print(f"Observed mean: {obs.mean():.2f} mm, std: {obs.std():.2f}")
print(f"Simulated mean: {sim.mean():.2f} mm, std: {sim.std():.2f}")
print(
    f"Observed wet days (≥{rain_threshold}mm): {(obs >= rain_threshold).sum()}")
print(
    f"Simulated wet days (≥{rain_threshold}mm): {(sim >= rain_threshold).sum()}")

# Training mask (aligned obs + sim)
train_mask = (~obs.isna()) & (~sim.isna())
obs_train = obs[train_mask].values
sim_train = sim[train_mask].values

# Full simulation array for correction (keep NaN for missing)
sim_full = sim.values

# ============================================================
# BIAS CORRECTION METHODS
# ============================================================


def linear_scaling(obs_t, sim_t, sim_f):
    """Multiply all values by constant correction factor"""
    # Avoid division by zero
    if np.nanmean(sim_t) == 0:
        cf = 1
    else:
        cf = np.nanmean(obs_t) / np.nanmean(sim_t)

    print(f"  Linear Scaling - CF: {cf:.4f}")
    corrected = sim_f * cf
    corrected[corrected < 0] = 0
    return corrected, cf, "*"


def local_intensity_scaling(obs_t, sim_t, sim_f):
    """Scale only wet days"""
    wet = sim_t >= rain_threshold
    if np.sum(wet) > 0:
        cf = np.nanmean(obs_t[wet]) / np.nanmean(sim_t[wet])
    else:
        cf = 1

    print(
        f"  Local Intensity - CF: {cf:.4f} (based on {np.sum(wet)} wet days)")
    corrected = np.where(sim_f >= rain_threshold, sim_f * cf, sim_f)
    corrected[corrected < 0] = 0
    return corrected, cf, "* (wet-day)"


def power_transformation(obs_t, sim_t, sim_f):
    """Power transformation for wet days"""
    wet = sim_t >= rain_threshold
    if np.sum(wet) > 0 and np.nanmean(sim_t[wet]) > 0:
        alpha = np.log(np.nanmean(obs_t[wet]) + 1) / \
            np.log(np.nanmean(sim_t[wet]) + 1)
    else:
        alpha = 1

    print(f"  Power Transform - α: {alpha:.4f}")
    corrected = np.where(sim_f >= rain_threshold,
                         (sim_f + 1) ** alpha - 1,
                         sim_f)
    corrected[corrected < 0] = 0
    return corrected, alpha, "power"


def gamma_quantile_mapping(obs_t, sim_t, sim_f):
    """Gamma distribution based QM"""
    obs_pos = obs_t[obs_t > 0]
    sim_pos = sim_t[sim_t > 0]

    corrected = sim_f.copy()

    if len(obs_pos) < 2 or len(sim_pos) < 2:
        print(f"  Gamma QM - Not enough data for fitting")
        return corrected, np.nan, "CDF (Gamma)"

    try:
        o_shape, _, o_scale = gamma.fit(obs_pos, floc=0)
        s_shape, _, s_scale = gamma.fit(sim_pos, floc=0)

        print(f"  Gamma QM - Obs: shape={o_shape:.3f}, scale={o_scale:.3f}")
        print(f"  Gamma QM - Sim: shape={s_shape:.3f}, scale={s_scale:.3f}")

        for i, v in enumerate(sim_f):
            if v > 0:
                p = gamma.cdf(v, s_shape, scale=s_scale)
                corrected[i] = gamma.ppf(p, o_shape, scale=o_scale)

        corrected[corrected < 0] = 0
    except Exception as e:
        print(f"  Gamma QM - Error: {e}")

    return corrected, np.nan, "CDF (Gamma)"


def delta_quantile_mapping(obs_t, sim_t, sim_f):
    """Add delta from quantile mapping"""
    try:
        q = np.linspace(1, 99, 99)
        obs_q = np.nanpercentile(obs_t, q)
        sim_q = np.nanpercentile(sim_t, q)

        delta = obs_q - sim_q
        f = interp1d(sim_q, delta, bounds_error=False,
                     fill_value="extrapolate")

        corrected = sim_f + f(sim_f)
        corrected[corrected < 0] = 0

        print(f"  Delta QM - Mean delta: {np.nanmean(delta):.4f}")
        return corrected, np.nanmean(delta), "+ (ΔQ)"
    except Exception as e:
        print(f"  Delta QM - Error: {e}")
        return sim_f, np.nan, "+ (ΔQ)"


def empirical_quantile_mapping(obs_t, sim_t, sim_f):
    """Empirical CDF matching"""
    try:
        q = np.linspace(0, 100, 101)
        obs_q = np.nanpercentile(obs_t, q)
        sim_q = np.nanpercentile(sim_t, q)

        f = interp1d(sim_q, obs_q, bounds_error=False,
                     fill_value="extrapolate")
        corrected = f(sim_f)
        corrected[corrected < 0] = 0

        print(f"  Empirical QM - Applied")
        return corrected, np.nan, "CDF (Empirical)"
    except Exception as e:
        print(f"  Empirical QM - Error: {e}")
        return sim_f, np.nan, "CDF (Empirical)"


def mean_variance_scaling(obs_t, sim_t, sim_f):
    """Mean and variance scaling"""
    mu_o, mu_s = np.nanmean(obs_t), np.nanmean(sim_t)
    sd_o, sd_s = np.nanstd(obs_t), np.nanstd(sim_t)

    cf = sd_o / sd_s if sd_s > 0 else 1
    corrected = mu_o + (sim_f - mu_s) * cf
    corrected[corrected < 0] = 0

    print(
        f"  Mean-Variance - μ_obs: {mu_o:.3f}, μ_sim: {mu_s:.3f}, CF: {cf:.3f}")
    return corrected, cf, "μ–σ"


methods = {
    "Linear_Scaling": linear_scaling,
    "Local_Intensity_Scaling": local_intensity_scaling,
    "Power_Transformation": power_transformation,
    "Gamma_QM": gamma_quantile_mapping,
    "Delta_QM": delta_quantile_mapping,
    "Empirical_QM": empirical_quantile_mapping,
    "Mean_Variance": mean_variance_scaling,
}

# ============================================================
# APPLY CORRECTIONS
# ============================================================
print(f"\n{'='*60}")
print("APPLYING BIAS CORRECTION METHODS")
print('='*60)

corrected = {}
factors = {}
operations = {}

for name, func in methods.items():
    print(f"\n{name}:")
    corrected[name], factors[name], operations[name] = func(
        obs_train, sim_train, sim_full
    )

    # Debug: Show some stats
    if not np.all(np.isnan(corrected[name])):
        valid = ~np.isnan(corrected[name])
        print(f"  Corrected mean: {np.nanmean(corrected[name][valid]):.3f}, "
              f"std: {np.nanstd(corrected[name][valid]):.3f}")

# ============================================================
# SAVE BIAS-CORRECTED DATA
# ============================================================
print(f"\n{'='*60}")
print("SAVING RESULTS")
print('='*60)

out_df = df.copy()

for name, data in corrected.items():
    out_df[f"{sim_col}_{name}"] = data

    # Save factor if it's a scalar
    if isinstance(factors[name], (int, float, np.number)):
        out_df[f"{name}_Factor"] = factors[name]
    else:
        out_df[f"{name}_Factor"] = np.nan

    out_df[f"{name}_Operation"] = operations[name]

out_csv = os.path.join(output_dir, f"Bias_Corrected_{sim_col}_FULL.csv")
out_df.to_csv(out_csv, index=False)
print(f"✅ Bias-corrected data saved: {out_csv}")

# Create summary CSV
summary_data = []
for name in methods.keys():
    if name in corrected and name in factors:
        valid = ~np.isnan(corrected[name])
        if np.sum(valid) > 0:
            summary_data.append({
                "Method": name,
                "Correction_Factor": factors[name] if isinstance(factors[name], (int, float, np.number)) else np.nan,
                "Original_Mean": np.nanmean(sim_full),
                "Corrected_Mean": np.nanmean(corrected[name][valid]),
                "Original_Std": np.nanstd(sim_full[~np.isnan(sim_full)]),
                "Corrected_Std": np.nanstd(corrected[name][valid]),
                "Operation": operations[name]
            })

summary_df = pd.DataFrame(summary_data)
summary_csv = os.path.join(output_dir, f"Correction_Summary_{sim_col}.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"✅ Correction summary saved: {summary_csv}")

# ============================================================
# METRICS + GOODNESS-OF-FIT
# ============================================================
print(f"\n{'='*60}")
print("CALCULATING METRICS")
print('='*60)

metrics = {}

obs_sorted = np.sort(obs.dropna())
obs_cdf = np.linspace(0, 1, len(obs_sorted))

for name, data in corrected.items():
    valid = (~np.isnan(data)) & (~obs.isna())
    o = obs[valid].values
    s = data[valid]

    if len(o) < 2 or len(s) < 2:
        print(f"⚠️  {name}: Not enough data for metrics")
        continue

    print(f"  {name}: n={len(o)}")

    # Statistics
    rmse = np.sqrt(mean_squared_error(o, s))
    mae = np.mean(np.abs(o - s))
    r = np.corrcoef(o, s)[0, 1] if len(o) > 1 else np.nan
    pbias = 100 * np.sum(s - o) / np.sum(o) if np.sum(o) != 0 else np.nan
    nse = 1 - np.sum((o - s) ** 2) / np.sum((o - np.mean(o)) **
                                            2) if np.sum((o - np.mean(o)) ** 2) != 0 else np.nan

    # Categorical
    obs_event = o >= rain_threshold
    sim_event = s >= rain_threshold
    hits = np.sum(obs_event & sim_event)
    misses = np.sum(obs_event & ~sim_event)
    false = np.sum(~obs_event & sim_event)

    pod = hits / (hits + misses) if hits + misses > 0 else np.nan
    far = false / (hits + false) if hits + false > 0 else np.nan
    csi = hits / (hits + misses + false) if hits + \
        misses + false > 0 else np.nan

    # CDF metrics
    try:
        sim_sorted = np.sort(s)
        sim_cdf = np.linspace(0, 1, len(sim_sorted))
        f_sim = interp1d(sim_sorted, sim_cdf,
                         bounds_error=False, fill_value=(0, 1))
        cdf_rmse = np.sqrt(np.mean((obs_cdf - f_sim(obs_sorted)) ** 2))
    except:
        cdf_rmse = np.nan

    ks_stat, ks_p = ks_2samp(o, s) if len(
        o) > 0 and len(s) > 0 else (np.nan, np.nan)

    metrics[name] = {
        "RMSE": rmse,
        "MAE": mae,
        "R": r,
        "PBIAS (%)": pbias,
        "NSE": nse,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "CDF_RMSE": cdf_rmse,
        "KS_Statistic": ks_stat,
        "KS_pvalue": ks_p,
        "n_samples": len(o)
    }

metrics_df = pd.DataFrame(metrics).T
metrics_csv = os.path.join(
    output_dir, f"Bias_Correction_Metrics_{sim_col}.csv")
metrics_df.to_csv(metrics_csv)
print(f"\n✅ Metrics saved: {metrics_csv}")

# ============================================================
# CDF PLOT
# ============================================================
print(f"\n{'='*60}")
print("CREATING PLOTS")
print('='*60)

plt.figure(figsize=(12, 8))

colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

# Plot observed CDF
plt.plot(obs_sorted, obs_cdf, "k", lw=3, label="Observed", zorder=10)

# Plot corrected CDFs
for idx, (name, data) in enumerate(corrected.items()):
    valid = ~np.isnan(data)
    if np.sum(valid) > 0:
        sorted_data = np.sort(data[valid])
        plt.plot(sorted_data,
                 np.linspace(0, 1, np.sum(valid)),
                 label=name,
                 color=colors[idx % len(colors)],
                 alpha=0.8,
                 linewidth=2)

plt.xlabel("Rainfall (mm)", fontsize=12)
plt.ylabel("Cumulative Probability", fontsize=12)
plt.title(
    f"CDF Comparison – Bias Correction Methods\nObserved vs Corrected {sim_col}", fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)

cdf_file = os.path.join(output_dir, f"CDF_Plot_{sim_col}.png")
plt.savefig(cdf_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ CDF plot saved: {cdf_file}")

# ============================================================
# SCATTER PLOT
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

method_names = list(methods.keys())

for i, ax in enumerate(axes):
    if i < len(method_names):
        name = method_names[i]
        if name in corrected:
            valid = (~np.isnan(corrected[name])) & (~obs.isna())
            if np.sum(valid) > 0:
                # First 1000 points for clarity
                o_scatter = obs[valid].values[:1000]
                s_scatter = corrected[name][valid][:1000]

                ax.scatter(o_scatter, s_scatter, alpha=0.5, s=10)
                ax.plot([0, max(o_scatter.max(), s_scatter.max())],
                        [0, max(o_scatter.max(), s_scatter.max())],
                        'r--', alpha=0.5)
                ax.set_xlabel("Observed (mm)")
                ax.set_ylabel("Corrected (mm)")
                ax.set_title(name, fontsize=10)
                ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(len(method_names), len(axes)):
    axes[i].set_visible(False)

plt.suptitle(f"Observed vs Corrected Rainfall - Scatter Plots", fontsize=14)
plt.tight_layout()

scatter_file = os.path.join(output_dir, f"Scatter_Plots_{sim_col}.png")
plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Scatter plots saved: {scatter_file}")

print(f"\n{'='*60}")
print("✅ PROCESSING COMPLETE!")
print('='*60)
print(f"Output files in: {output_dir}")
print(f"1. {out_csv}")
print(f"2. {summary_csv}")
print(f"3. {metrics_csv}")
print(f"4. {cdf_file}")
print(f"5. {scatter_file}")
print('='*60)
