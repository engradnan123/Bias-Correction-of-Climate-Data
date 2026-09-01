from scipy.stats import gamma, ks_2samp
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
input_file = r"E:\BEST Pak\Meteorological Data\Bias Correction\Corrected_Data\Rainfall_2005_2025.csv"
obs_col = "Observed_Rainfall"
sim_cols = [f"S{i}" for i in range(1, 21)] + ["Avg"]
output_dir = os.path.dirname(input_file)

durations = {'5min': 5, '15min': 15, '30min': 30, '1hr': 60,
             '6hr': 360, '12hr': 720, '24hr': 1440, '48hr': 2880, '72hr': 4320}
return_periods = [1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 1000]

dist_dict = {
    'Normal': stats.norm, 'Log-Normal': stats.lognorm, 'Gumbel': stats.gumbel_r,
    'GEV': stats.genextreme, 'Pearson3': stats.pearson3,
    'Gen_Pareto': stats.genpareto, 'Weibull': stats.weibull_min
}

# ==========================================
# 2. DATA LOADING & BIAS CORRECTION
# ==========================================
df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.dropna(subset=[obs_col]).sort_values("Date")

# Correcting ONLY the 'Avg' column for IDF analysis
q = np.linspace(0, 100, 101)
obs_q = np.nanpercentile(df[obs_col], q)
sim_q = np.nanpercentile(df["Avg"], q)
sq, idx = np.unique(sim_q, return_index=True)
f_interp = interp1d(
    sq, obs_q[idx], bounds_error=False, fill_value="extrapolate")

df["Avg_Corrected"] = np.maximum(f_interp(df["Avg"]), 0)

# ==========================================
# 3. AMS & DISAGGREGATION
# ==========================================
df['Year'] = df['Date'].dt.year
daily_ams = df.groupby('Year')["Avg_Corrected"].max()

ams_table = pd.DataFrame({'Year': daily_ams.index})
for d_name, d_min in durations.items():
    # Mononobe Formula
    ams_table[d_name] = daily_ams.values * (d_min / 1440)**0.33

# ==========================================
# 4. IDF FITTING & BEST FIT LOGIC
# ==========================================
idf_results, gof_results, best_fits = [], [], []

for d_name, d_min in durations.items():
    data = ams_table[d_name].values
    best_ks = float('inf')
    best_dist = None

    for d_label, d_obj in dist_dict.items():
        try:
            # Fitting: LP3 logic requires log transformation
            if d_label == 'Pearson3':
                log_data = np.log10(data[data > 0])
                params = d_obj.fit(log_data)
                ks_s, ks_p = stats.kstest(log_data, 'pearson3', args=params)
            else:
                params = d_obj.fit(data)
                ks_s, ks_p = stats.kstest(data, d_obj.name, args=params)

            gof_results.append(
                {'Duration': d_name, 'Distribution': d_label, 'KS_Stat': ks_s, 'P_Value': ks_p})

            # Identify Best Fit
            if ks_s < best_ks:
                best_ks = ks_s
                best_dist = d_label

            # Generate Intensities
            for rp in return_periods:
                p = 1 - (1/rp) if rp > 1 else 0.5
                val = 10**(d_obj.ppf(p, *params)
                           ) if d_label == 'Pearson3' else d_obj.ppf(p, *params)
                idf_results.append(
                    {'Dist': d_label, 'Duration': d_min, 'RP': rp, 'Intensity': (val / d_min) * 60})
        except:
            continue

    best_fits.append(
        {'Duration': d_name, 'Best_Distribution': best_dist, 'Min_KS': best_ks})

# ==========================================
# 5. OUTPUTS & VISUALIZATION
# ==========================================
idf_df = pd.DataFrame(idf_results)
best_df = pd.DataFrame(best_fits)

# Plot IDF Curves for the Best Fit Distribution per duration

plt.figure(figsize=(10, 6))
for rp in [2, 10, 50, 100]:
    x, y = [], []
    for d_name, d_min in durations.items():
        best_d = best_df[best_df['Duration'] ==
                         d_name]['Best_Distribution'].values[0]
        val = idf_df[(idf_df['Duration'] == d_min) & (idf_df['Dist'] == best_d) & (
            idf_df['RP'] == rp)]['Intensity'].values[0]
        x.append(d_min)
        y.append(val)
    plt.plot(x, y, '-o', label=f"{rp}-yr RP")

plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.title("IDF Curves for Avg Corrected (Best Fit per Duration)")
plt.xlabel("Duration (min)")
plt.ylabel("Intensity (mm/hr)")
plt.legend()
plt.savefig(os.path.join(output_dir, "Avg_IDF_Plot.png"))

with pd.ExcelWriter(os.path.join(output_dir, "Avg_Only_IDF_Report.xlsx")) as writer:
    ams_table.to_excel(writer, sheet_name='AMS_Data', index=False)
    best_df.to_excel(writer, sheet_name='Best_Dist_Summary', index=False)
    pd.DataFrame(gof_results).to_excel(
        writer, sheet_name='Detailed_GOF', index=False)
    idf_df.pivot_table(index=['Dist', 'Duration'], columns='RP',
                       values='Intensity').to_excel(writer, sheet_name='IDF_Tables')

print(f"✅ Analysis for Column 'Avg' complete. Files saved in {output_dir}")

# ============================================================
# SETTINGS & DATA LOADING
# ============================================================
input_file = r"E:\BEST Pak\Meteorological Data\Bias Correction\Corrected_Data\Rainfall_2005_2025.csv"
obs_col = "Observed_Rainfall"
sim_cols = [f"S{i}" for i in range(1, 21)] + ["Avg"]
rain_threshold = 1.0

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=[obs_col]).sort_values("Date")

# ============================================================
# BIAS CORRECTION FUNCTIONS (Returning Values + Factors)
# ============================================================


def get_linear_scaling(obs, sim):
    factor = np.mean(obs) / np.mean(sim) if np.mean(sim) != 0 else 1
    return sim * factor, factor, "P_corr = P_sim * Factor"


def get_loci(obs, sim):
    wet_sim = sim >= rain_threshold
    factor = np.mean(obs[obs >= rain_threshold]) / \
        np.mean(sim[wet_sim]) if np.sum(wet_sim) > 0 else 1
    corrected = np.where(sim >= rain_threshold, sim * factor, 0)
    return corrected, factor, "P_corr = P_sim * Factor (Only if P_sim >= 1mm)"


def get_power_trans(obs, sim):
    wet_sim = sim >= rain_threshold
    # Solving for b in Mean(Obs) = Mean(Sim^b)
    alpha = np.log(np.mean(obs[obs >= rain_threshold])+1) / \
        np.log(np.mean(sim[wet_sim])+1) if np.sum(wet_sim) > 0 else 1
    corrected = np.where(sim >= rain_threshold, (sim + 1)**alpha - 1, 0)
    return corrected, alpha, "P_corr = (P_sim + 1)^Alpha - 1"


def get_mean_var(obs, sim):
    mu_o, mu_s = np.mean(obs), np.mean(sim)
    std_o, std_s = np.std(obs), np.std(sim)
    factor = std_o / std_s if std_s > 0 else 1
    corrected = np.maximum(mu_o + (sim - mu_s) * factor, 0)
    return corrected, factor, "P_corr = Mean_Obs + (P_sim - Mean_Sim) * (Std_Obs/Std_Sim)"


def get_empirical_qm(obs, sim):
    q = np.linspace(0, 100, 101)
    obs_q = np.nanpercentile(obs, q)
    sim_q = np.nanpercentile(sim, q)
    sq, idx = np.unique(sim_q, return_index=True)
    f = interp1d(sq, obs_q[idx], bounds_error=False, fill_value="extrapolate")
    return np.maximum(f(sim), 0), np.nan, "P_corr = Direct_Quantile_Mapping"


# ============================================================
# PROCESSING LOOP
# ============================================================
results_df = df[["Date", obs_col]].copy()
full_metrics = []

methods = {
    "Linear_Scaling": get_linear_scaling,
    "LOCI": get_loci,
    "Power_Trans": get_power_trans,
    "Mean_Variance": get_mean_var,
    "Empirical_QM": get_empirical_qm
}

for col in sim_cols:
    obs_data = df[obs_col].values
    sim_data = df[col].values

    for name, func in methods.items():
        corrected, factor, logic = func(obs_data, sim_data)

        # Add to time series (naming convention: S1_Linear_Scaling)
        results_df[f"{col}_{name}"] = corrected

        # Calculate Metrics
        rmse = np.sqrt(mean_squared_error(obs_data, corrected))
        nse = 1 - (np.sum((obs_data - corrected)**2) /
                   np.sum((obs_data - np.mean(obs_data))**2))
        ks_stat, ks_p = ks_2samp(obs_data, corrected)

        full_metrics.append({
            "Source_Column": col,
            "Method": name,
            "Correction_Factor": factor if not np.isnan(factor) else "N/A",
            "Application_Logic": logic,
            "RMSE": rmse,
            "NSE": nse,
            "KS_Statistic": ks_stat,
            "KS_pvalue": ks_p,
            "Mean_Corrected": np.mean(corrected),
            "Std_Dev_Corrected": np.std(corrected)
        })

# ============================================================
# EXPORT TO EXCEL
# ============================================================
output_path = os.path.join(os.path.dirname(
    input_file), "Detailed_MultiMethod_Bias_Correction.xlsx")
with pd.ExcelWriter(output_path) as writer:
    pd.DataFrame(full_metrics).to_excel(
        writer, sheet_name='Metrics_&_Factors', index=False)
    results_df.to_excel(writer, sheet_name='All_Corrected_Series', index=False)

print(
    f"✅ Processing complete. Metrics for all 21 columns and all methods saved to: {output_path}")
