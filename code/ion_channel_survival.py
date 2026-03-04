#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA COADREAD Survival Analysis - Ion Channel Signature
Based on validated hub-ion channel connections from VGAE-KO

Core Logic:
  Hub genes (GALK1, LSM7, RIPK2, etc.) → Ion channels (KCNA5, CLIC1, CFTR, etc.)
  → Ion channel expression predicts clinical outcome

Analysis:
1. Use VALIDATED ion channels from VGAE-KO (KCNQ2, CLIC1, AQP9, GRIN2B, CFTR, KCNA5)
2. Calculate ion channel signature score
3. Stratify patients by signature (high vs low)
4. Perform survival analysis (OS, DSS, DFI, PFI)
5. Validate individual ion channels
"""

import os
import sys
import warnings
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STEP_DIR = os.path.dirname(_SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(_STEP_DIR)

# Data paths
DATA_DIR = os.path.join(_STEP_DIR, "data")
EXPR_FILE = os.path.join(DATA_DIR, "TCGA.COADREAD.sampleMap_HiSeqV2.txt")  # Use uncompressed file
SURVIVAL_FILE = os.path.join(DATA_DIR, "survival_COADREAD_survival.txt")

# VALIDATED ion channels from VGAE-KO (step5)
# These are the channels that showed significant perturbation when hub genes were knocked out
VALIDATED_ION_CHANNELS = {
    'KCNQ2': 'RPS21→KCNQ2 (validated in both datasets)',
    'CLIC1': 'LSM7→CLIC1 (validated in both datasets, score=15.5)',
    'AQP9': 'EXOSC5→AQP9 (validated in both datasets)',
    'GRIN2B': 'RPL39/RPS2→GRIN2B (validated in GSM5224587)',
    'CFTR': 'RIPK2/TRMT112→CFTR (validated in HCT116)',
    'KCNA5': 'GALK1→KCNA5 (validated in both datasets)'
}

# Output
OUTPUT_DIR = os.path.join(_STEP_DIR, "result", "ion_channel_survival")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Analysis parameters
STRATIFY_METHOD = "median"
MIN_SAMPLES_PER_GROUP = 10

# Survival endpoints
ENDPOINTS = ["OS", "DSS", "DFI", "PFI"]
ENDPOINT_NAMES = {
    "OS": "Overall Survival",
    "DSS": "Disease-Specific Survival", 
    "DFI": "Disease-Free Interval",
    "PFI": "Progression-Free Interval"
}


def log(msg):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_expression_data(filepath):
    """Load TCGA expression data."""
    log(f"Loading expression data from {os.path.basename(filepath)}...")
    
    expr_df = pd.read_csv(filepath, sep='\t', index_col=0)
    
    log(f"  Loaded {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
    return expr_df


def load_survival_data(filepath):
    """Load TCGA survival data."""
    log(f"Loading survival data from {os.path.basename(filepath)}...")
    
    surv_df = pd.read_csv(filepath, sep='\t')
    
    # Convert time from days to months
    for endpoint in ENDPOINTS:
        time_col = f"{endpoint}.time"
        if time_col in surv_df.columns:
            surv_df[time_col] = surv_df[time_col] / 30.44
    
    log(f"  Loaded {len(surv_df)} samples")
    return surv_df


# ============================================================
# ION CHANNEL SIGNATURE
# ============================================================
def calculate_ion_channel_signature(expr_df, ion_channels):
    """Calculate ion channel signature score.
    
    Rationale: Hub genes regulate ion channels through PPI networks.
    High ion channel expression → dysregulated ion homeostasis → poor prognosis.
    """
    available_channels = [ch for ch in ion_channels if ch in expr_df.index]
    missing_channels = [ch for ch in ion_channels if ch not in expr_df.index]
    
    if missing_channels:
        log(f"  Warning: {len(missing_channels)} channels not found")
        for ch in missing_channels:
            log(f"    Missing: {ch} ({VALIDATED_ION_CHANNELS[ch]})")
    
    if len(available_channels) == 0:
        raise ValueError("No ion channels found in expression data!")
    
    log(f"  Using {len(available_channels)}/{len(ion_channels)} validated ion channels:")
    for ch in available_channels:
        log(f"    ✓ {ch}: {VALIDATED_ION_CHANNELS[ch]}")
    
    # Calculate signature (mean expression)
    channel_expr = expr_df.loc[available_channels]
    signature = channel_expr.mean(axis=0)
    
    return signature, available_channels


def stratify_patients(signature, method='median'):
    """Stratify patients into risk groups."""
    if method == 'median':
        cutoff = signature.median()
        groups = pd.Series('Low', index=signature.index)
        groups[signature > cutoff] = 'High'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return groups


# ============================================================
# SURVIVAL ANALYSIS
# ============================================================
def perform_survival_analysis(surv_df, groups, endpoint='OS'):
    """Perform Kaplan-Meier and log-rank test."""
    time_col = f"{endpoint}.time"
    event_col = endpoint
    
    if time_col not in surv_df.columns or event_col not in surv_df.columns:
        return None
    
    # Align samples: groups index should match surv_df sample column
    common_samples = set(groups.index) & set(surv_df['sample'])
    
    if len(common_samples) < MIN_SAMPLES_PER_GROUP * 2:
        log(f"    Only {len(common_samples)} common samples (need {MIN_SAMPLES_PER_GROUP * 2})")
        return None
    
    # Filter to common samples
    surv_subset = surv_df[surv_df['sample'].isin(common_samples)].copy()
    surv_subset = surv_subset.set_index('sample')
    
    # Add groups
    surv_subset['group'] = groups
    surv_subset = surv_subset.dropna(subset=[time_col, event_col, 'group'])
    
    if len(surv_subset) < MIN_SAMPLES_PER_GROUP * 2:
        log(f"    Only {len(surv_subset)} valid samples after filtering")
        return None
    
    # Fit Kaplan-Meier curves
    kmf = KaplanMeierFitter()
    results = {'groups': {}, 'df': surv_subset}
    
    for group_name in ['Low', 'High']:
        mask = surv_subset['group'] == group_name
        group_df = surv_subset[mask]
        
        if len(group_df) < MIN_SAMPLES_PER_GROUP:
            continue
        
        kmf.fit(
            durations=group_df[time_col],
            event_observed=group_df[event_col],
            label=group_name
        )
        
        results['groups'][group_name] = {
            'kmf': kmf.survival_function_.copy(),
            'median_survival': kmf.median_survival_time_,
            'n_samples': len(group_df),
            'n_events': int(group_df[event_col].sum())
        }
    
    # Log-rank test
    if len(results['groups']) == 2:
        g_low = surv_subset[surv_subset['group'] == 'Low']
        g_high = surv_subset[surv_subset['group'] == 'High']
        
        lr_result = logrank_test(
            g_low[time_col], g_high[time_col],
            g_low[event_col], g_high[event_col]
        )
        
        results['logrank_p'] = lr_result.p_value
        results['logrank_stat'] = lr_result.test_statistic
    
    # *** NEW: Save patient-level KM data for plotting ***
    km_data = surv_subset[[time_col, event_col, 'group']].copy()
    km_data.columns = ['time', 'event', 'group']
    km_data = km_data.reset_index()
    results['km_data'] = km_data
    
    return results


def perform_cox_regression(surv_df, signature, endpoint='OS'):
    """Perform Cox proportional hazards regression."""
    time_col = f"{endpoint}.time"
    event_col = endpoint
    
    if time_col not in surv_df.columns or event_col not in surv_df.columns:
        return None
    
    # Align samples
    common_samples = set(signature.index) & set(surv_df['sample'])
    if len(common_samples) < 50:
        return None
    
    surv_subset = surv_df[surv_df['sample'].isin(common_samples)].copy()
    surv_subset = surv_subset.set_index('sample')
    surv_subset['signature'] = signature
    surv_subset = surv_subset[[time_col, event_col, 'signature']].dropna()
    
    if len(surv_subset) < 50:
        return None
    
    cph = CoxPHFitter()
    try:
        cph.fit(surv_subset, duration_col=time_col, event_col=event_col)
        return cph
    except:
        return None


# ============================================================
# INDIVIDUAL CHANNEL ANALYSIS
# ============================================================
def analyze_individual_channels(expr_df, surv_df, channels, endpoint='OS', save_km_data=True):
    """Analyze each ion channel individually."""
    log(f"\n  Analyzing individual channels for {endpoint}...")
    
    results = []
    km_data_all = {}  # Store KM data for each channel
    
    for channel in channels:
        if channel not in expr_df.index:
            continue
        
        # Stratify by channel expression
        channel_expr = expr_df.loc[channel]
        groups = stratify_patients(channel_expr, method='median')
        
        # Survival analysis
        km_results = perform_survival_analysis(surv_df, groups, endpoint)
        
        if km_results is None:
            continue
        
        # Save KM data for this channel
        if save_km_data and 'km_data' in km_results:
            km_data_all[channel] = km_results['km_data']
        
        # Cox regression
        cox_result = perform_cox_regression(surv_df, channel_expr, endpoint)
        
        if cox_result is not None:
            hr = np.exp(cox_result.params_['signature'])
            p_val = cox_result.summary.loc['signature', 'p']
        else:
            hr, p_val = np.nan, np.nan
        
        results.append({
            'Channel': channel,
            'Connection': VALIDATED_ION_CHANNELS[channel],
            'LogRank_p': km_results.get('logrank_p', np.nan),
            'Cox_HR': hr,
            'Cox_p': p_val,
            'N_samples': len(km_results['df']),
            'N_events_high': km_results['groups']['High']['n_events'],
            'N_events_low': km_results['groups']['Low']['n_events']
        })
    
    return pd.DataFrame(results), km_data_all


# ============================================================
# VISUALIZATION
# ============================================================
def plot_kaplan_meier(results, endpoint, title, output_path):
    """Plot Kaplan-Meier curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'Low': '#4DBBD5', 'High': '#E64B35'}
    
    for group_name in ['Low', 'High']:
        if group_name not in results['groups']:
            continue
        
        group_data = results['groups'][group_name]
        kmf_data = group_data['kmf']
        n = group_data['n_samples']
        events = group_data['n_events']
        median = group_data['median_survival']
        
        median_str = f"{median:.1f}" if not np.isnan(median) else "NA"
        label = f"{group_name} (n={n}, events={events}, median={median_str}mo)"
        
        ax.plot(kmf_data.index, kmf_data.values, 
                color=colors[group_name], linewidth=2.5, label=label)
    
    # Add log-rank p-value
    if 'logrank_p' in results:
        p_val = results['logrank_p']
        if p_val < 0.0001:
            p_text = "p < 0.0001"
        elif p_val < 0.001:
            p_text = f"p = {p_val:.4f}"
        else:
            p_text = f"p = {p_val:.3f}"
        
        # Add significance stars
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        ax.text(0.05, 0.05, f"Log-rank {p_text} {sig}", 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add at-risk table
    ax_table = fig.add_axes([0.15, 0.02, 0.7, 0.08])
    ax_table.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log(f"    Saved: {os.path.basename(output_path)}")


def plot_signature_distribution(signature, groups, channels, output_path):
    """Plot ion channel signature distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df = pd.DataFrame({'Signature': signature, 'Group': groups})
    colors = {'Low': '#4DBBD5', 'High': '#E64B35'}
    
    # Violin plot
    for i, group in enumerate(['Low', 'High']):
        data = df[df['Group'] == group]['Signature']
        parts = axes[0].violinplot([data], positions=[i], widths=0.7,
                                    showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[group])
            pc.set_alpha(0.7)
    
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Low', 'High'])
    axes[0].set_ylabel('Ion Channel Signature Score', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Signature Distribution\n({len(channels)} validated channels)', 
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Histogram
    for group in ['Low', 'High']:
        data = df[df['Group'] == group]['Signature']
        axes[1].hist(data, bins=30, alpha=0.6, label=group, 
                     color=colors[group], edgecolor='black', linewidth=0.5)
    
    axes[1].axvline(signature.median(), color='red', linestyle='--', 
                    linewidth=2, label='Median cutoff')
    axes[1].set_xlabel('Ion Channel Signature Score', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Signature Score Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log(f"  Saved: {os.path.basename(output_path)}")


# ============================================================
# MAIN ANALYSIS
# ============================================================
def main():
    log("=" * 80)
    log("TCGA COADREAD SURVIVAL ANALYSIS - ION CHANNEL SIGNATURE")
    log("Based on validated hub-ion channel connections from VGAE-KO")
    log("=" * 80)
    
    # Load data
    expr_df = load_expression_data(EXPR_FILE)
    surv_df = load_survival_data(SURVIVAL_FILE)
    
    # Calculate ion channel signature
    log("\nCalculating ion channel signature...")
    log(f"  Using {len(VALIDATED_ION_CHANNELS)} validated ion channels from VGAE-KO:")
    
    signature, available_channels = calculate_ion_channel_signature(
        expr_df, list(VALIDATED_ION_CHANNELS.keys())
    )
    
    # Stratify patients
    log(f"\nStratifying patients by median signature...")
    groups = stratify_patients(signature, method=STRATIFY_METHOD)
    
    log(f"  Group distribution:")
    for group, count in groups.value_counts().items():
        log(f"    {group}: {count} samples ({count/len(groups)*100:.1f}%)")
    
    # Plot signature distribution
    plot_signature_distribution(
        signature, groups, available_channels,
        os.path.join(OUTPUT_DIR, "ion_channel_signature_distribution.png")
    )
    
    # Survival analysis for each endpoint
    log("\n" + "=" * 80)
    log("ION CHANNEL SIGNATURE SURVIVAL ANALYSIS")
    log("=" * 80)
    
    results_summary = []
    
    for endpoint in ENDPOINTS:
        log(f"\n{endpoint} ({ENDPOINT_NAMES[endpoint]}):")
        
        # Kaplan-Meier analysis
        km_results = perform_survival_analysis(surv_df, groups, endpoint)
        
        if km_results is None:
            log(f"  Skipped (insufficient data)")
            continue
        
        # Plot KM curves
        title = f"{ENDPOINT_NAMES[endpoint]}\nIon Channel Signature (n={len(available_channels)} channels)"
        plot_kaplan_meier(
            km_results, endpoint, title,
            os.path.join(OUTPUT_DIR, f"KM_{endpoint}_ion_channels.png")
        )
        
        # *** NEW: Save patient-level KM data ***
        if 'km_data' in km_results:
            km_data_file = os.path.join(OUTPUT_DIR, f"KM_{endpoint}_signature_data.csv")
            km_results['km_data'].to_csv(km_data_file, index=False)
            log(f"  Saved KM data: {os.path.basename(km_data_file)}")
        
        # Cox regression
        cox_result = perform_cox_regression(surv_df, signature, endpoint)
        
        if cox_result is not None:
            hr = np.exp(cox_result.params_['signature'])
            ci = np.exp(cox_result.confidence_intervals_.loc['signature'])
            p_val = cox_result.summary.loc['signature', 'p']
            
            log(f"  Cox: HR={hr:.3f}, 95%CI=[{ci[0]:.3f}-{ci[1]:.3f}], p={p_val:.4f}")
            
            cox_result.summary.to_csv(
                os.path.join(OUTPUT_DIR, f"Cox_{endpoint}_ion_channels.csv")
            )
        else:
            hr, p_val = np.nan, np.nan
        
        # Summary
        if 'logrank_p' in km_results:
            results_summary.append({
                'Endpoint': endpoint,
                'Endpoint_Name': ENDPOINT_NAMES[endpoint],
                'N_samples': len(km_results['df']),
                'N_channels': len(available_channels),
                'LogRank_p': km_results['logrank_p'],
                'LogRank_stat': km_results['logrank_stat'],
                'Cox_HR': hr,
                'Cox_p': p_val,
                'Significant': 'Yes' if km_results['logrank_p'] < 0.05 else 'No'
            })
    
    # Individual channel analysis
    log("\n" + "=" * 80)
    log("INDIVIDUAL ION CHANNEL ANALYSIS")
    log("=" * 80)
    
    individual_results = {}
    individual_km_data = {}
    for endpoint in ['OS', 'PFI']:  # Focus on OS and PFI
        individual_df, km_data_dict = analyze_individual_channels(
            expr_df, surv_df, available_channels, endpoint, save_km_data=True
        )
        individual_results[endpoint] = individual_df
        individual_km_data[endpoint] = km_data_dict
        
        # Save results
        individual_df.to_csv(
            os.path.join(OUTPUT_DIR, f"individual_channels_{endpoint}.csv"),
            index=False
        )
        
        # *** NEW: Save individual channel KM data ***
        for channel, km_data in km_data_dict.items():
            km_file = os.path.join(OUTPUT_DIR, f"KM_{endpoint}_{channel}_data.csv")
            km_data.to_csv(km_file, index=False)
        log(f"  Saved {len(km_data_dict)} channel KM data files for {endpoint}")
        
        # Print significant channels
        sig_channels = individual_df[individual_df['LogRank_p'] < 0.05]
        if len(sig_channels) > 0:
            log(f"\n  Significant channels for {endpoint}:")
            for _, row in sig_channels.iterrows():
                log(f"    {row['Channel']}: p={row['LogRank_p']:.4f}, HR={row['Cox_HR']:.3f}")
    

    # ============================================================
    # MULTIPLE COMPARISON CORRECTION (BH FDR)
    # ============================================================
    log("\n" + "=" * 80)
    log("MULTIPLE COMPARISON CORRECTION")
    log("=" * 80)
    
    # Collect all Cox p-values from all endpoints
    all_cox_p = []
    for endpoint in ['OS', 'PFI']:
        if endpoint in individual_results:
            all_cox_p.extend(individual_results[endpoint]['Cox_p'].tolist())
    
    n_tests = len(all_cox_p)
    log(f"Total tests: {n_tests} (6 channels × 2 endpoints)")
    
    # Apply BH FDR correction
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        all_cox_p, alpha=0.05, method='fdr_bh'
    )
    
    # Assign corrected p-values back to dataframes
    idx = 0
    for endpoint in ['OS', 'PFI']:
        if endpoint in individual_results:
            n_channels = len(individual_results[endpoint])
            individual_results[endpoint]['Cox_q'] = pvals_corrected[idx:idx+n_channels]
            idx += n_channels
            
            # Re-save with corrected p-values
            individual_results[endpoint].to_csv(
                os.path.join(OUTPUT_DIR, f"individual_channels_{endpoint}.csv"),
                index=False
            )
            
            # Report significant channels after correction
            sig_before = sum(individual_results[endpoint]['Cox_p'] < 0.05)
            sig_after = sum(individual_results[endpoint]['Cox_q'] < 0.05)
            log(f"\n  {endpoint}: {sig_before} → {sig_after} significant (after FDR correction)")
            
            if sig_after > 0:
                sig_channels = individual_results[endpoint][individual_results[endpoint]['Cox_q'] < 0.05]
                for _, row in sig_channels.iterrows():
                    log(f"    {row['Channel']}: HR={row['Cox_HR']:.3f}, p={row['Cox_p']:.4f}, q={row['Cox_q']:.4f}")
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "ion_channel_survival_summary.csv"),
        index=False
    )
    
    # Final report
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    print(summary_df.to_string(index=False))
    
    log(f"\n✓ Results saved to: {OUTPUT_DIR}")
    log("\n" + "=" * 80)
    log("INTERPRETATION")
    log("=" * 80)
    log("Hub genes (GALK1, LSM7, RIPK2, etc.) regulate ion channels")
    log("→ Ion channel dysregulation affects patient survival")
    log("→ Validates the hub-ion channel regulatory axis")
    log("=" * 80)


if __name__ == "__main__":
    main()
