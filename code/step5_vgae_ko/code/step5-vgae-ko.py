"""
VGAE-KO: Virtual Gene Knockout Validation Pipeline
Following GenKI (Yang et al. 2023, Nucleic Acids Research) methodology with robust statistics.

Pipeline:
  1. Train VGAE ONCE on WT (Wild Type) graph per dataset
  2. Obtain Z_WT (latent representation) using trained model
  3. Create KO graph by removing edges connected to target gene + zeroing expression
  4. Obtain Z_KO by passing KO data through FROZEN WT model
  5. Compute KL divergence between Z_WT and Z_KO distributions

Statistical Methods (Hybrid Approach):
  1. GenKI Bagging (stability criterion):
     - Bootstrap cell labels N times, compute KL divergence for each permutation
     - For each permutation, identify genes in top 5% by KL magnitude
     - A gene is "significant" if it appears in top 5% for >= 95% of permutations

  2. Robust MAD-based Statistics (magnitude criterion):
     - KL distributions exhibit extreme outliers (max/median > 10^8)
     - Standard mean/variance estimation breaks down (breakdown point = 0%)
     - MAD (Median Absolute Deviation) is robust to 50% contamination
     - Z_robust = (KL - median) / (1.4826 * MAD)
     - Convert to p-values and apply Benjamini-Hochberg FDR correction
     - Significant if FDR < 0.05

  Combined: A gene is validated if EITHER criterion is met.

Usage:
  python vgae_ko_pipeline.py              # skip existing results
  python vgae_ko_pipeline.py --force      # rerun everything
  python vgae_ko_pipeline.py --report     # report only (no VGAE runs)
"""

import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STEP_DIR = os.path.dirname(_SCRIPT_DIR)  # step5_vgae_ko
_DATA_DIR = os.path.join(_STEP_DIR, "data")
DATASETS = {
    "HCT116": os.path.join(_DATA_DIR, "scTenifoldKnk_HCT116_count.csv"),
    "GSM5224587": os.path.join(_DATA_DIR, "scTenifoldKnk_GSM5224587_count.csv"),
}
OUTPUT_DIR = os.path.join(_STEP_DIR, "result", "VGAE_KO_Unified")
# Raw annotation source for GSM5224587 (auto-converted if needed)
GSM5224587_RAW_GZ = os.path.join(_DATA_DIR, "GSM5224587", "GSM5225487_HCT116-mock_anno.csv.gz")
MART_EXPORT_CSV = os.path.join(_DATA_DIR, "mart_export.csv")

BRIDGE_PATHS = [
    # All paths from Figure 4C (SourceData_Fig4C_IonBridgePaths.csv)
    # Using full local STRING v12.0 database — 16 distinct ion channels
    {
        "ko": "APBB1IP",
        "channel": "KCNA5",
        "path": "APBB1IP->SRC->KCNA5",
        "intermediates": ["SRC"],
    },
    {
        "ko": "CCDC167",
        "channel": "PKD2",
        "path": "CCDC167->PRKCSH->PKD2",
        "intermediates": ["PRKCSH"],
    },
    {
        "ko": "CD27",
        "channel": "KCNN3",
        "path": "CD27->KCNN3",
        "intermediates": [],
    },
    {
        "ko": "CD6",
        "channel": "GRIK2",
        "path": "CD6->SDCBP->GRIK2",
        "intermediates": ["SDCBP"],
    },
    {
        "ko": "EXOSC5",
        "channel": "AQP9",
        "path": "EXOSC5->EXOSC8->AQP9",
        "intermediates": ["EXOSC8"],
    },
    {
        "ko": "FCRL5",
        "channel": "RYR3",
        "path": "FCRL5->CD38->RYR3",
        "intermediates": ["CD38"],
    },
    {
        "ko": "GALK1",
        "channel": "KCNA5",
        "path": "GALK1->TPI1->KCNA5",
        "intermediates": ["TPI1"],
    },
    {
        "ko": "ITGAL",
        "channel": "GRIN2A",
        "path": "ITGAL->SRC->GRIN2A",
        "intermediates": ["SRC"],
    },
    {
        "ko": "LAG3",
        "channel": "RYR3",
        "path": "LAG3->CD38->RYR3",
        "intermediates": ["CD38"],
    },
    {
        "ko": "LAGE3",
        "channel": "ANO4",
        "path": "LAGE3->TP53RK->ANO4",
        "intermediates": ["TP53RK"],
    },
    {
        "ko": "LSM7",
        "channel": "CLIC1",
        "path": "LSM7->LSM1->CLIC1",
        "intermediates": ["LSM1"],
    },
    {
        "ko": "NAA10",
        "channel": "GRIN1",
        "path": "NAA10->GRIN1",
        "intermediates": [],
    },
    {
        "ko": "PDCD5",
        "channel": "TRPC3",
        "path": "PDCD5->KAT5->TRPC3",
        "intermediates": ["KAT5"],
    },
    {
        "ko": "PFDN4",
        "channel": "CLIC2",
        "path": "PFDN4->VBP1->CLIC2",
        "intermediates": ["VBP1"],
    },
    {
        "ko": "RIPK2",
        "channel": "CFTR",
        "path": "RIPK2->HSPA8->CFTR",
        "intermediates": ["HSPA8"],
    },
    {
        "ko": "RPL12",
        "channel": "KCNA3",
        "path": "RPL12->KCNA3",
        "intermediates": [],
    },
    {
        "ko": "RPL39",
        "channel": "GRIN2B",
        "path": "RPL39->RACK1->GRIN2B",
        "intermediates": ["RACK1"],
    },
    {
        "ko": "RPS19",
        "channel": "KCNA3",
        "path": "RPS19->KCNA3",
        "intermediates": [],
    },
    {
        "ko": "RPS2",
        "channel": "GRIN2B",
        "path": "RPS2->RACK1->GRIN2B",
        "intermediates": ["RACK1"],
    },
    {
        "ko": "RPS21",
        "channel": "KCNQ2",
        "path": "RPS21->KCNQ2",
        "intermediates": [],
    },
    {
        "ko": "SNRPD2",
        "channel": "GABRB3",
        "path": "SNRPD2->SNRPN->GABRB3",
        "intermediates": ["SNRPN"],
    },
    {
        "ko": "TRMT112",
        "channel": "CFTR",
        "path": "TRMT112->RPS27A->CFTR",
        "intermediates": ["RPS27A"],
    },
]

N_TOP_GENES = 2000
MIN_CELLS = 10
MIN_EXPR_PCT = 1.0
K_NEIGHBORS = 15
LATENT_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 300
LR = 0.01
BETA = 1e-4  # beta-VAE weight for KL term (from GenKI)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_all_forced_genes():
    forced = set()
    for bp in BRIDGE_PATHS:
        forced.add(bp["ko"])
        forced.add(bp["channel"])
        forced.update(bp["intermediates"])
    return forced


# ============================================================
# VGAE MODEL
# ============================================================
class VGAEEncoder(nn.Module):
    """Encoder following GenKI: outputs mu and logstd (not logvar)."""

    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)  # logstd, not logvar

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class VGAE(nn.Module):
    """VGAE model following official GenKI implementation.

    Key difference from original: stores mu/logstd and uses deterministic
    inference in eval mode (no sampling) to ensure comparable latent spaces.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.__mu__ = None
        self.__logstd__ = None

    def encode(self, x, edge_index):
        """Encode and store mu/logstd. Returns z (sampled in train, mu in eval)."""
        self.__mu__, self.__logstd__ = self.encoder(x, edge_index)
        self.__logstd__ = self.__logstd__.clamp(max=10)  # MAX_LOGSTD from GenKI
        if self.training:
            z = self.__mu__ + torch.randn_like(self.__logstd__) * torch.exp(
                self.__logstd__
            )
        else:
            z = self.__mu__  # Deterministic in eval mode - CRITICAL for KO comparison
        return z

    def forward(self, x, edge_index):
        """Forward pass for training."""
        z = self.encode(x, edge_index)
        adj_pred = torch.sigmoid(z @ z.t())
        return adj_pred, self.__mu__, self.__logstd__, z


def vgae_loss(adj_pred, edge_index, n_nodes, mu, logstd, beta=1e-4):
    """VGAE loss with beta-VAE weighting (following GenKI)."""
    pos_src, pos_dst = edge_index
    pos_loss = -torch.log(adj_pred[pos_src, pos_dst] + 1e-15).mean()
    neg_ei = negative_sampling(
        edge_index, num_nodes=n_nodes, num_neg_samples=edge_index.size(1)
    )
    neg_loss = -torch.log(1 - adj_pred[neg_ei[0], neg_ei[1]] + 1e-15).mean()
    # KL loss using logstd (not logvar) - matches GenKI
    kl = -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1)
    )
    return pos_loss + neg_loss + beta * kl


# ============================================================
# DATA LOADING & GRN
# ============================================================
def convert_gsm5224587():
    """Convert GSM5224587 raw annotation GZ into scTenifoldKnk-compatible CSV.
    Resolves ENSEMBL IDs to gene symbols using the Gene_name column (if present)
    and mart_export.csv as fallback. Duplicate symbols are summed.
    Skips conversion if the output file already exists.
    """
    output_path = DATASETS["GSM5224587"]
    if os.path.exists(output_path):
        log(f"GSM5224587 count matrix already exists, skipping conversion")
        return
    log(f"Converting GSM5224587 annotation GZ -> scTenifoldKnk CSV...")
    df = pd.read_csv(GSM5224587_RAW_GZ, index_col=0)
    log(f"  Raw shape: {df.shape[0]} rows x {df.shape[1]} cols")
    # Extract Gene_name column (last col) as primary symbol source
    if "Gene_name" in df.columns:
        gene_name_col = df["Gene_name"].copy()
        df = df.drop(columns=["Gene_name"])
    else:
        gene_name_col = pd.Series([np.nan] * len(df), index=df.index)
    # Build ENSEMBL -> symbol mapping from mart_export.csv
    mart = pd.read_csv(MART_EXPORT_CSV)
    mart.columns = ["ensembl_id", "gene_symbol"]
    mart = mart.dropna(subset=["gene_symbol"])
    mart = mart[mart["gene_symbol"].str.strip() != ""]
    mart_map = mart.drop_duplicates(subset="ensembl_id", keep="first")
    ensembl_to_symbol = dict(zip(mart_map["ensembl_id"], mart_map["gene_symbol"]))
    # Resolve: Gene_name col > mart_export > keep ENSEMBL as-is
    resolved = []
    for eid, gname in zip(df.index, gene_name_col):
        if pd.notna(gname) and str(gname).strip() != "":
            resolved.append(str(gname).strip())
        elif eid in ensembl_to_symbol:
            resolved.append(ensembl_to_symbol[eid])
        else:
            resolved.append(str(eid))
    df.index = resolved
    # Merge duplicate symbols by summing counts
    if df.index.nunique() < len(df):
        n_dup = len(df) - df.index.nunique()
        log(f"  Merging {n_dup} duplicate gene symbols by summing counts")
        df = df.groupby(df.index).sum()
    # Remove zero-count genes
    nonzero = df.sum(axis=1) > 0
    n_zero = (~nonzero).sum()
    if n_zero > 0:
        df = df[nonzero]
        log(f"  Removed {n_zero} zero-count genes")
    df.to_csv(output_path)
    log(f"  Saved {df.shape[0]} genes x {df.shape[1]} cells -> {os.path.basename(output_path)}")


def load_and_preprocess(csv_path, forced_genes):
    log(f"Loading {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path, index_col=0)
    ncells = (df > 0).sum(axis=1)
    df = df[(ncells >= MIN_CELLS) | df.index.isin(forced_genes)]
    forced_present = forced_genes & set(df.index)
    counts = df.values.astype(np.float32)
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    norm = np.log1p(counts / lib * np.median(lib))
    gene_names = df.index.tolist()
    selected = set(np.argsort(-norm.var(axis=1))[:N_TOP_GENES].tolist())
    for g in forced_present:
        selected.add(gene_names.index(g))
    selected = sorted(selected)
    norm = norm[selected, :]
    gene_names = [gene_names[i] for i in selected]
    expr_pct = {
        g: (norm[i] > 0).sum() / norm.shape[1] * 100 for i, g in enumerate(gene_names)
    }
    log(
        f"  {len(gene_names)}g x {df.shape[1]}c, {len(forced_present)}/{len(forced_genes)} forced present"
    )
    return norm, gene_names, expr_pct, df.shape[1]


def build_grn(expr, k=15):
    n = expr.shape[0]
    c = expr - expr.mean(axis=1, keepdims=True)
    nrm = np.sqrt((c**2).sum(axis=1, keepdims=True))
    nrm[nrm == 0] = 1
    c /= nrm
    corr = np.abs(c @ c.T)
    np.fill_diagonal(corr, 0)
    edges = set()
    for i in range(n):
        for j in np.argsort(-corr[i])[:k]:
            if corr[i, j] > 0:
                edges.add((min(i, j), max(i, j)))
    src, dst = [], []
    for s, d in edges:
        src.extend([s, d])
        dst.extend([d, s])
    return torch.tensor([src, dst], dtype=torch.long), len(edges)


# ============================================================
# VGAE TRAINING & INFERENCE (GenKI-aligned)
# ============================================================
def train_vgae(expr, edge_index, label="", epochs=EPOCHS, lr=LR, seed=8096):
    """Train VGAE on graph and return the trained model (frozen).

    Following GenKI: train ONCE, then use frozen model for all KO inference.
    This prevents latent space misalignment from random weight initialization.
    """
    n = expr.shape[0]
    x = torch.tensor(expr, dtype=torch.float32).to(DEVICE)
    ei = edge_index.to(DEVICE)

    if seed is not None:
        torch.manual_seed(seed)

    model = VGAE(VGAEEncoder(x.shape[1], HIDDEN_DIM, LATENT_DIM)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=9e-4)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        ap, mu, logstd, z = model(x, ei)
        loss = vgae_loss(ap, ei, n, mu, logstd)
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0 or epoch == 0:
            log(f"    {label} epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

    model.eval()  # CRITICAL: switch to eval mode for deterministic inference

    # Save trained weights
    weights_dir = os.path.join(OUTPUT_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    safe_label = label.replace(" ", "_")
    weights_path = os.path.join(weights_dir, f"{safe_label}_weights.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim": x.shape[1],
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "label": label,
    }, weights_path)
    log(f"    {label} training complete, model frozen, weights saved to {weights_path}")

    return model, x, ei


def get_latent_vars(model, x, edge_index):
    """Get latent variables (mu, variance) from trained model.

    Following GenKI: model.eval() ensures deterministic output (no sampling).
    Returns variance (not logstd) for KL computation.
    """
    model.eval()
    with torch.no_grad():
        _ = model.encode(x, edge_index)
        mu = model.__mu__.cpu().numpy()
        # Return variance (sigma^2), not logstd - matches GenKI utils.get_distance
        variance = (model.__logstd__.exp() ** 2).cpu().numpy()
    return mu, variance


def create_ko_data(x_tensor, edge_index, gene_names, ko_gene, device):
    """Create KO data by removing edges AND zeroing expression (following GenKI).

    GenKI _KO_data_init does TWO things:
    1. Remove edges connected to target gene
    2. Set target gene's expression to zero

    Returns: ko_x (with zeroed expression), ko_edge_index, n_removed_edges
    """
    ko_idx = gene_names.index(ko_gene)

    # 1. Remove edges connected to KO gene
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    mask = (src != ko_idx) & (dst != ko_idx)
    ko_ei = torch.tensor([src[mask], dst[mask]], dtype=torch.long).to(device)
    n_removed = (~mask).sum() // 2

    # 2. Zero out the KO gene's expression (CRITICAL - from GenKI)
    ko_x = x_tensor.clone()
    ko_x[ko_idx, :] = 0  # Set all expression values for KO gene to 0

    return ko_x, ko_ei, n_removed


def compute_kl(mu_wt, var_wt, mu_ko, var_ko):
    """Compute KL divergence between KO and WT latent distributions.

    Following GenKI utils.get_distance exactly.
    For diagonal covariance, KL simplifies significantly.

    Input: mu and variance (sigma^2) arrays of shape (n_genes, latent_dim).
    """
    # Ensure numerical stability
    var_wt = np.maximum(var_wt, 1e-10)
    var_ko = np.maximum(var_ko, 1e-10)

    # KL divergence for diagonal Gaussians (vectorized)
    # KL(KO || WT) = 0.5 * sum_d [ var_ko/var_wt + (mu_wt - mu_ko)^2/var_wt - 1 + log(var_wt/var_ko) ]
    kl = 0.5 * np.sum(
        var_ko / var_wt + (mu_wt - mu_ko) ** 2 / var_wt - 1 + np.log(var_wt / var_ko),
        axis=1,
    )
    return np.maximum(kl, 0)  # Ensure non-negative


# ============================================================
# STATISTICS: GenKI-style Significance Testing
# ============================================================
# Following official GenKI implementation (yjgeno/GenKI):
# - NO chi-square test, NO empirical p-values
# - Bagging approach: gene is significant if in top 5% for >=95% of permutations
# - Final ranking by observed KL divergence magnitude
N_PERMUTATIONS = 100  # Number of permutations for bagging
BAGGING_TOP_PCT = 0.05  # Top 5% genes considered "hits" per permutation
BAGGING_CUTOFF = 0.95  # Gene must be in top 5% for 95% of permutations


def genki_significance_test(
    model,
    x_wt,
    edge_index_wt,
    x_ko,
    edge_index_ko,
    wt_mu,
    wt_var,
    n_perm=N_PERMUTATIONS,
):
    """GenKI-style significance testing via bagging.

    Following official GenKI implementation exactly:
    1. Compute observed KL divergence between WT and KO latent distributions
    2. Bootstrap cell labels N times, recompute KL for each permutation
    3. For each permutation, identify genes in top 5% by KL
    4. A gene is "significant" if it appears in top 5% for >=95% of permutations
    5. Rank significant genes by observed KL magnitude

    This tests: "Is this gene's KL divergence consistently high across
    different cell samplings?" - a robustness/stability criterion.

    Returns:
        observed_kl: Original KL divergence values (n_genes,)
        hits: Number of times each gene was in top 5% across permutations (n_genes,)
        is_significant: Boolean mask for genes passing bagging threshold (n_genes,)
    """
    # Get observed KL between WT and KO
    ko_mu, ko_var = get_latent_vars(model, x_ko, edge_index_ko)
    observed_kl = compute_kl(wt_mu, wt_var, ko_mu, ko_var)

    n_genes = len(observed_kl)
    n_cells = x_wt.shape[1]

    # Bagging via cell bootstrap (following GenKI pmt() exactly)
    np.random.seed(0)  # GenKI uses seed=0
    top_k = max(1, int(n_genes * BAGGING_TOP_PCT))  # Top 5%
    hits = np.zeros(n_genes, dtype=int)

    for i in range(n_perm):
        # Bootstrap cell labels (sample with replacement) - exactly as GenKI
        idx_pmt = np.random.choice(n_cells, size=n_cells, replace=True)

        # Create permuted data tensors
        x_wt_perm = x_wt[:, idx_pmt].clone().detach()
        x_ko_perm = x_ko[:, idx_pmt].clone().detach()

        # Get latent vars for permuted data
        wt_mu_p, wt_var_p = get_latent_vars(model, x_wt_perm, edge_index_wt)
        ko_mu_p, ko_var_p = get_latent_vars(model, x_ko_perm, edge_index_ko)

        # Compute KL for this permutation
        perm_kl = compute_kl(wt_mu_p, wt_var_p, ko_mu_p, ko_var_p)

        # Bagging: count genes in top 5% (following GenKI get_generank)
        top_genes_idx = np.argsort(perm_kl)[-top_k:]
        hits[top_genes_idx] += 1

        if (i + 1) % 20 == 0:
            log(f"      Permutation {i + 1}/{n_perm}")

    # Significance: gene must be in top 5% for >= 95% of permutations
    hit_threshold = int(n_perm * BAGGING_CUTOFF)
    is_significant = (hits >= hit_threshold) & (observed_kl > 0)

    return observed_kl, hits, is_significant


# ============================================================
# ROBUST STATISTICS: MAD-based Z-scores for heavy-tailed data
# ============================================================


def compute_robust_statistics(kl_values):
    """Compute robust Z-scores and p-values using MAD (Median Absolute Deviation).

    Rationale: KL divergence distributions exhibit extreme outliers (max/median > 10^8),
    which violate chi-square assumptions and inflate standard deviation estimates.
    MAD-based estimation is robust to up to 50% contamination (breakdown point = 50%).

    The 1.4826 scaling factor makes MAD consistent with standard deviation under normality:
        MAD × 1.4826 ≈ σ (for Gaussian data)

    Note: Z-scores can be very large (millions) for extreme outliers. This is mathematically
    correct - it reflects how many "noise units" the value is from the median. The p-values
    and FDR are computed from raw Z-scores and remain valid for significance testing.

    Args:
        kl_values: Array of KL divergence values

    Returns:
        z_robust: Robust Z-scores (uncapped, can be very large for outliers)
        p_values: One-tailed p-values (testing for high KL)
        fdr: Benjamini-Hochberg FDR-adjusted p-values
    """
    kl = np.asarray(kl_values, dtype=np.float64)

    # Robust location (median) and scale (MAD)
    median_kl = np.median(kl)
    mad = np.median(np.abs(kl - median_kl))

    # Prevent division by zero (if >50% of values are identical)
    robust_sigma = max(1.4826 * mad, 1e-10)

    # Robust Z-score (uncapped)
    z_robust = (kl - median_kl) / robust_sigma

    # Convert to one-tailed p-values (we only care about high KL)
    # Using survival function (1 - CDF) for numerical stability
    p_values = norm.sf(z_robust)

    # Ensure minimum p-value for numerical stability
    p_values = np.clip(p_values, 1e-300, 1.0)

    # Benjamini-Hochberg FDR correction
    fdr = benjamini_hochberg(p_values)

    return z_robust, p_values, fdr


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg FDR correction.

    Args:
        pvalues: Array of p-values

    Returns:
        Adjusted p-values (FDR)
    """
    pv = np.asarray(pvalues, dtype=np.float64)
    n = len(pv)

    # Sort p-values
    sorted_idx = np.argsort(pv)
    sorted_pv = pv[sorted_idx]

    # Compute adjusted p-values
    adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        # BH formula: p_adj = min(p * n / rank, previous_p_adj)
        cummin = min(cummin, sorted_pv[i] * n / (i + 1))
        adjusted[sorted_idx[i]] = min(cummin, 1.0)

    return adjusted


# ============================================================
# PHASE 1: RUN ALL KOs (GenKI-aligned: train once, infer many)
# ============================================================
def run_all_kos(force=False):
    """Run all KO experiments following GenKI methodology.

    CRITICAL FIX: Train VGAE once per dataset on WT graph, then use the
    FROZEN model to infer latent vars for both WT and KO graphs.
    This ensures comparable latent spaces (no random init misalignment).

    Now includes GenKI-style permutation testing for robust significance.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Auto-convert GSM5224587 if needed
    convert_gsm5224587()
    forced = get_all_forced_genes()
    unique_kos = sorted(set(bp["ko"] for bp in BRIDGE_PATHS))

    # Load datasets
    ds_data = {}
    for name, path in DATASETS.items():
        expr, genes, expr_pct, nc = load_and_preprocess(path, forced)
        ei, ne = build_grn(expr, K_NEIGHBORS)
        log(f"  {name}: GRN {len(genes)} nodes, {ne} edges")
        ds_data[name] = {
            "expr": expr,
            "genes": genes,
            "expr_pct": expr_pct,
            "edge_index": ei,
            "n_cells": nc,
        }

    # Cache for trained models and WT latent vars per dataset
    model_cache = {}  # {ds_name: (model, x_tensor, wt_edge_index)}
    wt_latent_cache = {}  # {ds_name: (wt_mu, wt_var)}

    for ds_name, ds in ds_data.items():
        # Check if any KO needs to be run for this dataset
        kos_to_run = []
        for ko in unique_kos:
            out_path = os.path.join(OUTPUT_DIR, f"VGAE_KO_{ko}_{ds_name}.csv")
            if os.path.exists(out_path) and not force:
                continue
            pct = ds["expr_pct"].get(ko, 0)
            if pct < MIN_EXPR_PCT or ko not in ds["genes"]:
                log(f"  Skipping {ko} on {ds_name}: not in gene list or low expression")
                continue
            kos_to_run.append(ko)

        if not kos_to_run:
            log(f"  No KOs to run for {ds_name}, skipping training")
            continue

        # Train WT model ONCE for this dataset
        log(f"\n{'=' * 50}")
        log(
            f"Training WT VGAE for {ds_name} (will be used for {len(kos_to_run)} KOs)..."
        )
        model, x_tensor, wt_ei = train_vgae(
            ds["expr"], ds["edge_index"], f"WT-{ds_name}"
        )
        model_cache[ds_name] = (model, x_tensor, wt_ei)

        # Get WT latent vars using frozen model
        wt_mu, wt_var = get_latent_vars(model, x_tensor, wt_ei)
        wt_latent_cache[ds_name] = (wt_mu, wt_var)
        log(
            f"  WT latent vars cached: mu shape={wt_mu.shape}, var shape={wt_var.shape}"
        )

        # Run each KO using the FROZEN WT model
        for ko in kos_to_run:
            out_path = os.path.join(OUTPUT_DIR, f"VGAE_KO_{ko}_{ds_name}.csv")
            pct = ds["expr_pct"].get(ko, 0)

            log(f"\n  KO {ko} on {ds_name} ({ds['n_cells']}c, expr={pct:.1f}%)")

            # Create KO data: remove edges AND zero expression (following GenKI)
            ko_x, ko_ei, n_removed = create_ko_data(
                x_tensor, ds["edge_index"], ds["genes"], ko, DEVICE
            )
            log(f"    Removed {n_removed} edges for {ko}, zeroed expression")

            # Run GenKI-style significance test (bagging, no p-values)
            log(
                f"    Running GenKI significance test ({N_PERMUTATIONS} permutations)..."
            )
            observed_kl, hits, is_significant = genki_significance_test(
                model, x_tensor, wt_ei, ko_x, ko_ei, wt_mu, wt_var, N_PERMUTATIONS
            )

            # Compute robust statistics (MAD-based Z-scores, p-values, FDR)
            z_robust, p_values, fdr = compute_robust_statistics(observed_kl)

            # Combined significance: bagging OR robust FDR < 0.05
            is_sig_combined = is_significant | (fdr < 0.05)

            # Save results with both GenKI bagging and robust statistics
            res = pd.DataFrame(
                {
                    "gene": ds["genes"],
                    "KL": observed_kl,
                    "hits": hits,
                    "Z_robust": z_robust,
                    "pvalue": p_values,
                    "FDR": fdr,
                    "sig_bagging": is_significant.astype(int),
                    "sig_robust": (fdr < 0.05).astype(int),
                    "significant": is_sig_combined.astype(int),
                }
            )
            res = res.sort_values("KL", ascending=False).reset_index(drop=True)
            res["Rank"] = range(1, len(res) + 1)
            res.to_csv(out_path, index=False)

            # Report top hits
            n_sig_bagging = is_significant.sum()
            n_sig_robust = (fdr < 0.05).sum()
            n_sig_combined = is_sig_combined.sum()
            log(f"    Saved {os.path.basename(out_path)}")
            log(
                f"    Significant: {n_sig_bagging} (bagging), {n_sig_robust} (FDR<0.05), {n_sig_combined} (combined)"
            )

    log("\n" + "=" * 50)
    log("Starting Negative Control Analysis (Permutation-based)...")
    # ── Permutation-Based Negative Control Analysis ──
    # Replaces the original "bottom 10%" control with a permutation approach.
    # For each dataset, randomly select N_CTRL non-hub/non-channel genes from
    # the SAME gene set, KO each using the SAME frozen model, and compare
    # the target channel's KL divergence against hub KO results.
    # This controls for the KO procedure itself and avoids the expression-
    # matching problem (hub genes are high-expression housekeeping genes while
    # HVGs are predominantly sparse/variable).
    #
    # Statistical test: one-tailed Mann-Whitney U (hub KL > control KL distribution)
    # Also report empirical percentile of hub KL within the control distribution.
    from scipy.stats import mannwhitneyu

    N_CTRL = 50  # number of random control KOs per dataset

    nc_rows = []
    for ds_name, ds in ds_data.items():
        # If model not in cache, reload from saved weights
        if ds_name not in model_cache:
            weights_path = os.path.join(OUTPUT_DIR, "weights", f"WT-{ds_name}_weights.pt")
            if not os.path.exists(weights_path):
                log(f"  NegCtrl: skipping {ds_name} (no cached model and no saved weights)")
                continue
            log(f"  NegCtrl: loading saved weights for {ds_name}...")
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            n_cells = ds["expr"].shape[1]
            encoder = VGAEEncoder(n_cells, checkpoint["hidden_dim"], checkpoint["latent_dim"])
            model = VGAE(encoder).to(DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            x_tensor = torch.tensor(ds["expr"], dtype=torch.float32).to(DEVICE)
            wt_ei = ds["edge_index"].to(DEVICE)
            wt_mu, wt_var = get_latent_vars(model, x_tensor, wt_ei)
            model_cache[ds_name] = (model, x_tensor, wt_ei)
            wt_latent_cache[ds_name] = (wt_mu, wt_var)
            log(f"  NegCtrl: {ds_name} model restored from weights")
        else:
            model, x_tensor, wt_ei = model_cache[ds_name]
            wt_mu, wt_var = wt_latent_cache[ds_name]

        genes = ds["genes"]
        expr_pct = ds["expr_pct"]

        # Build reserved set
        reserved = set()
        for bp in BRIDGE_PATHS:
            reserved.add(bp["ko"])
            reserved.add(bp["channel"])
            reserved.update(bp["intermediates"])

        gene_to_idx = {g: i for i, g in enumerate(genes)}

        # Eligible control genes: non-reserved, expressed above threshold
        eligible_ctrls = [g for g in genes
                          if g not in reserved and expr_pct.get(g, 0) >= MIN_EXPR_PCT]

        if len(eligible_ctrls) < N_CTRL:
            log(f"  NegCtrl: only {len(eligible_ctrls)} eligible controls in {ds_name}, using all")

        # Select random controls (deterministic seed per dataset)
        # Priority: load seed from saved JSON > compute from hash
        import json
        seed_path = os.path.join(OUTPUT_DIR, "weights", f"negctrl_seed_{ds_name}.json")
        if os.path.exists(seed_path):
            with open(seed_path, "r", encoding="utf-8") as sf:
                seed_info = json.load(sf)
            ctrl_seed = seed_info["seed"]
            log(f"  NegCtrl: loaded seed={ctrl_seed} from {seed_path}")
        else:
            ctrl_seed = hash(ds_name) % (2**31)
            log(f"  NegCtrl: computed seed={ctrl_seed} (no saved JSON found)")
        np.random.seed(ctrl_seed)
        ctrl_genes = list(np.random.choice(
            eligible_ctrls, size=min(N_CTRL, len(eligible_ctrls)), replace=False
        ))

        # Save seed and control gene list for reproducibility
        seed_info = {
            "dataset": ds_name,
            "seed": ctrl_seed,
            "seed_source": f'hash("{ds_name}") % 2^31',
            "n_controls": len(ctrl_genes),
            "n_eligible": len(eligible_ctrls),
            "control_genes": sorted(ctrl_genes),
        }
        with open(seed_path, "w", encoding="utf-8") as sf:
            json.dump(seed_info, sf, indent=2, ensure_ascii=False)
        log(f"  NegCtrl: seed info saved to {seed_path}")

        # Pre-compute KL for all control KOs (reuse across hub-channel pairs)
        log(f"  NegCtrl: running {len(ctrl_genes)} control KOs for {ds_name}...")
        ctrl_kl_matrix = {}  # {ctrl_gene: kl_array (n_genes,)}
        for ci, ctrl_gene in enumerate(ctrl_genes):
            ctrl_x, ctrl_ei, _ = create_ko_data(
                x_tensor, wt_ei, genes, ctrl_gene, DEVICE
            )
            ctrl_mu, ctrl_var = get_latent_vars(model, ctrl_x, ctrl_ei)
            ctrl_kl_matrix[ctrl_gene] = compute_kl(wt_mu, wt_var, ctrl_mu, ctrl_var)
            if (ci + 1) % 10 == 0:
                log(f"    Control KO {ci + 1}/{len(ctrl_genes)}")

        for bp in BRIDGE_PATHS:
            ko_gene = bp["ko"]
            channel = bp["channel"]
            if ko_gene not in gene_to_idx or channel not in gene_to_idx:
                continue

            ch_idx = gene_to_idx[channel]

            # Read hub KO result for this channel's KL
            hub_csv = os.path.join(OUTPUT_DIR, f"VGAE_KO_{ko_gene}_{ds_name}.csv")
            if not os.path.exists(hub_csv):
                continue
            hub_df = pd.read_csv(hub_csv)
            hub_ch_row = hub_df[hub_df["gene"] == channel]
            if hub_ch_row.empty:
                continue
            hub_kl = hub_ch_row["KL"].values[0]

            # Collect control KLs for this target channel
            ctrl_kls = np.array([ctrl_kl_matrix[cg][ch_idx] for cg in ctrl_genes])
            median_ctrl = np.median(ctrl_kls)
            fold_change = hub_kl / median_ctrl if median_ctrl > 0 else float("inf")

            # Empirical percentile: fraction of controls with KL < hub_kl
            empirical_pctl = (ctrl_kls < hub_kl).sum() / len(ctrl_kls) * 100

            # Mann-Whitney U test
            if len(ctrl_kls) >= 3:
                _, pval = mannwhitneyu([hub_kl], ctrl_kls, alternative="greater")
            else:
                pval = float("nan")

            nc_rows.append({
                "dataset": ds_name,
                "ko_gene": ko_gene,
                "target_channel": channel,
                "hub_KL": hub_kl,
                "median_ctrl_KL": median_ctrl,
                "fold_change": fold_change,
                "pvalue": pval,
                "empirical_percentile": empirical_pctl,
                "n_controls": len(ctrl_genes),
                "seed": ctrl_seed,
                "control_genes": ";".join(sorted(ctrl_genes)),
                "ctrl_KLs": ";".join(f"{v:.6e}" for v in ctrl_kls),
            })
            log(f"  NegCtrl: {ko_gene}->{channel} [{ds_name}] "
                f"hub_KL={hub_kl:.2e}, ctrl_median={median_ctrl:.2e}, "
                f"FC={fold_change:.1f}x, p={pval:.2e}, pctl={empirical_pctl:.0f}%")

    if nc_rows:
        nc_df = pd.DataFrame(nc_rows)
        nc_path = os.path.join(OUTPUT_DIR, "VGAE_KO_NegativeControl_Analysis.csv")
        nc_df.to_csv(nc_path, index=False)
        log(f"  NegCtrl analysis saved: {nc_path}")
    else:
        log("  NegCtrl: no results generated")


# ============================================================
# PHASE 2: REPORT (Hybrid: GenKI bagging + Robust statistics)
# ============================================================
def generate_report():
    """Generate validation report using hybrid methodology.

    Significance criteria (combined):
    1. GenKI bagging: gene in top 5% for >= 95% of permutations (stability)
    2. Robust statistics: FDR < 0.05 using MAD-based Z-scores (magnitude)

    A gene is validated if EITHER criterion is met.
    """
    UNTESTABLE = set()  # pipeline handles low-expr skipping automatically
    CHANNELS = [
        "KCNA5",
        "PKD2",
        "KCNN3",
        "GRIK2",
        "AQP9",
        "RYR3",
        "GRIN2A",
        "ANO4",
        "CLIC1",
        "GRIN1",
        "TRPC3",
        "CLIC2",
        "CFTR",
        "KCNA3",
        "GRIN2B",
        "KCNQ2",
        "GABRB3",
    ]  # 16 distinct ion channels from full STRING v12.0
    HIT_THRESHOLD = int(N_PERMUTATIONS * BAGGING_CUTOFF)

    # Build rows: one per (path x dataset)
    # When the endpoint channel (e.g. KCNA5) is not expressed in the dataset,
    # we fall back to checking intermediate genes in the path. If KO of gene A
    # significantly perturbs intermediate B (which connects to the channel via
    # STRING PPI), that validates the predicted path.
    rows = []
    for bp in BRIDGE_PATHS:
        ko, ch = bp["ko"], bp["channel"]
        intermediates = bp.get("intermediates", [])
        for ds in DATASETS:
            row = {
                "dataset": ds,
                "ko_gene": ko,
                "target_channel": ch,
                "path": bp["path"],
            }
            nans = {
                "KL": np.nan,
                "Z_robust": np.nan,
                "pvalue": np.nan,
                "FDR": np.nan,
                "hits": np.nan,
                "rank": np.nan,
                "total_genes": np.nan,
                "n_sig": np.nan,
            }

            if ko in UNTESTABLE:
                rows.append({**row, **nans, "status": "UNTESTABLE_LOW_KO_EXPR"})
                continue

            fpath = os.path.join(OUTPUT_DIR, f"VGAE_KO_{ko}_{ds}.csv")
            if not os.path.exists(fpath):
                rows.append({**row, **nans, "status": "KO_NOT_RUN"})
                continue

            df = pd.read_csv(fpath)

            # Count significant genes
            if "significant" in df.columns:
                n_sig = int(df["significant"].sum())
            elif "hits" in df.columns:
                n_sig = int((df["hits"] >= HIT_THRESHOLD).sum())
            else:
                n_sig = 0

            df = df.sort_values("KL", ascending=False).reset_index(drop=True)
            df["Rank"] = range(1, len(df) + 1)

            # Try channel first, then fall back to intermediates (closest to channel first)
            target_candidates = [ch] + list(reversed(intermediates))
            target_row = None
            validated_gene = None
            for candidate in target_candidates:
                cand_row = df[df["gene"] == candidate]
                if len(cand_row) > 0:
                    target_row = cand_row.iloc[0]
                    validated_gene = candidate
                    break

            if target_row is None:
                rows.append(
                    {
                        **row,
                        **nans,
                        "total_genes": len(df),
                        "n_sig": n_sig,
                        "status": "NO_PATH_GENE_IN_GENESET",
                    }
                )
                continue

            r = target_row

            # Check significance by both methods
            is_sig_bagging = False
            is_sig_robust = False

            if "sig_bagging" in r:
                is_sig_bagging = bool(r["sig_bagging"])
            elif "hits" in r and not np.isnan(r["hits"]):
                is_sig_bagging = r["hits"] >= HIT_THRESHOLD

            if "FDR" in r and not np.isnan(r["FDR"]):
                is_sig_robust = r["FDR"] < 0.05

            # Combined: either method
            is_significant = is_sig_bagging or is_sig_robust

            # Note whether we used channel or intermediate
            via_note = "" if validated_gene == ch else f" (via {validated_gene})"
            status = "VALIDATED" if is_significant else "NOT_SIGNIFICANT"

            rows.append(
                {
                    **row,
                    "KL": r["KL"],
                    "Z_robust": r["Z_robust"] if "Z_robust" in r else np.nan,
                    "pvalue": r["pvalue"] if "pvalue" in r else np.nan,
                    "FDR": r["FDR"] if "FDR" in r else np.nan,
                    "hits": int(r["hits"])
                    if "hits" in r and not np.isnan(r["hits"])
                    else 0,
                    "rank": int(r["Rank"]),
                    "total_genes": len(df),
                    "n_sig": n_sig,
                    "sig_bagging": int(is_sig_bagging),
                    "sig_robust": int(is_sig_robust),
                    "validated_gene": validated_gene,
                    "status": status,
                }
            )

    report = pd.DataFrame(rows)
    report_path = os.path.join(OUTPUT_DIR, "VGAE_KO_Report.csv")
    try:
        report.to_csv(report_path, index=False)
    except PermissionError:
        # Try alternative filename if file is locked
        report_path = os.path.join(OUTPUT_DIR, f"VGAE_KO_Report_{int(time.time())}.csv")
        report.to_csv(report_path, index=False)

    # ── Print ──
    print(f"\n{'=' * 100}")
    print(
        f"  VGAE-KO VALIDATION REPORT (Hybrid: GenKI Bagging + Robust MAD-based Statistics)"
    )
    print(
        f"  Significance: hits >= {HIT_THRESHOLD}/{N_PERMUTATIONS} (bagging) OR FDR < 0.05 (robust)"
    )
    print(f"{'=' * 100}")

    for ds in DATASETS:
        ds_df = report[report["dataset"] == ds]
        part = "I" if ds == "HCT116" else "II"
        print(f"\n{'#' * 100}")
        print(f"  PART {part}: {ds}")
        print(f"{'#' * 100}")

        for ch in CHANNELS:
            ch_df = ds_df[ds_df["target_channel"] == ch]
            if len(ch_df) == 0:
                continue
            any_val = any(ch_df["status"] == "VALIDATED")
            print(f"\n  [{'VALIDATED' if any_val else 'UNVALIDATED'}] {ch}")
            print(
                f"  {'KO':<10} {'Path':<45} {'KL':>10} {'Z_rob':>8} {'FDR':>10} {'hits':>5} {'Rank':>10} Status"
            )
            print(
                f"  {'-' * 10} {'-' * 45} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 5} {'-' * 10} {'-' * 25}"
            )
            for _, r in ch_df.iterrows():
                if r["status"] in ("VALIDATED", "NOT_SIGNIFICANT"):
                    hits_str = (
                        f"{int(r['hits']):>5}" if not np.isnan(r["hits"]) else "    -"
                    )
                    z_str = (
                        f"{r['Z_robust']:8.2f}"
                        if not np.isnan(r["Z_robust"])
                        else "       -"
                    )
                    fdr_str = (
                        f"{r['FDR']:10.2e}" if not np.isnan(r["FDR"]) else "         -"
                    )

                    # Show which criterion was met
                    sig_marker = ""
                    if r["status"] == "VALIDATED":
                        if r.get("sig_bagging", 0) and r.get("sig_robust", 0):
                            sig_marker = " [B+R]"
                        elif r.get("sig_bagging", 0):
                            sig_marker = " [B]"
                        elif r.get("sig_robust", 0):
                            sig_marker = " [R]"

                    # Show validated gene if different from channel
                    via_note = ""
                    vg = r.get("validated_gene", ch)
                    if vg and vg != ch:
                        via_note = f" (via {vg})"

                    print(
                        f"  {r['ko_gene']:<10} {r['path']:<45} {r['KL']:10.2e} {z_str} {fdr_str} {hits_str} "
                        f"{int(r['rank']):>4}/{int(r['total_genes']):<4} {r['status']}{sig_marker}{via_note}"
                    )
                else:
                    print(
                        f"  {r['ko_gene']:<10} {r['path']:<45} {'':>10} {'':>8} {'':>10} {'':>5} {'':>10} {r['status']}"
                    )

        tested = ds_df[ds_df["status"].isin(["VALIDATED", "NOT_SIGNIFICANT"])]
        n_val = len(set(tested.loc[tested["status"] == "VALIDATED", "target_channel"]))
        print(f"\n  Summary: {n_val}/{len(CHANNELS)} channels validated in {ds}")
        print(f"  Legend: [B]=Bagging, [R]=Robust FDR, [B+R]=Both")

    # ── Overall verdict ──
    print(f"\n{'=' * 100}")
    print(f"  OVERALL CHANNEL VERDICT (best across both datasets)")
    print(f"{'=' * 100}")
    for ch in CHANNELS:
        ch_df = report[report["target_channel"] == ch]
        val = ch_df[ch_df["status"] == "VALIDATED"]
        if len(val) > 0:
            # Sort by KL (higher is better) to find best
            best = val.sort_values("KL", ascending=False).iloc[0]
            fdr_str = f", FDR={best['FDR']:.2e}" if not np.isnan(best["FDR"]) else ""
            sig_type = ""
            if best.get("sig_bagging", 0) and best.get("sig_robust", 0):
                sig_type = " [B+R]"
            elif best.get("sig_bagging", 0):
                sig_type = " [B]"
            elif best.get("sig_robust", 0):
                sig_type = " [R]"
            via_note = ""
            vg = best.get("validated_gene", ch)
            if vg and vg != ch:
                via_note = f" via {vg}"
            print(
                f"  VALIDATED   {ch:<8}  best: {best['path']} "
                f"(KL={best['KL']:.2e}, hits={int(best['hits'])}{fdr_str}){sig_type}{via_note} [{best['dataset']}]"
            )
        else:
            reasons = sorted(ch_df["status"].unique())
            print(f"  UNVALIDATED {ch:<8}  reasons: {', '.join(reasons)}")

    n_total = len(set(report.loc[report["status"] == "VALIDATED", "target_channel"]))
    print(f"\n  {n_total}/{len(CHANNELS)} channels validated")
    print(f"  Full report: {report_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGAE-KO Validation Pipeline")
    parser.add_argument(
        "--force", action="store_true", help="Rerun all KOs even if results exist"
    )
    parser.add_argument(
        "--report", action="store_true", help="Report only, skip VGAE runs"
    )
    args = parser.parse_args()

    log(f"Device: {DEVICE}, PyTorch: {torch.__version__}")

    if not args.report:
        run_all_kos(force=args.force)

    generate_report()
