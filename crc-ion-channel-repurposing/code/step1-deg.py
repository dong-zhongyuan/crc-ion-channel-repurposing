#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1 Generation Pipeline
Generated automatically - combines all parts
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1 Generation Script (16 Panels A-P)
Nature Style - ZERO-FAKE Policy
"""

import os
import sys
import logging
import hashlib
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# 尝试导入 UMAP
try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed, will use t-SNE instead")

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
# Use directory of this script as base for relative paths
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
_STEP_DIR = os.path.dirname(SKILL_DIR)  # step1_deg_analysis
PROJECT_ROOT = os.path.dirname(_STEP_DIR)  # PROJECT_ROOT
OUTPUT_DIR = f"{_STEP_DIR}/result"
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "data.csv")

PANELS_DIR = f"{OUTPUT_DIR}/panels16"
COMPOSITE_DIR = f"{OUTPUT_DIR}/composite"
SOURCEDATA_DIR = f"{OUTPUT_DIR}/sourcedata16"
CODE_DIR = f"{OUTPUT_DIR}/code"
RAW_DIR = f"{OUTPUT_DIR}/raw"
LOGS_DIR = f"{OUTPUT_DIR}/logs"
MANIFESTS_DIR = f"{OUTPUT_DIR}/manifests"

# GENCODE hg38 gene coordinates (ZERO-FAKE: real coordinates only)
# Note: User may need to provide this file or it will show placeholder for panel G
GENCODE_COORDS_FILE = os.path.join(PROJECT_ROOT, "data", "hg38_gene_symbol_lookup.tsv")

# KEY_TARGETS 将在 DEG 分析后动态选择
# 不再使用预设基因列表，而是根据 |Log2FC| 自动选择 Top DEGs
KEY_TARGETS = []  # 初始化为空，后续由 select_top_degs() 填充

# Top 基因选择参数
TOP_GENES_CONFIG = {
    "n_label": 10,  # Volcano图标注的Top基因数量
    "n_heatmap": 30,  # Heatmap显示的Top基因数量
    "n_lollipop": 15,  # Lollipop图显示的Top基因数量
    "n_violin": 6,  # Violin图显示的Top基因数量
    "n_effect": 20,  # Effect size图显示的Top基因数量
}

# DEG 阈值
FDR_THRESHOLD = 0.05
LOG2FC_THRESHOLD = np.log2(1.2)  # ~0.263

# 随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Nature 风格配色
NPG_COLORS = {
    "red": "#E64B35",
    "blue": "#3C5488",
    "green": "#00A087",
    "purple": "#8491B4",
    "orange": "#F39B7F",
    "cyan": "#4DBBD5",
    "gray": "#7E6148",
    "yellow": "#B09C85",
}

# 设置日志
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOGS_DIR}/run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Nature 风格设置
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


# ============================================================
# 辅助函数
# ============================================================
def add_panel_label(ax, label, x=-0.12, y=1.08):
    """添加面板标签 (A, B, C, ...)"""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def calculate_md5(filepath):
    """计算文件 MD5"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_sourcedata(df, panel_name, description=""):
    """保存源数据"""
    filepath = f"{SOURCEDATA_DIR}/SourceData_{panel_name}.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved: {filepath}")
    return filepath


# ============================================================
# Step 1: 加载数据
# ============================================================
def detect_data_preprocessing_status(df, logger):
    """
    Detect and report data preprocessing status.

    Checks:
    1. Whether data appears to be log2 transformed
    2. Whether data appears to be normalized
    3. Data range and distribution characteristics

    This information is critical for method selection:
    - If data is log2 normalized: Welch t-test is appropriate
    - If data is raw counts: Should use DESeq2/edgeR/limma-voom instead
    """
    logger.info("Detecting data preprocessing status...")

    # Get data statistics
    data_min = df.min().min()
    data_max = df.max().max()
    data_mean = df.mean().mean()
    data_median = df.median().median()

    # Check for log2 transformation
    # Log2 data typically has:
    # - Values in range 0-20 (for microarray) or 0-25 (for RNA-seq TPM/FPKM)
    # - No extremely large values (>1000)
    # - Relatively symmetric distribution

    is_likely_log2 = False
    is_likely_normalized = False
    data_type = "Unknown"

    if data_min >= 0 and data_max < 30:
        # Likely log2 transformed
        is_likely_log2 = True
        if data_max < 20:
            data_type = "Microarray (log2 normalized)"
        else:
            data_type = "RNA-seq (log2 TPM/FPKM)"
    elif data_min >= 0 and data_max > 1000:
        # Likely raw counts or TPM
        if data_max > 100000:
            data_type = "RNA-seq (raw counts) - WARNING: Consider DESeq2/edgeR"
        else:
            data_type = "RNA-seq (TPM/FPKM) - Consider log2 transformation"
    elif data_min < 0:
        # Negative values suggest already normalized/centered
        is_likely_normalized = True
        data_type = "Normalized (centered)"

    # Check for quantile normalization (similar distributions across samples)
    sample_medians = df.median()
    median_cv = (
        sample_medians.std() / sample_medians.mean()
        if sample_medians.mean() != 0
        else 0
    )

    if median_cv < 0.1:
        is_likely_normalized = True
        normalization_status = "Quantile/RMA normalized (low inter-sample variance)"
    elif median_cv < 0.3:
        normalization_status = "Partially normalized"
    else:
        normalization_status = "Not normalized (high inter-sample variance)"

    # Log findings
    logger.info(f"  Data range: [{data_min:.2f}, {data_max:.2f}]")
    logger.info(f"  Data mean: {data_mean:.2f}, median: {data_median:.2f}")
    logger.info(f"  Detected data type: {data_type}")
    logger.info(f"  Log2 transformed: {'Yes' if is_likely_log2 else 'No/Uncertain'}")
    logger.info(f"  Normalization status: {normalization_status}")

    # Return preprocessing info for documentation
    preprocessing_info = {
        "data_type": data_type,
        "is_log2": is_likely_log2,
        "is_normalized": is_likely_normalized,
        "normalization_status": normalization_status,
        "data_range": [float(data_min), float(data_max)],
        "data_mean": float(data_mean),
        "data_median": float(data_median),
        "sample_median_cv": float(median_cv),
        "statistical_method_appropriate": "Welch t-test"
        if is_likely_log2
        else "Consider DESeq2/edgeR for raw counts",
    }

    # Warning if data doesn't look log2 transformed
    if not is_likely_log2:
        logger.warning("  WARNING: Data may not be log2 transformed!")
        logger.warning("  Welch t-test assumes approximately normal distribution.")
        logger.warning("  For raw counts, consider using DESeq2, edgeR, or limma-voom.")

    return preprocessing_info


def load_data():
    """加载表达矩阵"""
    logger.info("Loading expression data...")

    df = pd.read_csv(DATA_FILE)
    logger.info(f"Raw data shape: {df.shape}")

    # 检查数据结构
    # 第一列是 sample/gene 名，第一行可能是标签行
    first_col = df.columns[0]

    # 检查第二行是否是标签行
    if df.iloc[0, 0] == "label":
        # 提取标签行
        labels = df.iloc[0, 1:].values
        # 移除标签行
        df = df.iloc[1:].reset_index(drop=True)
        logger.info("Detected label row, extracted sample groups")
    else:
        labels = None

    # 设置基因名为索引
    df = df.set_index(first_col)
    df.index.name = "Gene"

    # 转换为数值
    df = df.astype(float)

    logger.info(f"Expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")

    return df, labels


# ============================================================
# Step 2: 生成/验证 metadata
# ============================================================
def get_metadata(df, labels=None):
    """生成或加载样本分组 metadata"""
    metadata_file = f"{OUTPUT_DIR}/metadata.csv"

    if os.path.exists(metadata_file):
        logger.info(f"Loading existing metadata: {metadata_file}")
        metadata = pd.read_csv(metadata_file)
        return metadata

    # 尝试从列名或标签推断
    samples = df.columns.tolist()
    groups = []

    if labels is not None:
        # 使用标签行
        for label in labels:
            label_lower = str(label).lower()
            if "control" in label_lower:
                groups.append("Control")
            elif "case" in label_lower:
                groups.append("Case")
            else:
                groups.append(None)
    else:
        # 从列名推断
        for sample in samples:
            sample_lower = sample.lower()
            if "control" in sample_lower:
                groups.append("Control")
            elif "case" in sample_lower:
                groups.append("Case")
            else:
                groups.append(None)

    # 检查是否成功推断
    if None in groups:
        # 生成模板
        template = pd.DataFrame({"SampleID": samples, "Group": [""] * len(samples)})
        template_file = f"{OUTPUT_DIR}/metadata_TEMPLATE.csv"
        template.to_csv(template_file, index=False)
        logger.error(f"Cannot infer sample groups. Please fill in: {template_file}")
        raise ValueError("Sample groups cannot be inferred. Please provide metadata.")

    metadata = pd.DataFrame({"SampleID": samples, "Group": groups})
    metadata.to_csv(metadata_file, index=False)
    logger.info(f"Generated metadata: {metadata_file}")
    logger.info(
        f"Control: {sum(g == 'Control' for g in groups)}, Case: {sum(g == 'Case' for g in groups)}"
    )

    return metadata


# ============================================================
# Step 3: 差异表达分析
# ============================================================
def perform_deg_analysis(df, metadata):
    """执行差异表达分析 (Welch t-test + BH FDR)"""
    logger.info("Performing differential expression analysis...")

    young_samples = metadata[metadata["Group"] == "Control"]["SampleID"].tolist()
    old_samples = metadata[metadata["Group"] == "Case"]["SampleID"].tolist()

    logger.info(
        f"Control samples: {len(young_samples)}, Case samples: {len(old_samples)}"
    )

    results = []
    for gene in df.index:
        young_expr = df.loc[gene, young_samples].values
        old_expr = df.loc[gene, old_samples].values

        mean_young = np.mean(young_expr)
        mean_old = np.mean(old_expr)

        # Log2FC = Mean_old - Mean_young (数据已是 log 值)
        log2fc = mean_old - mean_young

        # Welch t-test
        t_stat, pvalue = stats.ttest_ind(old_expr, young_expr, equal_var=False)

        # 标准误 (用于置信区间)
        se_young = stats.sem(young_expr)
        se_old = stats.sem(old_expr)
        se_diff = np.sqrt(se_young**2 + se_old**2)

        # Cohen's d effect size (pooled standard deviation)
        n1, n2 = len(young_expr), len(old_expr)
        var1, var2 = np.var(young_expr, ddof=1), np.var(old_expr, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (mean_old - mean_young) / pooled_std if pooled_std > 0 else 0

        results.append(
            {
                "Gene": gene,
                "Mean_young": mean_young,
                "Mean_old": mean_old,
                "Log2FC": log2fc,
                "SE": se_diff,
                "t_statistic": t_stat,
                "pvalue": pvalue,
                "cohens_d": cohens_d,
            }
        )

    deg_df = pd.DataFrame(results)

    # BH FDR 校正
    _, padj, _, _ = multipletests(deg_df["pvalue"].fillna(1), method="fdr_bh")
    deg_df["padj"] = padj

    # 显著性标记
    deg_df["SignificantFlag"] = "NS"
    deg_df.loc[
        (deg_df["padj"] < FDR_THRESHOLD) & (deg_df["Log2FC"] > LOG2FC_THRESHOLD),
        "SignificantFlag",
    ] = "Up"
    deg_df.loc[
        (deg_df["padj"] < FDR_THRESHOLD) & (deg_df["Log2FC"] < -LOG2FC_THRESHOLD),
        "SignificantFlag",
    ] = "Down"

    # 统计汇总
    n_total = len(deg_df)
    n_up = sum(deg_df["SignificantFlag"] == "Up")
    n_down = sum(deg_df["SignificantFlag"] == "Down")
    n_ns = sum(deg_df["SignificantFlag"] == "NS")

    logger.info(f"DEG Analysis Complete:")
    logger.info(f"  Total genes: {n_total}")
    logger.info(f"  Upregulated (Case vs Control): {n_up}")
    logger.info(f"  Downregulated: {n_down}")
    logger.info(f"  Not significant: {n_ns}")
    logger.info(
        f"  Thresholds: FDR < {FDR_THRESHOLD}, |Log2FC| > {LOG2FC_THRESHOLD:.3f}"
    )

    # 保存完整 DEG 表
    deg_file = f"{SOURCEDATA_DIR}/SourceData_Fig1_DEG_full.csv"
    deg_df.to_csv(deg_file, index=False)
    logger.info(f"Saved DEG table: {deg_file}")

    return deg_df


# ============================================================
# Step 3.5: 自动选择 Top DEGs
# ============================================================
def select_top_degs(deg_df, n_up=5, n_down=5):
    """
    从 DEG 结果中自动选择 Top 基因

    策略: 选择 |Log2FC| 最大的显著基因
    - n_up: 上调基因数量
    - n_down: 下调基因数量

    Returns:
        list: Top 基因名称列表
    """
    global KEY_TARGETS

    logger.info(f"Selecting Top DEGs (up={n_up}, down={n_down})...")

    # 筛选显著基因
    sig_up = deg_df[deg_df["SignificantFlag"] == "Up"].copy()
    sig_down = deg_df[deg_df["SignificantFlag"] == "Down"].copy()

    # 按 |Log2FC| 排序
    sig_up = sig_up.sort_values("Log2FC", ascending=False)
    sig_down = sig_down.sort_values("Log2FC", ascending=True)  # 下调基因 Log2FC 为负

    # 选择 Top 基因
    top_up = sig_up.head(n_up)["Gene"].tolist()
    top_down = sig_down.head(n_down)["Gene"].tolist()

    # 合并
    top_genes = top_up + top_down

    logger.info(f"  Top Up-regulated: {top_up}")
    logger.info(f"  Top Down-regulated: {top_down}")
    logger.info(f"  Total Top DEGs selected: {len(top_genes)}")

    # 更新全局变量
    KEY_TARGETS = top_genes

    return top_genes


def get_top_genes_for_panel(deg_df, panel_type, config=None):
    """
    根据面板类型获取相应数量的 Top 基因

    Args:
        deg_df: DEG 分析结果
        panel_type: 面板类型 ('volcano', 'heatmap', 'lollipop', 'violin', 'effect')
        config: 配置字典，默认使用 TOP_GENES_CONFIG

    Returns:
        list: Top 基因名称列表
    """
    if config is None:
        config = TOP_GENES_CONFIG

    # 获取对应面板的基因数量
    n_map = {
        "volcano": config.get("n_label", 10),
        "heatmap": config.get("n_heatmap", 30),
        "lollipop": config.get("n_lollipop", 15),
        "violin": config.get("n_violin", 6),
        "effect": config.get("n_effect", 20),
    }

    n_total = n_map.get(panel_type, 10)
    n_up = n_total // 2
    n_down = n_total - n_up

    # 筛选显著基因
    sig_up = deg_df[deg_df["SignificantFlag"] == "Up"].copy()
    sig_down = deg_df[deg_df["SignificantFlag"] == "Down"].copy()

    # 按 |Log2FC| 排序
    sig_up = sig_up.sort_values("Log2FC", ascending=False)
    sig_down = sig_down.sort_values("Log2FC", ascending=True)

    # 选择 Top 基因
    top_up = sig_up.head(n_up)["Gene"].tolist()
    top_down = sig_down.head(n_down)["Gene"].tolist()

    return top_up + top_down


# ============================================================
# Step 4: 获取基因组坐标 (ZERO-FAKE: 使用GENCODE真实坐标)
# ============================================================
def load_gencode_coordinates(genes):
    """
    Load gene coordinates from GENCODE annotation file.

    ZERO-FAKE POLICY: Only use real coordinates from GENCODE.
    If coordinates are not available, the gene will be excluded from
    Manhattan plot rather than using simulated positions.

    Args:
        genes: List of gene symbols to look up

    Returns:
        DataFrame with columns: symbol, chrom, start, end, strand
        Only includes genes that have real coordinates.
    """
    logger.info("Loading gene coordinates from GENCODE annotation...")

    if not os.path.exists(GENCODE_COORDS_FILE):
        logger.error(f"GENCODE coordinates file not found: {GENCODE_COORDS_FILE}")
        logger.error("Please run Resources/annotation/parse_gtf_to_coords.py first")
        logger.error(
            "Download GTF from: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/"
        )
        return None

    # Load GENCODE coordinates
    coords_df = pd.read_csv(GENCODE_COORDS_FILE, sep="\t")
    logger.info(f"  Loaded {len(coords_df)} gene coordinates from GENCODE v49")

    # Filter to requested genes
    gene_set = set(genes)
    matched = coords_df[coords_df["symbol"].isin(gene_set)].copy()

    # Calculate coverage
    n_matched = len(matched)
    n_total = len(gene_set)
    coverage = n_matched / n_total * 100 if n_total > 0 else 0

    logger.info(f"  Matched {n_matched}/{n_total} genes ({coverage:.1f}%)")

    if coverage < 50:
        logger.warning(
            f"  Low coverage ({coverage:.1f}%). Check if gene symbols match GENCODE annotation."
        )

    return matched


def get_gene_coordinates(genes):
    """获取基因 hg38 坐标 (优先使用GENCODE，备选MyGene.info API)"""
    logger.info("Fetching gene coordinates...")

    # 首先尝试GENCODE坐标（推荐）
    coords_df = load_gencode_coordinates(genes)
    if coords_df is not None and len(coords_df) > 0:
        return coords_df

    # 检查缓存
    cache_file = f"{RAW_DIR}/gene_coords_hg38.tsv"

    # 检查缓存
    if os.path.exists(cache_file):
        logger.info(f"Loading cached coordinates: {cache_file}")
        coords_df = pd.read_csv(cache_file, sep="	")
        return coords_df

    # 检查用户提供的文件
    user_file = f"{PROJECT_ROOT}/data/gene_coords_hg38.tsv"
    if os.path.exists(user_file):
        logger.info(f"Loading user-provided coordinates: {user_file}")
        coords_df = pd.read_csv(user_file, sep="	")
        coords_df.to_csv(cache_file, sep="	", index=False)
        return coords_df

    # 使用 MyGene.info API
    try:
        import requests

        coords_list = []
        batch_size = 1000
        gene_list = list(genes)

        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i : i + batch_size]
            logger.info(
                f"Querying batch {i // batch_size + 1}/{(len(gene_list) - 1) // batch_size + 1}..."
            )

            url = "https://mygene.info/v3/query"
            params = {
                "q": ",".join(batch),
                "scopes": "symbol",
                "fields": "symbol,genomic_pos_hg19,genomic_pos",
                "species": "human",
                "size": batch_size,
            }

            response = requests.post(url, data=params, timeout=60)
            if response.status_code == 200:
                results = response.json()

                for hit in results:
                    if isinstance(hit, dict) and "genomic_pos" in hit:
                        gpos = hit["genomic_pos"]
                        if isinstance(gpos, list):
                            gpos = gpos[0]
                        if isinstance(gpos, dict):
                            coords_list.append(
                                {
                                    "GeneSymbol": hit.get(
                                        "symbol", hit.get("query", "")
                                    ),
                                    "chr": str(gpos.get("chr", "")),
                                    "start": gpos.get("start", 0),
                                    "end": gpos.get("end", 0),
                                }
                            )

        coords_df = pd.DataFrame(coords_list)

        # 检查覆盖率
        coverage = len(coords_df) / len(genes) * 100
        logger.info(
            f"Coordinate coverage: {len(coords_df)}/{len(genes)} ({coverage:.1f}%)"
        )

        if coverage < 85:
            logger.warning(
                f"Coverage {coverage:.1f}% < 85%. Manhattan plot may be incomplete."
            )
            # 仍然保存已获取的坐标并继续

        coords_df.to_csv(cache_file, sep="\t", index=False)
        logger.info(f"Saved coordinates: {cache_file}")

        return coords_df

    except Exception as e:
        logger.warning(f"Failed to fetch coordinates: {e}")
        logger.warning("Will generate Manhattan plot without real coordinates")
        return None


# ============================================================
# Panel A: PCA
# ============================================================
def plot_panel_A(df, metadata):
    """PCA with 95% confidence ellipse"""
    logger.info("Generating Panel A: PCA...")

    # 准备数据
    X = df.T.values  # samples x genes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_result = pca.fit_transform(X_scaled)

    # 创建结果 DataFrame
    pca_df = pd.DataFrame(
        {"SampleID": df.columns, "PC1": pca_result[:, 0], "PC2": pca_result[:, 1]}
    )
    pca_df = pca_df.merge(metadata, on="SampleID")

    # 绘图
    fig, ax = plt.subplots(figsize=(5, 4.5))

    colors = {"Control": NPG_COLORS["blue"], "Case": NPG_COLORS["red"]}

    for group in ["Control", "Case"]:
        mask = pca_df["Group"] == group
        data = pca_df[mask]

        ax.scatter(
            data["PC1"],
            data["PC2"],
            c=colors[group],
            s=80,
            alpha=0.8,
            label=group,
            edgecolors="white",
            linewidth=0.5,
        )

        # 95% 置信椭圆
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms

        if len(data) > 2:
            mean_x, mean_y = data["PC1"].mean(), data["PC2"].mean()
            cov = np.cov(data["PC1"], data["PC2"])

            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)

            ellipse = Ellipse(
                (0, 0),
                width=ell_radius_x * 2,
                height=ell_radius_y * 2,
                facecolor=colors[group],
                alpha=0.15,
                edgecolor=colors[group],
                linewidth=1.5,
            )

            scale_x = np.sqrt(cov[0, 0]) * 2.447  # 95% CI
            scale_y = np.sqrt(cov[1, 1]) * 2.447

            transf = (
                transforms.Affine2D()
                .rotate_deg(45)
                .scale(scale_x, scale_y)
                .translate(mean_x, mean_y)
            )

            ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "A")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1A.png", dpi=600)
    plt.close()

    # 保存源数据
    pca_df["PC1_var_explained"] = pca.explained_variance_ratio_[0]
    pca_df["PC2_var_explained"] = pca.explained_variance_ratio_[1]
    save_sourcedata(pca_df, "Fig1A_PCA")

    logger.info("Panel A complete")
    return pca_df


# ============================================================
# Panel B: UMAP/t-SNE
# ============================================================
def plot_panel_B(df, metadata):
    """UMAP or t-SNE embedding"""
    logger.info("Generating Panel B: UMAP/t-SNE...")

    X = df.T.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2, random_state=RANDOM_SEED, n_neighbors=5, min_dist=0.3
        )
        embedding = reducer.fit_transform(X_scaled)
        method = "UMAP"
    else:
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=5)
        embedding = reducer.fit_transform(X_scaled)
        method = "t-SNE"

    embed_df = pd.DataFrame(
        {
            "SampleID": df.columns,
            f"{method}1": embedding[:, 0],
            f"{method}2": embedding[:, 1],
        }
    )
    embed_df = embed_df.merge(metadata, on="SampleID")

    fig, ax = plt.subplots(figsize=(5, 4.5))

    colors = {"Control": NPG_COLORS["blue"], "Case": NPG_COLORS["red"]}

    for group in ["Control", "Case"]:
        mask = embed_df["Group"] == group
        data = embed_df[mask]
        ax.scatter(
            data[f"{method}1"],
            data[f"{method}2"],
            c=colors[group],
            s=80,
            alpha=0.8,
            label=group,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel(f"{method}1")
    ax.set_ylabel(f"{method}2")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "B")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1B.png", dpi=600)
    plt.close()

    save_sourcedata(embed_df, f"Fig1B_{method}")
    logger.info("Panel B complete")
    return embed_df


# ============================================================
# Panel C: Volcano Plot
# ============================================================
def plot_panel_C(deg_df):
    """Volcano plot with Top DEGs labeled (auto-selected by |Log2FC|)"""
    logger.info("Generating Panel C: Volcano Plot...")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # 计算 -log10(FDR)
    deg_df["-log10FDR"] = -np.log10(deg_df["padj"].clip(lower=1e-300))

    # 分组绘制
    ns = deg_df[deg_df["SignificantFlag"] == "NS"]
    up = deg_df[deg_df["SignificantFlag"] == "Up"]
    down = deg_df[deg_df["SignificantFlag"] == "Down"]

    ax.scatter(
        ns["Log2FC"],
        ns["-log10FDR"],
        c=NPG_COLORS["gray"],
        s=8,
        alpha=0.4,
        label=f"NS ({len(ns)})",
    )
    ax.scatter(
        up["Log2FC"],
        up["-log10FDR"],
        c=NPG_COLORS["red"],
        s=12,
        alpha=0.6,
        label=f"Up ({len(up)})",
    )
    ax.scatter(
        down["Log2FC"],
        down["-log10FDR"],
        c=NPG_COLORS["blue"],
        s=12,
        alpha=0.6,
        label=f"Down ({len(down)})",
    )

    # 阈值线
    ax.axhline(
        -np.log10(FDR_THRESHOLD), color="gray", linestyle="--", linewidth=0.8, alpha=0.5
    )
    ax.axvline(LOG2FC_THRESHOLD, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(
        -LOG2FC_THRESHOLD, color="gray", linestyle="--", linewidth=0.8, alpha=0.5
    )

    # 自动选择 Top DEGs 进行标注
    top_genes_to_label = get_top_genes_for_panel(deg_df, "volcano")
    logger.info(f"  Labeling {len(top_genes_to_label)} top genes on volcano plot")

    from adjustText import adjust_text

    texts = []
    for gene in top_genes_to_label:
        if gene in deg_df["Gene"].values:
            row = deg_df[deg_df["Gene"] == gene].iloc[0]
            ax.scatter(
                row["Log2FC"],
                row["-log10FDR"],
                c=NPG_COLORS["orange"],
                s=60,
                edgecolors="black",
                linewidth=1,
                zorder=10,
            )
            texts.append(
                ax.text(
                    row["Log2FC"], row["-log10FDR"], gene, fontsize=8, fontweight="bold"
                )
            )

    try:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    except:
        pass  # adjustText 可能未安装

    ax.set_xlabel("Log2 Fold Change (Case/Control)")
    ax.set_ylabel("-log10(FDR)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "C")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1C.png", dpi=600)
    plt.close()

    # 保存源数据
    volcano_df = deg_df[
        ["Gene", "Log2FC", "pvalue", "padj", "-log10FDR", "SignificantFlag"]
    ].copy()
    save_sourcedata(volcano_df, "Fig1C_Volcano")

    logger.info("Panel C complete")


# ============================================================
# Panel D: MA Plot
# ============================================================
def plot_panel_D(deg_df):
    """MA plot"""
    logger.info("Generating Panel D: MA Plot...")

    fig, ax = plt.subplots(figsize=(5, 4.5))

    # A = average expression
    deg_df["AvgExpr"] = (deg_df["Mean_young"] + deg_df["Mean_old"]) / 2

    ns = deg_df[deg_df["SignificantFlag"] == "NS"]
    sig = deg_df[deg_df["SignificantFlag"] != "NS"]

    ax.scatter(
        ns["AvgExpr"], ns["Log2FC"], c=NPG_COLORS["gray"], s=8, alpha=0.3, label="NS"
    )

    up = sig[sig["SignificantFlag"] == "Up"]
    down = sig[sig["SignificantFlag"] == "Down"]
    ax.scatter(
        up["AvgExpr"], up["Log2FC"], c=NPG_COLORS["red"], s=12, alpha=0.6, label="Up"
    )
    ax.scatter(
        down["AvgExpr"],
        down["Log2FC"],
        c=NPG_COLORS["blue"],
        s=12,
        alpha=0.6,
        label="Down",
    )

    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(LOG2FC_THRESHOLD, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(
        -LOG2FC_THRESHOLD, color="gray", linestyle="--", linewidth=0.8, alpha=0.5
    )

    ax.set_xlabel("Average Expression (A)")
    ax.set_ylabel("Log2 Fold Change (M)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "D")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1D.png", dpi=600)
    plt.close()

    ma_df = deg_df[["Gene", "AvgExpr", "Log2FC", "SignificantFlag"]].copy()
    save_sourcedata(ma_df, "Fig1D_MA")

    logger.info("Panel D complete")


# ============================================================
# Panel E: DEG Counts Bar Chart
# ============================================================
def plot_panel_E(deg_df):
    """DEG counts bar chart"""
    logger.info("Generating Panel E: DEG Counts...")

    n_up = sum(deg_df["SignificantFlag"] == "Up")
    n_down = sum(deg_df["SignificantFlag"] == "Down")

    fig, ax = plt.subplots(figsize=(4, 4.5))

    bars = ax.bar(
        ["Upregulated", "Downregulated"],
        [n_up, n_down],
        color=[NPG_COLORS["red"], NPG_COLORS["blue"]],
        width=0.6,
    )

    # 柱顶数值
    for bar, val in zip(bars, [n_up, n_down]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Number of DEGs")
    ax.set_title(
        f"FDR < {FDR_THRESHOLD}, |Log2FC| > {LOG2FC_THRESHOLD:.2f}", fontsize=9
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "E")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1E.png", dpi=600)
    plt.close()

    counts_df = pd.DataFrame(
        {"Category": ["Upregulated", "Downregulated"], "Count": [n_up, n_down]}
    )
    save_sourcedata(counts_df, "Fig1E_DEG_counts")

    logger.info("Panel E complete")


# ============================================================
# Panel F: Top 30 DEG Heatmap
# ============================================================
def plot_panel_F(df, deg_df, metadata):
    """Top 30 DE genes heatmap"""
    logger.info("Generating Panel F: Top 30 DEG Heatmap...")

    # 选择 Top 30 by |Log2FC|
    sig_genes = deg_df[deg_df["SignificantFlag"] != "NS"].copy()
    sig_genes["absLog2FC"] = sig_genes["Log2FC"].abs()
    top30 = sig_genes.nlargest(30, "absLog2FC")["Gene"].tolist()

    if len(top30) < 30:
        # 如果显著基因不足30个，补充最大变化的基因
        remaining = 30 - len(top30)
        other_genes = deg_df[~deg_df["Gene"].isin(top30)].copy()
        other_genes["absLog2FC"] = other_genes["Log2FC"].abs()
        extra = other_genes.nlargest(remaining, "absLog2FC")["Gene"].tolist()
        top30.extend(extra)

    # 准备热图数据
    heatmap_data = df.loc[top30].copy()

    # 按组排序样本
    young_samples = metadata[metadata["Group"] == "Control"]["SampleID"].tolist()
    old_samples = metadata[metadata["Group"] == "Case"]["SampleID"].tolist()
    ordered_samples = young_samples + old_samples
    heatmap_data = heatmap_data[ordered_samples]

    # Z-score 标准化
    heatmap_z = (heatmap_data.T - heatmap_data.mean(axis=1)) / heatmap_data.std(axis=1)
    heatmap_z = heatmap_z.T

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # 创建颜色条
    group_colors = [NPG_COLORS["blue"]] * len(young_samples) + [
        NPG_COLORS["red"]
    ] * len(old_samples)

    # 绘制热图
    im = ax.imshow(heatmap_z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Z-score")

    # 设置标签
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30, fontsize=7)
    ax.set_xticks([])

    # 添加组别色条
    for i, color in enumerate(group_colors):
        ax.add_patch(plt.Rectangle((i - 0.5, -1.5), 1, 1, color=color, clip_on=False))

    # 图例
    young_patch = mpatches.Patch(color=NPG_COLORS["blue"], label="Control")
    old_patch = mpatches.Patch(color=NPG_COLORS["red"], label="Case")
    ax.legend(
        handles=[young_patch, old_patch],
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        frameon=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "F")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1F.png", dpi=600)
    plt.close()

    # 保存源数据
    heatmap_z.to_csv(f"{SOURCEDATA_DIR}/SourceData_Fig1F_Heatmap.csv")

    logger.info("Panel F complete")


# =============================================================================
# Panel G: Manhattan-like Plot (ZERO-FAKE: Real coordinates only)
# =============================================================================
def plot_panel_G(deg_df, gene_coords, output_dir):
    """
    Manhattan-like plot by chromosome position.

    ZERO-FAKE POLICY: Only uses real GENCODE coordinates.
    If gene_coords is None, this panel will show a placeholder message
    rather than simulated data.
    """
    logger.info("Generating Panel G: Manhattan-like plot...")

    # ZERO-FAKE: Do not generate simulated positions
    if gene_coords is None or len(gene_coords) == 0:
        logger.warning("  No gene coordinates available. Generating placeholder panel.")
        logger.warning("  To fix: ensure GENCODE coordinates file exists at:")
        logger.warning(f"    {GENCODE_COORDS_FILE}")

        # Create placeholder figure
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "Manhattan Plot\n\nGene coordinates not available.\nPlease provide GENCODE annotation.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        add_panel_label(ax, "G")

        output_path = f"{output_dir}/panels16/Fig1G_Manhattan.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        # Save empty source data with explanation
        empty_df = pd.DataFrame(
            {
                "Gene": [],
                "chrom": [],
                "start": [],
                "neg_log10_fdr": [],
                "note": ["No coordinates available - GENCODE file required"],
            }
        )
        save_sourcedata(empty_df, "Fig1G_Manhattan")
        logger.info("  Panel G: Placeholder generated (no simulated data)")
        return

    # Merge DEG results with REAL coordinates
    plot_df = deg_df.merge(
        gene_coords, left_on="Gene", right_on="GeneSymbol", how="inner"
    )

    n_matched = len(plot_df)
    n_total = len(deg_df)
    coverage = n_matched / n_total * 100 if n_total > 0 else 0
    logger.info(
        f"  Matched {n_matched}/{n_total} genes ({coverage:.1f}%) with GENCODE coordinates"
    )

    # Handle chromosome naming (GENCODE uses chr prefix)
    # Rename 'chr' to 'chrom' if needed
    if "chrom" not in plot_df.columns and "chr" in plot_df.columns:
        # Rename 'chr' to 'chrom'
        plot_df = plot_df.rename(columns={"chr": "chrom"})

    # Ensure chromosome names have "chr" prefix
    plot_df["chrom"] = plot_df["chrom"].astype(str)
    plot_df.loc[~plot_df["chrom"].str.startswith("chr"), "chrom"] = (
        "chr" + plot_df.loc[~plot_df["chrom"].str.startswith("chr"), "chrom"]
    )

    # Define chromosome order (with chr prefix)
    chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    plot_df = plot_df[plot_df["chrom"].isin(chrom_order)]
    plot_df["chrom_num"] = plot_df["chrom"].map(
        {c: i for i, c in enumerate(chrom_order)}
    )
    plot_df = plot_df.sort_values(["chrom_num", "start"])

    # Calculate cumulative position
    chrom_lengths = plot_df.groupby("chrom_num")["start"].max().to_dict()
    cumsum = 0
    chrom_offsets = {}
    chrom_centers = {}
    for c in sorted(chrom_lengths.keys()):
        chrom_offsets[c] = cumsum
        chrom_centers[c] = cumsum + chrom_lengths.get(c, 0) / 2
        cumsum += chrom_lengths.get(c, 0) + 1e7  # Gap between chromosomes

    plot_df["cumpos"] = plot_df.apply(
        lambda r: chrom_offsets.get(r["chrom_num"], 0) + r["start"], axis=1
    )
    plot_df["neg_log10_fdr"] = -np.log10(plot_df["padj"].clip(lower=1e-300))

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))

    # Color by chromosome and direction
    colors_up = [
        NPG_COLORS["red"] if i % 2 == 0 else NPG_COLORS["orange"]
        for i in plot_df["chrom_num"]
    ]
    colors_down = [
        NPG_COLORS["blue"] if i % 2 == 0 else NPG_COLORS["cyan"]
        for i in plot_df["chrom_num"]
    ]

    # Plot non-significant
    ns_mask = plot_df["SignificantFlag"] == "NS"
    ax.scatter(
        plot_df.loc[ns_mask, "cumpos"],
        plot_df.loc[ns_mask, "neg_log10_fdr"],
        c=NPG_COLORS["gray"],
        s=8,
        alpha=0.3,
        rasterized=True,
    )

    # Plot up-regulated
    up_mask = plot_df["SignificantFlag"] == "Up"
    if up_mask.sum() > 0:
        ax.scatter(
            plot_df.loc[up_mask, "cumpos"],
            plot_df.loc[up_mask, "neg_log10_fdr"],
            c=NPG_COLORS["red"],
            s=15,
            alpha=0.7,
            label="Up-regulated",
        )

    # Plot down-regulated
    down_mask = plot_df["SignificantFlag"] == "Down"
    if down_mask.sum() > 0:
        ax.scatter(
            plot_df.loc[down_mask, "cumpos"],
            plot_df.loc[down_mask, "neg_log10_fdr"],
            c=NPG_COLORS["blue"],
            s=15,
            alpha=0.7,
            label="Down-regulated",
        )

    # Highlight Top DEGs (auto-selected)
    top_genes_manhattan = get_top_genes_for_panel(
        deg_df, "volcano"
    )  # 使用与 volcano 相同的 top genes
    key_df = plot_df[plot_df["Gene"].isin(top_genes_manhattan)]
    if len(key_df) > 0:
        ax.scatter(
            key_df["cumpos"],
            key_df["neg_log10_fdr"],
            c=NPG_COLORS["purple"],
            s=60,
            marker="D",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
            label="Top DEGs",
        )

        # Add labels
        texts = []
        for _, row in key_df.iterrows():
            texts.append(
                ax.text(
                    row["cumpos"],
                    row["neg_log10_fdr"],
                    row["Gene"],
                    fontsize=8,
                    fontweight="bold",
                )
            )
        if texts:
            try:
                from adjustText import adjust_text

                adjust_text(
                    texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
                )
            except:
                pass

    # Significance threshold
    sig_line = -np.log10(FDR_THRESHOLD)
    ax.axhline(y=sig_line, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # X-axis labels
    ax.set_xticks(
        [chrom_centers[i] for i in sorted(chrom_centers.keys()) if i in chrom_centers]
    )
    ax.set_xticklabels(
        [chrom_order[i] for i in sorted(chrom_centers.keys()) if i < len(chrom_order)],
        fontsize=7,
        rotation=45,
    )

    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel("-log10(FDR)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "G")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1G.png", dpi=600)
    plt.close()

    # Save source data
    save_sourcedata(
        plot_df[
            [
                "Gene",
                "chrom",
                "start",
                "end",
                "Log2FC",
                "padj",
                "SignificantFlag",
                "cumpos",
            ]
        ],
        "Fig1G_Manhattan",
    )
    logger.info(f"Panel G complete ({len(plot_df)} genes plotted)")
    return fig


# =============================================================================
# Panel H: QQ Plot
# =============================================================================
def plot_panel_H(deg_df, output_dir):
    """QQ plot of p-values to detect systematic bias"""
    print("  Plotting Panel H: QQ plot...")

    # Get p-values (not FDR)
    pvals = deg_df["pvalue"].dropna()
    pvals = pvals[pvals > 0]  # Remove zeros

    # Calculate expected vs observed
    n = len(pvals)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(np.sort(pvals))

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot points
    ax.scatter(
        expected, observed, c=NPG_COLORS["blue"], s=10, alpha=0.5, rasterized=True
    )

    # Diagonal line (null expectation)
    max_val = max(expected.max(), observed.max())
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Expected (null)")

    # Calculate genomic inflation factor (lambda)
    median_chi2 = np.median(stats.chi2.ppf(1 - pvals, df=1))
    lambda_gc = median_chi2 / stats.chi2.ppf(0.5, df=1)

    ax.text(
        0.05,
        0.95,
        f"λGC = {lambda_gc:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Expected -log10(p)", fontsize=10)
    ax.set_ylabel("Observed -log10(p)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    add_panel_label(ax, "H")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1H.png", dpi=600)
    plt.close()

    # Save source data
    qq_data = pd.DataFrame(
        {"expected_neglog10p": expected, "observed_neglog10p": observed}
    )
    save_sourcedata(qq_data, "Fig1H_QQplot")
    print(f"    Saved: Fig1H.png (λGC = {lambda_gc:.3f})")
    return fig, lambda_gc


# =============================================================================
# Panel I: Top Up-regulated Genes Lollipop Chart
# =============================================================================
def plot_panel_I(deg_df, output_dir, top_n=None):
    """Lollipop chart of top up-regulated genes (auto-selected by Log2FC)"""
    print("  Plotting Panel I: Top up-regulated lollipop...")

    # 使用配置中的数量，如果未指定
    if top_n is None:
        top_n = TOP_GENES_CONFIG.get("n_lollipop", 15)

    # Get top up-regulated genes
    up_genes = deg_df[deg_df["SignificantFlag"] == "Up"].nlargest(top_n, "Log2FC")

    if len(up_genes) == 0:
        print("    WARNING: No up-regulated genes found!")
        return None

    fig, ax = plt.subplots(figsize=(5, 5))

    y_pos = np.arange(len(up_genes))

    # Draw stems
    ax.hlines(
        y=y_pos,
        xmin=0,
        xmax=up_genes["Log2FC"].values,
        color=NPG_COLORS["red"],
        alpha=0.7,
        linewidth=2,
    )

    # Draw dots
    ax.scatter(
        up_genes["Log2FC"].values,
        y_pos,
        color=NPG_COLORS["red"],
        s=80,
        zorder=3,
        edgecolors="white",
        linewidths=1,
    )

    # 不再高亮预设基因，所有显示的都是 Top DEGs

    ax.set_yticks(y_pos)
    ax.set_yticklabels(up_genes["Gene"].values, fontsize=9)
    ax.set_xlabel("Log2 Fold Change", fontsize=10)
    ax.set_ylabel("")
    ax.axvline(
        x=LOG2FC_THRESHOLD, color="gray", linestyle="--", linewidth=0.8, alpha=0.5
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    add_panel_label(ax, "I")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1I.png", dpi=600)
    plt.close()

    # Save source data
    save_sourcedata(
        up_genes[["Gene", "Log2FC", "padj", "SignificantFlag"]], "Fig1I_Lollipop"
    )
    print(f"    Saved: Fig1I.png ({len(up_genes)} genes)")
    return fig


# =============================================================================
# Panel J: Top DEGs Expression Violin Plot
# =============================================================================
def plot_panel_J(expr_df, metadata, output_dir, deg_df=None):
    """Violin plot of Top DEGs expression by group (auto-selected by |Log2FC|)"""
    print("  Plotting Panel J: Top DEGs violin plot...")

    # 自动选择 Top DEGs 用于 violin plot
    n_violin = TOP_GENES_CONFIG.get("n_violin", 6)
    if deg_df is not None:
        top_genes_violin = get_top_genes_for_panel(deg_df, "violin")
    else:
        # 如果没有 deg_df，使用全局 KEY_TARGETS（应该已被 select_top_degs 更新）
        top_genes_violin = KEY_TARGETS[:n_violin] if KEY_TARGETS else []

    if not top_genes_violin:
        print("    WARNING: No Top DEGs available for violin plot!")
        return None

    # Filter for Top DEGs - expr_df has Gene as index
    key_expr = expr_df.loc[expr_df.index.isin(top_genes_violin)].copy()
    key_expr = key_expr.reset_index()  # Move Gene from index to column
    key_expr = key_expr.rename(columns={"index": "Gene"})

    if len(key_expr) == 0:
        print("    WARNING: No Top DEGs found in expression data!")
        return None

    # Melt to long format
    sample_cols = [c for c in key_expr.columns if c != "Gene"]
    key_long = key_expr.melt(
        id_vars=["Gene"],
        value_vars=sample_cols,
        var_name="SampleID",
        value_name="expression",
    )

    # Add group info
    key_long["Group"] = key_long["SampleID"].map(
        metadata.set_index("SampleID")["Group"]
    )

    # Order genes by |Log2FC| (保持选择顺序)
    gene_order = [g for g in top_genes_violin if g in key_expr["Gene"].values]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Create violin plot
    palette = {"Control": NPG_COLORS["blue"], "Case": NPG_COLORS["red"]}

    sns.violinplot(
        data=key_long,
        x="Gene",
        y="expression",
        hue="Group",
        order=gene_order,
        palette=palette,
        split=True,
        inner="quart",
        ax=ax,
        linewidth=1,
    )

    # Add individual points
    sns.stripplot(
        data=key_long,
        x="Gene",
        y="expression",
        hue="Group",
        order=gene_order,
        palette=palette,
        dodge=True,
        size=4,
        alpha=0.6,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("", fontsize=10)
    ax.set_ylabel("Expression (log2)", fontsize=10)
    ax.legend(title="Group", loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right")

    add_panel_label(ax, "J")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1J.png", dpi=600)
    plt.close()

    # Save source data
    save_sourcedata(key_long, "Fig1J_Violin")
    print(f"    Saved: Fig1J.png ({len(gene_order)} genes: {gene_order})")
    return fig


# =============================================================================
# Panel K: Effect Size (Cohen d) Plot
# =============================================================================
def plot_panel_K(deg_df, output_dir, top_n=None):
    """Effect size plot for top DEGs (auto-selected by |Cohen's d|)"""
    print("  Plotting Panel K: Effect size plot...")

    # 使用配置中的数量，如果未指定
    if top_n is None:
        top_n = TOP_GENES_CONFIG.get("n_effect", 20)

    # Get top DEGs by absolute effect size
    deg_sig = deg_df[deg_df["SignificantFlag"] != "NS"].copy()
    deg_sig["abs_cohend"] = deg_sig["cohens_d"].abs()
    top_effect = deg_sig.nlargest(top_n, "abs_cohend")

    if len(top_effect) == 0:
        print("    WARNING: No significant DEGs found!")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))

    # Sort by Cohen's d
    top_effect = top_effect.sort_values("cohens_d")
    y_pos = np.arange(len(top_effect))

    # Color by direction
    colors = [
        NPG_COLORS["red"] if d > 0 else NPG_COLORS["blue"]
        for d in top_effect["cohens_d"]
    ]

    bars = ax.barh(
        y_pos, top_effect["cohens_d"].values, color=colors, height=0.7, alpha=0.8
    )

    # 不再高亮预设基因，所有显示的都是 Top DEGs by effect size

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_effect["Gene"].values, fontsize=9)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(
        x=0.8,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="Large effect",
    )
    ax.axvline(x=-0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "K")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1K_effect_size.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{PANELS_DIR}/Fig1K_effect_size.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Save source data
    save_sourcedata(
        top_effect[["Gene", "cohens_d", "Log2FC", "padj", "SignificantFlag"]],
        "Fig1K_EffectSize",
    )
    print(f"    Saved: Fig1K.png ({len(top_effect)} genes)")
    return fig


# =============================================================================
# Panel L: Chromosome Enrichment (ZERO-FAKE: Real coordinates only)
# =============================================================================
def plot_panel_L(deg_df, gene_coords, output_dir):
    """
    Chromosome enrichment analysis for DEGs.

    ZERO-FAKE POLICY: Only uses real GENCODE coordinates.
    If gene_coords is None, this panel will show a placeholder message
    rather than simulated data.
    """
    logger.info("Generating Panel L: Chromosome enrichment...")

    # ZERO-FAKE: Do not generate simulated positions
    if gene_coords is None or len(gene_coords) == 0:
        logger.warning("  No gene coordinates available. Generating placeholder panel.")

        # Create placeholder figure
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "Chromosome Enrichment\n\nGene coordinates not available.\nPlease provide GENCODE annotation.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        add_panel_label(ax, "L")

        output_path = f"{output_dir}/panels16/Fig1L_ChromEnrich.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        # Save empty source data with explanation
        empty_df = pd.DataFrame(
            {
                "chrom": [],
                "total": [],
                "up": [],
                "down": [],
                "note": ["No coordinates available - GENCODE file required"],
            }
        )
        save_sourcedata(empty_df, "Fig1L_ChromEnrich")
        logger.info("  Panel L: Placeholder generated (no simulated data)")
        return

    # Merge DEG with REAL coordinates
    merged = deg_df.merge(
        gene_coords, left_on="Gene", right_on="GeneSymbol", how="inner"
    )

    # Handle chromosome naming (GENCODE uses chr prefix)
    # Rename 'chr' to 'chrom' if needed
    if "chrom" not in merged.columns and "chr" in merged.columns:
        # Rename 'chr' to 'chrom'
        merged = merged.rename(columns={"chr": "chrom"})

    # Ensure chromosome names have "chr" prefix
    merged["chrom"] = merged["chrom"].astype(str)
    merged.loc[~merged["chrom"].str.startswith("chr"), "chrom"] = (
        "chr" + merged.loc[~merged["chrom"].str.startswith("chr"), "chrom"]
    )

    # Define chromosome order (with chr prefix for GENCODE)
    chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    merged = merged[merged["chrom"].isin(chrom_order)]

    n_matched = len(merged)
    logger.info(f"  Using {n_matched} genes with GENCODE coordinates")

    # Count DEGs per chromosome
    chrom_counts = []
    for chrom in chrom_order:
        chrom_data = merged[merged["chrom"] == chrom]
        total = len(chrom_data)
        up = len(chrom_data[chrom_data["SignificantFlag"] == "Up"])
        down = len(chrom_data[chrom_data["SignificantFlag"] == "Down"])
        ns = len(chrom_data[chrom_data["SignificantFlag"] == "NS"])
        chrom_counts.append(
            {
                "chrom": chrom,
                "total": total,
                "up": up,
                "down": down,
                "ns": ns,
                "deg_pct": (up + down) / total * 100 if total > 0 else 0,
            }
        )

    chrom_df = pd.DataFrame(chrom_counts)

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(chrom_order))
    width = 0.35

    bars_up = ax.bar(
        x - width / 2,
        chrom_df["up"],
        width,
        label="Up-regulated",
        color=NPG_COLORS["red"],
        alpha=0.8,
    )
    bars_down = ax.bar(
        x + width / 2,
        chrom_df["down"],
        width,
        label="Down-regulated",
        color=NPG_COLORS["blue"],
        alpha=0.8,
    )

    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel("Number of DEGs", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(chrom_order, fontsize=8, rotation=45)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "L")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1L.png", dpi=600)
    plt.close()

    # Save source data
    save_sourcedata(chrom_df, "Fig1L_ChromEnrich")
    logger.info("Panel L complete")
    return fig


# =============================================================================
# Panel M: Sample Correlation Heatmap
# =============================================================================
def plot_panel_M(expr_df, metadata, output_dir):
    """Sample-sample correlation heatmap"""
    logger.info("Generating Panel M: Sample correlation heatmap...")

    # expr_df already has Gene as index, columns are samples
    expr_matrix = expr_df

    # Calculate correlation
    corr_matrix = expr_matrix.corr(method="pearson")

    # Create annotation colors
    sample_order = metadata.sort_values("Group")["SampleID"].tolist()
    # Filter to only samples that exist in corr_matrix
    sample_order = [s for s in sample_order if s in corr_matrix.columns]
    corr_matrix = corr_matrix.loc[sample_order, sample_order]

    # Color annotation
    group_colors = metadata.set_index("SampleID")["Group"].map(
        {"Control": NPG_COLORS["blue"], "Case": NPG_COLORS["red"]}
    )
    group_colors = group_colors.loc[sample_order]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0.9,
        vmin=0.8,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        ax=ax,
        annot=False,
    )

    # Add group color bar on top
    for i, (sample, color) in enumerate(group_colors.items()):
        ax.add_patch(plt.Rectangle((i, -0.5), 1, 0.5, color=color, clip_on=False))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    add_panel_label(ax, "M")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1M.png", dpi=600)
    plt.close()

    # Save source data
    save_sourcedata(corr_matrix.reset_index(), "Fig1M_SampleCorr")
    logger.info("Panel M complete")
    return fig


# =============================================================================
# Panel N: Sample Dendrogram
# =============================================================================
def plot_panel_N(expr_df, metadata, output_dir):
    """Hierarchical clustering dendrogram of samples"""
    logger.info("Generating Panel N: Sample dendrogram...")

    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    # expr_df already has Gene as index, transpose to get samples as rows
    expr_matrix = expr_df.T

    # Calculate distance and linkage
    distances = pdist(expr_matrix.values, metric="correlation")
    linkage_matrix = linkage(distances, method="ward")

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Create color mapping for leaves
    sample_colors = metadata.set_index("SampleID")["Group"].map(
        {"Control": NPG_COLORS["blue"], "Case": NPG_COLORS["red"]}
    )

    # Build short labels: extract a compact ID from the long sample name
    sample_list = expr_matrix.index.tolist()
    short_labels = []
    for s in sample_list:
        # e.g. "GSE196006_X15.018_L7_G821_htseq.out_control" -> "196006_X15.018"
        parts = s.split("_")
        gse_short = (
            parts[0].replace("GSE", "") if parts[0].startswith("GSE") else parts[0]
        )
        sample_bit = parts[1] if len(parts) > 1 else ""
        short_labels.append(f"{gse_short}_{sample_bit}")

    # Plot dendrogram — thin lines, tiny truncated labels
    with plt.rc_context({"lines.linewidth": 0.4}):
        dend = dendrogram(
            linkage_matrix,
            labels=short_labels,
            leaf_rotation=90,
            leaf_font_size=3,
            ax=ax,
            color_threshold=0,
            above_threshold_color="#999999",
        )

    # Color the leaf labels by group
    xlbls = ax.get_xticklabels()
    leaf_order = dend["leaves"]
    for lbl, leaf_idx in zip(xlbls, leaf_order):
        sample = sample_list[leaf_idx]
        if sample in sample_colors.index:
            lbl.set_color(sample_colors[sample])
        lbl.set_fontsize(2.5)

    ax.set_ylabel("Distance (Ward)", fontsize=8)
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=NPG_COLORS["blue"], label="Control"),
        Patch(facecolor=NPG_COLORS["red"], label="Case"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=7)

    add_panel_label(ax, "N")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1N.png", dpi=600)
    plt.close()

    # Save source data (linkage matrix)
    linkage_df = pd.DataFrame(
        linkage_matrix, columns=["cluster1", "cluster2", "distance", "n_samples"]
    )
    save_sourcedata(linkage_df, "Fig1N_Dendrogram")
    logger.info("Panel N complete")
    return fig


# =============================================================================
# Panel O: Effect Size Distribution
# =============================================================================
def plot_panel_O(deg_df, output_dir):
    """Distribution of effect sizes (Cohen d) for all genes"""
    print("  Plotting Panel O: Effect size distribution...")

    fig, ax = plt.subplots(figsize=(5, 4))

    # Get Cohen's d values
    cohend_values = deg_df["cohens_d"].dropna()

    # Plot histogram
    bins = np.linspace(-3, 3, 61)

    # Separate by DEG status (SignificantFlag column)
    ns_vals = deg_df[deg_df["SignificantFlag"] == "NS"]["cohens_d"].dropna()
    up_vals = deg_df[deg_df["SignificantFlag"] == "Up"]["cohens_d"].dropna()
    down_vals = deg_df[deg_df["SignificantFlag"] == "Down"]["cohens_d"].dropna()

    ax.hist(
        ns_vals,
        bins=bins,
        color=NPG_COLORS["gray"],
        alpha=0.5,
        label="NS",
        density=True,
    )
    ax.hist(
        up_vals, bins=bins, color=NPG_COLORS["red"], alpha=0.7, label="Up", density=True
    )
    ax.hist(
        down_vals,
        bins=bins,
        color=NPG_COLORS["blue"],
        alpha=0.7,
        label="Down",
        density=True,
    )

    # Add effect size thresholds
    ax.axvline(x=0.2, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(x=-0.2, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(x=0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=-0.8, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    ax.text(0.25, ax.get_ylim()[1] * 0.9, "Small", fontsize=7, color="gray")
    ax.text(0.85, ax.get_ylim()[1] * 0.9, "Large", fontsize=7, color="gray")

    ax.set_xlabel("Cohen's d", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_panel_label(ax, "O")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1O_effect_dist.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{PANELS_DIR}/Fig1O_effect_dist.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Save source data
    effect_summary = pd.DataFrame(
        {
            "statistic": [
                "mean",
                "median",
                "std",
                "n_small_pos",
                "n_small_neg",
                "n_large_pos",
                "n_large_neg",
            ],
            "value": [
                cohend_values.mean(),
                cohend_values.median(),
                cohend_values.std(),
                ((cohend_values > 0.2) & (cohend_values <= 0.8)).sum(),
                ((cohend_values < -0.2) & (cohend_values >= -0.8)).sum(),
                (cohend_values > 0.8).sum(),
                (cohend_values < -0.8).sum(),
            ],
        }
    )
    save_sourcedata(effect_summary, "Fig1O_EffectDist")
    print(f"    Saved: Fig1O.png")
    return fig


# =============================================================================
# Panel P: Study Design Schematic
# =============================================================================
def plot_panel_P(metadata, output_dir):
    """Study design schematic showing sample groups"""
    print("  Plotting Panel P: Study design schematic...")

    fig, ax = plt.subplots(figsize=(6, 4))

    # Count samples per group (Group column with Control/Case values)
    young_n = len(metadata[metadata["Group"] == "Control"])
    old_n = len(metadata[metadata["Group"] == "Case"])

    # Create schematic
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(
        5, 5.5, "Study Design", fontsize=14, fontweight="bold", ha="center", va="center"
    )

    # Control group box
    young_box = plt.Rectangle(
        (0.5, 2),
        3.5,
        2.5,
        fill=True,
        facecolor=NPG_COLORS["blue"],
        alpha=0.3,
        edgecolor=NPG_COLORS["blue"],
        linewidth=2,
    )
    ax.add_patch(young_box)
    ax.text(
        2.25,
        4,
        "Control",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color=NPG_COLORS["blue"],
    )
    ax.text(2.25, 3.2, f"n = {young_n}", fontsize=11, ha="center", va="center")
    ax.text(2.25, 2.5, "samples", fontsize=10, ha="center", va="center")

    # Case group box
    old_box = plt.Rectangle(
        (6, 2),
        3.5,
        2.5,
        fill=True,
        facecolor=NPG_COLORS["red"],
        alpha=0.3,
        edgecolor=NPG_COLORS["red"],
        linewidth=2,
    )
    ax.add_patch(old_box)
    ax.text(
        7.75,
        4,
        "Case",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color=NPG_COLORS["red"],
    )
    ax.text(7.75, 3.2, f"n = {old_n}", fontsize=11, ha="center", va="center")
    ax.text(7.75, 2.5, "samples", fontsize=10, ha="center", va="center")

    # Arrow and comparison
    ax.annotate(
        "",
        xy=(5.8, 3.25),
        xytext=(4.2, 3.25),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
    )
    ax.text(
        5, 3.7, "DEG Analysis", fontsize=10, ha="center", va="center", fontweight="bold"
    )

    # Analysis details
    ax.text(
        5,
        1.2,
        f"Welch t-test + BH FDR correction",
        fontsize=9,
        ha="center",
        va="center",
        style="italic",
    )
    ax.text(
        5,
        0.7,
        f"FDR < {FDR_THRESHOLD}, |Log2FC| > {LOG2FC_THRESHOLD:.3f}",
        fontsize=9,
        ha="center",
        va="center",
    )

    add_panel_label(ax, "P")

    plt.tight_layout()
    plt.savefig(f"{PANELS_DIR}/Fig1P_study_design.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{PANELS_DIR}/Fig1P_study_design.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Save source data
    design_data = pd.DataFrame(
        {
            "group": ["Control", "Case"],
            "n_samples": [young_n, old_n],
            "fdr_threshold": [FDR_THRESHOLD, FDR_THRESHOLD],
            "log2fc_threshold": [LOG2FC_THRESHOLD, LOG2FC_THRESHOLD],
        }
    )
    save_sourcedata(design_data, "Fig1P_StudyDesign")
    print(f"    Saved: Fig1P.png")
    return fig


# =============================================================================
# Composite Figure Generation
# =============================================================================
def create_composite_figure(output_dir):
    """Create 4x4 composite figure from individual panels"""
    print("\n[COMPOSITE] Creating 4x4 composite figure...")

    from PIL import Image
    import os

    # Panel layout (4 rows x 4 columns)
    panel_layout = [
        ["A", "B", "C", "D"],
        ["E", "F", "G", "H"],
        ["I", "J", "K", "L"],
        ["M", "N", "O", "P"],
    ]

    # Load all panel images
    panels = {}
    panel_dir = PANELS_DIR  # Use the global PANELS_DIR constant

    for row in panel_layout:
        for panel_id in row:
            # Find the panel file - try multiple naming patterns
            found = False
            for f in os.listdir(panel_dir):
                # Match patterns like Fig1A.png, Fig1A_xxx.png
                if f.startswith(f"Fig1{panel_id}") and f.endswith(".png"):
                    img_path = os.path.join(panel_dir, f)
                    panels[panel_id] = Image.open(img_path)
                    found = True
                    break
            if not found:
                print(f"  WARNING: Panel {panel_id} not found")

    if len(panels) < 16:
        print(f"  WARNING: Only {len(panels)}/16 panels found")
        missing = [p for row in panel_layout for p in row if p not in panels]
        print(f"  Missing panels: {missing}")

    # Determine panel size (use largest dimensions)
    max_width = max(img.width for img in panels.values()) if panels else 1000
    max_height = max(img.height for img in panels.values()) if panels else 800

    # Create composite image
    composite_width = max_width * 4
    composite_height = max_height * 4
    composite = Image.new("RGB", (composite_width, composite_height), "white")

    # Paste panels
    for i, row in enumerate(panel_layout):
        for j, panel_id in enumerate(row):
            if panel_id in panels:
                img = panels[panel_id]
                # Resize to fit cell if needed
                img_resized = img.resize(
                    (max_width, max_height), Image.Resampling.LANCZOS
                )
                x = j * max_width
                y = i * max_height
                composite.paste(img_resized, (x, y))

    # Save composite
    composite.save(f"{COMPOSITE_DIR}/Figure1_composite.png", dpi=(300, 300))
    composite.save(
        f"{COMPOSITE_DIR}/Figure1_composite.tiff",
        dpi=(300, 300),
        compression="tiff_lzw",
    )
    print(f"  Saved: {COMPOSITE_DIR}/Figure1_composite.png")
    print(f"  Saved: {COMPOSITE_DIR}/Figure1_composite.tiff")

    return composite


# =============================================================================
# Audit and Manifest Generation
# =============================================================================
def generate_audit_files(output_dir, deg_df, gene_coords, metadata, lambda_gc):
    """Generate audit trail and manifest files"""
    print("\n[AUDIT] Generating audit files...")

    import datetime
    import hashlib
    import os

    # Calculate checksums for all output files
    checksums = {}
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            filepath = os.path.join(root, f)
            checksums[os.path.relpath(filepath, output_dir)] = calculate_md5(filepath)

    # Create manifest
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(),
        "input_file": DATA_FILE,
        "output_dir": output_dir,
        "top_degs_auto_selected": KEY_TARGETS,
        "thresholds": {"fdr": FDR_THRESHOLD, "log2fc": LOG2FC_THRESHOLD},
        "results_summary": {
            "total_genes": len(deg_df),
            "deg_up": len(deg_df[deg_df["SignificantFlag"] == "Up"]),
            "deg_down": len(deg_df[deg_df["SignificantFlag"] == "Down"]),
            "deg_ns": len(deg_df[deg_df["SignificantFlag"] == "NS"]),
            "genes_with_coords": len(gene_coords) if gene_coords is not None else 0,
            "coord_coverage_pct": len(gene_coords) / len(deg_df) * 100
            if gene_coords is not None and len(deg_df) > 0
            else 0,
            "lambda_gc": lambda_gc,
            "n_samples_young": len(metadata[metadata["Group"] == "Control"]),
            "n_samples_old": len(metadata[metadata["Group"] == "Case"]),
        },
        "file_checksums": checksums,
    }

    # Save manifest as JSON
    with open(f"{MANIFESTS_DIR}/MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {MANIFESTS_DIR}/MANIFEST.json")

    # Create human-readable summary
    summary_lines = [
        "=" * 70,
        "FIGURE 1 GENERATION SUMMARY",
        "=" * 70,
        f"Generated: {manifest['generated_at']}",
        f"Input: {DATA_FILE}",
        f"Output: {output_dir}",
        "",
        "TOP DEGs (auto-selected by |Log2FC|):",
        f"  {KEY_TARGETS}",
        "",
        "THRESHOLDS:",
        f"  FDR < {FDR_THRESHOLD}",
        f"  |Log2FC| > {LOG2FC_THRESHOLD:.4f}",
        "",
        "RESULTS:",
        f"  Total genes analyzed: {manifest['results_summary']['total_genes']:,}",
        f"  Up-regulated DEGs: {manifest['results_summary']['deg_up']:,}",
        f"  Down-regulated DEGs: {manifest['results_summary']['deg_down']:,}",
        f"  Non-significant: {manifest['results_summary']['deg_ns']:,}",
        f"  Genes with coordinates: {manifest['results_summary']['genes_with_coords']:,} ({manifest['results_summary']['coord_coverage_pct']:.1f}%)",
        f"  Genomic inflation (λGC): {manifest['results_summary']['lambda_gc']:.3f}",
        "",
        "SAMPLES:",
        f"  Control: {manifest['results_summary']['n_samples_young']}",
        f"  Case: {manifest['results_summary']['n_samples_old']}",
        "",
        "OUTPUT FILES:",
    ]

    for filepath in sorted(checksums.keys()):
        summary_lines.append(f"  {filepath}")

    summary_lines.extend(["", "=" * 70])

    with open(f"{MANIFESTS_DIR}/SUMMARY.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Saved: {MANIFESTS_DIR}/SUMMARY.txt")

    return manifest


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def copy_source_code(output_code_dir):
    """Copy source code files to output directory for reproducibility."""
    import shutil
    import glob

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Copy all .py files from script directory to output code directory
    py_files = glob.glob(os.path.join(script_dir, "*.py"))
    for py_file in py_files:
        shutil.copy2(py_file, output_code_dir)

    print(f"  Copied {len(py_files)} code files to {output_code_dir}")


def main():
    """Main execution function"""
    # Create output directories
    for subdir in [
        PANELS_DIR,
        COMPOSITE_DIR,
        SOURCEDATA_DIR,
        CODE_DIR,
        RAW_DIR,
        LOGS_DIR,
        MANIFESTS_DIR,
    ]:
        os.makedirs(subdir, exist_ok=True)

    # Copy source code to output directory
    copy_source_code(CODE_DIR)

    print("=" * 70)
    print("FIGURE 1 GENERATION PIPELINE")
    print("=" * 70)
    print(f"Input: {DATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Top DEGs: (will be auto-selected after DEG analysis)")
    print("=" * 70)

    # Step 1: Load data
    print("")
    print("[STEP 1] Loading expression data...")
    expr_df, labels = load_data()
    print(f"  Loaded {len(expr_df)} genes x {len(expr_df.columns)} samples")

    # Step 1b: Detect data preprocessing status
    print("")
    print("[STEP 1b] Detecting data preprocessing status...")
    preprocessing_info = detect_data_preprocessing_status(expr_df, logger)
    print(f"  Data type: {preprocessing_info['data_type']}")
    print(
        f"  Log2 transformed: {'Yes' if preprocessing_info['is_log2'] else 'No/Uncertain'}"
    )
    print(f"  Normalization: {preprocessing_info['normalization_status']}")
    print(
        f"  Statistical method: {preprocessing_info['statistical_method_appropriate']}"
    )

    # Save preprocessing info
    preprocessing_file = f"{OUTPUT_DIR}/data_preprocessing_status.json"
    with open(preprocessing_file, "w") as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"  Saved preprocessing info: {preprocessing_file}")

    # Step 2: Get/validate metadata
    print("")
    print("[STEP 2] Processing metadata...")
    metadata = get_metadata(expr_df, labels)
    print(f"  Control: {len(metadata[metadata['Group'] == 'Control'])} samples")
    print(f"  Case: {len(metadata[metadata['Group'] == 'Case'])} samples")

    # Step 3: DEG analysis
    print("")
    print("[STEP 3] Performing DEG analysis...")
    deg_df = perform_deg_analysis(expr_df, metadata)
    n_up = len(deg_df[deg_df["SignificantFlag"] == "Up"])
    n_down = len(deg_df[deg_df["SignificantFlag"] == "Down"])
    print(f"  DEGs: {n_up} up, {n_down} down")

    # Auto-select Top DEGs for visualization
    top_degs = select_top_degs(deg_df, n_up=5, n_down=5)
    print(f"  Top DEGs selected: {top_degs}")

    # Step 4: Load gene coordinates from GENCODE (ZERO-FAKE)
    print("")
    print("[STEP 4] Loading gene coordinates from GENCODE...")
    gene_coords = get_gene_coordinates(deg_df["Gene"].tolist())
    if gene_coords is not None:
        print(f"  Loaded coordinates for {len(gene_coords)} genes")
    else:
        print(
            "  WARNING: No coordinates available. Panel G and L will show placeholders."
        )
    lambda_gc = 1.0

    # Step 5: Generate panels
    print("")
    print("[STEP 5] Generating panels...")

    # Row 1: A-D
    print("")
    print("  --- Row 1 ---")
    plot_panel_A(expr_df, metadata)
    plot_panel_B(expr_df, metadata)
    plot_panel_C(deg_df)
    plot_panel_D(deg_df)

    # Row 2: E-H
    print("")
    print("  --- Row 2 ---")
    plot_panel_E(deg_df)
    plot_panel_F(expr_df, deg_df, metadata)
    plot_panel_G(deg_df, gene_coords, OUTPUT_DIR)
    fig_h, lambda_gc = plot_panel_H(deg_df, OUTPUT_DIR)

    # Row 3: I-L
    print("")
    print("  --- Row 3 ---")
    plot_panel_I(deg_df, OUTPUT_DIR)
    plot_panel_J(expr_df, metadata, OUTPUT_DIR, deg_df)
    plot_panel_K(deg_df, OUTPUT_DIR)
    plot_panel_L(deg_df, gene_coords, OUTPUT_DIR)

    # Row 4: M-P
    print("")
    print("  --- Row 4 ---")
    plot_panel_M(expr_df, metadata, OUTPUT_DIR)
    plot_panel_N(expr_df, metadata, OUTPUT_DIR)
    plot_panel_O(deg_df, OUTPUT_DIR)
    plot_panel_P(metadata, OUTPUT_DIR)

    # Step 6: Create composite figure
    print("")
    print("[STEP 6] Creating composite figure...")
    create_composite_figure(OUTPUT_DIR)

    # Step 7: Generate audit files
    print("")
    print("[STEP 7] Generating audit files...")
    manifest = generate_audit_files(
        OUTPUT_DIR, deg_df, gene_coords, metadata, lambda_gc
    )

    # Final summary
    print("")
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("")
    print(f"Output directory: {OUTPUT_DIR}")
    print("")
    print("Generated files:")
    print("  - Figure1_composite.png/tiff")
    print("  - 16 individual panels (panels/)")
    print("  - Source data files (sourcedata/)")
    print("  - MANIFEST.json")
    print("  - SUMMARY.txt")
    print("")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Figure 1: DEG Analysis Pipeline")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Override OUTPUT_DIR if provided
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        PANELS_DIR = f"{OUTPUT_DIR}/panels16"
        COMPOSITE_DIR = f"{OUTPUT_DIR}/composite"
        SOURCEDATA_DIR = f"{OUTPUT_DIR}/sourcedata16"
        CODE_DIR = f"{OUTPUT_DIR}/code"
        RAW_DIR = f"{OUTPUT_DIR}/raw"
        LOGS_DIR = f"{OUTPUT_DIR}/logs"
        MANIFESTS_DIR = f"{OUTPUT_DIR}/manifests"

    main()
