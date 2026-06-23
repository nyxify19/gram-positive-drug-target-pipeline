<p align="center">
  <h1 align="center">🧬 Drug Target Discovery Pipeline</h1>
  <p align="center">
    <strong>A multi-evidence computational pipeline for identifying and prioritising antibacterial drug targets in Gram-positive bacteria.</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#usage">Usage</a> •
    <a href="#project-structure">Structure</a> •
    <a href="#outputs">Outputs</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## Overview

This pipeline integrates **proteomics**, **structural biology**, **comparative genomics**, and **machine learning** to systematically discover and rank candidate drug targets from bacterial proteomes. It was developed for Gram-positive organisms but is configurable for any bacterial taxonomy.

Starting from a set of NCBI taxonomy IDs, the pipeline:

1. Retrieves the reviewed proteome from **UniProt**
2. Computes physicochemical and conservation features
3. Fetches **AlphaFold** structural confidence scores
4. Performs **subtractive genomics** against the human proteome (MMseqs2)
5. Scores gene **essentiality** from curated databases or heuristics
6. Predicts **binding-pocket druggability** (P2Rank / fpocket)
7. Trains a calibrated **XGBoost** druggability classifier
8. Produces a **weighted composite ranking** with Monte Carlo sensitivity analysis
9. Generates **publication-quality diagnostic plots**

---

## Features

| Stage | Method | Fallback |
|-------|--------|----------|
| Proteome retrieval | UniProt REST API with pagination + disk cache | — |
| Structural confidence | AlphaFold pLDDT (parallelised, SQLite-cached) | Median imputation |
| Host selectivity | MMseqs2 vs. human reference proteome | Neutral score (0.5) |
| Gene essentiality | DEG table lookup | Keyword heuristic |
| Pocket druggability | P2Rank → fpocket | Gaussian length proxy |
| ML classifier | XGBoost with `scale_pos_weight` | Random Forest (`class_weight="balanced"`) |
| Calibration | Isotonic (≥50 positives) / Sigmoid | — |
| Sensitivity analysis | Dirichlet-perturbed Monte Carlo (1000 samples) | — |

### Design Highlights

- **Graceful degradation** — every external tool (MMseqs2, P2Rank, fpocket) is optional; the pipeline warns and falls back automatically
- **Disk caching** — AlphaFold and pocket results are SQLite-cached; re-runs skip already-fetched data
- **Reproducibility** — a JSON manifest captures exact versions, config, and metrics for every run
- **5-tier positive labels** — DrugBank hits, PDB+enzyme, antibiotic keywords, virulence terms, and functional text evidence

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- (Optional) [MMseqs2](https://github.com/soedinglab/MMseqs2), [P2Rank](https://github.com/rdk/p2rank), [fpocket](https://github.com/Discngine/fpocket)

### Installation

```bash
# Clone the repository
git clone https://github.com/nyxify19/gram-positive-drug-target-pipeline.git
cd gram-positive-drug-target-pipeline

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as a package (includes xgboost)
pip install -e ".[boost]"
```

### Install optional bioinformatics tools

```bash
conda install -c bioconda mmseqs2 p2rank
conda install -c conda-forge fpocket
```

> **Note:** On Windows, run the pipeline from WSL2 so it can access Linux builds of these tools.

---

## Usage

```bash
# Run with defaults (Firmicutes + Actinobacteria)
python run_pipeline.py

# Target a specific organism
python run_pipeline.py --taxa 1313 --outdir results_strep/

# Supply curated essentiality data
python run_pipeline.py --deg-file deg_essential.tsv

# See all options
python run_pipeline.py --help
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--outdir` | `drugtarget_out` | Output directory |
| `--taxa` | `1239,201174` | Comma-separated NCBI taxonomy IDs |
| `--deg-file` | `deg_essential.tsv` | Path to DEG essentiality table |
| `--no-pocket` | off | Disable pocket detection entirely |
| `--hard-host-gate` | off | Remove (not just penalise) human homologs |
| `--host-identity` | `35.0` | Identity % threshold for host homolog call |
| `--af-workers` | `8` | Parallel threads for AlphaFold fetching |
| `--refresh-af` | off | Re-fetch previously failed AlphaFold entries |
| `--pocket-max` | `0` (all) | Cap number of structures for pocket analysis |
| `--tier-mode` | `percentile` | `percentile` (cohort-relative) or `absolute` (fixed cut-offs) |
| `--tier1-pct` | `0.05` | Top fraction → Tier 1 (percentile mode) |
| `--tier2-pct` | `0.20` | Cumulative fraction → Tier 2 (percentile mode) |
| `--seed` | `42` | Random seed for full reproducibility |

---

## Project Structure

```
gram-positive-drug-target-pipeline/
│
├── run_pipeline.py              # Entry point
├── pipeline/                    # Core package
│   ├── __init__.py              #   Package metadata + public API
│   ├── config.py                #   Constants, weights, Config dataclass
│   ├── utils.py                 #   HTTP sessions, SQLite, parallel fetching
│   ├── proteome.py              #   UniProt retrieval, parsing, cleaning
│   ├── features.py              #   Annotations, physicochemistry, conservation
│   ├── alphafold.py             #   AlphaFold pLDDT confidence features
│   ├── selectivity.py           #   Host off-target selectivity (MMseqs2)
│   ├── essentiality.py          #   Gene essentiality scoring
│   ├── pockets.py               #   P2Rank / fpocket pocket detection
│   ├── classifier.py            #   ML druggability model (XGBoost / RF)
│   ├── scoring.py               #   Composite scores, tiers, Monte Carlo
│   ├── visualization.py         #   Diagnostic plots + QC reporting
│   ├── manifest.py              #   Reproducibility JSON manifest
│   └── cli.py                   #   Argument parsing + pipeline orchestrator
│
├── drug_target_pipeline.py      # Legacy monolithic script
├── pyproject.toml               # Packaging and tool configuration
├── requirements.txt             # Python dependencies
├── LICENSE                      # Apache License 2.0
└── .gitignore
```

---

## Outputs

All outputs are written to `--outdir` (default `drugtarget_out/`):

### Data

| File | Description |
|------|-------------|
| `grampos_final_results.csv` | Full scored and ranked protein table with all features |
| `run_manifest.json` | Reproducibility manifest — config, versions, metrics, feature availability |

### Visualizations

| Plot | What it shows |
|------|---------------|
| `model_evaluation.png` | Confusion matrix + permutation feature importances |
| `ml_performance_curves.png` | ROC curve and Precision-Recall curve |
| `feature_correlation.png` | Spearman correlation heatmap of scoring features |
| `tier_distribution.png` | Composite score histogram coloured by priority tier |
| `radar_top_candidates.png` | Radar charts comparing top candidates across all axes |
| `top_targets.png` | Horizontal bar chart of the top 20 ranked targets |
| `selectivity_vs_score.png` | Host selectivity vs. composite score scatter plot |
| `monte_carlo.png` | Ranking stability under Dirichlet weight perturbation |
| `pipeline_funnel.png` | Subtractive genomics funnel chart (LinkedIn friendly) |
| `proteome_landscape.png` | 2D t-SNE projection of the proteome highlighting top targets |

---

## Composite Scoring

Targets are ranked by a weighted composite score combining seven evidence axes:

| Component | Weight | Source |
|-----------|--------|--------|
| ML druggability probability | 30% | XGBoost classifier |
| Cross-strain conservation | 18% | Gene-name ortholog grouping |
| Host selectivity | 18% | MMseqs2 vs. human proteome |
| Gene essentiality | 12% | DEG table or keyword heuristic |
| Pocket druggability | 12% | P2Rank / fpocket / length proxy |
| AlphaFold structural confidence | 7% | Mean pLDDT score |
| Functional annotation present | 3% | UniProt function field |

Ranking stability is validated via **Monte Carlo simulation** (1000 Dirichlet-perturbed weight samples), producing per-protein `rank_std` and `tier_stability` metrics.

---

## Acknowledgements & References

This pipeline relies on the following excellent tools, databases, and libraries.

### External Tools (Software)
| Tool | Usage in Pipeline | Citation |
|------|-------------------|----------|
| **MMseqs2** | Host homology search (`easy-search` command) | Steinegger & Söding, *Nature Biotechnology* 35, 1026–1028 (2017) · [github.com/soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) |
| **P2Rank** | Binding-pocket detection from AlphaFold PDB models (preferred) | Krivák & Hoksza, *J Cheminform* 10, 39 (2018) · [github.com/rdk/p2rank](https://github.com/rdk/p2rank) |
| **fpocket** | Binding-pocket detection (fallback to P2Rank) | Le Guilloux et al., *BMC Bioinformatics* 10, 168 (2009) · [github.com/Discngine/fpocket](https://github.com/Discngine/fpocket) |

### Databases & APIs (Data Sources)
| Resource | Usage in Pipeline | Citation |
|----------|-------------------|----------|
| **UniProtKB / Swiss-Prot** | Proteome download via REST API; sequence, function, keywords, localization, DrugBank cross-refs | The UniProt Consortium, *Nucleic Acids Research* 2023 |
| **AlphaFold EBI API** | Per-residue pLDDT confidence scores | Varadi et al., *Nucleic Acids Research* 50, D439–D444 (2022) |
| **AlphaFold PDB models** | Structural models downloaded for pocket detection | Jumper et al., *Nature* 596, 583–589 (2021) |
| **DrugBank** | Cross-references used as Tier 1 positive training labels | Wishart et al., *Nucleic Acids Research* 2018 |
| **RCSB PDB** | `xref_pdb` field used as Tier 4 labelling evidence | Berman et al., *Nucleic Acids Research* 28, 235–242 (2000) |
| **DEG** | Database of Essential Genes for optional `--deg-file` input | Zhang & Lin, *Nucleic Acids Research* 37, D455–D458 (2009) |
| **Human Reference Proteome** | UP000005640 downloaded for host-homology search | The UniProt Consortium, *Nucleic Acids Research* 2023 |

### Python Libraries
| Library | Usage in Pipeline | Citation |
|---------|-------------------|----------|
| **Biopython** | `ProteinAnalysis` (MW, pI, GRAVY, aromaticity, instability) | Cock et al., *Bioinformatics* 25, 1422–1423 (2009) |
| **scikit-learn** | Random Forest, Calibration, CV, metrics, and importances | Pedregosa et al., *JMLR* 12, 2825–2830 (2011) |
| **XGBoost** | Preferred classifier for extreme class imbalance | Chen & Guestrin, *KDD* 2016 |
| **NumPy** | Array operations, Monte Carlo Dirichlet sampling | Harris et al., *Nature* 585, 357–362 (2020) |
| **pandas** | Core data manipulation | The Pandas Development Team, Zenodo |
| **Matplotlib** | Diagnostic visualizations | Hunter, *Computing in Science & Engineering* 9, 90–95 (2007) |

### Methodology Concepts
| Concept | Usage in Pipeline | Reference |
|---------|-------------------|-----------|
| **Subtractive Genomics** | Entire host-selectivity stage filters out human homologs | e.g. Sarangi et al. / foundational subtractive genomics papers |
| **pLDDT as confidence** | Thresholds 70 (confident) and 50 (disordered) | Jumper et al., *Nature* 596, 583–589 (2021) |
| **Dirichlet perturbation** | Monte Carlo sensitivity analysis | Applied Dirichlet sensitivity method |

---

## Citation

If you use this pipeline in your research, please cite it as:

```
Drug Target Discovery Pipeline (2025).
https://github.com/nyxify19/gram-positive-drug-target-pipeline
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
