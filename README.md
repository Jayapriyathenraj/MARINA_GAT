# Marina-GAT

**Current-Informed Graph Attention Networks for Predicting Marine Species-Richness Patterns on the India East Coast**

Marina-GAT predicts marine species-richness patterns along India's east coast using OBIS biodiversity records, Copernicus Marine oceanographic data, and graph-based learning.

## Repository structure

- `src/` — data ingestion, graph construction, GAT/GCN models, baselines, evaluation, explainability, and visualization modules.
- `notebooks/` — the main Marina-GAT pipeline notebook.
- `results/` — selected result figures, plots, and evaluation summaries.
- `models/` — saved model checkpoint.
- `requirements.txt` — Python dependencies.
- `.gitignore` — files excluded from version control.

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Some notebook cells are designed for Google Colab. PyTorch Geometric installation may depend on the installed PyTorch version and hardware; use its official installation guidance if needed.

## Data access and credentials

The project uses OBIS species-occurrence records and Copernicus Marine oceanographic data. Review the scripts and notebook for data queries, study area, time windows, and access requirements. Copernicus Marine access may require an account. Never commit passwords, API tokens, or personal access tokens.

## Running the project

Open `notebooks/Marina_GAT_pipeline.ipynb` and follow the cells in order after checking the configuration and data-access requirements. Python modules are in `src/`; depending on the environment, you may need to add `src/` to the Python path or run the notebook from the repository root.

## Reproducibility

The `results/` directory contains selected outputs from the supplied project archive. Re-run the workflow to verify results before citing metrics in academic work.
