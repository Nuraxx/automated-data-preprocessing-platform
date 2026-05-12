# AI-Powered Data Preprocessing Web App (Streamlit)

A production-style Streamlit application for uploading CSV datasets and performing automated preprocessing, cleaning, analysis, transformation, and interactive visualization — with AI-style rule-based recommendations.

## Features

- CSV upload (drag & drop supported) with dataset preview and metadata
- Automated profiling: missing values, duplicates, types, unique counts, memory usage
- Data quality score (0–100) + health breakdown
- Cleaning tools:
  - Missing value handling (mean/median/mode/ffill/bfill/drop rows/drop columns)
  - Duplicate detection + removal
  - Automatic datatype detection & intelligent conversions
  - IQR-based outlier detection + optional removal
- Transformations:
  - Categorical encoding (Label / One-Hot)
  - Feature scaling (Standard / MinMax / Robust)
  - Reusable sklearn `Pipeline` + `ColumnTransformer`
- Interactive dashboard (Plotly): histograms, box plots, scatter, correlations, missingness heatmaps, pie charts
- Export:
  - Download cleaned CSV
  - Download transformed (encoded/scaled) dataset
  - Download preprocessing report (JSON/Markdown, and PDF if enabled)

## Project Structure

```
project/
├── app.py
├── preprocessing.py
├── visualization.py
├── utils.py
├── report_generator.py
├── requirements.txt
├── assets/
└── sample_data/
```

## Run Locally

```bash
cd /home/nuraxx/Documents/Data_preprocessing_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Try the Sample Dataset

Use the file in `sample_data/sample_employee_data.csv`.

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On Streamlit Community Cloud, create a new app.
3. Select the repository and set the main file path to `app.py`.
4. Ensure `requirements.txt` is in the repo root.

