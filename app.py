"""Automated Data Preprocessing Web Application (Streamlit).

Run:
  streamlit run app.py

This app is intentionally modular:
- preprocessing.py: core cleaning + pipeline logic
- visualization.py: plotly charts
- report_generator.py: exportable report
- utils.py: helpers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from preprocessing import (
    ColumnTypeGroups,
    apply_missing_value_strategy,
    auto_convert_dtypes,
    build_preprocessing_pipeline,
    compute_quality_score,
    correlated_feature_pairs,
    count_duplicates,
    dataset_memory_usage_bytes,
    detect_column_types,
    drop_correlated_features_greedy,
    fit_transform_to_dataframe,
    generate_ai_recommendations,
    iqr_outlier_summary,
    remove_duplicates,
    remove_outliers_iqr,
    skewness_summary,
    summarize_missing_values,
    unique_values_summary,
)
from report_generator import (
    generate_preprocessing_report,
    report_to_json_bytes,
    report_to_markdown,
    report_to_pdf_bytes,
)
from utils import human_readable_bytes, read_csv_with_fallbacks, utc_now_iso
from visualization import (
    box_plot,
    correlation_heatmap,
    empty_figure,
    histogram,
    missing_value_heatmap,
    null_percentage_bar,
    pie_chart_categorical,
    quality_score_gauge,
    scatter_plot,
    value_distribution,
)

APP_VERSION = "1.0.0"


# -----------------------------
# App helpers
# -----------------------------


def _load_css() -> None:
    """Load optional local CSS tweaks."""

    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "raw_df": None,
        "df": None,
        "file_meta": None,
        "type_conversion_report": {"conversions": []},
        "raw_analysis": None,
        "analysis": None,
        "history": [],
        "pipeline_config": {},
        "transformed_df": None,
        "transformed_info": None,
        "last_error": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_app() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def _log(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    st.session_state.history.append(
        {
            "timestamp": utc_now_iso(),
            "action": action,
            "details": details or {},
        }
    )


def _require_dataset() -> bool:
    if st.session_state.df is None:
        st.info("Upload a CSV dataset to get started.")
        return False
    return True


def _compute_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute and return a full analysis package for a dataset."""

    types = detect_column_types(df)
    missing_summary = summarize_missing_values(df)
    dup_count = count_duplicates(df)
    memory_bytes = dataset_memory_usage_bytes(df)
    uniq_summary = unique_values_summary(df)
    skew_df = skewness_summary(df, types.numerical)
    outlier_summary, outlier_mask = iqr_outlier_summary(df, numerical_cols=types.numerical, multiplier=1.5)
    corr_pairs = correlated_feature_pairs(df, numerical_cols=types.numerical, threshold=0.90)

    quality = compute_quality_score(df, detected_types=types)
    recs = generate_ai_recommendations(df, types)

    return {
        "types": types,
        "missing_summary": missing_summary,
        "duplicates": dup_count,
        "memory_bytes": memory_bytes,
        "unique_summary": uniq_summary,
        "skewness": skew_df,
        "outlier_summary": outlier_summary,
        "outlier_row_pct": float(outlier_mask.mean()) if len(outlier_mask) else 0.0,
        "correlated_pairs": corr_pairs,
        "quality": quality,
        "recommendations": recs,
    }


def _refresh_analysis() -> None:
    if st.session_state.raw_df is not None and st.session_state.raw_analysis is None:
        st.session_state.raw_analysis = _compute_analysis(st.session_state.raw_df)
    if st.session_state.df is not None:
        st.session_state.analysis = _compute_analysis(st.session_state.df)


def _download_button_from_df(label: str, df: pd.DataFrame, file_name: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=file_name, mime="text/csv")


# -----------------------------
# Pages
# -----------------------------


def page_upload() -> None:
    st.title("📁 Upload & Overview")
    st.caption("Upload a CSV to automatically analyze, clean, transform, and export a high-quality dataset.")

    uploaded = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        accept_multiple_files=False,
        help="Drag & drop supported",
    )

    sample_path = Path(__file__).parent / "sample_data" / "sample_employee_data.csv"
    with st.expander("Try the included sample dataset", expanded=False):
        st.write("You can download and upload the sample file to explore all features.")
        if sample_path.exists():
            st.download_button(
                label="Download sample_employee_data.csv",
                data=sample_path.read_bytes(),
                file_name="sample_employee_data.csv",
                mime="text/csv",
            )

    if uploaded is None:
        if st.session_state.df is None:
            st.warning("No dataset loaded yet.")
            return

        # Dataset already loaded in session; show overview without requiring re-upload.
        df = st.session_state.df
        _refresh_analysis()
        analysis = st.session_state.analysis

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Duplicate Rows", f"{analysis['duplicates']:,}")
        size_bytes = int((st.session_state.file_meta or {}).get("bytes", 0))
        c5.metric("File Size", human_readable_bytes(size_bytes))

        st.divider()
        tab1, tab2, tab3 = st.tabs(["Preview", "Columns", "Schema"])
        with tab1:
            st.subheader("First rows")
            st.dataframe(df.head(20), use_container_width=True)
        with tab2:
            st.subheader("Column names")
            st.write(df.columns.tolist())
        with tab3:
            st.subheader("Data types")
            st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values}), use_container_width=True)

        return

    try:
        with st.spinner("Reading CSV and preparing dataset…"):
            raw_df, meta = read_csv_with_fallbacks(uploaded)
            st.session_state.file_meta = meta

            # Keep original untouched
            st.session_state.raw_df = raw_df.copy()

            # Reset analyses so they recompute for the newly uploaded dataset
            st.session_state.raw_analysis = None
            st.session_state.analysis = None

            # Auto-convert obvious dtypes for a better preprocessing experience
            converted_df, report = auto_convert_dtypes(raw_df)
            st.session_state.df = converted_df
            st.session_state.type_conversion_report = report

            st.session_state.transformed_df = None
            st.session_state.transformed_info = None
            st.session_state.pipeline_config = {}

            st.session_state.history = []
            _log("Dataset loaded", {"file": meta.get("file_name"), "encoding": meta.get("encoding")})
            if report.get("conversions"):
                _log("Auto type conversion", {"conversions": report.get("conversions")})

            _refresh_analysis()

        st.success("Dataset loaded successfully.")

    except Exception as e:  # noqa: BLE001
        st.session_state.last_error = str(e)
        st.error(f"Failed to read CSV: {e}")
        return

    df = st.session_state.df
    analysis = st.session_state.analysis

    # Overview cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{analysis['duplicates']:,}")

    size_bytes = int((st.session_state.file_meta or {}).get("bytes", 0))
    c5.metric("File Size", human_readable_bytes(size_bytes))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Preview", "Columns", "Schema"])

    with tab1:
        st.subheader("First rows")
        st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        st.subheader("Column names")
        st.write(df.columns.tolist())

    with tab3:
        st.subheader("Data types")
        st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values}), use_container_width=True)

    conversions = st.session_state.type_conversion_report.get("conversions", [])
    if conversions:
        st.divider()
        st.subheader("Automatic data type conversions")
        st.caption("These conversions are applied to the working copy only (the raw dataset is preserved).")
        st.dataframe(pd.DataFrame(conversions), use_container_width=True)


def page_analysis() -> None:
    st.title("🔎 Dataset Analysis")
    st.caption("Automated dataset profiling, quality scoring, and AI-style preprocessing suggestions.")

    if not _require_dataset():
        return

    with st.spinner("Computing analysis…"):
        _refresh_analysis()

    df = st.session_state.df
    analysis = st.session_state.analysis
    types: ColumnTypeGroups = analysis["types"]
    quality = analysis["quality"]

    # High-level metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", f"{df.shape[1]:,}")
    m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    m4.metric("Duplicate rows", f"{analysis['duplicates']:,}")
    m5.metric("Memory usage", human_readable_bytes(int(analysis["memory_bytes"])))

    st.divider()

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Column types")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Numerical", len(types.numerical))
        t2.metric("Categorical", len(types.categorical))
        t3.metric("Boolean", len(types.boolean))
        t4.metric("Datetime", len(types.datetime))

        with st.expander("See columns by type", expanded=False):
            st.write({
                "numerical": types.numerical,
                "categorical": types.categorical,
                "boolean": types.boolean,
                "datetime": types.datetime,
                "other": types.other,
            })

    with right:
        st.subheader("Dataset health score")
        st.plotly_chart(quality_score_gauge(quality.score), use_container_width=True)
        st.progress(quality.score / 100)

        with st.expander("Score breakdown", expanded=False):
            st.write(
                {
                    "score": quality.score,
                    "missing_penalty": round(quality.missing_penalty, 2),
                    "duplicate_penalty": round(quality.duplicate_penalty, 2),
                    "outlier_penalty": round(quality.outlier_penalty, 2),
                    "dtype_penalty": round(quality.dtype_penalty, 2),
                }
            )

    st.divider()

    # Missing values
    st.subheader("Missing values")
    st.plotly_chart(null_percentage_bar(df), use_container_width=True)
    st.dataframe(analysis["missing_summary"].head(50), use_container_width=True)

    # Duplicates
    st.subheader("Duplicates")
    if analysis["duplicates"] == 0:
        st.success("No duplicate rows detected.")
    else:
        st.warning(f"Duplicate rows detected: {analysis['duplicates']:,}")

    # Outliers
    st.subheader("Outliers (IQR)")
    if analysis["outlier_summary"].empty:
        st.info("No numeric columns available for outlier detection.")
    else:
        st.dataframe(analysis["outlier_summary"].head(30), use_container_width=True)

    # Skewness (bonus)
    st.subheader("Skewness (numeric)")
    if analysis["skewness"].empty:
        st.info("No numeric columns available for skewness analysis.")
    else:
        st.dataframe(analysis["skewness"].head(30), use_container_width=True)

    # Unique values
    st.subheader("Unique values per column")
    st.dataframe(analysis["unique_summary"].head(50), use_container_width=True)

    st.divider()

    # AI recommendations
    st.subheader(" AI-style recommendations")
    recs: List[str] = analysis["recommendations"]
    if not recs:
        st.info("No major issues detected. Your dataset looks healthy.")
    else:
        st.info("\n".join([f"- {r}" for r in recs]))


def page_preprocess() -> None:
    st.title(" Preprocessing & Cleaning")
    st.caption("Apply preprocessing steps and track a reproducible history.")

    if not _require_dataset():
        return

    df = st.session_state.df

    tabs = st.tabs(["Data Types", "Missing Values", "Duplicates", "Outliers", "Pipeline & Transform", "History"])

    # --- Data Types ---
    with tabs[0]:
        st.subheader("Automatic data type detection")
        types = detect_column_types(df)
        st.write(types.to_dict())

        st.subheader("Current schema")
        st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values}), use_container_width=True)

        st.subheader("Type conversion log")
        conversions = st.session_state.type_conversion_report.get("conversions", [])
        if conversions:
            st.dataframe(pd.DataFrame(conversions), use_container_width=True)
        else:
            st.info("No automatic conversions were applied.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Re-run type auto-conversion", type="primary"):
                with st.spinner("Converting dtypes…"):
                    converted, report = auto_convert_dtypes(st.session_state.df)
                    st.session_state.df = converted
                    st.session_state.type_conversion_report = report
                    _log("Auto type conversion", {"conversions": report.get("conversions", [])})
                    _refresh_analysis()
                st.success("Type conversion completed.")

        with c2:
            if st.button("Reset working copy to raw dataset"):
                if st.session_state.raw_df is None:
                    st.warning("Raw dataset is not available in this session.")
                    st.stop()

                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.type_conversion_report = {"conversions": []}
                st.session_state.transformed_df = None
                st.session_state.transformed_info = None
                st.session_state.pipeline_config = {}
                _log("Reset working copy", {})
                _refresh_analysis()
                st.success("Working dataset reset.")

    # --- Missing Values ---
    with tabs[1]:
        st.subheader("Missing value handling")
        missing_summary = summarize_missing_values(df)
        st.dataframe(missing_summary.head(60), use_container_width=True)

        cols_with_missing = missing_summary.loc[missing_summary["missing_count"] > 0, "column"].tolist()
        selected_cols = st.multiselect(
            "Target columns",
            options=df.columns.tolist(),
            default=cols_with_missing[: min(30, len(cols_with_missing))],
            help="Choose which columns the missing-value strategy applies to",
        )

        strategy_label = st.selectbox(
            "Choose a strategy",
            options=[
                "Fill with mean (numeric)",
                "Fill with median (numeric)",
                "Fill with mode (most frequent)",
                "Forward fill",
                "Backward fill",
                "Drop rows",
                "Drop columns",
            ],
        )

        strategy_map = {
            "Fill with mean (numeric)": "mean",
            "Fill with median (numeric)": "median",
            "Fill with mode (most frequent)": "mode",
            "Forward fill": "forward_fill",
            "Backward fill": "backward_fill",
            "Drop rows": "drop_rows",
            "Drop columns": "drop_columns",
        }
        strategy = strategy_map[strategy_label]

        before_missing = int(df.isna().sum().sum())
        before_shape = df.shape

        if st.button("Apply missing value strategy", type="primary"):
            details: Dict[str, Any] = {"applied": False, "strategy": strategy, "columns": selected_cols}
            with st.spinner("Applying missing value strategy…"):
                try:
                    new_df, details = apply_missing_value_strategy(df, strategy=strategy, columns=selected_cols)
                    if details.get("applied"):
                        st.session_state.df = new_df
                        _log("Missing values handled", details)
                        _refresh_analysis()
                except Exception as e:  # noqa: BLE001
                    details = {"applied": False, "reason": str(e), "strategy": strategy, "columns": selected_cols}
                    st.error(f"Failed to apply missing value strategy: {e}")

            if not details.get("applied"):
                st.warning(details.get("reason", "Selected strategy could not be applied."))
                skipped = details.get("skipped_columns")
                if skipped:
                    with st.expander("Columns skipped for this strategy", expanded=False):
                        st.write(skipped)
            else:
                after_missing = int(st.session_state.df.isna().sum().sum())
                after_shape = st.session_state.df.shape

                st.success("Missing value strategy applied.")
                b1, b2, a1, a2 = st.columns(4)
                b1.metric("Before missing", f"{before_missing:,}")
                b2.metric("Before shape", f"{before_shape[0]:,} × {before_shape[1]:,}")
                a1.metric("After missing", f"{after_missing:,}")
                a2.metric("After shape", f"{after_shape[0]:,} × {after_shape[1]:,}")

                skipped = details.get("skipped_columns")
                if skipped:
                    with st.expander("Columns skipped for this strategy", expanded=False):
                        st.write(skipped)

                st.subheader("Before vs After (preview)")
                col_l, col_r = st.columns(2)
                with col_l:
                    st.caption("Before")
                    st.dataframe(df.head(10), use_container_width=True)
                with col_r:
                    st.caption("After")
                    st.dataframe(st.session_state.df.head(10), use_container_width=True)

    # --- Duplicates ---
    with tabs[2]:
        st.subheader("Duplicate rows")
        dup_count = count_duplicates(df)
        st.metric("Duplicate rows detected", f"{dup_count:,}")

        if dup_count == 0:
            st.success("No duplicates detected.")
        else:
            if st.button("Remove duplicates", type="primary"):
                with st.spinner("Removing duplicates…"):
                    new_df, details = remove_duplicates(df)
                    st.session_state.df = new_df
                    _log("Duplicates removed", details)
                    _refresh_analysis()
                st.success(f"Removed {details['removed']:,} duplicate rows.")

    # --- Outliers ---
    with tabs[3]:
        st.subheader("IQR-based outlier detection")
        types = detect_column_types(df)
        if not types.numerical:
            st.info("No numeric columns available.")
        else:
            multiplier = st.slider("IQR multiplier", min_value=0.5, max_value=4.0, value=1.5, step=0.1)
            summary, _ = iqr_outlier_summary(df, numerical_cols=types.numerical, multiplier=multiplier)
            st.dataframe(summary, use_container_width=True)

            st.subheader("Box plot")
            selected_box_col = st.selectbox("Select a numeric column", options=types.numerical)
            st.plotly_chart(box_plot(df, selected_box_col), use_container_width=True)

            st.subheader("Remove outliers")
            selected_cols = st.multiselect("Columns to consider", options=types.numerical, default=types.numerical)
            mode = st.selectbox("Row removal mode", options=["any", "all"], help="Drop rows that are outliers in ANY vs ALL selected columns")

            if st.button("Remove outliers (IQR)", type="primary"):
                with st.spinner("Removing outliers…"):
                    new_df, details = remove_outliers_iqr(df, numerical_cols=selected_cols, multiplier=multiplier, mode=mode)
                    st.session_state.df = new_df
                    _log("Outliers removed", details)
                    _refresh_analysis()
                st.success(f"Removed {details['removed']:,} rows flagged as outliers.")

    # --- Pipeline ---
    with tabs[4]:
        st.subheader("Reusable preprocessing pipeline (sklearn)")
        st.caption("Build a ColumnTransformer + Pipeline based on detected column types.")

        types = detect_column_types(df)

        left, right = st.columns(2)
        with left:
            numeric_imputation = st.selectbox("Numeric imputation (pipeline)", options=["median", "mean"], index=0)
            encoding = st.selectbox("Categorical encoding", options=["One Hot Encoding", "Label Encoding"], index=0)
        with right:
            scaler = st.selectbox("Feature scaling", options=["StandardScaler", "MinMaxScaler", "RobustScaler", "None"], index=0)
            corr_threshold = st.slider("Correlation feature selection threshold (bonus)", 0.70, 0.99, 0.90, 0.01)

        encoding_key = "onehot" if encoding == "One Hot Encoding" else "label"
        scaler_key = {
            "StandardScaler": "standard",
            "MinMaxScaler": "minmax",
            "RobustScaler": "robust",
            "None": "none",
        }[scaler]

        with st.expander("Correlation-based feature selection (bonus)", expanded=False):
            pairs = correlated_feature_pairs(df, numerical_cols=types.numerical, threshold=corr_threshold)
            if pairs.empty:
                st.info("No highly correlated pairs found at the selected threshold.")
            else:
                st.dataframe(pairs, use_container_width=True)
                if st.button("Drop correlated features (greedy)"):
                    new_df, details = drop_correlated_features_greedy(df, numerical_cols=types.numerical, threshold=corr_threshold)
                    st.session_state.df = new_df
                    _log("Correlated features dropped", details)
                    _refresh_analysis()
                    st.success(f"Dropped {len(details['dropped'])} correlated features.")

        if st.button("Build pipeline and transform dataset", type="primary"):
            with st.spinner("Building pipeline and transforming…"):
                try:
                    pipeline = build_preprocessing_pipeline(
                        types,
                        numeric_imputation=numeric_imputation,
                        categorical_encoding=encoding_key,
                        scaling=scaler_key,
                    )

                    transformed_df, info = fit_transform_to_dataframe(df, pipeline)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Failed to build/transform pipeline: {e}")
                else:
                    st.session_state.transformed_df = transformed_df
                    st.session_state.transformed_info = info
                    st.session_state.pipeline_config = {
                        "numeric_imputation": numeric_imputation,
                        "categorical_encoding": encoding_key,
                        "scaling": scaler_key,
                        "detected_types": types.to_dict(),
                        "estimator": "sklearn.Pipeline(ColumnTransformer)",
                    }
                    _log("Pipeline generated", st.session_state.pipeline_config)

                    st.success("Transformed dataset generated.")

        if st.session_state.transformed_df is not None:
            tdf = st.session_state.transformed_df
            st.subheader("Transformed dataset preview")
            st.write(f"Shape: {tdf.shape[0]:,} rows × {tdf.shape[1]:,} columns")
            st.dataframe(tdf.head(20), use_container_width=True)

    # --- History ---
    with tabs[5]:
        st.subheader("Preprocessing history")
        if not st.session_state.history:
            st.info("No steps applied yet.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)


def page_dashboard() -> None:
    st.title("Visualization Dashboard")
    st.caption("Interactive charts powered by Plotly.")

    if not _require_dataset():
        return

    df = st.session_state.df
    analysis = st.session_state.analysis or _compute_analysis(df)
    types: ColumnTypeGroups = analysis["types"]

    tabs = st.tabs(["Distributions", "Box Plots", "Correlation", "Missingness", "Scatter", "Categorical"])

    with tabs[0]:
        st.subheader("Value distributions")
        col = st.selectbox("Choose a column", options=df.columns.tolist())
        st.plotly_chart(value_distribution(df, col), use_container_width=True)

    with tabs[1]:
        st.subheader("Box plots")
        if not types.numerical:
            st.info("No numerical columns available.")
        else:
            col = st.selectbox("Choose a numerical column", options=types.numerical, key="box_col")
            st.plotly_chart(box_plot(df, col), use_container_width=True)

    with tabs[2]:
        st.subheader("Correlation heatmap")
        if len(types.numerical) < 2:
            st.info("Need at least two numerical columns.")
        else:
            st.plotly_chart(correlation_heatmap(df, types.numerical), use_container_width=True)

    with tabs[3]:
        st.subheader("Missingness")
        st.plotly_chart(null_percentage_bar(df), use_container_width=True)
        st.plotly_chart(missing_value_heatmap(df), use_container_width=True)

    with tabs[4]:
        st.subheader("Scatter plots")
        if len(types.numerical) < 2:
            st.info("Need at least two numerical columns.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                x = st.selectbox("X", options=types.numerical, index=0)
            with c2:
                y = st.selectbox("Y", options=types.numerical, index=1)
            with c3:
                color = st.selectbox("Color (optional)", options=["(none)"] + types.categorical + types.boolean + types.datetime)
                color = None if color == "(none)" else color

            st.plotly_chart(scatter_plot(df, x=x, y=y, color=color), use_container_width=True)

    with tabs[5]:
        st.subheader("Categorical insights")
        if not types.categorical:
            st.info("No categorical columns available.")
        else:
            col = st.selectbox("Choose a categorical column", options=types.categorical, key="pie_col")
            st.plotly_chart(pie_chart_categorical(df, col), use_container_width=True)


def page_export() -> None:
    st.title("⬇️ Export & Report")
    st.caption("Download cleaned data and export a detailed preprocessing report.")

    if not _require_dataset():
        return

    cleaned_df: pd.DataFrame = st.session_state.df
    raw_df: pd.DataFrame = st.session_state.raw_df if st.session_state.raw_df is not None else cleaned_df

    with st.spinner("Refreshing analysis for export…"):
        _refresh_analysis()

    raw_analysis = st.session_state.raw_analysis
    analysis = st.session_state.analysis

    # Fallback safety: if raw analysis isn't available, reuse current analysis.
    if raw_analysis is None:
        raw_analysis = analysis

    st.subheader("Dataset comparison (Before vs After)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before (Raw)**")
        st.metric("Rows", f"{len(raw_df):,}")
        st.metric("Columns", f"{raw_df.shape[1]:,}")
        st.metric("Missing cells", f"{int(raw_df.isna().sum().sum()):,}")
        st.metric("Duplicate rows", f"{count_duplicates(raw_df):,}")
        st.plotly_chart(
            quality_score_gauge(raw_analysis["quality"].score),
            use_container_width=True,
            key="export_quality_raw_gauge",
        )

    with c2:
        st.markdown("**After (Cleaned)**")
        st.metric("Rows", f"{len(cleaned_df):,}")
        st.metric("Columns", f"{cleaned_df.shape[1]:,}")
        st.metric("Missing cells", f"{int(cleaned_df.isna().sum().sum()):,}")
        st.metric("Duplicate rows", f"{analysis['duplicates']:,}")
        st.plotly_chart(
            quality_score_gauge(analysis["quality"].score),
            use_container_width=True,
            key="export_quality_cleaned_gauge",
        )

    st.divider()

    st.subheader("Download cleaned dataset")
    _download_button_from_df("Download cleaned CSV", cleaned_df, file_name="cleaned_dataset.csv")

    if st.session_state.transformed_df is not None:
        st.subheader("Download transformed dataset (encoded/scaled)")
        _download_button_from_df("Download transformed CSV", st.session_state.transformed_df, file_name="transformed_dataset.csv")

    st.divider()

    st.subheader("Preprocessing report")
    st.caption("Exports a summary of the dataset and transformations applied.")

    generate_pdf = st.checkbox("Also generate PDF (bonus)", value=True)

    if st.button("Generate report", type="primary"):
        with st.spinner("Generating report…"):
            types = analysis["types"].to_dict()
            report = generate_preprocessing_report(
                generated_at=utc_now_iso(),
                app_version=APP_VERSION,
                raw_df=raw_df,
                cleaned_df=cleaned_df,
                raw_quality=raw_analysis["quality"].to_dict(),
                cleaned_quality=analysis["quality"].to_dict(),
                column_types=types,
                transformations=st.session_state.history,
                pipeline_config=st.session_state.pipeline_config,
                recommendations=analysis["recommendations"],
                transformed_dataset_info=st.session_state.transformed_info,
            )

            md = report_to_markdown(report)
            st.session_state["latest_report"] = report
            st.session_state["latest_report_md"] = md

            st.success("Report generated.")

    if "latest_report" in st.session_state:
        report = st.session_state["latest_report"]
        md = st.session_state.get("latest_report_md", "")

        st.markdown("### Report preview")
        st.markdown(md)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download report (JSON)", data=report_to_json_bytes(report), file_name="preprocessing_report.json", mime="application/json")
        with c2:
            st.download_button("Download report (Markdown)", data=md.encode("utf-8"), file_name="preprocessing_report.md", mime="text/markdown")
        with c3:
            if generate_pdf:
                st.download_button("Download report (PDF)", data=report_to_pdf_bytes(report), file_name="preprocessing_report.pdf", mime="application/pdf")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    st.set_page_config(
        page_title="Auto Data Preprocessing Studio",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()
    _load_css()

    with st.sidebar:
        st.title("Auto-Preprocess")
        st.caption("Automated preprocessing • Visual analytics • Exportable reports")

        page = st.radio(
            "Navigation",
            options=[
                "Upload & Overview",
                "Analysis",
                "Preprocess",
                "Dashboard",
                "⬇Export",
            ],
        )

        with st.expander("History tracker", expanded=False):
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history).tail(20), use_container_width=True, height=220)
            else:
                st.write("No actions yet")

        st.divider()
        if st.button("Reset app", help="Clears the session state"):
            _reset_app()
            st.rerun()

    # Route
    if page == "Upload & Overview":
        page_upload()
    elif page == "Analysis":
        page_analysis()
    elif page == "Preprocess":
        page_preprocess()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "⬇Export":
        page_export()
    else:
        st.plotly_chart(empty_figure("Unknown page"))


if __name__ == "__main__":
    main()
