"""Core preprocessing and profiling logic.

This module contains:
- Dataset profiling (missing values, duplicates, memory, unique counts)
- Automatic type detection + intelligent conversions
- Missing value / duplicate / outlier handling
- Sklearn preprocessing pipeline builder (Pipeline + ColumnTransformer)
- Rule-based "AI" recommendations

The Streamlit UI in `app.py` calls these functions and stores results in
`st.session_state`.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


# -----------------------------
# Profiling helpers
# -----------------------------


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing count and percentage."""

    if df.empty:
        return pd.DataFrame(columns=["column", "dtype", "missing_count", "missing_pct"])  # pragma: no cover

    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    summary = (
        pd.DataFrame(
            {
                "column": missing_count.index,
                "dtype": df.dtypes.astype(str).values,
                "missing_count": missing_count.values,
                "missing_pct": missing_pct.values,
            }
        )
        .sort_values(["missing_count", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return summary


def count_duplicates(df: pd.DataFrame) -> int:
    """Count duplicate rows."""

    if df.empty:
        return 0
    return int(df.duplicated().sum())


def dataset_memory_usage_bytes(df: pd.DataFrame) -> int:
    """Estimate DataFrame memory usage in bytes (deep)."""

    return int(df.memory_usage(deep=True).sum()) if not df.empty else 0


def unique_values_summary(df: pd.DataFrame, preview_max_items: int = 12) -> pd.DataFrame:
    """Return unique counts per column with a small preview of values."""

    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        sample_vals = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        preview = sample_vals[:preview_max_items]
        preview_str = ", ".join(preview)
        if nunique > preview_max_items:
            preview_str = f"{preview_str} … (+{nunique - preview_max_items} more)"

        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "unique_values": nunique,
                "preview": preview_str,
            }
        )

    return pd.DataFrame(rows).sort_values(["unique_values", "column"], ascending=[False, True]).reset_index(drop=True)


# -----------------------------
# Type detection + conversions
# -----------------------------


@dataclass(frozen=True)
class ColumnTypeGroups:
    numerical: List[str]
    categorical: List[str]
    boolean: List[str]
    datetime: List[str]
    other: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return asdict(self)


def _is_boolean_like(series: pd.Series) -> bool:
    if series.empty:
        return False

    # Consider only non-null values
    s = series.dropna().astype(str).str.strip().str.lower()
    if s.empty:
        return False

    allowed = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}
    ratio = (s.isin(list(allowed))).mean()
    return ratio >= 0.95


def _to_boolean(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False,
        "t": True,
        "f": False,
    }
    out = s.map(mapping)
    # Keep NaNs where mapping failed
    return out.astype("boolean")


def _try_to_numeric(series: pd.Series, threshold: float = 0.90) -> Tuple[Optional[pd.Series], float]:
    """Attempt numeric conversion. Returns (converted, parse_ratio)."""

    non_null = series.dropna()
    if non_null.empty:
        return None, 0.0

    converted = pd.to_numeric(series, errors="coerce")
    parse_ratio = float(converted.notna().sum() / max(int(series.notna().sum()), 1))

    if parse_ratio >= threshold:
        return converted, parse_ratio

    return None, parse_ratio


def _try_to_datetime(series: pd.Series, threshold: float = 0.90) -> Tuple[Optional[pd.Series], float]:
    """Attempt datetime conversion. Returns (converted, parse_ratio)."""

    non_null = series.dropna()
    if non_null.empty:
        return None, 0.0

    # Pandas may warn when it cannot infer a single datetime format.
    # For our heuristic detection, this is expected and safe to ignore.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Could not infer format.*", category=UserWarning)
        converted = pd.to_datetime(series, errors="coerce", utc=False)
    parse_ratio = float(converted.notna().sum() / max(int(series.notna().sum()), 1))

    if parse_ratio >= threshold:
        return converted, parse_ratio

    return None, parse_ratio


def auto_convert_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convert obvious mis-typed columns (object -> bool/number/datetime).

    This function is conservative: it only converts a column if conversion
    succeeds for at least `threshold` fraction of non-null values.

    Returns:
        converted_df, report

    report contains a list under `conversions` with column-level details.
    """

    converted_df = df.copy()
    conversions: List[Dict[str, Any]] = []

    for col in converted_df.columns:
        s = converted_df[col]
        before = str(s.dtype)

        # Skip if already a strong dtype
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue

        # Consider only object/string-like
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        # 1) Boolean-like
        if _is_boolean_like(s):
            converted_df[col] = _to_boolean(s)
            conversions.append(
                {
                    "column": col,
                    "from": before,
                    "to": str(converted_df[col].dtype),
                    "method": "boolean_map",
                    "parse_ratio": 1.0,
                }
            )
            continue

        # 2) Numeric-like
        numeric, numeric_ratio = _try_to_numeric(s)
        if numeric is not None:
            converted_df[col] = numeric
            conversions.append(
                {
                    "column": col,
                    "from": before,
                    "to": str(converted_df[col].dtype),
                    "method": "to_numeric",
                    "parse_ratio": numeric_ratio,
                }
            )
            continue

        # 3) Datetime-like
        dt, dt_ratio = _try_to_datetime(s)
        if dt is not None:
            converted_df[col] = dt
            conversions.append(
                {
                    "column": col,
                    "from": before,
                    "to": str(converted_df[col].dtype),
                    "method": "to_datetime",
                    "parse_ratio": dt_ratio,
                }
            )
            continue

    report = {"conversions": conversions}
    return converted_df, report


def detect_column_types(df: pd.DataFrame) -> ColumnTypeGroups:
    """Detect high-level column groups for preprocessing."""

    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    boolean_cols = df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    used = set(datetime_cols) | set(boolean_cols) | set(numerical_cols)

    categorical_cols = [
        c
        for c in df.columns
        if c not in used
        and (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
        )
    ]

    used |= set(categorical_cols)
    other_cols = [c for c in df.columns if c not in used]

    return ColumnTypeGroups(
        numerical=numerical_cols,
        categorical=categorical_cols,
        boolean=boolean_cols,
        datetime=datetime_cols,
        other=other_cols,
    )


def skewness_summary(df: pd.DataFrame, numerical_cols: Sequence[str]) -> pd.DataFrame:
    """Compute skewness for numeric columns."""

    rows: List[Dict[str, Any]] = []
    for col in numerical_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            continue
        skew = float(s.skew())
        rows.append({"column": col, "skewness": skew, "abs_skewness": abs(skew)})

    if not rows:
        return pd.DataFrame(columns=["column", "skewness", "abs_skewness"])

    return pd.DataFrame(rows).sort_values("abs_skewness", ascending=False).reset_index(drop=True)


# -----------------------------
# Missing values + duplicates
# -----------------------------


def apply_missing_value_strategy(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply a missing value handling strategy.

    Supported strategies:
        - mean
        - median
        - mode
        - forward_fill
        - backward_fill
        - drop_rows
        - drop_columns

    Notes:
        - mean/median only apply to numerical columns in `columns`.
        - mode applies to all column types.
    """

    if strategy not in {
        "mean",
        "median",
        "mode",
        "forward_fill",
        "backward_fill",
        "drop_rows",
        "drop_columns",
    }:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")

    working = df.copy()
    target_cols = list(columns) if columns is not None and len(columns) > 0 else working.columns.tolist()

    before_shape = working.shape
    before_missing_total = int(working.isna().sum().sum())

    applied_details: Dict[str, Any] = {"strategy": strategy, "columns": target_cols}

    if strategy in {"mean", "median"}:
        # Pandas treats boolean as numeric, but mean/median imputation
        # with a float fill value will fail for extension dtype 'boolean'.
        numeric_cols: List[str] = []
        skipped_cols: List[str] = []
        for c in target_cols:
            if pd.api.types.is_bool_dtype(working[c]):
                skipped_cols.append(c)
                continue
            if pd.api.types.is_numeric_dtype(working[c]):
                numeric_cols.append(c)
            else:
                skipped_cols.append(c)

        if not numeric_cols:
            return working, {
                "applied": False,
                "reason": "Mean/median imputation applies to numeric (non-boolean) columns only.",
                "skipped_columns": skipped_cols,
                **applied_details,
            }

        for col in numeric_cols:
            if working[col].isna().any():
                series_numeric = pd.to_numeric(working[col], errors="coerce").astype("float64")
                fill_val = float(series_numeric.median() if strategy == "median" else series_numeric.mean())
                working[col] = series_numeric.fillna(fill_val)

        applied_details["numeric_columns"] = numeric_cols
        if skipped_cols:
            applied_details["skipped_columns"] = skipped_cols

    elif strategy == "mode":
        for col in target_cols:
            if working[col].isna().any():
                modes = working[col].mode(dropna=True)
                if not modes.empty:
                    working[col] = working[col].fillna(modes.iloc[0])

    elif strategy == "forward_fill":
        working[target_cols] = working[target_cols].ffill()

    elif strategy == "backward_fill":
        working[target_cols] = working[target_cols].bfill()

    elif strategy == "drop_rows":
        working = working.dropna(subset=target_cols, how="any")

    elif strategy == "drop_columns":
        cols_to_drop = [c for c in target_cols if working[c].isna().any()]
        working = working.drop(columns=cols_to_drop)
        applied_details["dropped_columns"] = cols_to_drop

    after_shape = working.shape
    after_missing_total = int(working.isna().sum().sum())

    return working, {
        "applied": True,
        **applied_details,
        "before_shape": before_shape,
        "after_shape": after_shape,
        "missing_total_before": before_missing_total,
        "missing_total_after": after_missing_total,
    }


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop duplicate rows and report how many were removed."""

    before = len(df)
    working = df.drop_duplicates().copy()
    removed = before - len(working)
    return working, {"removed": int(removed), "before_rows": int(before), "after_rows": int(len(working))}


# -----------------------------
# Outliers (IQR)
# -----------------------------


def iqr_outlier_summary(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute IQR outlier counts per numerical column.

    Returns:
        (summary_df, row_mask)

    row_mask is True for rows that contain an outlier in ANY numerical column.
    """

    if df.empty or len(numerical_cols) == 0:
        return (
            pd.DataFrame(columns=["column", "outliers", "outlier_pct", "lower", "upper"]),
            pd.Series(False, index=df.index),
        )

    per_col_rows: List[Dict[str, Any]] = []
    any_outlier_mask = pd.Series(False, index=df.index)

    for col in numerical_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            lower = q1
            upper = q3
        else:
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

        mask = (s < lower) | (s > upper)
        outliers = int(mask.sum())
        outlier_pct = float(outliers / len(df) * 100)

        any_outlier_mask = any_outlier_mask | mask.fillna(False)

        per_col_rows.append(
            {
                "column": col,
                "outliers": outliers,
                "outlier_pct": outlier_pct,
                "lower": lower,
                "upper": upper,
            }
        )

    summary = pd.DataFrame(per_col_rows).sort_values(["outliers", "column"], ascending=[False, True]).reset_index(drop=True)
    return summary, any_outlier_mask


def remove_outliers_iqr(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    multiplier: float = 1.5,
    mode: str = "any",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove outliers using IQR rules.

    mode:
        - "any": drop rows that are outliers in ANY selected numeric column
        - "all": drop rows that are outliers in ALL selected numeric columns
    """

    if mode not in {"any", "all"}:
        raise ValueError("mode must be 'any' or 'all'")

    summary, _ = iqr_outlier_summary(df, numerical_cols=numerical_cols, multiplier=multiplier)
    if summary.empty:
        return df.copy(), {"removed": 0, "before_rows": len(df), "after_rows": len(df), "mode": mode}

    masks: List[pd.Series] = []
    for col in numerical_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        masks.append(((s < lower) | (s > upper)).fillna(False))

    if not masks:
        return df.copy(), {"removed": 0, "before_rows": len(df), "after_rows": len(df), "mode": mode}

    if mode == "any":
        outlier_rows = masks[0].copy()
        for m in masks[1:]:
            outlier_rows = outlier_rows | m
    else:
        outlier_rows = masks[0].copy()
        for m in masks[1:]:
            outlier_rows = outlier_rows & m

    before_rows = len(df)
    cleaned = df.loc[~outlier_rows].copy()
    removed = before_rows - len(cleaned)

    return cleaned, {
        "removed": int(removed),
        "before_rows": int(before_rows),
        "after_rows": int(len(cleaned)),
        "mode": mode,
        "multiplier": float(multiplier),
        "columns": list(numerical_cols),
    }


# -----------------------------
# Correlation-based feature selection (bonus)
# -----------------------------


def correlated_feature_pairs(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    threshold: float = 0.90,
) -> pd.DataFrame:
    """Return highly-correlated numeric feature pairs above a threshold."""

    if len(numerical_cols) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    corr = df[list(numerical_cols)].corr(numeric_only=True)
    pairs: List[Dict[str, Any]] = []

    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr.iloc[i, j])
            if abs(val) >= threshold:
                pairs.append({"feature_1": cols[i], "feature_2": cols[j], "correlation": val})

    if not pairs:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    return pd.DataFrame(pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def drop_correlated_features_greedy(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    threshold: float = 0.90,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Greedily drop one feature from each highly-correlated pair."""

    pairs = correlated_feature_pairs(df, numerical_cols=numerical_cols, threshold=threshold)
    if pairs.empty:
        return df.copy(), {"dropped": [], "threshold": threshold, "pairs": []}

    dropped: List[str] = []
    present = set(df.columns)

    for _, row in pairs.iterrows():
        a = str(row["feature_1"])
        b = str(row["feature_2"])
        if a in present and b in present:
            # Drop the second feature by default.
            dropped.append(b)
            present.remove(b)

    cleaned = df.drop(columns=sorted(set(dropped)), errors="ignore").copy()

    return cleaned, {"dropped": sorted(set(dropped)), "threshold": threshold, "pairs": pairs.to_dict("records")}


# -----------------------------
# Data quality score + AI recommendations
# -----------------------------


@dataclass(frozen=True)
class QualityBreakdown:
    score: int
    missing_penalty: float
    duplicate_penalty: float
    outlier_penalty: float
    dtype_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _dtype_inconsistency_ratio(df: pd.DataFrame) -> float:
    """Heuristic: object columns with mixed numeric/text are considered inconsistent."""

    if df.empty:
        return 0.0

    object_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    if not object_cols:
        return 0.0

    inconsistent = 0
    for col in object_cols:
        s = df[col]
        # If many values parse as numeric but not enough to confidently cast, it indicates mixing.
        _, ratio = _try_to_numeric(s, threshold=0.95)
        if 0.30 <= ratio < 0.90:
            inconsistent += 1

    return float(inconsistent / len(object_cols))


def compute_quality_score(
    df: pd.DataFrame,
    detected_types: Optional[ColumnTypeGroups] = None,
) -> QualityBreakdown:
    """Compute a dataset quality score (0–100) based on common issues."""

    if df.empty:
        return QualityBreakdown(score=0, missing_penalty=40, duplicate_penalty=20, outlier_penalty=25, dtype_penalty=15)

    rows = len(df)

    # Missing
    total_cells = rows * max(int(df.shape[1]), 1)
    missing_cells = int(df.isna().sum().sum())
    missing_ratio = missing_cells / max(total_cells, 1)

    # Duplicates
    dup_rows = count_duplicates(df)
    dup_ratio = dup_rows / max(rows, 1)

    # Outliers (IQR, numeric only)
    if detected_types is None:
        detected_types = detect_column_types(df)

    outlier_rows_pct = 0.0
    if len(detected_types.numerical) > 0:
        _, outlier_mask = iqr_outlier_summary(df, numerical_cols=detected_types.numerical, multiplier=1.5)
        outlier_rows_pct = float(outlier_mask.mean())

    # Dtype consistency heuristic
    dtype_inconsistency = _dtype_inconsistency_ratio(df)

    # Weighted penalties
    missing_penalty = 40.0 * missing_ratio
    duplicate_penalty = 20.0 * dup_ratio
    outlier_penalty = 25.0 * outlier_rows_pct
    dtype_penalty = 15.0 * dtype_inconsistency

    score = int(round(100.0 - (missing_penalty + duplicate_penalty + outlier_penalty + dtype_penalty)))
    score = max(0, min(100, score))

    return QualityBreakdown(
        score=score,
        missing_penalty=missing_penalty,
        duplicate_penalty=duplicate_penalty,
        outlier_penalty=outlier_penalty,
        dtype_penalty=dtype_penalty,
    )


def generate_ai_recommendations(
    df: pd.DataFrame,
    types: ColumnTypeGroups,
    missing_threshold_pct: float = 5.0,
    high_cardinality_threshold: int = 50,
) -> List[str]:
    """Rule-based suggestions that simulate AI recommendations."""

    recs: List[str] = []
    if df.empty:
        return recs

    missing = summarize_missing_values(df)
    skew_df = skewness_summary(df, types.numerical)

    skew_map: Dict[str, float] = {}
    if not skew_df.empty:
        skew_map = dict(zip(skew_df["column"].tolist(), skew_df["skewness"].tolist()))

    for _, row in missing.iterrows():
        col = str(row["column"])
        pct = float(row["missing_pct"])
        if pct < missing_threshold_pct:
            continue

        if col in types.numerical:
            skew = abs(float(skew_map.get(col, 0.0)))
            if skew >= 1.0:
                recs.append(f"Column '{col}' contains {pct:.1f}% missing values. Median imputation recommended (robust to skew).")
            else:
                recs.append(f"Column '{col}' contains {pct:.1f}% missing values. Mean imputation recommended.")
        elif col in types.categorical:
            recs.append(f"Column '{col}' contains {pct:.1f}% missing values. Mode (most frequent) imputation recommended.")
        elif col in types.datetime:
            recs.append(f"Column '{col}' contains {pct:.1f}% missing values. Consider forward-fill/back-fill or dropping rows if appropriate.")
        else:
            recs.append(f"Column '{col}' contains {pct:.1f}% missing values. Consider mode imputation or dropping affected rows.")

    if count_duplicates(df) > 0:
        recs.append("Duplicate rows detected. Removing duplicates is recommended.")

    # Skewness suggestions
    if not skew_df.empty:
        for _, row in skew_df.head(5).iterrows():
            col = str(row["column"])
            skew = float(row["skewness"])
            if abs(skew) >= 1.0:
                recs.append(f"Column '{col}' is highly skewed (skewness {skew:.2f}). Consider RobustScaler or normalization.")

    # Cardinality suggestions
    for col in types.categorical:
        nunique = int(df[col].nunique(dropna=True))
        if nunique >= high_cardinality_threshold:
            recs.append(
                f"Column '{col}' has high cardinality ({nunique} unique values). One-hot encoding may create many features; consider label encoding or feature hashing."
            )

    # Correlated numeric pairs (bonus)
    pairs = correlated_feature_pairs(df, numerical_cols=types.numerical, threshold=0.90)
    if not pairs.empty:
        top = pairs.iloc[0]
        recs.append(
            f"Highly correlated features detected: '{top['feature_1']}' vs '{top['feature_2']}' (corr {float(top['correlation']):.2f}). Consider dropping one (correlation-based feature selection)."
        )

    return recs


# -----------------------------
# Sklearn preprocessing pipeline
# -----------------------------


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract basic datetime features from one or more datetime columns."""

    def __init__(self, features: Optional[Sequence[str]] = None) -> None:
        # IMPORTANT (sklearn compatibility): do not modify/copy constructor
        # parameters here. Store them as-is; derive defaults in transform.
        self.features = features

    def fit(self, X: Any, y: Any = None) -> "DateTimeFeatureExtractor":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            frame = X
        else:
            frame = pd.DataFrame(X)

        default_features = ("year", "month", "day", "dayofweek")
        features = set(self.features or default_features)

        out_cols: List[np.ndarray] = []
        for col in frame.columns:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Could not infer format.*", category=UserWarning)
                s = pd.to_datetime(frame[col], errors="coerce")
            if "year" in features:
                out_cols.append(s.dt.year.to_numpy(dtype=float))
            if "month" in features:
                out_cols.append(s.dt.month.to_numpy(dtype=float))
            if "day" in features:
                out_cols.append(s.dt.day.to_numpy(dtype=float))
            if "dayofweek" in features:
                out_cols.append(s.dt.dayofweek.to_numpy(dtype=float))

        if not out_cols:
            return np.empty((len(frame), 0))

        out = np.vstack(out_cols).T
        # Missing datetimes become NaN; keep as NaN for imputers/scalers downstream if needed.
        return out

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        if input_features is None:
            input_features = ["datetime"]

        default_features = ("year", "month", "day", "dayofweek")
        features = set(self.features or default_features)

        names: List[str] = []
        for col in list(input_features):
            if "year" in features:
                names.append(f"{col}__year")
            if "month" in features:
                names.append(f"{col}__month")
            if "day" in features:
                names.append(f"{col}__day")
            if "dayofweek" in features:
                names.append(f"{col}__dayofweek")

        return np.array(names, dtype=object)


def _make_one_hot_encoder() -> OneHotEncoder:
    """Compatibility helper across sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _to_float_array(X: Any) -> np.ndarray:
    """Cast an array-like object to float numpy array.

    Used to convert boolean features to numeric 0/1 for ML-friendly outputs.
    """

    return np.asarray(X, dtype=float)


def build_preprocessor(
    types: ColumnTypeGroups,
    numeric_imputation: str = "median",
    categorical_encoding: str = "onehot",
    scaling: str = "standard",
) -> ColumnTransformer:
    """Build a ColumnTransformer based on detected types.

    Parameters:
        numeric_imputation: "mean" | "median"
        categorical_encoding: "label" | "onehot"
        scaling: "standard" | "minmax" | "robust" | "none"
    """

    if numeric_imputation not in {"mean", "median"}:
        raise ValueError("numeric_imputation must be 'mean' or 'median'")

    if categorical_encoding not in {"label", "onehot"}:
        raise ValueError("categorical_encoding must be 'label' or 'onehot'")

    if scaling not in {"standard", "minmax", "robust", "none"}:
        raise ValueError("scaling must be 'standard', 'minmax', 'robust', or 'none'")

    # Scaling
    scaler: Optional[Any]
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    else:
        scaler = None

    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=numeric_imputation))]
    if scaler is not None:
        num_steps.append(("scaler", scaler))

    numeric_pipeline = Pipeline(steps=num_steps)

    if categorical_encoding == "onehot":
        encoder = _make_one_hot_encoder()
    else:
        # Label-encoding style per column
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    # Datetime columns -> basic derived features
    datetime_pipeline = Pipeline(
        steps=[
            ("features", DateTimeFeatureExtractor()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # Treat boolean as numeric 0/1 for ML pipelines
    boolean_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_float", FunctionTransformer(_to_float_array, feature_names_out="one-to-one")),
        ]
    )

    transformers: List[Tuple[str, Any, List[str]]] = []
    if types.numerical:
        transformers.append(("num", numeric_pipeline, types.numerical))
    if types.categorical:
        transformers.append(("cat", categorical_pipeline, types.categorical))
    if types.boolean:
        transformers.append(("bool", boolean_pipeline, types.boolean))
    if types.datetime:
        transformers.append(("dt", datetime_pipeline, types.datetime))

    remainder = "drop" if transformers else "passthrough"
    transformer = ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        verbose_feature_names_out=True,
    )

    return transformer


def build_preprocessing_pipeline(
    types: ColumnTypeGroups,
    numeric_imputation: str = "median",
    categorical_encoding: str = "onehot",
    scaling: str = "standard",
) -> Pipeline:
    """Build a reusable sklearn Pipeline that wraps the ColumnTransformer."""

    preprocessor = build_preprocessor(
        types,
        numeric_imputation=numeric_imputation,
        categorical_encoding=categorical_encoding,
        scaling=scaling,
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def fit_transform_to_dataframe(
    df: pd.DataFrame,
    estimator: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit+transform and return a DataFrame with feature names.

    `estimator` can be a ColumnTransformer or a Pipeline that wraps one.
    """

    transformed = estimator.fit_transform(df)

    feature_names: np.ndarray
    try:
        if hasattr(estimator, "get_feature_names_out"):
            feature_names = estimator.get_feature_names_out()
        elif hasattr(estimator, "named_steps") and "preprocessor" in estimator.named_steps:
            feature_names = estimator.named_steps["preprocessor"].get_feature_names_out()
        else:
            raise AttributeError("No get_feature_names_out")
    except Exception:  # pragma: no cover
        feature_names = np.array([f"feature_{i}" for i in range(transformed.shape[1])], dtype=object)

    out_df = pd.DataFrame(transformed, columns=[str(c) for c in feature_names], index=df.index)
    return out_df, {"n_features": int(out_df.shape[1]), "feature_names": [str(c) for c in feature_names]}
