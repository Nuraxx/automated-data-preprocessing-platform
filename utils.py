"""Utility helpers for the Streamlit preprocessing app.

This module intentionally has no Streamlit dependency, so it can be imported
from anywhere (including tests/scripts) without UI side-effects.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def human_readable_bytes(num_bytes: int) -> str:
    """Convert a byte count into a human-friendly string."""

    if num_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1

    if unit == 0:
        return f"{int(size)} {units[unit]}"
    return f"{size:.2f} {units[unit]}"


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_with_fallbacks(
    uploaded_file: Any,
    **read_csv_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read CSV from a Streamlit UploadedFile (or any file-like) robustly.

    The Streamlit UploadedFile is a file-like wrapper with a `.getvalue()` method.
    This helper reads the underlying bytes once, then retries parsing with
    common encodings.

    Returns:
        (df, meta)

    meta includes:
        - file_name
        - bytes
        - encoding
    """

    if uploaded_file is None:
        raise ValueError("No file provided")

    file_name = getattr(uploaded_file, "name", "uploaded.csv")

    raw_bytes: Optional[bytes] = None
    if hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()

    if raw_bytes is None:
        # Fall back to reading from the file-like object directly.
        raw_bytes = uploaded_file.read()

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1"]
    last_error: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            buf = io.BytesIO(raw_bytes)
            df = pd.read_csv(buf, encoding=enc, low_memory=False, **read_csv_kwargs)
            return df, {"file_name": file_name, "bytes": len(raw_bytes), "encoding": enc}
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:  # noqa: BLE001 - surface original error
            last_error = e
            break

    if last_error is not None:
        raise last_error

    raise RuntimeError("Failed to read CSV")
