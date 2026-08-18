"""
Data Quality Engine for DataLens AI
Performs missing value audits, duplicate row detection, deterministic outlier analysis (IQR & Z-score),
and provides automated data cleaning utilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataQualityEngine:
    """
    Evaluates dataset quality and detects data cleanliness anomalies deterministically.
    """

    def __init__(self, df: pd.DataFrame, profiler_summary: Optional[Dict[str, Any]] = None):
        self.df = df.copy()
        self.profiler_summary = profiler_summary
        self.total_rows = len(df)
        self.quality_report: Dict[str, Any] = {}
        self._audit()

    def _detect_iqr_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Calculates outliers using the standard Interquartile Range (IQR) method."""
        clean_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(clean_series) < 4:
            return {
                "outlier_count": 0,
                "outlier_pct": 0.0,
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_bound": None,
                "upper_bound": None,
                "outlier_indices": [],
                "outlier_values": [],
            }

        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        iqr = float(q3 - q1)
        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)

        outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
        outlier_series = clean_series[outlier_mask]
        outlier_count = int(len(outlier_series))
        outlier_pct = round((outlier_count / len(clean_series)) * 100, 2)

        return {
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "lower_bound": round(lower_bound, 3),
            "upper_bound": round(upper_bound, 3),
            "outlier_indices": list(outlier_series.index),
            "outlier_values": [round(float(v), 3) for v in outlier_series.tolist()],
        }

    def _detect_zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Calculates outliers using Z-score method (|z| > threshold)."""
        clean_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(clean_series) < 4 or clean_series.std() == 0:
            return {"outlier_count": 0, "outlier_pct": 0.0, "threshold": threshold}

        mean = clean_series.mean()
        std = clean_series.std()
        z_scores = (clean_series - mean) / std
        outliers = clean_series[np.abs(z_scores) > threshold]

        return {
            "outlier_count": int(len(outliers)),
            "outlier_pct": round((len(outliers) / len(clean_series)) * 100, 2),
            "threshold": threshold,
            "mean": round(float(mean), 3),
            "std": round(float(std), 3),
        }

    def _audit(self) -> None:
        """Runs the complete data quality inspection."""
        # 1. Missing values
        missing_by_col = []
        for col in self.df.columns:
            m_cnt = int(self.df[col].isna().sum())
            if m_cnt > 0:
                missing_by_col.append({
                    "column": str(col),
                    "missing_count": m_cnt,
                    "missing_pct": round((m_cnt / self.total_rows) * 100, 2) if self.total_rows > 0 else 0.0,
                })
        missing_by_col.sort(key=lambda x: x["missing_count"], reverse=True)

        # 2. Duplicate rows
        dup_count = int(self.df.duplicated().sum())
        dup_pct = round((dup_count / self.total_rows) * 100, 2) if self.total_rows > 0 else 0.0
        dup_indices = list(self.df[self.df.duplicated(keep=False)].index)

        # 3. Numeric Outliers (IQR + Z-Score)
        outliers_by_col = {}
        total_outliers_found = 0

        # Look at numeric columns
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Skip identifier columns if identifiable
                col_name_lower = str(col).lower()
                if col_name_lower.endswith("id") or "_id" in col_name_lower:
                    continue

                iqr_res = self._detect_iqr_outliers(self.df[col])
                z_res = self._detect_zscore_outliers(self.df[col])
                
                if iqr_res["outlier_count"] > 0 or z_res["outlier_count"] > 0:
                    outliers_by_col[str(col)] = {
                        "iqr": iqr_res,
                        "z_score": z_res,
                    }
                    total_outliers_found += iqr_res["outlier_count"]

        # 4. Overall Health Score (0 - 100)
        total_cells = max(1, self.total_rows * len(self.df.columns))
        missing_cells = int(self.df.isna().sum().sum())
        missing_penalty = (missing_cells / total_cells) * 40
        dup_penalty = (dup_count / max(1, self.total_rows)) * 30
        outlier_penalty = min(30.0, (total_outliers_found / max(1, self.total_rows)) * 15)
        
        health_score = max(0, min(100, round(100 - (missing_penalty + dup_penalty + outlier_penalty), 1)))

        self.quality_report = {
            "health_score": health_score,
            "total_rows": self.total_rows,
            "total_columns": len(self.df.columns),
            "missing_columns_count": len(missing_by_col),
            "missing_details": missing_by_col,
            "total_missing_cells": missing_cells,
            "duplicate_rows": dup_count,
            "duplicate_pct": dup_pct,
            "duplicate_indices": dup_indices,
            "outliers": outliers_by_col,
            "total_outliers_detected": total_outliers_found,
        }

    def clean_dataset(
        self,
        drop_duplicates: bool = True,
        missing_strategy: str = "none",  # 'none', 'drop_rows', 'impute_mean_mode', 'impute_median'
        outlier_strategy: str = "none",  # 'none', 'trim_iqr', 'cap_iqr'
        target_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies deterministic data cleaning operations and returns cleaned dataframe + change log.
        """
        df_cleaned = self.df.copy()
        log = {"removed_duplicates": 0, "missing_handled": 0, "outliers_adjusted": 0}

        # 1. Deduplication
        if drop_duplicates:
            before_len = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            log["removed_duplicates"] = before_len - len(df_cleaned)

        # 2. Missing values
        if missing_strategy == "drop_rows":
            before_len = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            log["missing_handled"] = before_len - len(df_cleaned)
        elif missing_strategy in ["impute_mean_mode", "impute_median"]:
            for col in df_cleaned.columns:
                if df_cleaned[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        fill_val = df_cleaned[col].mean() if missing_strategy == "impute_mean_mode" else df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                    else:
                        mode_val = df_cleaned[col].mode()
                        if len(mode_val) > 0:
                            df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                    log["missing_handled"] += 1

        # 3. Outlier handling
        if outlier_strategy in ["trim_iqr", "cap_iqr"]:
            num_cols = target_columns or [c for c in df_cleaned.columns if pd.api.types.is_numeric_dtype(df_cleaned[c])]
            for col in num_cols:
                col_name_lower = str(col).lower()
                if col_name_lower.endswith("id") or "_id" in col_name_lower:
                    continue
                series = pd.to_numeric(df_cleaned[col], errors="coerce")
                if series.dropna().empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                if outlier_strategy == "trim_iqr":
                    mask = (df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)
                    outlier_cnt = (~mask).sum()
                    df_cleaned = df_cleaned[mask | df_cleaned[col].isna()]
                    log["outliers_adjusted"] += int(outlier_cnt)
                elif outlier_strategy == "cap_iqr":
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].clip(lower=lower, upper=upper)
                    adjusted = (before != df_cleaned[col]).sum()
                    log["outliers_adjusted"] += int(adjusted)

        return df_cleaned, log

    def to_dict(self) -> Dict[str, Any]:
        return self.quality_report
