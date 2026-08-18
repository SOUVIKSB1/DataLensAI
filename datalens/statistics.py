"""
Statistical Analysis Engine for DataLens AI
Computes descriptive statistics, distributions, skewness, kurtosis,
and Pearson/Spearman correlation matrices.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class StatisticalEngine:
    """
    Computes rigorous univariate and bivariate descriptive statistics.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.stats_report: Dict[str, Any] = {}
        self._compute()

    def _get_numerical_summary(self) -> Dict[str, Dict[str, Any]]:
        """Calculates comprehensive descriptive statistics for all numeric columns."""
        summary = {}
        num_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            col_name_lower = str(col).lower()
            if col_name_lower.endswith("id") or "_id" in col_name_lower:
                continue

            series = pd.to_numeric(self.df[col].dropna(), errors="coerce").dropna()
            if len(series) == 0:
                continue

            q1 = float(series.quantile(0.25))
            q2 = float(series.quantile(0.50))  # Median
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            mode_vals = series.mode()
            mode_val = float(mode_vals.iloc[0]) if len(mode_vals) > 0 else None

            # Skewness and kurtosis
            skewness = float(series.skew()) if len(series) > 2 else 0.0
            kurt = float(series.kurtosis()) if len(series) > 3 else 0.0

            # Skewness interpretation
            if abs(skewness) < 0.5:
                skew_label = "Fairly Symmetrical"
            elif skewness > 0.5:
                skew_label = "Positively Skewed (Right Tail)"
            else:
                skew_label = "Negatively Skewed (Left Tail)"

            summary[str(col)] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 3),
                "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
                "variance": round(float(series.var()), 3) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 3),
                "q1_25": round(q1, 3),
                "median_50": round(q2, 3),
                "q3_75": round(q3, 3),
                "max": round(float(series.max()), 3),
                "iqr": round(iqr, 3),
                "mode": round(mode_val, 3) if mode_val is not None else None,
                "skewness": round(skewness, 3),
                "skewness_label": skew_label,
                "kurtosis": round(kurt, 3),
            }

        return summary

    def _get_categorical_summary(self) -> Dict[str, Dict[str, Any]]:
        """Calculates frequency distributions and top values for non-numeric columns."""
        summary = {}
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns

        for col in cat_cols:
            series = self.df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            unique_cnt = int(series.nunique())
            val_counts = series.value_counts()
            top_val = str(val_counts.index[0]) if len(val_counts) > 0 else None
            top_freq = int(val_counts.iloc[0]) if len(val_counts) > 0 else 0
            top_pct = round((top_freq / len(series)) * 100, 2) if len(series) > 0 else 0.0

            summary[str(col)] = {
                "count": int(len(series)),
                "unique": unique_cnt,
                "top": top_val,
                "freq": top_freq,
                "top_percentage": top_pct,
                "frequencies": {str(k): int(v) for k, v in val_counts.head(8).items()},
            }

        return summary

    def _get_correlation_analysis(self) -> Dict[str, Any]:
        """Calculates Pearson and Spearman correlation matrices and identifies notable pairs."""
        num_df = self.df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return {
                "pearson_matrix": {},
                "spearman_matrix": {},
                "strong_correlations": [],
            }

        pearson_corr = num_df.corr(method="pearson").round(3)
        spearman_corr = num_df.corr(method="spearman").round(3)

        # Extract top pairs
        strong_pairs = []
        cols = list(num_df.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                
                # Ignore identifier pairs
                if str(col1).lower().endswith("id") or str(col2).lower().endswith("id"):
                    continue

                p_val = pearson_corr.loc[col1, col2]
                s_val = spearman_corr.loc[col1, col2]

                if not np.isnan(p_val):
                    strength = "None"
                    if abs(p_val) >= 0.7:
                        strength = "Strong"
                    elif abs(p_val) >= 0.4:
                        strength = "Moderate"
                    elif abs(p_val) >= 0.2:
                        strength = "Weak"

                    strong_pairs.append({
                        "col1": str(col1),
                        "col2": str(col2),
                        "pearson": float(p_val),
                        "spearman": float(s_val) if not np.isnan(s_val) else float(p_val),
                        "strength": strength,
                        "direction": "Positive" if p_val > 0 else "Negative",
                    })

        # Sort by absolute pearson correlation
        strong_pairs.sort(key=lambda x: abs(x["pearson"]), reverse=True)

        return {
            "pearson_matrix": pearson_corr.to_dict(),
            "spearman_matrix": spearman_corr.to_dict(),
            "strong_correlations": strong_pairs,
        }

    def _compute(self) -> None:
        """Runs the statistical processing pipeline."""
        num_summary = self._get_numerical_summary()
        cat_summary = self._get_categorical_summary()
        corr_summary = self._get_correlation_analysis()

        self.stats_report = {
            "numerical": num_summary,
            "categorical": cat_summary,
            "correlations": corr_summary,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.stats_report
