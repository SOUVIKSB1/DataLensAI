"""
Unit and Integration Test Suite for DataLens AI Platform
"""

import os
import pandas as pd

from datalens.profiler import DataProfiler, ColumnType
from datalens.quality import DataQualityEngine
from datalens.statistics import StatisticalEngine
from datalens.visualizer import VisualizerEngine
from datalens.ml_engine import MLEngine
from datalens.ai_engine import AIEngine


def test_profiler_and_types():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
    df = pd.read_csv(csv_path)
    profiler = DataProfiler(df)
    summary = profiler.to_dict()

    assert summary["total_rows"] == len(df)
    assert summary["total_cols"] == len(df.columns)
    assert "Employee_ID" in profiler.get_columns_by_type(ColumnType.IDENTIFIER)
    assert "Salary" in profiler.get_columns_by_type(ColumnType.NUMERICAL)
    assert "Department" in profiler.get_columns_by_type(ColumnType.CATEGORICAL)
    print("✅ Profiler and Column Type Tests PASSED")


def test_quality_and_iqr_outliers():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
    df = pd.read_csv(csv_path)
    quality = DataQualityEngine(df)
    rep = quality.to_dict()

    assert rep["duplicate_rows"] == 1  # Row 1005 is duplicated in test CSV
    assert rep["total_missing_cells"] > 0
    # Salary 180000 should be detected as an outlier
    salary_outliers = rep["outliers"].get("Salary", {})
    assert salary_outliers.get("iqr", {}).get("outlier_count", 0) >= 1

    # Test cleaning pipeline
    df_clean, log = quality.clean_dataset(drop_duplicates=True, missing_strategy="impute_median", outlier_strategy="cap_iqr")
    assert len(df_clean) == len(df) - 1
    assert df_clean.isna().sum().sum() == 0
    print("✅ Data Quality & Cleaning Tests PASSED")


def test_statistics_and_correlations():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
    df = pd.read_csv(csv_path)
    stats = StatisticalEngine(df)
    rep = stats.to_dict()

    assert "Salary" in rep["numerical"]
    assert "Department" in rep["categorical"]
    # Correlation between Experience_Years and Salary
    corrs = rep["correlations"]["strong_correlations"]
    assert len(corrs) > 0
    print("✅ Statistical Analysis & Correlation Tests PASSED")


def test_ml_engine():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
    df = pd.read_csv(csv_path)
    ml = MLEngine(df)

    # Test Regression
    reg_res = ml.train_regression("Salary")
    assert "metrics" in reg_res
    assert "R2_Score" in reg_res["metrics"]

    # Test Classification
    clf_res = ml.train_classification("Department")
    assert "metrics" in clf_res
    assert "Accuracy" in clf_res["metrics"]

    # Test Clustering
    clust_res = ml.train_clustering(n_clusters=3)
    assert "silhouette_score" in clust_res
    print("✅ Machine Learning Engine Tests PASSED")


def test_ai_engine():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
    df = pd.read_csv(csv_path)
    p = DataProfiler(df).to_dict()
    q = DataQualityEngine(df).to_dict()
    s = StatisticalEngine(df).to_dict()

    ai = AIEngine(df, p, q, s)
    insights = ai.generate_executive_insights()
    assert "Executive Summary" in insights or "Executive" in insights

    # Test Question Answering
    ans = ai.answer_query("What is the average salary by Department?")
    assert ans["data"] is not None or len(ans["answer"]) > 10
    print("✅ AI Engine & Q&A Router Tests PASSED")


if __name__ == "__main__":
    test_profiler_and_types()
    test_quality_and_iqr_outliers()
    test_statistics_and_correlations()
    test_ml_engine()
    test_ai_engine()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
