"""
DataLens AI - Production-Grade Grounded Analytics & AI Studio
Phase 1-13 Production Web Frontend (Streamlit)
"""

import os
import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from datalens.loader import DataLoader
from datalens.privacy import PrivacyScanner
from datalens.profiler import DataProfiler, ColumnType
from datalens.quality import DataQualityEngine
from datalens.statistics import StatisticalEngine
from datalens.visualizer import VisualizerEngine
from datalens.ml_engine import MLEngine
from datalens.ai_engine import AIEngine
from datalens.rag_engine import RAGEngine
from datalens.agents import AgentOrchestrator
from datalens.reports import ReportGenerator
from datalens.logger import app_logger

# Streamlit Page Config
st.set_page_config(
    page_title="DataLens AI | Grounded Data Science & AI Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System (Clean, modern slate UI with cards, badges, and responsive typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
        margin-top: 0.3rem;
    }
    
    .evidence-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    
    .safety-banner {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def init_state():
    """Initializes session state keys."""
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "cleaning_log" not in st.session_state:
        st.session_state.cleaning_log = {}
    if "ml_results" not in st.session_state:
        st.session_state.ml_results = None
    if "privacy_report" not in st.session_state:
        st.session_state.privacy_report = None


init_state()


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/data-configuration.png", width=56)
    st.title("DataLens AI")
    st.caption("Production Grounded Analytics & AI Studio")

    st.markdown("---")
    st.subheader("🧭 Navigation")
    
    nav_selection = st.radio(
        "Select View",
        [
            "🏠 Home & Ingestion",
            "📋 Overview & Profiling",
            "🧼 Data Quality & Cleaning",
            "🔢 Statistical Analysis",
            "📈 Visualizations & Auto-Charts",
            "🔗 Correlation Dynamics",
            "🚨 Anomalies & Outliers",
            "🧠 Machine Learning Studio",
            "💡 AI Insights & Evidence",
            "🤖 AI Data Analyst Chatbot",
            "📚 RAG Knowledge Base",
            "📑 Reports & Reproducibility",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("🔑 AI Configuration")
    api_key_input = st.text_input(
        "Google Gemini API Key (Optional)",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="If not provided, DataLens AI uses its built-in deterministic intelligence engine.",
    )

    st.markdown("---")
    if st.session_state.raw_df is not None:
        active_dataset_label = "Cleaned Data" if st.session_state.cleaned_df is not None else "Raw Data"
        st.success(f"📊 Active: **{st.session_state.dataset_name}** ({active_dataset_label})")
        if st.session_state.cleaned_df is not None:
            if st.button("↺ Reset to Raw Dataset", use_container_width=True):
                st.session_state.cleaned_df = None
                st.rerun()

    st.markdown(
        "<div style='font-size: 0.75rem; color: #94A3B8; text-align: center; margin-top: 1rem;'>"
        "DataLens AI v1.0<br>Privacy Safe • Grounded • Deterministic"
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------- DATASET SELECTION -----------------
current_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

if current_df is not None:
    # Run Cached Analysis Pipeline
    profiler = DataProfiler(current_df)
    prof_dict = profiler.to_dict()

    quality_engine = DataQualityEngine(current_df, profiler_summary=prof_dict)
    qual_dict = quality_engine.to_dict()

    stats_engine = StatisticalEngine(current_df)
    stats_dict = stats_engine.to_dict()

    ai_engine = AIEngine(
        current_df,
        profiler_dict=prof_dict,
        quality_dict=qual_dict,
        stats_dict=stats_dict,
        api_key=api_key_input,
    )
    st.session_state.privacy_report = ai_engine.privacy_report


# =========================================================
# 1. HOME & INGESTION
# =========================================================
if nav_selection == "🏠 Home & Ingestion":
    st.markdown("<h1 class='main-title'>🔬 Welcome to DataLens AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>A grounded, privacy-preserving analytical engine that turns tabular datasets into actionable intelligence.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📁 Upload Your Dataset")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Auto-detects encodings (UTF-8, Latin-1, CP1252) and delimiters.")
        sample_choice = st.button("📊 Load Sample HR & Attrition Dataset", use_container_width=True)

        if sample_choice:
            sample_path = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
            df_loaded, err = DataLoader.load_dataset(sample_path)
            if df_loaded is not None:
                st.session_state.raw_df = df_loaded
                st.session_state.cleaned_df = None
                st.session_state.dataset_name = "auto_data_analyzer_test.csv"
                st.session_state.chat_history = []
                st.success("Sample dataset successfully loaded!")
                st.rerun()

        if uploaded_file is not None:
            df_loaded, err = DataLoader.load_dataset(uploaded_file, sample_if_large=True)
            if err:
                st.error(err)
            else:
                st.session_state.raw_df = df_loaded
                st.session_state.cleaned_df = None
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.chat_history = []
                st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df_loaded):,} rows)!")
                st.rerun()

    with col2:
        st.subheader("🛡️ AI Safety & Privacy Guard")
        if current_df is not None and st.session_state.privacy_report:
            p_rep = st.session_state.privacy_report
            score = p_rep["privacy_safety_score"]
            color = "#16A34A" if score >= 80 else ("#D97706" if score >= 50 else "#DC2626")
            
            st.markdown(f"""
            <div class='safety-banner'>
                <b>Privacy Safety Score: <span style='color:{color}; font-size: 1.1rem;'>{score}/100</span></b><br>
                Sensitive Columns Detected: <b>{p_rep['sensitive_columns_count']}</b><br>
                Total PII Matches Masked: <b>{p_rep['total_pii_occurrences']}</b>
            </div>
            """, unsafe_allow_html=True)

            if p_rep["flagged_columns"]:
                with st.expander("View Privacy Scan Findings"):
                    st.write(p_rep["flagged_columns"])
        else:
            st.info("Upload a dataset to automatically run the AI Safety & PII scanner before analysis.")


# =========================================================
# 2. OVERVIEW & PROFILING
# =========================================================
elif nav_selection == "📋 Overview & Profiling":
    if current_df is None:
        st.warning("Please upload or select a dataset on the Home page first.")
    else:
        st.markdown(f"<h1 class='main-title'>📋 Dataset Profile & Schema</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='subtitle'>Dataset: <b>{st.session_state.dataset_name}</b></p>", unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Rows</div><div class='metric-val'>{prof_dict['total_rows']:,}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Columns</div><div class='metric-val'>{prof_dict['total_cols']}</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Memory</div><div class='metric-val'>{prof_dict['memory_mb']} MB</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Missing Cells</div><div class='metric-val'>{prof_dict['total_missing_cells']}</div></div>", unsafe_allow_html=True)
        with k5:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Duplicates</div><div class='metric-val'>{prof_dict['duplicate_rows']}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Semantic Column Classification")
        col_table = []
        for c in prof_dict["columns"]:
            col_table.append({
                "Column Name": c["column_name"],
                "Semantic Type": c["semantic_type"],
                "Dtype": c["pandas_dtype"],
                "Missing Values": f"{c['missing_count']} ({c['missing_pct']}%)",
                "Unique Values": c["unique_count"],
                "Sample Values": ", ".join(c["sample_values"][:3]),
            })
        st.dataframe(pd.DataFrame(col_table), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Interactive Data Explorer")
        st.dataframe(current_df, use_container_width=True)


# =========================================================
# 3. DATA QUALITY & CLEANING
# =========================================================
elif nav_selection == "🧼 Data Quality & Cleaning":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>🧼 Data Quality Audit & Cleaning</h1>", unsafe_allow_html=True)
        
        q1, q2 = st.columns(2)
        with q1:
            st.subheader("Missing Values Audit")
            miss_fig = VisualizerEngine.create_missing_values_bar(current_df)
            if miss_fig:
                st.plotly_chart(miss_fig, use_container_width=True)
            else:
                st.success("✅ Dataset is 100% complete with 0 missing values!")

        with q2:
            st.subheader("Duplicate Records")
            st.metric("Duplicate Rows", f"{qual_dict['duplicate_rows']} ({qual_dict['duplicate_pct']}%)")
            if qual_dict['duplicate_rows'] > 0:
                st.warning(f"Detected {qual_dict['duplicate_rows']} duplicate row(s) that should be removed.")
            else:
                st.success("✅ No duplicate rows detected.")

        st.markdown("---")
        st.subheader("🧹 One-Click Data Cleaning Studio")
        c1, c2, c3 = st.columns(3)
        with c1:
            drop_dups = st.checkbox("Deduplicate Dataset", value=True)
        with c2:
            missing_mode = st.selectbox("Handle Missing Values", ["none", "impute_mean_mode", "impute_median", "drop_rows"])
        with c3:
            outlier_mode = st.selectbox("Handle Outliers (IQR)", ["none", "cap_iqr", "trim_iqr"])

        if st.button("🚀 Apply Cleaning Transformations", type="primary"):
            cleaned, log = quality_engine.clean_dataset(
                drop_duplicates=drop_dups,
                missing_strategy=missing_mode,
                outlier_strategy=outlier_mode,
            )
            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_log = log
            st.success(f"Cleaned! Removed {log['removed_duplicates']} dups, handled {log['missing_handled']} missing cols, adjusted {log['outliers_adjusted']} outliers.")
            st.rerun()


# =========================================================
# 4. STATISTICAL ANALYSIS
# =========================================================
elif nav_selection == "🔢 Statistical Analysis":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>🔢 Statistical Analysis Engine</h1>", unsafe_allow_html=True)
        
        st.subheader("Numerical Summary Statistics")
        num_stats = stats_dict.get("numerical", {})
        if num_stats:
            df_num = pd.DataFrame(num_stats).T
            st.dataframe(df_num, use_container_width=True)
        else:
            st.info("No numerical features available.")

        st.markdown("---")
        st.subheader("Categorical Feature Distributions")
        cat_stats = stats_dict.get("categorical", {})
        if cat_stats:
            for col, d in cat_stats.items():
                with st.expander(f"Distribution: {col} (Top: {d['top']} - {d['top_percentage']}%)"):
                    st.write(pd.DataFrame(list(d["frequencies"].items()), columns=["Category", "Frequency"]))
        else:
            st.info("No categorical features found.")


# =========================================================
# 5. VISUALIZATIONS & AUTO-CHARTS
# =========================================================
elif nav_selection == "📈 Visualizations & Auto-Charts":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>📈 Automated & Custom Visualizations</h1>", unsafe_allow_html=True)
        
        st.subheader("✨ Automated Recommendations")
        recs = VisualizerEngine.recommend_visualizations(current_df, profiler.column_profiles)
        for i in range(0, len(recs), 2):
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(recs[i]["fig"], use_container_width=True)
                st.caption(f"**{recs[i]['title']}**: {recs[i]['description']}")
            if i + 1 < len(recs):
                with c2:
                    st.plotly_chart(recs[i+1]["fig"], use_container_width=True)
                    st.caption(f"**{recs[i+1]['title']}**: {recs[i+1]['description']}")

        st.markdown("---")
        st.subheader("🎨 Custom Multi-Axis Chart Builder")
        c_type = st.selectbox("Select Chart Type", ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Chart"])
        num_cols = profiler.get_columns_by_type(ColumnType.NUMERICAL)
        cat_cols = profiler.get_columns_by_type(ColumnType.CATEGORICAL)
        date_cols = profiler.get_columns_by_type(ColumnType.DATE)

        if c_type == "Histogram" and num_cols:
            h_col = st.selectbox("Numerical Column", num_cols)
            bins = st.slider("Bins", 5, 50, 20)
            st.plotly_chart(VisualizerEngine.create_histogram(current_df, h_col, bins), use_container_width=True)
        elif c_type == "Bar Chart" and cat_cols:
            b_col = st.selectbox("Categorical Column", cat_cols)
            st.plotly_chart(VisualizerEngine.create_bar_chart(current_df, b_col), use_container_width=True)
        elif c_type == "Scatter Plot" and len(num_cols) >= 2:
            sc1, sc2, sc3 = st.columns(3)
            with sc1: x_c = st.selectbox("X-Axis", num_cols, index=0)
            with sc2: y_c = st.selectbox("Y-Axis", num_cols, index=min(1, len(num_cols)-1))
            with sc3: color_c = st.selectbox("Color Grouping", [None] + cat_cols)
            trend = st.checkbox("OLS Trendline", value=False)
            st.plotly_chart(VisualizerEngine.create_scatter_plot(current_df, x_c, y_c, color_c, trend), use_container_width=True)
        elif c_type == "Box Plot" and num_cols:
            bx1, bx2 = st.columns(2)
            with bx1: bx_y = st.selectbox("Value Column", num_cols)
            with bx2: bx_x = st.selectbox("Group By (Optional)", [None] + cat_cols)
            st.plotly_chart(VisualizerEngine.create_box_plot(current_df, bx_y, bx_x), use_container_width=True)
        elif c_type == "Line Chart" and (date_cols or num_cols):
            ln_d = st.selectbox("Date / Index", date_cols if date_cols else list(current_df.columns))
            ln_v = st.selectbox("Value Column", num_cols if num_cols else list(current_df.columns))
            st.plotly_chart(VisualizerEngine.create_line_chart(current_df, ln_d, ln_v), use_container_width=True)


# =========================================================
# 6. CORRELATION DYNAMICS
# =========================================================
elif nav_selection == "🔗 Correlation Dynamics":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>🔗 Correlation Dynamics & Collinearity</h1>", unsafe_allow_html=True)
        
        corr_method = st.radio("Correlation Method", ["pearson", "spearman"], horizontal=True)
        heat_fig = VisualizerEngine.create_correlation_heatmap(current_df, method=corr_method)
        if heat_fig:
            st.plotly_chart(heat_fig, use_container_width=True)
        else:
            st.info("At least 2 numerical columns required for correlation matrix.")

        st.subheader("Key Pairwise Relationships")
        corrs = stats_dict.get("correlations", {}).get("strong_correlations", [])
        if corrs:
            st.dataframe(pd.DataFrame(corrs), use_container_width=True, hide_index=True)
        else:
            st.info("No strong pairwise correlations detected.")


# =========================================================
# 7. ANOMALIES & OUTLIERS
# =========================================================
elif nav_selection == "🚨 Anomalies & Outliers":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>🚨 Anomaly & Outlier Engine</h1>", unsafe_allow_html=True)
        
        outliers = qual_dict.get("outliers", {})
        if outliers:
            out_rows = []
            for col, info in outliers.items():
                iqr = info["iqr"]
                if iqr["outlier_count"] > 0:
                    out_rows.append({
                        "Column": col,
                        "IQR Outlier Count": iqr["outlier_count"],
                        "Outlier %": f"{iqr['outlier_pct']}%",
                        "Lower Bound": iqr["lower_bound"],
                        "Upper Bound": iqr["upper_bound"],
                        "Outlier Values": str(iqr["outlier_values"][:5]),
                    })
            if out_rows:
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
            else:
                st.success("✅ Zero statistical outliers detected using IQR method.")
        else:
            st.success("✅ No numerical features with outliers detected.")


# =========================================================
# 8. MACHINE LEARNING STUDIO
# =========================================================
elif nav_selection == "🧠 Machine Learning Studio":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>🧠 Machine Learning Studio</h1>", unsafe_allow_html=True)
        
        ml_engine = MLEngine(current_df)
        potential_targets = [c for c in current_df.columns if not str(c).lower().endswith("id")]

        if potential_targets:
            target_col = st.selectbox("Select Target Variable to Predict", potential_targets, index=len(potential_targets)-2 if len(potential_targets) > 2 else 0)
            problem_type = ml_engine.detect_problem_type(target_col)
            st.info(f"🎯 **Detected Problem**: **{problem_type}** for `{target_col}`.")

            m1, m2 = st.columns(2)
            with m1:
                model_algo = st.selectbox("Algorithm", ["Random Forest", "Linear Regression"] if problem_type == "Regression" else ["Random Forest", "Logistic Regression"])
            with m2:
                feature_choices = [c for c in potential_targets if c != target_col]
                selected_features = st.multiselect("Predictor Features", feature_choices, default=feature_choices)

            if st.button("🚀 Train & Evaluate ML Model", type="primary"):
                with st.spinner("Training model..."):
                    if problem_type == "Regression":
                        res = ml_engine.train_regression(target_col, selected_features, model_algo)
                    else:
                        res = ml_engine.train_classification(target_col, selected_features, model_algo)

                    st.session_state.ml_results = res

            if st.session_state.ml_results:
                res = st.session_state.ml_results
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success(f"Model ({res['model_name']}) evaluated successfully!")
                    
                    # Metrics Row
                    m_cols = st.columns(len(res["metrics"]))
                    for idx, (m_name, m_val) in enumerate(res["metrics"].items()):
                        with m_cols[idx]:
                            st.metric(m_name, m_val)

                    # Charts
                    p1, p2 = st.columns(2)
                    with p1:
                        if "fig_prediction" in res: st.plotly_chart(res["fig_prediction"], use_container_width=True)
                        elif "fig_confusion_matrix" in res: st.plotly_chart(res["fig_confusion_matrix"], use_container_width=True)
                    with p2:
                        if "fig_importance" in res: st.plotly_chart(res["fig_importance"], use_container_width=True)

        st.markdown("---")
        st.subheader("🔮 Unsupervised K-Means Clustering")
        k_clusters = st.slider("Number of Clusters (k)", 2, 8, 3)
        if st.button("Run K-Means Clustering"):
            c_res = ml_engine.train_clustering(n_clusters=k_clusters)
            if "error" in c_res:
                st.error(c_res["error"])
            else:
                st.metric("Silhouette Score", c_res["silhouette_score"])
                st.write("**Cluster Membership Breakdown:**", c_res["cluster_counts"])
                if c_res.get("fig_clusters"):
                    st.plotly_chart(c_res["fig_clusters"], use_container_width=True)


# =========================================================
# 9. AI INSIGHTS & EVIDENCE
# =========================================================
elif nav_selection == "💡 AI Insights & Evidence":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>💡 Autonomous AI Insights & Grounded Evidence</h1>", unsafe_allow_html=True)
        
        if st.button("⚡ Generate Autonomous Multi-Agent Briefing", type="primary"):
            with st.spinner("Synthesizing multi-agent findings..."):
                briefing = ai_engine.generate_executive_insights()
                st.session_state.executive_briefing = briefing

        if "executive_briefing" not in st.session_state:
            st.session_state.executive_briefing = ai_engine.generate_executive_insights()

        st.markdown(st.session_state.executive_briefing)


# =========================================================
# 10. AI DATA ANALYST CHATBOT
# =========================================================
elif nav_selection == "🤖 AI Data Analyst Chatbot":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>🤖 AI Data Analyst Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Ask questions in plain English. Calculations are computed deterministically with zero hallucination.</p>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "data" in msg and msg["data"] is not None:
                    st.dataframe(msg["data"], use_container_width=True)

        user_q = st.chat_input("Ask about your dataset (e.g. 'Which department has the highest average salary?')")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    res = ai_engine.answer_query(user_q)
                    st.markdown(res["answer"])
                    if res.get("data") is not None:
                        st.dataframe(res["data"], use_container_width=True)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "data": res.get("data"),
                    })


# =========================================================
# 11. RAG KNOWLEDGE BASE
# =========================================================
elif nav_selection == "📚 RAG Knowledge Base":
    st.markdown("<h1 class='main-title'>📚 RAG Statistical Knowledge Base</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Retrieve verified definitions, mathematical formulas, and metric interpretations.</p>", unsafe_allow_html=True)

    search_query = st.text_input("Search Statistical / ML Concepts", placeholder="e.g. 'Pearson correlation', 'IQR outlier detection', 'R2 score'")
    if search_query:
        docs = RAGEngine.search(search_query, top_k=5)
        if docs:
            for d in docs:
                with st.expander(f"📖 {d['title']}", expanded=True):
                    st.write(d["content"])
                    st.caption(f"Keywords: {', '.join(d['keywords'])}")
        else:
            st.info("No matching articles found in knowledge repository.")
    else:
        st.write("### Indexed Knowledge Documents")
        for d in RAGEngine.KNOWLEDGE_DOCUMENTS:
            with st.expander(f"📖 {d['title']}"):
                st.write(d["content"])
                st.caption(f"Keywords: {', '.join(d['keywords'])}")


# =========================================================
# 12. REPORTS & REPRODUCIBILITY
# =========================================================
elif nav_selection == "📑 Reports & Reproducibility":
    if current_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.markdown("<h1 class='main-title'>📑 Reports & Reproducibility Engine</h1>", unsafe_allow_html=True)
        
        briefing_content = st.session_state.get("executive_briefing", ai_engine.generate_executive_insights())

        r_col1, r_col2, r_col3 = st.columns(3)

        # 1. Download Cleaned CSV
        with r_col1:
            st.subheader("1. Cleaned CSV")
            csv_bytes = current_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Cleaned CSV",
                data=csv_bytes,
                file_name=f"datalens_cleaned_{st.session_state.dataset_name}",
                mime="text/csv",
                use_container_width=True,
            )

        # 2. Download PDF Report
        with r_col2:
            st.subheader("2. PDF Audit Report")
            pdf_bytes = ReportGenerator.generate_pdf_report(
                dataset_name=st.session_state.dataset_name,
                df=current_df,
                profiler_dict=prof_dict,
                quality_dict=qual_dict,
                stats_dict=stats_dict,
                ai_briefing=briefing_content,
                privacy_dict=st.session_state.privacy_report,
            )
            st.download_button(
                "📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"datalens_report_{st.session_state.dataset_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # 3. Download JSON Reproducibility Audit
        with r_col3:
            st.subheader("3. JSON Audit Checksum")
            json_str = ReportGenerator.generate_reproducibility_json(
                dataset_name=st.session_state.dataset_name,
                df=current_df,
                profiler_dict=prof_dict,
                quality_dict=qual_dict,
                stats_dict=stats_dict,
                cleaning_log=st.session_state.cleaning_log,
                ml_results=st.session_state.ml_results,
            )
            st.download_button(
                "📥 Download Reproducibility JSON",
                data=json_str,
                file_name=f"datalens_reproducibility_{st.session_state.dataset_name}.json",
                mime="application/json",
                use_container_width=True,
            )
