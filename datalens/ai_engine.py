"""
Grounded AI Engine for DataLens AI
Integrates LLM reasoning with deterministic calculation tools, Privacy Scanner,
Evidence Verification, and RAG Knowledge Base.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from .logger import app_logger
from .privacy import PrivacyScanner
from .rag_engine import RAGEngine
from .agents import AgentOrchestrator


class AIEngine:
    """
    Grounded AI Analyst that combines deterministic calculations, privacy protection,
    RAG domain knowledge, and evidence-backed LLM interpretation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        profiler_dict: Dict[str, Any],
        quality_dict: Dict[str, Any],
        stats_dict: Dict[str, Any],
        api_key: Optional[str] = None,
    ):
        self.raw_df = df.copy()
        # 1. Privacy Safety Scan & Masking
        self.privacy_scanner = PrivacyScanner(df)
        self.df = self.privacy_scanner.mask_dataframe()
        self.privacy_report = self.privacy_scanner.to_dict()

        self.profiler = profiler_dict
        self.quality = quality_dict
        self.stats = stats_dict
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        
        # 2. Agent Orchestrator
        self.orchestrator = AgentOrchestrator(self.df, self.profiler, self.quality, self.stats)
        self._init_llm_client()

    def _init_llm_client(self):
        """Initializes Google GenAI / Gemini client safely."""
        self.client = None
        if not self.api_key:
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            app_logger.info("Initialized Google GenAI client.")
        except Exception:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self.client = legacy_genai.GenerativeModel("gemini-1.5-flash")
                app_logger.info("Initialized legacy Google GenerativeAI client.")
            except Exception as e:
                app_logger.warning(f"Could not initialize LLM client: {e}")
                self.client = None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Versatile caller supporting models.generate_content, interactions.create, and legacy SDKs."""
        if not self.client:
            return None

        # Candidate models to try in sequence
        models_to_try = [self.model_name, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-3.7-flash"]
        models_to_try = list(dict.fromkeys(models_to_try))

        # 1. Try modern google-genai SDK
        for m in models_to_try:
            try:
                # Priority 1: models.generate_content
                if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                    resp = self.client.models.generate_content(model=m, contents=prompt)
                    if resp and resp.text:
                        return resp.text.strip()

                # Priority 2: interactions.create
                if hasattr(self.client, "interactions") and hasattr(self.client.interactions, "create"):
                    inter = self.client.interactions.create(model=m, input=prompt)
                    if inter and hasattr(inter, "output_text") and inter.output_text:
                        return inter.output_text.strip()

                # Priority 3: Legacy generate_content
                if hasattr(self.client, "generate_content"):
                    resp = self.client.generate_content(prompt)
                    if resp and resp.text:
                        return resp.text.strip()
            except Exception as e:
                app_logger.warning(f"Model {m} attempt failed: {e}")
                continue

        return None

    def _build_grounded_context(self) -> str:
        """Constructs a factual, deterministic, privacy-sanitized summary context."""
        lines = []
        lines.append("=== PRIVACY & DATASET HEALTH ===")
        lines.append(f"- Privacy Safety Score: {self.privacy_report.get('privacy_safety_score')}/100")
        lines.append(f"- Health Score: {self.quality.get('health_score')}/100")
        lines.append(f"- Total Rows: {self.profiler.get('total_rows')}")
        lines.append(f"- Total Columns: {self.profiler.get('total_cols')}")
        lines.append(f"- Duplicate Rows: {self.quality.get('duplicate_rows')} ({self.quality.get('duplicate_pct')}%)")
        lines.append(f"- Missing Cells: {self.quality.get('total_missing_cells')} ({self.profiler.get('missing_cells_pct')}%)")

        lines.append("\n=== STATISTICAL GROUND TRUTH ===")
        num_stats = self.stats.get("numerical", {})
        for col, s in list(num_stats.items())[:6]:
            lines.append(f"- `{col}`: Mean={s.get('mean')}, Median={s.get('median_50')}, Std={s.get('std')}, Min={s.get('min')}, Max={s.get('max')}, Skewness={s.get('skewness')} ({s.get('skewness_label')})")

        strong_corrs = self.stats.get("correlations", {}).get("strong_correlations", [])
        if strong_corrs:
            lines.append("\n=== PROVEN CORRELATIONS ===")
            for c in strong_corrs[:4]:
                lines.append(f"- `{c['col1']}` <-> `{c['col2']}`: Pearson r = {c['pearson']} ({c['strength']} {c['direction']})")

        outliers = self.quality.get("outliers", {})
        if outliers:
            lines.append("\n=== PROVEN IQR OUTLIERS ===")
            for c, info in outliers.items():
                if info["iqr"]["outlier_count"] > 0:
                    lines.append(f"- `{c}`: {info['iqr']['outlier_count']} outliers (Bounds: [{info['iqr']['lower_bound']}, {info['iqr']['upper_bound']}]) -> Values: {info['iqr']['outlier_values']}")

        return "\n".join(lines)

    def generate_executive_insights(self) -> str:
        """
        Generates an executive briefing with grounded evidence citations.
        """
        context = self._build_grounded_context()

        # If LLM is available, query Gemini
        if self.client:
            prompt = f"""You are DataLens AI, an elite Principal Data Scientist, ML Architect, and Enterprise Business Strategist.
Conduct an extensive, high-impact Executive Intelligence Briefing on the following mathematically verified dataset:

{context}

Generate an authoritative, boardroom-ready report in GitHub-flavored Markdown covering:

# 💡 Executive Key Takeaways & Core Drivers
- High-level business interpretation of the data.
- Core metric findings with exact evidence citations (e.g. `[Evidence: metric=value]`).

# ⚠️ Data Quality, Outliers & Operational Hazards
- Critical anomalies, missing data risks, and IQR outlier analysis.
- Operational impact if left uncleaned.

# 📈 Correlation Dynamics & Behavioral Trends
- Deep breakdown of strong Pearson correlations and feature relationships.
- Business meaning behind the statistical interactions.

# 🎯 Strategic Action Plan & Predictive Modeling Roadmap
- 3 high-impact business decisions to execute immediately based on the data.
- Recommended ML models (e.g. Random Forest Classifier/Regressor) and primary feature drivers.

RULE: Every critical metric claim MUST include a bracketed evidence tag `[Evidence: metric=value]` to maintain 100% mathematical verifiability."""
            result = self._call_gemini(prompt)
            if result:
                return result

        # Fallback to deterministic multi-agent audit
        audit = self.orchestrator.run_autonomous_audit()
        return audit["executive_briefing"]

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """
        Answers user questions using deterministic calculation tools + RAG knowledge retrieval.
        """
        app_logger.info(f"Processing query: '{user_query}'")
        q = user_query.strip().lower()

        # 1. RAG search for conceptual/domain queries
        rag_docs = RAGEngine.search(user_query)
        rag_context = "\n".join([f"• **{d['title']}**: {d['content']}" for d in rag_docs]) if rag_docs else ""

        # 2. Highest / Lowest / Average group queries
        agg_match = re.search(r"(highest|maximum|max|lowest|minimum|min|average|avg|mean)\s+([\w_]+)(?:\s+(?:by|in|for|per)\s+([\w_]+))?", q)
        if agg_match:
            op, target_term, group_term = agg_match.groups()
            target_col = self._match_column_name(target_term)
            group_col = self._match_column_name(group_term) if group_term else None

            if target_col and pd.api.types.is_numeric_dtype(self.raw_df[target_col]):
                if group_col and group_col in self.raw_df.columns:
                    if op in ["highest", "maximum", "max"]:
                        res = self.raw_df.groupby(group_col)[target_col].max().reset_index().sort_values(by=target_col, ascending=False)
                        top_grp = res.iloc[0][group_col]
                        top_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The `{group_col}` with the highest `{target_col}` is **{top_grp}** with a value of **{top_val:,.2f}**.\n\n`[Evidence: Group Maximum calculation]`",
                            "data": res,
                        }
                    elif op in ["lowest", "minimum", "min"]:
                        res = self.raw_df.groupby(group_col)[target_col].min().reset_index().sort_values(by=target_col, ascending=True)
                        low_grp = res.iloc[0][group_col]
                        low_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The `{group_col}` with the lowest `{target_col}` is **{low_grp}** with a value of **{low_val:,.2f}**.\n\n`[Evidence: Group Minimum calculation]`",
                            "data": res,
                        }
                    else:  # Average
                        res = self.raw_df.groupby(group_col)[target_col].mean().round(2).reset_index().sort_values(by=target_col, ascending=False)
                        top_grp = res.iloc[0][group_col]
                        top_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The average `{target_col}` across `{group_col}` is highest in **{top_grp}** at **{top_val:,.2f}**.\n\n`[Evidence: Group Mean calculation]`",
                            "data": res,
                        }
                else:
                    if op in ["highest", "maximum", "max"]:
                        val = self.raw_df[target_col].max()
                        return {"answer": f"The maximum value of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Max={val}]`", "data": None}
                    elif op in ["lowest", "minimum", "min"]:
                        val = self.raw_df[target_col].min()
                        return {"answer": f"The minimum value of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Min={val}]`", "data": None}
                    else:
                        val = self.raw_df[target_col].mean()
                        return {"answer": f"The overall average (mean) of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Mean={val:,.2f}]`", "data": None}

        # 3. Missing values query
        if "missing" in q or "null" in q or "na" in q:
            missing_data = self.raw_df.isna().sum().reset_index()
            missing_data.columns = ["Column", "Missing Count"]
            missing_data = missing_data[missing_data["Missing Count"] > 0]
            if missing_data.empty:
                return {"answer": "There are **0 missing values** across all columns in this dataset.\n\n`[Evidence: 100% Completeness]`", "data": None}
            else:
                return {
                    "answer": f"Detected missing values in **{len(missing_data)} column(s)**:\n\n`[Evidence: Total Missing Cells = {self.quality.get('total_missing_cells')}]`",
                    "data": missing_data,
                }

        # 4. Duplicate rows query
        if "duplicate" in q or "repeat" in q:
            dups = self.quality.get("duplicate_rows", 0)
            return {"answer": f"The dataset contains **{dups} duplicate row(s)** ({self.quality.get('duplicate_pct')}%).\n\n`[Evidence: Exact row duplication check]`", "data": None}

        # 5. Outliers / Anomalies query
        if "outlier" in q or "anomaly" in q or "extreme" in q:
            outliers = self.quality.get("outliers", {})
            outlier_rows = []
            for col, o in outliers.items():
                if o["iqr"]["outlier_count"] > 0:
                    outlier_rows.append({
                        "Column": col,
                        "Outliers": o["iqr"]["outlier_count"],
                        "Lower Bound": o["iqr"]["lower_bound"],
                        "Upper Bound": o["iqr"]["upper_bound"],
                        "Outlier Values": str(o["iqr"]["outlier_values"]),
                    })
            if outlier_rows:
                return {
                    "answer": f"Detected IQR outliers in **{len(outlier_rows)} column(s)**:\n\n`[Evidence: Interquartile Range method]`",
                    "data": pd.DataFrame(outlier_rows),
                }
            else:
                return {"answer": "No statistical outliers detected via IQR method.\n\n`[Evidence: Zero values outside [Q1-1.5IQR, Q3+1.5IQR]]`", "data": None}

        # 6. Fallback to Gemini with RAG domain context
        if self.client:
            grounded_ctx = self._build_grounded_context()
            prompt = f"""You are DataLens AI Data Analyst.
Dataset Context:
{grounded_ctx}

Relevant Knowledge Base Articles:
{rag_context}

User question: {user_query}

Provide a concise, grounded answer. Ensure any claims are backed by data facts or domain knowledge."""
            res_text = self._call_gemini(prompt)
            if res_text:
                return {"answer": res_text, "data": None}

        # If RAG found an article for conceptual query
        if rag_docs:
            top_d = rag_docs[0]
            return {
                "answer": f"### 📚 Knowledge Base: {top_d['title']}\n\n{top_d['content']}\n\n*Retrieved via DataLens RAG Engine.*",
                "data": None,
            }

        return {
            "answer": f"I parsed your question regarding **'{user_query}'**. Try asking questions like:\n- *'What is the average Salary by Department?'*\n- *'Show outliers in Experience_Years'*\n- *'What does Pearson correlation mean?'*\n- *'Check for duplicate rows'*\n- *'Highest Performance_Score by City'*",
            "data": None,
        }

    def _match_column_name(self, term: Optional[str]) -> Optional[str]:
        """Fuzzy matches user terms against DataFrame column names."""
        if not term:
            return None
        term_clean = re.sub(r"[^\w]", "", term.lower())
        for col in self.raw_df.columns:
            col_clean = re.sub(r"[^\w]", "", str(col).lower())
            if term_clean == col_clean or term_clean in col_clean or col_clean in term_clean:
                return str(col)
        return None
