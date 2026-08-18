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
from .local_ai import call_local_llm, get_local_ai_config, local_ai_enabled


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
        self.local_ai_base_url, self.local_model_name = get_local_ai_config()
        
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

    def _call_external_llm(self, prompt: str) -> Optional[str]:
        """Versatile Gemini caller used only when an API key/client is available."""
        if not self.client:
            return None

        # Candidate models to try in sequence
        models_to_try = [self.model_name, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
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
                err_str = str(e)
                app_logger.warning(f"Model {m} attempt: {e}")
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    break
                continue

        return None

    def _call_llm(self, prompt: str, *, prefer_local: bool = True) -> Optional[str]:
        """Returns best-effort generated text from local AI first, then optional Gemini."""
        if prefer_local and local_ai_enabled():
            local_text = call_local_llm(prompt)
            if local_text:
                return local_text

        return self._call_external_llm(prompt)

    def get_engine_status(self) -> Dict[str, Any]:
        """Exposes AI availability for API/UI status badges."""
        return {
            "local_ai_enabled": local_ai_enabled(),
            "local_model": self.local_model_name,
            "local_base_url": self.local_ai_base_url,
            "external_llm_available": bool(self.client),
            "external_model": self.model_name if self.client else None,
        }

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

        # Try local LLM first, then optional external LLM if configured.
        if local_ai_enabled() or self.client:
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
            result = self._call_llm(prompt)
            if result:
                return result

        # Fallback to deterministic multi-agent audit
        audit = self.orchestrator.run_autonomous_audit()
        return audit["executive_briefing"]

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """
        Answers user questions using deterministic calculation tools + Gemini reasoning + RAG knowledge retrieval.
        """
        app_logger.info(f"Processing query: '{user_query}'")
        q = user_query.strip().lower()

        # 0. Conversational Greetings Handler
        if q in ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "who are you", "what can you do"]:
            rows = self.profiler.get("total_rows", len(self.raw_df))
            cols = self.profiler.get("total_cols", len(self.raw_df.columns))
            return {
                "answer": f"Hello! 👋 I am your **DataLens AI Executive Analyst**.\n\nI am actively analyzing your dataset with **{rows:,} rows** and **{cols} features**.\n\nHere are practical questions you can ask me right now:\n- 📈 *'What are the strongest correlations in this data?'*\n- 💰 *'What is the highest and lowest salary by department?'*\n- 🚨 *'Show outliers in Experience or Performance'* \n- 🧼 *'Summarize data hygiene and missingness risks'*\n- 🧠 *'Train an ML model on Salary or Attrition'*",
                "data": None,
            }

        # 1. Deterministic Correlation Queries
        if "correlation" in q or "relationship" in q or "correlated" in q:
            strong_corrs = self.stats.get("correlations", {}).get("strong_correlations", [])
            if strong_corrs:
                corr_rows = []
                for c in strong_corrs:
                    corr_rows.append({
                        "Feature 1": c["col1"],
                        "Feature 2": c["col2"],
                        "Pearson (r)": c["pearson"],
                        "Strength": f"{c['strength'].capitalize()} {c['direction']}"
                    })
                top_c = strong_corrs[0]
                return {
                    "answer": f"The strongest linear correlation is between **`{top_c['col1']}`** and **`{top_c['col2']}`** with a Pearson **$r = {top_c['pearson']}$** ({top_c['strength']} {top_c['direction']}).\n\n`[Evidence: Pearson Correlation Matrix calculation]`",
                    "data": corr_rows,
                }
            else:
                return {
                    "answer": "No strong linear correlations ($|r| \\ge 0.5$) were detected between numerical columns in this dataset.\n\n`[Evidence: Full pairwise correlation matrix evaluated]`",
                    "data": None,
                }

        # 2. Deterministic calculations before any generative model.
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
                            "data": res.to_dict(orient="records"),
                        }
                    if op in ["lowest", "minimum", "min"]:
                        res = self.raw_df.groupby(group_col)[target_col].min().reset_index().sort_values(by=target_col, ascending=True)
                        low_grp = res.iloc[0][group_col]
                        low_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The `{group_col}` with the lowest `{target_col}` is **{low_grp}** with a value of **{low_val:,.2f}**.\n\n`[Evidence: Group Minimum calculation]`",
                            "data": res.to_dict(orient="records"),
                        }

                    res = self.raw_df.groupby(group_col)[target_col].mean().round(2).reset_index().sort_values(by=target_col, ascending=False)
                    top_grp = res.iloc[0][group_col]
                    top_val = res.iloc[0][target_col]
                    return {
                        "answer": f"The average `{target_col}` across `{group_col}` is highest in **{top_grp}** at **{top_val:,.2f}**.\n\n`[Evidence: Group Mean calculation]`",
                        "data": res.to_dict(orient="records"),
                    }

                if op in ["highest", "maximum", "max"]:
                    val = self.raw_df[target_col].max()
                    return {"answer": f"The maximum value of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Max={val}]`", "data": None}
                if op in ["lowest", "minimum", "min"]:
                    val = self.raw_df[target_col].min()
                    return {"answer": f"The minimum value of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Min={val}]`", "data": None}

                val = self.raw_df[target_col].mean()
                return {"answer": f"The overall average (mean) of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Mean={val:,.2f}]`", "data": None}

        if "missing" in q or "null" in q or "na" in q:
            missing_data = self.raw_df.isna().sum().reset_index()
            missing_data.columns = ["Column", "Missing Count"]
            missing_data = missing_data[missing_data["Missing Count"] > 0]
            if missing_data.empty:
                return {"answer": "There are **0 missing values** across all columns in this dataset.\n\n`[Evidence: 100% Completeness]`", "data": None}
            return {
                "answer": f"Detected missing values in **{len(missing_data)} column(s)**:\n\n`[Evidence: Total Missing Cells = {self.quality.get('total_missing_cells')}]`",
                "data": missing_data.to_dict(orient="records"),
            }

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
                    "data": outlier_rows,
                }
            return {"answer": "No statistical outliers detected via IQR method.\n\n`[Evidence: Zero values outside [Q1-1.5IQR, Q3+1.5IQR]]`", "data": None}

        # 3. DataLens AI Grounded Reasoning with Full Dataset Context
        grounded_ctx = self._build_grounded_context()
        rag_docs = RAGEngine.search(user_query)
        rag_context = "\n".join([f"• **{d['title']}**: {d['content']}" for d in rag_docs]) if rag_docs else ""

        prompt = f"""You are DataLens AI, an expert, friendly, and practical data scientist.
Dataset Context:
{grounded_ctx}

Relevant Knowledge Base Articles:
{rag_context}

User question: {user_query}

Provide a direct, crystal-clear, and helpful answer in Markdown. Keep explanations simple, practical, and grounded in the data. Cite exact numbers and column names where helpful. Never mention third-party AI provider names, refer exclusively to DataLens AI."""

        res_text = self._call_llm(prompt)
        if res_text:
            return {"answer": res_text, "data": None}

        # 3. Deterministic Aggregation Fallback (Highest / Lowest / Average group queries)
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
                            "data": res.to_dict(orient="records"),
                        }
                    elif op in ["lowest", "minimum", "min"]:
                        res = self.raw_df.groupby(group_col)[target_col].min().reset_index().sort_values(by=target_col, ascending=True)
                        low_grp = res.iloc[0][group_col]
                        low_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The `{group_col}` with the lowest `{target_col}` is **{low_grp}** with a value of **{low_val:,.2f}**.\n\n`[Evidence: Group Minimum calculation]`",
                            "data": res.to_dict(orient="records"),
                        }
                    else:  # Average
                        res = self.raw_df.groupby(group_col)[target_col].mean().round(2).reset_index().sort_values(by=target_col, ascending=False)
                        top_grp = res.iloc[0][group_col]
                        top_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The average `{target_col}` across `{group_col}` is highest in **{top_grp}** at **{top_val:,.2f}**.\n\n`[Evidence: Group Mean calculation]`",
                            "data": res.to_dict(orient="records"),
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

        # 4. Missing values query
        if "missing" in q or "null" in q or "na" in q:
            missing_data = self.raw_df.isna().sum().reset_index()
            missing_data.columns = ["Column", "Missing Count"]
            missing_data = missing_data[missing_data["Missing Count"] > 0]
            if missing_data.empty:
                return {"answer": "There are **0 missing values** across all columns in this dataset.\n\n`[Evidence: 100% Completeness]`", "data": None}
            else:
                return {
                    "answer": f"Detected missing values in **{len(missing_data)} column(s)**:\n\n`[Evidence: Total Missing Cells = {self.quality.get('total_missing_cells')}]`",
                    "data": missing_data.to_dict(orient="records"),
                }

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
                    "data": outlier_rows,
                }
            else:
                return {"answer": "No statistical outliers detected via IQR method.\n\n`[Evidence: Zero values outside [Q1-1.5IQR, Q3+1.5IQR]]`", "data": None}

        # 6. Conceptual definitions (only if explicitly asking "what is", "explain")
        if any(term in q for term in ["what is", "explain", "definition of", "how does"]):
            if rag_docs:
                top_d = rag_docs[0]
                return {
                    "answer": f"### 📚 Knowledge Base: {top_d['title']}\n\n{top_d['content']}\n\n*Retrieved via DataLens RAG Engine.*",
                    "data": None,
                }

        return {
            "answer": f"I parsed your question: **'{user_query}'**.\n\nHere are specific questions you can ask me:\n- *'What is the average Salary by Department?'*\n- *'Show outliers in Experience'* \n- *'What are the strongest correlations?'*\n- *'Check for duplicate rows'*\n- *'Highest Performance_Score by Department'*",
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
