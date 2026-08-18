"""
Agentic Multi-Agent Orchestration Engine for DataLens AI
Coordinates specialized autonomous agents (InsightAgent, AnomalyAgent, ChatAgent, ReportAgent).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from .logger import app_logger
from .rag_engine import RAGEngine


class InsightAgent:
    """Specialized agent for macro pattern recognition, feature correlations, and distributions."""

    def __init__(self, stats_dict: Dict[str, Any], profiler_dict: Dict[str, Any]):
        self.stats = stats_dict
        self.profiler = profiler_dict

    def run(self) -> List[Dict[str, Any]]:
        insights = []
        corrs = self.stats.get("correlations", {}).get("strong_correlations", [])
        for c in corrs[:3]:
            insights.append({
                "agent": "InsightAgent",
                "category": "Correlation Dynamics",
                "finding": f"Strong relationship between `{c['col1']}` and `{c['col2']}`.",
                "evidence": f"Pearson r = {c['pearson']} ({c['strength']} {c['direction']})",
                "importance": "High" if abs(c["pearson"]) >= 0.8 else "Medium",
            })

        num_stats = self.stats.get("numerical", {})
        for col, s in num_stats.items():
            if abs(s.get("skewness", 0)) > 1.0:
                insights.append({
                    "agent": "InsightAgent",
                    "category": "Distribution Asymmetry",
                    "finding": f"`{col}` has significant {s['skewness_label'].lower()}.",
                    "evidence": f"Skewness = {s['skewness']}, Mean = {s['mean']}, Median = {s['median_50']}",
                    "importance": "Medium",
                })

        return insights


class AnomalyAgent:
    """Specialized agent for detecting data hygiene hazards, duplicate records, and outliers."""

    def __init__(self, quality_dict: Dict[str, Any]):
        self.quality = quality_dict

    def run(self) -> List[Dict[str, Any]]:
        anomalies = []
        dups = self.quality.get("duplicate_rows", 0)
        if dups > 0:
            anomalies.append({
                "agent": "AnomalyAgent",
                "category": "Redundancy Risk",
                "finding": f"Detected {dups} duplicate row(s) in dataset.",
                "evidence": f"Duplicate % = {self.quality.get('duplicate_pct')}%",
                "action": "Deduplicate dataset before ML model training.",
            })

        missing = self.quality.get("missing_details", [])
        for m in missing:
            anomalies.append({
                "agent": "AnomalyAgent",
                "category": "Missing Data Gap",
                "finding": f"Column `{m['column']}` is missing {m['missing_count']} values.",
                "evidence": f"Missing % = {m['missing_pct']}%",
                "action": "Apply median or mode imputation strategy.",
            })

        outliers = self.quality.get("outliers", {})
        for col, info in outliers.items():
            iqr = info["iqr"]
            if iqr["outlier_count"] > 0:
                anomalies.append({
                    "agent": "AnomalyAgent",
                    "category": "IQR Statistical Outlier",
                    "finding": f"`{col}` contains {iqr['outlier_count']} extreme value(s).",
                    "evidence": f"Bounds: [{iqr['lower_bound']}, {iqr['upper_bound']}], Outliers: {iqr['outlier_values'][:3]}",
                    "action": "Consider IQR capping (Winsorization) to avoid model distortion.",
                })

        return anomalies


class ReportAgent:
    """Specialized agent for compiling multi-agent outputs into a coherent executive report."""

    def __init__(self, insight_agent: InsightAgent, anomaly_agent: AnomalyAgent):
        self.insights = insight_agent.run()
        self.anomalies = anomaly_agent.run()

    def compile(self) -> str:
        lines = []
        lines.append("## 🤖 Autonomous Multi-Agent Executive Briefing\n")

        lines.append("### 🔍 1. Macro Insights & Patterns (InsightAgent)")
        if self.insights:
            for item in self.insights:
                lines.append(f"- **{item['finding']}**  \n  `[Evidence: {item['evidence']}]`")
        else:
            lines.append("- No abnormal feature correlations detected.")

        lines.append("\n### ⚠️ 2. Anomaly & Hygiene Hazards (AnomalyAgent)")
        if self.anomalies:
            for item in self.anomalies:
                lines.append(f"- **{item['finding']}**  \n  `[Evidence: {item['evidence']}]` → *Action: {item['action']}*")
        else:
            lines.append("- Dataset is completely clean (zero missing, duplicates, or extreme outliers).")

        return "\n".join(lines)


class AgentOrchestrator:
    """Main orchestrator coordinating all specialized analysis sub-agents."""

    def __init__(self, df: pd.DataFrame, profiler_dict: Dict[str, Any], quality_dict: Dict[str, Any], stats_dict: Dict[str, Any]):
        self.df = df
        self.insight_agent = InsightAgent(stats_dict, profiler_dict)
        self.anomaly_agent = AnomalyAgent(quality_dict)
        self.report_agent = ReportAgent(self.insight_agent, self.anomaly_agent)
        app_logger.info("AgentOrchestrator initialized with 4 autonomous sub-agents.")

    def run_autonomous_audit(self) -> Dict[str, Any]:
        """Runs all sub-agents and aggregates findings."""
        return {
            "insights": self.insight_agent.run(),
            "anomalies": self.anomaly_agent.run(),
            "executive_briefing": self.report_agent.compile(),
        }
