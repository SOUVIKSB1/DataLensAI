"""
Visualization Engine for DataLens AI
Generates intelligent, interactive Plotly charts tailored to data types and relationships.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class VisualizerEngine:
    """
    Constructs responsive, theme-consistent Plotly visualizations and automated chart recommendations.
    """

    COLOR_PALETTE = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0", "#4895EF", "#560BAD"]
    THEME_TEMPLATE = "plotly_white"

    @staticmethod
    def create_histogram(df: pd.DataFrame, col: str, nbins: int = 25) -> go.Figure:
        """Builds a distribution histogram with marginal boxplot."""
        clean_df = df.dropna(subset=[col])
        fig = px.histogram(
            clean_df,
            x=col,
            nbins=nbins,
            marginal="box",
            title=f"Distribution of {col}",
            color_discrete_sequence=["#4361EE"],
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title=col,
            yaxis_title="Frequency",
            bargap=0.05,
        )
        return fig

    @staticmethod
    def create_box_plot(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None) -> go.Figure:
        """Builds an interactive box & whisker plot, optionally grouped by a category."""
        clean_df = df.dropna(subset=[y_col] + ([x_col] if x_col else []))
        fig = px.box(
            clean_df,
            y=y_col,
            x=x_col,
            color=x_col if x_col else None,
            points="outliers",
            title=f"Box Plot: {y_col}" + (f" grouped by {x_col}" if x_col else ""),
            color_discrete_sequence=VisualizerEngine.COLOR_PALETTE,
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, col: str, top_n: int = 15) -> go.Figure:
        """Builds a frequency bar chart for categorical variables."""
        counts = df[col].value_counts().head(top_n).reset_index()
        counts.columns = [col, "Count"]
        
        fig = px.bar(
            counts,
            x=col,
            y="Count",
            text="Count",
            title=f"Top Categories in {col}",
            color=col,
            color_discrete_sequence=VisualizerEngine.COLOR_PALETTE,
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            xaxis_tickangle=-30 if len(counts) > 5 else 0,
        )
        return fig

    @staticmethod
    def create_scatter_plot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        add_trendline: bool = False,
    ) -> go.Figure:
        """Builds a scatter plot for relationship analysis."""
        subset_cols = [x_col, y_col] + ([color_col] if color_col else [])
        clean_df = df.dropna(subset=subset_cols)

        fig = px.scatter(
            clean_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col else None,
            trendline="ols" if add_trendline else None,
            title=f"Relationship: {x_col} vs {y_col}",
            color_discrete_sequence=VisualizerEngine.COLOR_PALETTE,
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_line_chart(df: pd.DataFrame, date_col: str, val_col: str) -> go.Figure:
        """Builds a time-series line chart."""
        temp_df = df.dropna(subset=[date_col, val_col]).copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df = temp_df.dropna(subset=[date_col]).sort_values(by=date_col)

        fig = px.line(
            temp_df,
            x=date_col,
            y=val_col,
            markers=True,
            title=f"Trend Analysis: {val_col} over {date_col}",
            color_discrete_sequence=["#4361EE"],
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> Optional[go.Figure]:
        """Builds an annotated correlation heatmap."""
        num_df = df.select_dtypes(include=[np.number])
        # Filter out id columns
        filtered_cols = [c for c in num_df.columns if not (str(c).lower().endswith("id") or "_id" in str(c).lower())]
        if len(filtered_cols) < 2:
            return None

        corr_matrix = num_df[filtered_cols].corr(method=method).round(2)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"{method.capitalize()} Correlation Heatmap",
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_missing_values_bar(df: pd.DataFrame) -> Optional[go.Figure]:
        """Builds a bar chart showing missing values per column."""
        missing = df.isna().sum()
        missing = missing[missing > 0].reset_index()
        if missing.empty:
            return None

        missing.columns = ["Column", "MissingCount"]
        missing["Percentage"] = ((missing["MissingCount"] / len(df)) * 100).round(1)

        fig = px.bar(
            missing,
            x="Column",
            y="MissingCount",
            text="Percentage",
            title="Missing Values by Column (%)",
            color="MissingCount",
            color_continuous_scale="Reds",
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        return fig

    @classmethod
    def recommend_visualizations(cls, df: pd.DataFrame, column_profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determines the most insightful automated visualizations based on column types and correlations.
        """
        recommendations = []
        num_cols = [c for c, info in column_profiles.items() if info.get("semantic_type") == "Numerical"]
        cat_cols = [c for c, info in column_profiles.items() if info.get("semantic_type") == "Categorical"]
        date_cols = [c for c, info in column_profiles.items() if info.get("semantic_type") == "Date"]

        # 1. Distribution of key numerical variables
        for col in num_cols[:3]:
            recommendations.append({
                "type": "histogram",
                "title": f"Distribution of {col}",
                "description": f"Displays frequency spread, central tendency, and skewness for {col}.",
                "fig": cls.create_histogram(df, col),
            })

        # 2. Key categorical breakdowns
        for col in cat_cols[:2]:
            recommendations.append({
                "type": "bar",
                "title": f"Category Breakdown: {col}",
                "description": f"Shows distribution and most frequent classes in {col}.",
                "fig": cls.create_bar_chart(df, col),
            })

        # 3. Correlation heatmap if >= 2 numerical columns
        if len(num_cols) >= 2:
            heat_fig = cls.create_correlation_heatmap(df)
            if heat_fig:
                recommendations.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "description": "Exposes linear relationships and collinearity among numerical features.",
                    "fig": heat_fig,
                })

        # 4. Numerical vs Categorical Box Plot
        if num_cols and cat_cols:
            recommendations.append({
                "type": "box",
                "title": f"{num_cols[0]} by {cat_cols[0]}",
                "description": f"Analyzes variance and median of {num_cols[0]} across {cat_cols[0]} groups.",
                "fig": cls.create_box_plot(df, y_col=num_cols[0], x_col=cat_cols[0]),
            })

        # 5. Date trends if date column exists
        if date_cols and num_cols:
            recommendations.append({
                "type": "line",
                "title": f"{num_cols[0]} Trend Over Time",
                "description": f"Tracks progression of {num_cols[0]} across {date_cols[0]}.",
                "fig": cls.create_line_chart(df, date_col=date_cols[0], val_col=num_cols[0]),
            })

        return recommendations
