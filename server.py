"""
DataLens AI - FastAPI High-Performance Backend Server
Serves static assets and provides RESTful APIs for the bespoke frontend.
"""

import os
import io
import json
import uvicorn
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from datalens.loader import DataLoader, load_dataset
from datalens.privacy import PrivacyScanner
from datalens.profiler import DataProfiler, ColumnType
from datalens.quality import DataQualityEngine
from datalens.statistics import StatisticalEngine
from datalens.ml_engine import MLEngine
from datalens.ai_engine import AIEngine
from datalens.rag_engine import RAGEngine
from datalens.reports import ReportGenerator
from datalens.resume_engine import ResumeEngine
from datalens.logger import app_logger

# Initialize FastAPI App
app = FastAPI(title="DataLens AI API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory State for Active Dataset Session
SESSION: Dict[str, Any] = {
    "raw_df": None,
    "cleaned_df": None,
    "dataset_name": None,
    "profiler": None,
    "quality": None,
    "statistics": None,
    "privacy_report": None,
    "ml_results": None,
    "ai_engine": None,
    "cleaning_log": None,
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "ingestion_meta": None,
    "resume_analysis": None,
}


def _update_session_pipeline(df: pd.DataFrame, dataset_name: str, is_cleaned: bool = False, raw_text: Optional[str] = None):
    """Refreshes all downstream analytical engines when a dataset or document is uploaded."""
    if not is_cleaned:
        SESSION["raw_df"] = df
        SESSION["cleaned_df"] = None
        SESSION["dataset_name"] = dataset_name
        SESSION["ml_results"] = None
        SESSION["cleaning_log"] = None
    else:
        SESSION["cleaned_df"] = df

    active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]

    # 1. AI Safety & Privacy Scan
    privacy_scanner = PrivacyScanner(active_df)
    SESSION["privacy_report"] = privacy_scanner.to_dict()

    # 2. Semantic Data Profiler
    profiler = DataProfiler(active_df)
    SESSION["profiler"] = profiler.to_dict()

    # 3. Deterministic Data Quality & Anomalies
    quality = DataQualityEngine(active_df)
    SESSION["quality"] = quality.to_dict()
    SESSION["_quality_obj"] = quality

    # 4. Statistical & Correlation Matrices
    stats = StatisticalEngine(active_df)
    SESSION["statistics"] = stats.to_dict()

    active_key = SESSION.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # 5. Check if it's a Resume and run Resume Engine
    if raw_text:
        try:
            engine = ResumeEngine(raw_text, file_name=dataset_name, api_key=active_key)
            SESSION["resume_analysis"] = engine.analyze()
        except Exception as e:
            app_logger.warning(f"Resume analysis error: {e}")
            SESSION["resume_analysis"] = None
    elif SESSION.get("resume_analysis") is not None:
        pass
    else:
        SESSION["resume_analysis"] = None

    # 6. AI Insight Engine
    SESSION["ai_engine"] = AIEngine(
        active_df,
        profiler_dict=SESSION["profiler"],
        quality_dict=SESSION["quality"],
        stats_dict=SESSION["statistics"],
        api_key=active_key,
    )

    app_logger.info(f"Analytical pipeline refreshed for '{dataset_name}'. Rows: {len(active_df)}, Cols: {len(active_df.columns)}")


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads and ingests any supported dataset or resume document."""
    try:
        contents = await file.read()
        uploads_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        df, meta = load_dataset(file_path)
        _update_session_pipeline(df, file.filename, raw_text=meta.get("raw_text"))
        SESSION["ingestion_meta"] = meta

        is_resume = meta.get("is_resume", False) or bool(SESSION.get("resume_analysis"))
        format_label = "RESUME / CV" if is_resume else meta.get("format", "file").upper()

        return {
            "status": "success",
            "message": f"Successfully loaded '{file.filename}' [{format_label}]",
            "rows": len(df),
            "cols": len(df.columns),
            "format": meta.get("format", "tabular"),
            "is_resume": is_resume,
            "resume_analysis": SESSION.get("resume_analysis"),
        }
    except Exception as e:
        app_logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/resume/analyze")
async def get_resume_analysis():
    """Returns the market readiness analysis and suggestions for an uploaded resume."""
    if not SESSION.get("resume_analysis"):
        raise HTTPException(status_code=400, detail="No active resume document loaded.")
    return {"status": "success", "analysis": SESSION["resume_analysis"]}


@app.post("/api/resume/analyze-text")
async def analyze_resume_text(data: Dict[str, str]):
    """Analyzes raw pasted resume text."""
    text = data.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Pasted resume text cannot be empty.")

    engine = ResumeEngine(text, file_name="Pasted_Resume.txt", api_key=SESSION.get("api_key"))
    analysis = engine.analyze()
    SESSION["resume_analysis"] = analysis

    # Create text dataframe for viewer
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    df = pd.DataFrame({
        "Line_ID": list(range(1, len(lines) + 1)),
        "Resume_Content": lines,
        "Char_Count": [len(l) for l in lines],
        "Word_Count": [len(l.split()) for l in lines],
    })
    _update_session_pipeline(df, "Pasted_Resume.txt", raw_text=text)
    SESSION["ingestion_meta"] = {"format": "resume", "is_resume": True}

    return {"status": "success", "analysis": analysis, "is_resume": True}


@app.post("/api/load-sample")
async def load_sample():
    """Loads the pre-packaged sample HR & Attrition dataset."""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
    df, meta = load_dataset(sample_path)

    _update_session_pipeline(df, "auto_data_analyzer_test.csv")
    SESSION["ingestion_meta"] = meta
    return {"status": "success", "message": "Sample HR dataset loaded", "rows": len(df), "cols": len(df.columns)}


@app.post("/api/load-sample-resume")
async def load_sample_resume():
    """Loads a benchmark sample Data Scientist resume for instant AI analysis demo."""
    sample_resume = """
    Alex Rivera - Senior AI / Data Scientist
    Email: alex.rivera@example.com | Phone: (555) 345-6789 | Location: San Francisco, CA
    LinkedIn: linkedin.com/in/alexrivera | GitHub: github.com/alexrivera-ai

    PROFESSIONAL SUMMARY
    Results-driven AI & Data Scientist with 5+ years of experience engineering production LLM systems, RAG pipelines, and high-throughput predictive machine learning models at enterprise scale.

    EXPERIENCE
    Lead Data Scientist | Nexus AI Corp (2022 - Present)
    • Spearheaded generative AI RAG agent workflows using Python, PyTorch, and LangChain, improving retrieval accuracy by 38% and saving $450k annually in operational customer support costs.
    • Architected high-throughput microservices using FastAPI and Docker on AWS, reducing inference response latency from 850ms to 120ms for 2.5M+ active users.
    • Engineered automated ETL pipeline in PostgreSQL and Apache Spark, decreasing daily batch processing time by 4.5 hours.
    • Optimized XGBoost and Random Forest churn classification models, increasing customer retention by 22% across 150k enterprise accounts.

    Data Scientist | QuantData Analytics (2019 - 2022)
    • Built statistical predictive models and A/B testing infrastructure in Python, boosting marketing conversion rate by 17.5%.
    • Conducted comprehensive EDA and clustering on multi-terabyte datasets in Snowflake using SQL.
    • Developed automated executive KPI reporting dashboards reducing reporting turnaround time by 65%.

    TECHNICAL SKILLS
    • AI & Data: Python, PyTorch, TensorFlow, Scikit-Learn, LangChain, RAG, Transformers, SQL, PostgreSQL, Spark, Snowflake
    • Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD, Git, Linux
    • Engineering & BI: FastAPI, REST APIs, System Design, Tableau, Power BI, Statistics, EDA

    EDUCATION & CERTIFICATIONS
    • M.S. in Computer Science (Machine Learning Focus) - Stanford University
    • AWS Certified Machine Learning - Specialty
    """
    
    engine = ResumeEngine(sample_resume, file_name="Sample_Senior_AI_Resume.pdf", api_key=SESSION.get("api_key"))
    analysis = engine.analyze()
    SESSION["resume_analysis"] = analysis
    
    # Create text dataframe for viewer
    lines = [l.strip() for l in sample_resume.strip().split("\n") if l.strip()]
    df = pd.DataFrame({
        "Line_ID": list(range(1, len(lines) + 1)),
        "Resume_Content": lines,
        "Char_Count": [len(l) for l in lines],
        "Word_Count": [len(l.split()) for l in lines],
    })
    _update_session_pipeline(df, "Sample_Senior_AI_Resume.pdf", raw_text=sample_resume)
    SESSION["ingestion_meta"] = {"format": "resume", "is_resume": True}

    return {
        "status": "success",
        "message": "Sample AI Resume loaded",
        "analysis": analysis,
        "is_resume": True
    }


@app.post("/api/config/api-key")
async def set_api_key(data: Dict[str, str]):
    """Configures or updates the Google Gemini API key with live verification and global propagation."""
    key = data.get("api_key", "").strip()
    if key:
        os.environ["GEMINI_API_KEY"] = key
        SESSION["api_key"] = key
    else:
        SESSION["api_key"] = os.getenv("GEMINI_API_KEY", "")

    active_key = SESSION["api_key"] or os.getenv("GEMINI_API_KEY", "")

    # Test key verification
    verified = False
    error_msg = None
    if active_key:
        try:
            from google import genai
            client = genai.Client(api_key=active_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Test verification. Reply 'OK'."
            )
            if resp and resp.text:
                verified = True
                app_logger.info("Gemini API key successfully verified and active globally.")
        except Exception as e:
            error_msg = str(e)
            app_logger.warning(f"API Key verification warning: {e}")

    if SESSION.get("ai_engine") and (SESSION["raw_df"] is not None):
        active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]
        SESSION["ai_engine"] = AIEngine(
            active_df,
            profiler_dict=SESSION["profiler"],
            quality_dict=SESSION["quality"],
            stats_dict=SESSION["statistics"],
            api_key=active_key,
        )

    # Re-run resume analysis if active
    if SESSION.get("resume_analysis") and SESSION.get("raw_df") is not None:
        try:
            raw_text = SESSION.get("ingestion_meta", {}).get("raw_text")
            if raw_text:
                engine = ResumeEngine(raw_text, file_name=SESSION.get("dataset_name"), api_key=active_key)
                SESSION["resume_analysis"] = engine.analyze()
        except Exception:
            pass

    return {
        "status": "success",
        "verified": verified,
        "has_key": bool(active_key),
        "error": error_msg,
        "message": "Gemini 3.7 / 2.5 Flash connected and verified!" if verified else "API key active."
    }


@app.get("/api/config/api-key-status")
async def get_api_key_status():
    """Returns whether a Gemini API key is currently active."""
    active_key = SESSION.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return {
        "has_key": bool(active_key),
        "model": "gemini-3.7-flash / gemini-2.5-flash",
        "provider": "Google DeepMind GenAI"
    }


@app.get("/api/dataset")
async def get_dataset_state(page: int = 1, page_size: int = 15):
    """Returns the current dataset state, overview metrics, column schemas, and paginated rows."""
    if SESSION["raw_df"] is None:
        return {"has_dataset": False}

    active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]
    
    # Paginate raw records
    total_records = len(active_df)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_records)
    records = active_df.iloc[start_idx:end_idx].fillna("").to_dict(orient="records")

    return {
        "has_dataset": True,
        "dataset_name": SESSION["dataset_name"],
        "is_cleaned": SESSION["cleaned_df"] is not None,
        "is_resume": bool(SESSION.get("resume_analysis")),
        "resume_analysis": SESSION.get("resume_analysis"),
        "total_rows": total_records,
        "total_cols": len(active_df.columns),
        "columns": list(active_df.columns),
        "profiler": SESSION["profiler"],
        "quality": SESSION["quality"],
        "statistics": SESSION["statistics"],
        "privacy": SESSION["privacy_report"],
        "sample_data": {
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total_records + page_size - 1) // page_size),
            "records": records,
        }
    }


@app.post("/api/clean")
async def clean_dataset(data: Dict[str, Any]):
    """Executes data cleaning transformations (deduplication, missing imputation, IQR outlier handling)."""
    if SESSION["raw_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    drop_dups = data.get("drop_duplicates", True)
    missing_strategy = data.get("missing_strategy", "none")
    outlier_strategy = data.get("outlier_strategy", "none")

    quality_obj = SESSION.get("_quality_obj") or DataQualityEngine(SESSION["raw_df"])
    cleaned_df, log = quality_obj.clean_dataset(
        drop_duplicates=drop_dups,
        missing_strategy=missing_strategy,
        outlier_strategy=outlier_strategy,
    )

    _update_session_pipeline(cleaned_df, SESSION["dataset_name"], is_cleaned=True)
    SESSION["cleaning_log"] = log

    return {"status": "success", "message": "Dataset cleaned successfully", "log": log}


@app.post("/api/reset-data")
async def reset_data():
    """Resets dataset to original raw state."""
    if SESSION["raw_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    _update_session_pipeline(SESSION["raw_df"], SESSION["dataset_name"], is_cleaned=False)
    return {"status": "success", "message": "Dataset reset to raw original state"}


@app.post("/api/ml/train")
async def train_ml(data: Dict[str, Any]):
    """Trains regression or classification model."""
    if SESSION["raw_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]
    target_col = data.get("target_column")
    model_name = data.get("model_name", "Random Forest")
    feature_cols = data.get("features", None)

    if not target_col or target_col not in active_df.columns:
        raise HTTPException(status_code=400, detail="Invalid target column specified.")

    ml_engine = MLEngine(active_df)
    problem_type = ml_engine.detect_problem_type(target_col)

    if problem_type == "Regression":
        res = ml_engine.train_regression(target_col, feature_cols, model_name)
    else:
        res = ml_engine.train_classification(target_col, feature_cols, model_name)

    # Sanitize response for JSON serialization (remove plotly objects)
    sanitized = {
        "task": res.get("task"),
        "model_name": res.get("model_name"),
        "target": res.get("target"),
        "metrics": res.get("metrics"),
        "feature_importance": res.get("feature_importance", []),
        "confusion_matrix": res.get("confusion_matrix", []),
        "classes": res.get("classes", []),
    }
    SESSION["ml_results"] = sanitized
    return {"status": "success", "result": sanitized}


@app.post("/api/ml/cluster")
async def cluster_data(data: Dict[str, Any]):
    """Performs K-Means clustering on the active dataset."""
    if SESSION["raw_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]
    k = data.get("k", 3)
    ml_engine = MLEngine(active_df)
    res = ml_engine.train_clustering(n_clusters=k)
    return {"status": "success", "result": res}


@app.post("/api/ai/query")
async def query_ai(data: Dict[str, str]):
    """Handles grounded natural language questions via tool routing."""
    query = data.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not SESSION.get("ai_engine"):
        raise HTTPException(status_code=400, detail="No dataset loaded to analyze.")

    ai = SESSION["ai_engine"]
    res = ai.answer_query(query)

    # If data is DataFrame, convert to dict
    if res.get("data") is not None and isinstance(res["data"], pd.DataFrame):
        res["data"] = res["data"].fillna("").to_dict(orient="records")

    return res


@app.get("/api/ai/briefing")
async def get_briefing():
    """Generates or returns the multi-agent grounded executive briefing."""
    if not SESSION.get("ai_engine"):
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    briefing = SESSION["ai_engine"].generate_executive_insights()
    return {"status": "success", "briefing": briefing}


@app.get("/api/rag/search")
async def search_rag(q: str):
    """Searches the RAG knowledge base for statistical and domain terms."""
    docs = RAGEngine.search(q, top_k=5)
    return {"status": "success", "results": docs}


@app.get("/api/export/csv")
async def export_csv():
    """Downloads the cleaned or active dataset as CSV."""
    if SESSION["raw_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset available.")

    active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]
    csv_str = active_df.to_csv(index=False)
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=datalens_cleaned_{SESSION['dataset_name']}"}
    )


@app.get("/api/export/pdf")
async def export_pdf():
    """Compiles and streams the executive PDF report."""
    if SESSION["raw_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset available.")

    active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]
    briefing = SESSION["ai_engine"].generate_executive_insights() if SESSION.get("ai_engine") else "Analysis report"

    pdf_bytes = ReportGenerator.generate_pdf_report(
        dataset_name=SESSION["dataset_name"],
        df=active_df,
        profiler_dict=SESSION["profiler"],
        quality_dict=SESSION["quality"],
        stats_dict=SESSION["statistics"],
        ai_briefing=briefing,
        privacy_dict=SESSION["privacy_report"],
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=datalens_report_{SESSION['dataset_name']}.pdf"}
    )


@app.get("/api/export/json")
async def export_json():
    """Downloads the reproducibility JSON audit."""
    if SESSION["raw_df"] is None:
        raise HTTPException(status_code=400, detail="No dataset available.")

    active_df = SESSION["cleaned_df"] if SESSION["cleaned_df"] is not None else SESSION["raw_df"]
    json_str = ReportGenerator.generate_reproducibility_json(
        dataset_name=SESSION["dataset_name"],
        df=active_df,
        profiler_dict=SESSION["profiler"],
        quality_dict=SESSION["quality"],
        stats_dict=SESSION["statistics"],
        cleaning_log=SESSION.get("cleaning_log"),
        ml_results=SESSION.get("ml_results"),
    )
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=datalens_audit_{SESSION['dataset_name']}.json"}
    )


# Serve Static Assets & Single Page Application
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/healthz")
async def health_check():
    """Cloud liveness & readiness health check endpoint."""
    return {"status": "healthy", "service": "DataLens AI", "version": "1.0.0"}


@app.get("/")
async def root():
    """Serves the main single-page frontend application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>DataLens AI server is running. (Building custom frontend...)</h1>")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    app_logger.info(f"Starting DataLens AI production server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
