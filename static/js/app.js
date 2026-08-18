/**
 * DataLens AI - Custom Vanilla JavaScript Controller
 * Floating Top Navbar Hubs • Sub-Nav Pills • Liquid Glass Orange Dark Theme
 */

document.addEventListener("DOMContentLoaded", () => {
  let appState = {
    dataset: null,
    currentPage: 1,
    charts: {},
  };

  // DOM Elements
  const hubButtons = document.querySelectorAll(".hub-tab-btn");
  const viewPanels = document.querySelectorAll(".view-panel");
  const navCenterHubs = document.getElementById("navCenterHubs");
  const mobileNavToggle = document.getElementById("mobileNavToggle");
  const navBrandBtn = document.getElementById("navBrandBtn");
  const topDatasetPill = document.getElementById("topDatasetPill");
  const topDatasetName = document.getElementById("topDatasetName");
  const resetDataBtn = document.getElementById("resetDataBtn");
  const quickSampleBtn = document.getElementById("quickSampleBtn");
  const fileInput = document.getElementById("fileInput");
  const dropZone = document.getElementById("dropZone");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const saveApiKeyBtn = document.getElementById("saveApiKeyBtn");

  // Chart.js Global Dark Theme Defaults
  if (window.Chart) {
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
  }

  // =========================================================
  // 1. Primary Hub Navigation & Mobile Toggle
  // =========================================================
  function switchHub(targetHubId) {
    hubButtons.forEach((btn) => {
      if (btn.getAttribute("data-hub") === targetHubId) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    viewPanels.forEach((panel) => {
      if (panel.id === targetHubId) {
        panel.classList.add("active");
      } else {
        panel.classList.remove("active");
      }
    });

    // Close mobile drawer if open
    navCenterHubs.classList.remove("mobile-open");

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  hubButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const hubId = btn.getAttribute("data-hub");
      switchHub(hubId);
    });
  });

  navBrandBtn.addEventListener("click", () => switchHub("hub-home"));

  mobileNavToggle.addEventListener("click", () => {
    navCenterHubs.classList.toggle("mobile-open");
  });

  // =========================================================
  // 2. Sub-Navigation Pills (Inside Each Hub)
  // =========================================================
  document.querySelectorAll(".subnav-pills-bar").forEach((bar) => {
    bar.addEventListener("click", (e) => {
      const pill = e.target.closest(".subnav-pill");
      if (!pill) return;

      const parentHub = bar.closest(".view-panel");
      parentHub.querySelectorAll(".subnav-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");

      const targetSubViewId = pill.getAttribute("data-subview");
      parentHub.querySelectorAll(".subview-section").forEach((sec) => {
        if (sec.id === targetSubViewId) {
          sec.style.display = "block";
        } else {
          sec.style.display = "none";
        }
      });

      // If RAG subview selected and empty, auto search
      if (targetSubViewId === "sub-rag" && !document.getElementById("ragResultsContainer").hasChildNodes()) {
        searchRAG("");
      }
    });
  });

  // =========================================================
  // 3. Dataset State Sync & Conditional Navbar Lock
  // =========================================================
  async function fetchDatasetState(page = 1) {
    try {
      const res = await fetch(`/api/dataset?page=${page}`);
      const data = await res.json();
      
      if (!data.has_dataset) {
        // No dataset: lock and hide navbar hubs
        navCenterHubs.classList.add("hidden-nav");
        mobileNavToggle.style.display = "none";
        topDatasetPill.style.display = "none";
        resetDataBtn.style.display = "none";
        switchHub("hub-home");
        return;
      }

      // Dataset present: unlock navbar hubs
      navCenterHubs.classList.remove("hidden-nav");
      mobileNavToggle.style.display = "";
      appState.dataset = data;
      appState.currentPage = page;
      updateUI();
    } catch (err) {
      console.error("Error fetching dataset state:", err);
    }
  }

  function updateUI() {
    const d = appState.dataset;
    if (!d || !d.has_dataset) {
      navCenterHubs.classList.add("hidden-nav");
      mobileNavToggle.style.display = "none";
      topDatasetPill.style.display = "none";
      resetDataBtn.style.display = "none";
      return;
    }

    // Unlock and show full navigation hubs
    navCenterHubs.classList.remove("hidden-nav");
    mobileNavToggle.style.display = "";

    // 1. Top Bar Pill
    topDatasetPill.style.display = "inline-flex";
    topDatasetName.textContent = `${d.dataset_name} ${d.is_cleaned ? "(Cleaned)" : ""}`;
    resetDataBtn.style.display = d.is_cleaned ? "inline-flex" : "none";

    // 1.5 Handle Resume Analysis View
    const navResumeBtn = document.getElementById("navResumeBtn");
    if (d.is_resume && d.resume_analysis) {
      if (navResumeBtn) navResumeBtn.style.display = "inline-flex";
      renderResumeAnalysis(d.resume_analysis);
    } else {
      if (navResumeBtn) navResumeBtn.style.display = "none";
    }

    // 2. Privacy Scanner Box
    const pBox = document.getElementById("privacyStatusBox");
    if (d.privacy) {
      const score = d.privacy.privacy_safety_score;
      const color = score >= 80 ? "#10B981" : score >= 50 ? "#F59E0B" : "#EF4444";
      pBox.innerHTML = `
        <div style="font-weight: 700; color: ${color}; font-size: 1.15rem; margin-bottom: 0.4rem;">
          🛡️ Privacy Safety Score: ${score}/100
        </div>
        <p style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.6;">
          Flagged Sensitive Columns: <b style="color: var(--text-pure);">${d.privacy.sensitive_columns_count}</b> &nbsp;|&nbsp; 
          PII Matches Redacted: <b style="color: var(--text-pure);">${d.privacy.total_pii_occurrences}</b>
        </p>
      `;
    }

    // 3. Overview Metrics Grid
    const mGrid = document.getElementById("overviewMetrics");
    const health = d.quality ? d.quality.health_score : 100;
    const hColor = health >= 80 ? "#10B981" : health >= 60 ? "#F59E0B" : "#EF4444";

    mGrid.innerHTML = `
      <div class="glass-card kpi-card"><div class="kpi-label">Total Records</div><div class="kpi-val orange">${d.total_rows.toLocaleString()}</div></div>
      <div class="glass-card kpi-card"><div class="kpi-label">Features</div><div class="kpi-val">${d.total_cols}</div></div>
      <div class="glass-card kpi-card"><div class="kpi-label">Health Score</div><div class="kpi-val" style="color: ${hColor}">${health}/100</div></div>
      <div class="glass-card kpi-card"><div class="kpi-label">Missing Cells</div><div class="kpi-val">${d.profiler ? d.profiler.total_missing_cells : 0}</div></div>
      <div class="glass-card kpi-card"><div class="kpi-label">Duplicate Rows</div><div class="kpi-val">${d.quality ? d.quality.duplicate_rows : 0}</div></div>
    `;

    // 4. Schema Table
    const schemaTbody = document.querySelector("#schemaTable tbody");
    schemaTbody.innerHTML = "";
    if (d.profiler && d.profiler.columns) {
      d.profiler.columns.forEach((c) => {
        const tr = document.createElement("tr");
        const typeBadgeClass = getTypeBadgeClass(c.semantic_type);
        tr.innerHTML = `
          <td><strong style="color: #FFFFFF;">${c.column_name}</strong></td>
          <td><span class="badge ${typeBadgeClass}">${c.semantic_type}</span></td>
          <td><code>${c.pandas_dtype}</code></td>
          <td>${c.missing_count} (${c.missing_pct}%)</td>
          <td>${c.unique_count}</td>
          <td>${c.sample_values.slice(0, 3).join(", ")}</td>
        `;
        schemaTbody.appendChild(tr);
      });
    }

    // 5. Raw Data Table Pagination
    renderRawDataTable();

    // 6. Quality Missing Values Chart
    renderMissingValuesChart();

    // 7. Stats Table
    renderStatsTable();

    // 8. Correlations Table
    renderCorrelationsTable();

    // 9. Outliers Table
    renderOutliersTable();

    // 10. Populate Dropdowns
    populateColumnDropdowns();
  }

  function getTypeBadgeClass(type) {
    switch (type) {
      case "Identifier": return "badge-id";
      case "Numerical": return "badge-num";
      case "Categorical": return "badge-cat";
      case "Date": return "badge-date";
      case "Boolean": return "badge-bool";
      default: return "badge-id";
    }
  }

  function renderRawDataTable() {
    const sample = appState.dataset.sample_data;
    if (!sample) return;

    const thead = document.getElementById("rawDataTableHead");
    const tbody = document.getElementById("rawDataTableBody");
    const pageIndicator = document.getElementById("pageIndicator");

    thead.innerHTML = `<tr>${appState.dataset.columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
    tbody.innerHTML = "";

    sample.records.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = appState.dataset.columns.map((c) => `<td>${row[c] !== undefined ? row[c] : ""}</td>`).join("");
      tbody.appendChild(tr);
    });

    pageIndicator.textContent = `Page ${sample.page} of ${sample.total_pages}`;
  }

  // =========================================================
  // 4. Chart Rendering
  // =========================================================
  function renderMissingValuesChart() {
    const d = appState.dataset;
    if (!d || !d.quality) return;

    const ctx = document.getElementById("missingValuesChart").getContext("2d");
    if (appState.charts.missing) {
      appState.charts.missing.destroy();
    }

    const missingDetails = d.quality.missing_details || [];
    const labels = missingDetails.map((m) => m.column);
    const data = missingDetails.map((m) => m.missing_count);

    appState.charts.missing = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels.length ? labels : ["Zero Missing Values"],
        datasets: [{
          label: "Missing Count",
          data: data.length ? data : [0],
          backgroundColor: "#FF6B00",
          borderRadius: 8,
          borderColor: "#FF851B",
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: "rgba(255, 255, 255, 0.05)" } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function renderStatsTable() {
    const d = appState.dataset;
    if (!d || !d.statistics) return;

    const tbody = document.querySelector("#statsTable tbody");
    tbody.innerHTML = "";
    const numStats = d.statistics.numerical || {};

    Object.entries(numStats).forEach(([col, s]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="color: #FFFFFF;">${col}</strong></td>
        <td>${s.mean}</td>
        <td>${s.median_50}</td>
        <td>${s.std}</td>
        <td>${s.min}</td>
        <td>${s.q1_25}</td>
        <td>${s.q3_75}</td>
        <td>${s.max}</td>
        <td>${s.skewness}</td>
        <td><span class="badge ${Math.abs(s.skewness) > 1 ? 'badge-cat' : 'badge-bool'}">${s.skewness_label}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderCorrelationsTable() {
    const d = appState.dataset;
    if (!d || !d.statistics) return;

    const tbody = document.querySelector("#correlationTable tbody");
    tbody.innerHTML = "";
    const corrs = (d.statistics.correlations && d.statistics.correlations.strong_correlations) || [];

    if (!corrs.length) {
      tbody.innerHTML = "<tr><td colspan='6' style='text-align: center; color: var(--text-muted);'>No strong pairwise correlations detected.</td></tr>";
      return;
    }

    corrs.forEach((c) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="color: #FFFFFF;">${c.col1}</strong></td>
        <td><strong style="color: #FFFFFF;">${c.col2}</strong></td>
        <td><code style="color: var(--orange-bright);">${c.pearson}</code></td>
        <td><code>${c.spearman}</code></td>
        <td><span class="badge ${c.strength === 'Strong' ? 'badge-cat' : 'badge-id'}">${c.strength}</span></td>
        <td>${c.direction}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderOutliersTable() {
    const d = appState.dataset;
    if (!d || !d.quality) return;

    const tbody = document.querySelector("#outliersTable tbody");
    tbody.innerHTML = "";
    const outliers = d.quality.outliers || {};

    let hasOutliers = false;
    Object.entries(outliers).forEach(([col, info]) => {
      const iqr = info.iqr;
      if (iqr.outlier_count > 0) {
        hasOutliers = true;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong style="color: #FFFFFF;">${col}</strong></td>
          <td><span class="badge badge-cat">${iqr.outlier_count}</span></td>
          <td>${iqr.outlier_pct}%</td>
          <td>${iqr.lower_bound}</td>
          <td>${iqr.upper_bound}</td>
          <td>${iqr.outlier_values.slice(0, 4).join(", ")}</td>
        `;
        tbody.appendChild(tr);
      }
    });

    if (!hasOutliers) {
      tbody.innerHTML = "<tr><td colspan='6' style='text-align: center; color: #10B981;'>✅ Zero statistical outliers detected via IQR method.</td></tr>";
    }
  }

  function populateColumnDropdowns() {
    const d = appState.dataset;
    if (!d || !d.columns) return;

    const chartColSelect = document.getElementById("chartSelectCol");
    const mlTargetSelect = document.getElementById("mlTargetCol");

    chartColSelect.innerHTML = "";
    mlTargetSelect.innerHTML = "";

    d.columns.forEach((col) => {
      const opt1 = document.createElement("option");
      opt1.value = col;
      opt1.textContent = col;
      chartColSelect.appendChild(opt1);

      if (!col.toLowerCase().endsWith("id")) {
        const opt2 = document.createElement("option");
        opt2.value = col;
        opt2.textContent = col;
        mlTargetSelect.appendChild(opt2);
      }
    });
  }

  // =========================================================
  // 4b. Deep Thinking Resume & Career Intelligence Renderer
  // =========================================================
  function renderResumeAnalysis(analysis) {
    if (!analysis) return;

    // 1. Overall Score out of 10.0
    const rawScore = analysis.score_out_of_10 !== undefined ? analysis.score_out_of_10 : ((analysis.market_score || 70) / 10.0).toFixed(1);
    const scoreVal = Number(rawScore).toFixed(1);
    const sColor = scoreVal >= 8.5 ? "#10B981" : scoreVal >= 7.0 ? "#FF851B" : "#EF4444";

    const masterScoreEl = document.getElementById("resumeMasterScore");
    if (masterScoreEl) {
      masterScoreEl.textContent = scoreVal;
      masterScoreEl.style.color = sColor;
      masterScoreEl.style.textShadow = `0 0 32px ${sColor}88`;
    }

    const badgeEl = document.getElementById("resumePercentileBadge");
    if (badgeEl) {
      badgeEl.textContent = analysis.percentile_tier || "85th Percentile • Strong Market Contender";
      badgeEl.style.color = sColor;
      badgeEl.style.borderColor = `${sColor}44`;
      badgeEl.style.background = `${sColor}14`;
    }

    const p = analysis.profile || {};
    const docNameEl = document.getElementById("resumeDocName");
    if (docNameEl) docNameEl.textContent = p.file_name || "Resume.pdf";

    const bulletCountEl = document.getElementById("resumeBulletCount");
    if (bulletCountEl) bulletCountEl.textContent = p.total_bullet_points || 0;

    const quantRatioEl = document.getElementById("resumeQuantRatio");
    if (quantRatioEl) {
      const qPct = p.total_bullet_points ? Math.round((p.quantified_bullets_count / p.total_bullet_points) * 100) : 0;
      quantRatioEl.textContent = `${qPct}%`;
    }

    // 2. Dimensional Sub-Scores Grid (Out of 10.0)
    const rGrid = document.getElementById("resumeMetricsGrid");
    const sub = analysis.sub_scores || {};
    
    const dimensions = [
      { key: "impact", defaultScore: 7.5, label: "Business Impact & Metrics", icon: "📈", desc: "Numbers, $, %, scale metrics" },
      { key: "verbs", defaultScore: 7.8, label: "Executive Power Verbs", icon: "⚡", desc: "Decisive action vs passive tone" },
      { key: "skills", defaultScore: 8.2, label: "2026 Tech Stack Alignment", icon: "🧠", desc: "Modern AI, Cloud, Systems density" },
      { key: "ats", defaultScore: 9.0, label: "ATS Architecture & Structure", icon: "🛡️", desc: "Machine readability & contact completeness" },
      { key: "leadership", defaultScore: 7.0, label: "Seniority & Scope Signals", icon: "🎖️", desc: "Architecture, mentorship, ownership" },
    ];

    rGrid.innerHTML = dimensions.map((d) => {
      const scoreObj = sub[d.key] || {};
      const sc = (scoreObj.score !== undefined ? scoreObj.score : d.defaultScore).toFixed(1);
      const scColor = sc >= 8.5 ? "#10B981" : sc >= 7.0 ? "#FF851B" : "#EF4444";
      const progressWidth = Math.min(100, Math.max(10, Math.round((sc / 10.0) * 100)));

      return `
        <div class="glass-card kpi-card" style="text-align: left; padding: 1.25rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
            <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-dim); text-transform: uppercase;">${d.icon} ${d.label}</span>
            <span style="font-size: 1.25rem; font-weight: 800; color: ${scColor};">${sc} <small style="font-size: 0.75rem; color: var(--text-dim);">/ 10</small></span>
          </div>
          <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; margin: 0.4rem 0;">
            <div style="width: ${progressWidth}%; height: 100%; background: ${scColor}; border-radius: 3px; box-shadow: 0 0 10px ${scColor}88;"></div>
          </div>
          <div style="font-size: 0.75rem; color: var(--text-dim);">${d.desc}</div>
        </div>
      `;
    }).join("");

    // 3. Skills Match & Missing Radar
    const sContainer = document.getElementById("resumeSkillsContainer");
    if (sContainer) {
      sContainer.innerHTML = "";
      const matched = analysis.matched_skills || {};
      const recommended = analysis.recommended_keywords || {};

      Object.keys(matched).forEach((cat) => {
        const foundList = matched[cat] || [];
        const recList = recommended[cat] || [];
        const catDiv = document.createElement("div");
        catDiv.style.marginBottom = "1.25rem";
        catDiv.innerHTML = `
          <div style="font-size: 0.9rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.45rem;">${cat}</div>
          <div style="display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
            ${foundList.map((s) => `<span class="badge badge-bool">✓ ${s}</span>`).join("") || '<span style="font-size: 0.8rem; color: var(--text-dim);">No direct keywords detected</span>'}
          </div>
          ${recList.length ? `
            <div style="font-size: 0.8rem; color: var(--orange-bright); margin-top: 0.25rem;">
              Recommended 2026 Skills to Integrate: ${recList.map((s) => `<code style="color: var(--orange-bright); margin-right: 0.35rem; background: rgba(255,107,0,0.08); padding: 2px 6px; border-radius: 4px;">+ ${s}</code>`).join("")}
            </div>
          ` : ''}
        `;
        sContainer.appendChild(catDiv);
      });
    }

    // 4. Candidate ATS Diagnostic
    const pContainer = document.getElementById("resumeProfileContainer");
    if (pContainer) {
      pContainer.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 0.85rem; font-size: 0.92rem;">
          <div><b>📄 Uploaded File:</b> <span style="color: var(--text-pure);">${p.file_name || "Resume.pdf"}</span></div>
          <div><b>📧 Email Address:</b> <span style="color: var(--text-orange);">${p.detected_email || "Not found"}</span></div>
          <div><b>📱 Phone Number:</b> <span style="color: var(--text-muted);">${p.detected_phone || "Not found"}</span></div>
          <div><b>🌐 LinkedIn Profile:</b> <span class="badge ${p.has_linkedin ? 'badge-bool' : 'badge-id'}">${p.has_linkedin ? '✓ Linked' : '✗ Missing link'}</span></div>
          <div><b>💻 GitHub / Portfolio:</b> <span class="badge ${p.has_github ? 'badge-bool' : 'badge-id'}">${p.has_github ? '✓ Linked' : '✗ Missing link'}</span></div>
          <div style="border-top: 1px solid var(--border-glass); padding-top: 0.85rem; margin-top: 0.25rem;">
            <b>Total Bullet Statements Scanned:</b> <span style="color: var(--text-pure); font-weight: 700;">${p.total_bullet_points || 0}</span>
          </div>
          <div>
            <b>Quantified Metric Bullets ($ / % / Scale):</b> <span style="color: #10B981; font-weight: 700;">${p.quantified_bullets_count || 0}</span>
          </div>
          <div>
            <b>Executive Power Verbs:</b> <span style="color: var(--orange-bright); font-weight: 700;">${p.action_verb_count || 0}</span>
          </div>
          <div>
            <b>Passive Phrasing Traps Found:</b> <span style="color: #EF4444; font-weight: 700;">${p.weak_phrase_count || 0} (${(p.weak_phrases_found || []).join(", ") || "None"})</span>
          </div>
        </div>
      `;
    }

    // 5. Deep Thinking Executive Report
    const sugBox = document.getElementById("resumeSuggestionsContent");
    if (sugBox) {
      const insights = analysis.deep_insights || analysis.suggestions || {};
      if (insights.markdown) {
        sugBox.innerHTML = marked.parse(insights.markdown);
      } else {
        sugBox.innerHTML = "<p style='color: var(--text-muted);'>Executive analysis report generating...</p>";
      }
    }
  }

  // =========================================================
  // 5. File Ingestion & Sample Loading
  // =========================================================
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      uploadFile(e.target.files[0]);
    }
  });

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (res.ok) {
        await fetchDatasetState(1);
        if (data.is_resume) {
          switchHub("hub-resume");
        } else {
          switchHub("hub-data");
        }
      } else {
        alert(data.detail || "Failed to upload and parse file. Please verify file contains valid data.");
      }
    } catch (err) {
      console.error("Upload error:", err);
      alert("Error uploading file. Check that the document contains readable text or tabular data.");
    }
  }

  quickSampleBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/load-sample", { method: "POST" });
      if (res.ok) {
        await fetchDatasetState(1);
        switchHub("hub-data");
      }
    } catch (err) {
      console.error("Sample load error:", err);
    }
  });

  const heroSampleBtn = document.getElementById("heroSampleBtn");
  if (heroSampleBtn) {
    heroSampleBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/load-sample", { method: "POST" });
        if (res.ok) {
          await fetchDatasetState(1);
          switchHub("hub-data");
        }
      } catch (err) {
        console.error("Sample load error:", err);
      }
    });
  }

  const heroResumeBtn = document.getElementById("heroResumeBtn");
  if (heroResumeBtn) {
    heroResumeBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/load-sample-resume", { method: "POST" });
        if (res.ok) {
          await fetchDatasetState(1);
          switchHub("hub-resume");
        }
      } catch (err) {
        console.error("Sample resume error:", err);
      }
    });
  }

  const analyzePastedResumeBtn = document.getElementById("analyzePastedResumeBtn");
  const pastedResumeInput = document.getElementById("pastedResumeInput");
  if (analyzePastedResumeBtn && pastedResumeInput) {
    analyzePastedResumeBtn.addEventListener("click", async () => {
      const text = pastedResumeInput.value.trim();
      if (!text) {
        alert("Please paste your resume text before clicking analyze.");
        return;
      }
      analyzePastedResumeBtn.disabled = true;
      analyzePastedResumeBtn.textContent = "Analyzing...";

      try {
        const res = await fetch("/api/resume/analyze-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const data = await res.json();
        analyzePastedResumeBtn.disabled = false;
        analyzePastedResumeBtn.textContent = "⚡ Analyze Pasted Resume";

        if (res.ok) {
          await fetchDatasetState(1);
          renderResumeAnalysis(data.analysis);
        } else {
          alert(data.detail || "Resume analysis failed.");
        }
      } catch (err) {
        console.error("Paste analysis error:", err);
        analyzePastedResumeBtn.disabled = false;
        analyzePastedResumeBtn.textContent = "⚡ Analyze Pasted Resume";
      }
    });
  }

  resetDataBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/reset-data", { method: "POST" });
      if (res.ok) {
        await fetchDatasetState(1);
      }
    } catch (err) {
      console.error("Reset error:", err);
    }
  });

  // Pagination Handlers
  document.getElementById("prevPageBtn").addEventListener("click", () => {
    if (appState.currentPage > 1) {
      fetchDatasetState(appState.currentPage - 1);
    }
  });

  document.getElementById("nextPageBtn").addEventListener("click", () => {
    const sample = appState.dataset.sample_data;
    if (sample && appState.currentPage < sample.total_pages) {
      fetchDatasetState(appState.currentPage + 1);
    }
  });

  // =========================================================
  // 6. Data Cleaning Pipeline
  // =========================================================
  document.getElementById("applyCleaningBtn").addEventListener("click", async () => {
    const payload = {
      drop_duplicates: document.getElementById("cleanDropDups").checked,
      missing_strategy: document.getElementById("cleanMissingStrat").value,
      outlier_strategy: document.getElementById("cleanOutlierStrat").value,
    };

    try {
      const res = await fetch("/api/clean", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        alert(`Cleaned successfully! Removed ${data.log.removed_duplicates} dups, handled ${data.log.missing_handled} missing cols, adjusted ${data.log.outliers_adjusted} outliers.`);
        await fetchDatasetState(1);
      }
    } catch (err) {
      console.error("Cleaning error:", err);
    }
  });

  // =========================================================
  // 7. Custom Chart Builder
  // =========================================================
  document.getElementById("renderChartBtn").addEventListener("click", () => {
    const col = document.getElementById("chartSelectCol").value;
    const type = document.getElementById("chartSelectType").value;
    const d = appState.dataset;
    if (!d || !col) return;

    const ctx = document.getElementById("customChartCanvas").getContext("2d");
    if (appState.charts.custom) {
      appState.charts.custom.destroy();
    }

    document.getElementById("customChartTitle").textContent = `${type.toUpperCase()} Chart: ${col}`;

    const records = d.sample_data ? d.sample_data.records : [];
    const counts = {};
    records.forEach((r) => {
      const v = r[col];
      if (v !== undefined && v !== "") {
        counts[v] = (counts[v] || 0) + 1;
      }
    });

    const labels = Object.keys(counts).slice(0, 15);
    const values = Object.values(counts).slice(0, 15);

    appState.charts.custom = new Chart(ctx, {
      type: type === "line" ? "line" : "bar",
      data: {
        labels: labels,
        datasets: [{
          label: col,
          data: values,
          backgroundColor: "rgba(255, 107, 0, 0.75)",
          borderColor: "#FF851B",
          borderWidth: 1.5,
          fill: type === "line",
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: true, grid: { color: "rgba(255, 255, 255, 0.06)" } },
          x: { grid: { display: false } },
        },
      },
    });
  });

  // =========================================================
  // 8. ML Studio
  // =========================================================
  document.getElementById("trainMlBtn").addEventListener("click", async () => {
    const targetCol = document.getElementById("mlTargetCol").value;
    const modelName = document.getElementById("mlModelName").value;
    const btn = document.getElementById("trainMlBtn");

    btn.disabled = true;
    btn.textContent = "Training Model...";

    try {
      const res = await fetch("/api/ml/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: targetCol, model_name: modelName }),
      });
      const data = await res.json();
      btn.disabled = false;
      btn.textContent = "🚀 Train & Evaluate Model";

      if (res.ok) {
        const result = data.result;
        document.getElementById("mlResultsContainer").style.display = "block";

        // Render Metrics
        const grid = document.getElementById("mlMetricsGrid");
        grid.innerHTML = Object.entries(result.metrics).map(([k, v]) => `
          <div class="glass-card kpi-card">
            <div class="kpi-label">${k}</div>
            <div class="kpi-val orange">${v}</div>
          </div>
        `).join("");

        // Render Importance Chart
        if (result.feature_importance && result.feature_importance.length) {
          const ctx = document.getElementById("mlImportanceChart").getContext("2d");
          if (appState.charts.importance) appState.charts.importance.destroy();

          const labels = result.feature_importance.map((f) => f.Feature);
          const vals = result.feature_importance.map((f) => f.Importance);

          appState.charts.importance = new Chart(ctx, {
            type: "bar",
            data: {
              labels: labels,
              datasets: [{
                label: "Importance",
                data: vals,
                backgroundColor: "rgba(255, 107, 0, 0.8)",
                borderColor: "#FF851B",
                borderRadius: 6,
              }],
            },
            options: {
              indexAxis: "y",
              responsive: true,
              plugins: { legend: { display: false } },
              scales: {
                x: { grid: { color: "rgba(255,255,255,0.06)" } },
                y: { grid: { display: false } },
              },
            },
          });
        }
      } else {
        alert(data.detail || "ML Training failed.");
      }
    } catch (err) {
      console.error("ML Error:", err);
      btn.disabled = false;
      btn.textContent = "🚀 Train & Evaluate Model";
    }
  });

  // =========================================================
  // 9. AI Insights Briefing
  // =========================================================
  document.getElementById("refreshBriefingBtn").addEventListener("click", async () => {
    const box = document.getElementById("briefingContentBox");
    box.innerHTML = "<p style='color: var(--orange-bright);'>⚡ Compiling multi-agent executive briefing...</p>";

    try {
      const res = await fetch("/api/ai/briefing");
      const data = await res.json();
      if (res.ok) {
        box.innerHTML = marked.parse(data.briefing);
      } else {
        box.innerHTML = `<p style='color: var(--rose);'>${data.detail || "Failed to generate briefing."}</p>`;
      }
    } catch (err) {
      console.error("Briefing error:", err);
      box.innerHTML = "<p style='color: var(--rose);'>Error generating AI briefing.</p>";
    }
  });

  // =========================================================
  // 10. AI Data Analyst Chatbot
  // =========================================================
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatMessages = document.getElementById("chatMessages");

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    appendChatBubble("user", query);
    chatInput.value = "";

    const thinkingBubble = appendChatBubble("assistant", "Thinking...");

    try {
      const res = await fetch("/api/ai/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query }),
      });
      const data = await res.json();
      thinkingBubble.remove();

      if (res.ok) {
        let content = marked.parse(data.answer);
        if (data.data && Array.isArray(data.data) && data.data.length) {
          const keys = Object.keys(data.data[0]);
          const tableHtml = `
            <div class="table-glass-container" style="margin-top: 0.75rem;">
              <table>
                <thead><tr>${keys.map((k) => `<th>${k}</th>`).join("")}</tr></thead>
                <tbody>${data.data.slice(0, 5).map((row) => `<tr>${keys.map((k) => `<td>${row[k]}</td>`).join("")}</tr>`).join("")}</tbody>
              </table>
            </div>
          `;
          content += tableHtml;
        }
        appendChatBubble("assistant", content, true);
      } else {
        appendChatBubble("assistant", `<span style='color: var(--rose);'>${data.detail || "Query failed."}</span>`, true);
      }
    } catch (err) {
      thinkingBubble.remove();
      appendChatBubble("assistant", "<span style='color: var(--rose);'>Error communicating with AI Analyst.</span>", true);
    }
  });

  function appendChatBubble(role, htmlContent, isHtml = false) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role}`;
    if (isHtml) {
      bubble.innerHTML = htmlContent;
    } else {
      bubble.textContent = htmlContent;
    }
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  // =========================================================
  // 11. RAG Knowledge Search
  // =========================================================
  const ragInput = document.getElementById("ragSearchInput");
  ragInput.addEventListener("input", debounce(() => searchRAG(ragInput.value), 300));

  async function searchRAG(query) {
    try {
      const res = await fetch(`/api/rag/search?q=${encodeURIComponent(query)}`);
      const data = await res.json();
      const container = document.getElementById("ragResultsContainer");
      container.innerHTML = "";

      if (data.results && data.results.length) {
        data.results.forEach((d) => {
          const card = document.createElement("div");
          card.className = "glass-card";
          card.innerHTML = `
            <h3 style="font-size: 1.15rem; color: var(--orange-bright); margin-bottom: 0.5rem;">📖 ${d.title}</h3>
            <p style="font-size: 0.93rem; line-height: 1.6; color: var(--text-main);">${d.content}</p>
            <div style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-dim);">
              Keywords: <code style="color: var(--orange-bright);">${d.keywords.join(", ")}</code>
            </div>
          `;
          container.appendChild(card);
        });
      } else {
        container.innerHTML = "<p style='color: var(--text-muted);'>No matching concepts found.</p>";
      }
    } catch (err) {
      console.error("RAG search error:", err);
    }
  }

  // =========================================================
  // 12. Google Gemini Connection Status & Setup
  // =========================================================
  const geminiBadge = document.getElementById("geminiStatusBadge");
  const geminiNotice = document.getElementById("geminiFeaturesNotice");

  async function checkApiKeyStatus() {
    try {
      const res = await fetch("/api/config/api-key-status");
      const data = await res.json();
      if (data.has_key && geminiBadge) {
        geminiBadge.className = "badge badge-bool";
        geminiBadge.textContent = "⚡ Gemini 2.5 Active";
        if (geminiNotice) {
          geminiNotice.innerHTML = "<span style='color: #10B981;'>✓ Connected to Gemini 2.5 Flash. Vision OCR, Deep Thinking Resume Engine, and AI Analyst are fully activated.</span>";
        }
      }
    } catch (e) {
      console.warn("Could not check Gemini key status:", e);
    }
  }

  saveApiKeyBtn.addEventListener("click", async () => {
    const key = apiKeyInput.value.trim();
    if (!key) {
      alert("Please enter a valid Gemini API key (starts with AIzaSy...).");
      return;
    }

    saveApiKeyBtn.disabled = true;
    saveApiKeyBtn.textContent = "Verifying...";

    try {
      const res = await fetch("/api/config/api-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key }),
      });
      const data = await res.json();
      saveApiKeyBtn.disabled = false;
      saveApiKeyBtn.textContent = "Connect";

      if (res.ok) {
        if (data.verified) {
          if (geminiBadge) {
            geminiBadge.className = "badge badge-bool";
            geminiBadge.textContent = "⚡ Gemini 2.5 Active";
          }
          if (geminiNotice) {
            geminiNotice.innerHTML = "<span style='color: #10B981;'>✓ Verified! Gemini 2.5 Flash Vision OCR, Deep Thinking Career Suite, and AI Analyst are active.</span>";
          }
          alert("🎉 Google Gemini API key connected and verified successfully!");
        } else {
          alert("Key saved! (Note: Running in offline deterministic fallback if quota or verification fails).");
        }
      } else {
        alert(data.detail || "Failed to update API key.");
      }
    } catch (err) {
      console.error("Key save error:", err);
      saveApiKeyBtn.disabled = false;
      saveApiKeyBtn.textContent = "Connect";
    }
  });

  // Utility: Debounce
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Initial Data & API Key Status Fetch
  checkApiKeyStatus();
  fetchDatasetState(1);
});

