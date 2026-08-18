/**
 * DataLens AI - High-Performance Controller & UI Engine
 * Floating Top Navbar Hubs • Interactive Ingestion Tabs • Deep Thinking Resume 10.0 Suite • Gemini Integration
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
  const heroSampleBtn = document.getElementById("heroSampleBtn");
  const heroResumeBtn = document.getElementById("heroResumeBtn");
  const fileInput = document.getElementById("fileInput");
  const dropZone = document.getElementById("dropZone");

  // Ingestion Tabs
  const tabUploadFileBtn = document.getElementById("tabUploadFileBtn");
  const tabPasteTextBtn = document.getElementById("tabPasteTextBtn");
  const tabSampleDatasetsBtn = document.getElementById("tabSampleDatasetsBtn");
  const ingestTabUpload = document.getElementById("ingestTabUpload");
  const ingestTabPaste = document.getElementById("ingestTabPaste");
  const ingestTabSamples = document.getElementById("ingestTabSamples");
  const pasteResumeInput = document.getElementById("pasteResumeInput");
  const analyzePastedResumeBtn = document.getElementById("analyzePastedResumeBtn");

  // Gemini API Key Elements
  const navApiKeyBtn = document.getElementById("navApiKeyBtn");
  const apiKeyModal = document.getElementById("apiKeyModal");
  const closeApiKeyModalBtn = document.getElementById("closeApiKeyModalBtn");
  const cancelApiKeyModalBtn = document.getElementById("cancelApiKeyModalBtn");
  const saveModalApiKeyBtn = document.getElementById("saveModalApiKeyBtn");
  const modalApiKeyInput = document.getElementById("modalApiKeyInput");
  const modalKeyFeedback = document.getElementById("modalKeyFeedback");
  const navGeminiStatusIcon = document.getElementById("navGeminiStatusIcon");
  const navGeminiStatusText = document.getElementById("navGeminiStatusText");
  const homeGeminiStatusPill = document.getElementById("homeGeminiStatusPill");

  // Chart.js Global Dark Theme Defaults
  if (window.Chart) {
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
  }

  // =========================================================
  // 1. Hub Navigation
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

    if (navCenterHubs) navCenterHubs.classList.remove("mobile-open");
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  hubButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const hubId = btn.getAttribute("data-hub");
      switchHub(hubId);
    });
  });

  // In-Page Interactive Tool Action Buttons
  const actionBtnClean = document.getElementById("actionBtnClean");
  const actionBtnAi = document.getElementById("actionBtnAi");
  const actionBtnMl = document.getElementById("actionBtnMl");
  const actionBtnExport = document.getElementById("actionBtnExport");

  if (actionBtnClean) actionBtnClean.addEventListener("click", () => switchHub("hub-quality"));
  if (actionBtnAi) actionBtnAi.addEventListener("click", () => switchHub("hub-ai"));
  if (actionBtnMl) actionBtnMl.addEventListener("click", () => switchHub("hub-ml"));
  if (actionBtnExport) actionBtnExport.addEventListener("click", () => switchHub("hub-export"));

  if (navBrandBtn) navBrandBtn.addEventListener("click", () => switchHub("hub-home"));
  if (mobileNavToggle) {
    mobileNavToggle.addEventListener("click", () => {
      navCenterHubs.classList.toggle("mobile-open");
    });
  }

  // =========================================================
  // 2. Ingestion Hub Tabs
  // =========================================================
  function setIngestTab(tab) {
    [tabUploadFileBtn, tabPasteTextBtn, tabSampleDatasetsBtn].forEach((btn) => {
      if (btn) btn.className = "btn btn-sm btn-secondary";
    });
    [ingestTabUpload, ingestTabPaste, ingestTabSamples].forEach((sec) => {
      if (sec) sec.style.display = "none";
    });

    if (tab === "upload") {
      if (tabUploadFileBtn) tabUploadFileBtn.className = "btn btn-sm btn-outline-orange active";
      if (ingestTabUpload) ingestTabUpload.style.display = "block";
    } else if (tab === "paste") {
      if (tabPasteTextBtn) tabPasteTextBtn.className = "btn btn-sm btn-outline-orange active";
      if (ingestTabPaste) ingestTabPaste.style.display = "block";
    } else if (tab === "samples") {
      if (tabSampleDatasetsBtn) tabSampleDatasetsBtn.className = "btn btn-sm btn-outline-orange active";
      if (ingestTabSamples) ingestTabSamples.style.display = "block";
    }
  }

  if (tabUploadFileBtn) tabUploadFileBtn.addEventListener("click", () => setIngestTab("upload"));
  if (tabPasteTextBtn) tabPasteTextBtn.addEventListener("click", () => setIngestTab("paste"));
  if (tabSampleDatasetsBtn) tabSampleDatasetsBtn.addEventListener("click", () => setIngestTab("samples"));

  // =========================================================
  // 3. Gemini API Key Modal Management
  // =========================================================
  function openApiKeyModal() {
    if (apiKeyModal) apiKeyModal.style.display = "flex";
  }
  function closeApiKeyModal() {
    if (apiKeyModal) apiKeyModal.style.display = "none";
  }

  if (navApiKeyBtn) navApiKeyBtn.addEventListener("click", openApiKeyModal);
  if (closeApiKeyModalBtn) closeApiKeyModalBtn.addEventListener("click", closeApiKeyModal);
  if (cancelApiKeyModalBtn) cancelApiKeyModalBtn.addEventListener("click", closeApiKeyModal);

  async function checkGeminiStatus() {
    try {
      const res = await fetch("/api/config/api-key-status");
      const data = await res.json();
      if (data.has_key) {
        if (navGeminiStatusIcon) navGeminiStatusIcon.textContent = "⚡";
        if (navGeminiStatusText) navGeminiStatusText.textContent = "Gemini Active";
        if (homeGeminiStatusPill) homeGeminiStatusPill.innerHTML = "⚡ Gemini 3.7 / 2.5 Active";
      } else {
        if (navGeminiStatusIcon) navGeminiStatusIcon.textContent = "🔑";
        if (navGeminiStatusText) navGeminiStatusText.textContent = "Connect Key";
        if (homeGeminiStatusPill) homeGeminiStatusPill.innerHTML = "🔒 Offline Math Engine";
      }
    } catch (e) {
      console.warn("Could not fetch Gemini status:", e);
    }
  }

  if (saveModalApiKeyBtn) {
    saveModalApiKeyBtn.addEventListener("click", async () => {
      const key = modalApiKeyInput.value.trim();
      if (!key) {
        alert("Please enter a valid Gemini API key (starts with AIzaSy...).");
        return;
      }
      saveModalApiKeyBtn.disabled = true;
      saveModalApiKeyBtn.textContent = "Verifying...";
      if (modalKeyFeedback) modalKeyFeedback.innerHTML = "<span style='color: var(--orange-bright);'>Testing connection with Gemini 2.5/3.7 Flash...</span>";

      try {
        const res = await fetch("/api/config/api-key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: key }),
        });
        const data = await res.json();
        saveModalApiKeyBtn.disabled = false;
        saveModalApiKeyBtn.textContent = "Connect Key";

        if (res.ok) {
          if (data.verified) {
            checkGeminiStatus();
            if (modalKeyFeedback) modalKeyFeedback.innerHTML = "<span style='color: #10B981;'>✓ Connected & Verified with Gemini!</span>";
            setTimeout(closeApiKeyModal, 1200);
          } else {
            if (modalKeyFeedback) modalKeyFeedback.innerHTML = "<span style='color: var(--amber);'>Key saved! (Fallback mode active if quota exceeded).</span>";
            setTimeout(closeApiKeyModal, 1500);
          }
        }
      } catch (err) {
        saveModalApiKeyBtn.disabled = false;
        saveModalApiKeyBtn.textContent = "Connect Key";
        if (modalKeyFeedback) modalKeyFeedback.innerHTML = "<span style='color: var(--rose);'>Failed to connect.</span>";
      }
    });
  }

  const navUploadNewBtn = document.getElementById("navUploadNewBtn");
  if (navUploadNewBtn && fileInput) {
    navUploadNewBtn.addEventListener("click", () => fileInput.click());
  }

  const initialDropZoneHtml = dropZone ? dropZone.innerHTML : "";

  // =========================================================
  // 4. Ingestion Triggers (Dropzone, Browse, Paste, Sample)
  // =========================================================
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
      }
    });
  }

  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });
  }

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    if (dropZone) {
      dropZone.innerHTML = `
        <div class="drop-icon" style="animation: spin 1s infinite linear;">⚙️</div>
        <h4 style="color: #FFFFFF; font-size: 1.2rem;">Analyzing '${file.name}'...</h4>
        <p style="color: var(--text-muted); font-size: 0.88rem; margin-top: 0.5rem;">Extracting structure, PII scan & running intelligence pipeline...</p>
      `;
    }

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      // Reset input and dropzone so user never needs to reload
      if (fileInput) fileInput.value = "";
      if (dropZone) dropZone.innerHTML = initialDropZoneHtml;

      if (res.ok) {
        await fetchDatasetState(1);
        if (data.is_resume) {
          switchHub("hub-resume");
        } else if (data.is_marksheet) {
          switchHub("hub-marksheet");
        } else {
          switchHub("hub-data");
        }
      } else {
        alert(data.detail || "File processing failed.");
      }
    } catch (err) {
      console.error("Upload error:", err);
      if (fileInput) fileInput.value = "";
      if (dropZone) dropZone.innerHTML = initialDropZoneHtml;
      alert("Error uploading file.");
    }
  }

  // Paste Resume Action
  if (analyzePastedResumeBtn) {
    analyzePastedResumeBtn.addEventListener("click", async () => {
      const text = pasteResumeInput ? pasteResumeInput.value.trim() : "";
      if (!text || text.length < 30) {
        alert("Please paste more than 30 characters of resume or document content.");
        return;
      }
      analyzePastedResumeBtn.disabled = true;
      analyzePastedResumeBtn.textContent = "⚡ Analyzing Resume (Scoring 10.0)...";

      try {
        const res = await fetch("/api/resume/analyze-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text }),
        });
        const data = await res.json();
        analyzePastedResumeBtn.disabled = false;
        analyzePastedResumeBtn.textContent = "💼 Analyze & Score Resume (10.0)";

        if (res.ok) {
          await fetchDatasetState(1);
          switchHub("hub-resume");
        } else {
          alert(data.detail || "Analysis failed.");
        }
      } catch (err) {
        console.error("Paste error:", err);
        analyzePastedResumeBtn.disabled = false;
        analyzePastedResumeBtn.textContent = "💼 Analyze & Score Resume (10.0)";
      }
    });
  }

  // Load Sample HR Data
  async function loadSampleHRData() {
    try {
      const res = await fetch("/api/load-sample");
      const data = await res.json();
      if (res.ok) {
        await fetchDatasetState(1);
        switchHub("hub-data");
      }
    } catch (err) {
      console.error("Sample HR load error:", err);
    }
  }

  // Load Sample Resume Data
  async function loadSampleResumeData() {
    try {
      const res = await fetch("/api/load-sample-resume", { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        await fetchDatasetState(1);
        switchHub("hub-resume");
      }
    } catch (err) {
      console.error("Sample resume load error:", err);
    }
  }

  // Load Sample Marksheet Data
  async function loadSampleMarksheetData() {
    try {
      const res = await fetch("/api/load-sample-marksheet", { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        await fetchDatasetState(1);
        switchHub("hub-marksheet");
      }
    } catch (err) {
      console.error("Sample marksheet load error:", err);
    }
  }

  if (quickSampleBtn) quickSampleBtn.addEventListener("click", loadSampleHRData);
  if (heroSampleBtn) heroSampleBtn.addEventListener("click", loadSampleHRData);
  if (heroResumeBtn) heroResumeBtn.addEventListener("click", loadSampleResumeData);
  const heroMarksheetBtn = document.getElementById("heroMarksheetBtn");
  if (heroMarksheetBtn) heroMarksheetBtn.addEventListener("click", loadSampleMarksheetData);

  // =========================================================
  // 5. Dataset State Synchronization & UI Hydration
  // =========================================================
  async function fetchDatasetState(page = 1) {
    try {
      const res = await fetch(`/api/dataset?page=${page}`);
      const data = await res.json();
      
      if (!data.has_dataset) {
        if (navCenterHubs) navCenterHubs.classList.add("hidden-nav");
        if (topDatasetPill) topDatasetPill.style.display = "none";
        if (resetDataBtn) resetDataBtn.style.display = "none";
        switchHub("hub-home");
        return;
      }

      if (navCenterHubs) navCenterHubs.classList.remove("hidden-nav");
      appState.dataset = data;
      appState.currentPage = page;
      updateUI();
    } catch (err) {
      console.error("Error fetching dataset state:", err);
    }
  }

  function updateUI() {
    const d = appState.dataset;
    if (!d || !d.has_dataset) return;

    if (topDatasetPill) {
      topDatasetPill.style.display = "inline-flex";
      let typeBadge = "";
      if (d.is_resume) typeBadge = " [Resume]";
      else if (d.is_marksheet) typeBadge = " [Marksheet]";
      else typeBadge = " [CSV Data]";
      if (topDatasetName) topDatasetName.textContent = `${d.dataset_name}${typeBadge} ${d.is_cleaned ? "(Cleaned)" : ""}`;
    }
    if (resetDataBtn) resetDataBtn.style.display = d.is_cleaned ? "inline-flex" : "none";

    const navResumeBtn = document.getElementById("navResumeBtn");
    if (d.is_resume && d.resume_analysis) {
      if (navResumeBtn) navResumeBtn.style.display = "inline-flex";
      renderResumeAnalysis(d.resume_analysis);
    } else {
      if (navResumeBtn) navResumeBtn.style.display = "none";
    }

    const navMarksheetBtn = document.getElementById("navMarksheetBtn");
    if (d.is_marksheet && d.marksheet_analysis) {
      if (navMarksheetBtn) navMarksheetBtn.style.display = "inline-flex";
      renderMarksheetAnalysis(d.marksheet_analysis);
    } else {
      if (navMarksheetBtn) navMarksheetBtn.style.display = "none";
    }

    // Update KPI cards in Data Explorer
    const kpiRows = document.getElementById("kpiRows");
    const kpiCols = document.getElementById("kpiCols");
    const kpiMissing = document.getElementById("kpiMissing");
    const kpiMemory = document.getElementById("kpiMemory");

    if (kpiRows) kpiRows.textContent = d.total_rows.toLocaleString();
    if (kpiCols) kpiCols.textContent = d.total_cols;
    if (kpiMissing && d.profiler) kpiMissing.textContent = `${d.profiler.missing_cells_pct}%`;
    if (kpiMemory && d.profiler) kpiMemory.textContent = d.profiler.memory_usage_mb ? `${d.profiler.memory_usage_mb} MB` : "< 1 MB";

    // Column schema badges
    const columnSchemaBadges = document.getElementById("columnSchemaBadges");
    if (columnSchemaBadges && d.profiler && d.profiler.columns) {
      columnSchemaBadges.innerHTML = d.profiler.columns.map((c) => `
        <span class="badge ${getTypeBadgeClass(c.semantic_type)}" style="padding: 0.4rem 0.75rem; font-size: 0.85rem;">
          <strong>${c.column_name}</strong> &bull; ${c.semantic_type}
        </span>
      `).join("");
    }

    // Render paginated data table
    renderDataTable();

    // Render Quality Hub
    renderQualityHub();

    // Populate ML Target Column Dropdown
    populateMLDropdown();
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

  // =========================================================
  // 6. Resume Studio Sub-Tabs & Rendering
  // =========================================================
  const resumeTabAuditBtn = document.getElementById("resumeTabAuditBtn");
  const resumeTabRewritesBtn = document.getElementById("resumeTabRewritesBtn");
  const resumeTabTechBtn = document.getElementById("resumeTabTechBtn");
  const resumeTabProfileBtn = document.getElementById("resumeTabProfileBtn");

  const resumeSubViewAudit = document.getElementById("resumeSubViewAudit");
  const resumeSubViewRewrites = document.getElementById("resumeSubViewRewrites");
  const resumeSubViewTech = document.getElementById("resumeSubViewTech");
  const resumeSubViewProfile = document.getElementById("resumeSubViewProfile");

  function setResumeSubTab(tab) {
    [resumeTabAuditBtn, resumeTabRewritesBtn, resumeTabTechBtn, resumeTabProfileBtn].forEach((btn) => {
      if (btn) btn.className = "btn btn-sm btn-secondary";
    });
    [resumeSubViewAudit, resumeSubViewRewrites, resumeSubViewTech, resumeSubViewProfile].forEach((sec) => {
      if (sec) sec.style.display = "none";
    });

    if (tab === "audit") {
      if (resumeTabAuditBtn) resumeTabAuditBtn.className = "btn btn-sm btn-outline-orange active";
      if (resumeSubViewAudit) resumeSubViewAudit.style.display = "block";
    } else if (tab === "rewrites") {
      if (resumeTabRewritesBtn) resumeTabRewritesBtn.className = "btn btn-sm btn-outline-orange active";
      if (resumeSubViewRewrites) resumeSubViewRewrites.style.display = "block";
    } else if (tab === "tech") {
      if (resumeTabTechBtn) resumeTabTechBtn.className = "btn btn-sm btn-outline-orange active";
      if (resumeSubViewTech) resumeSubViewTech.style.display = "block";
    } else if (tab === "profile") {
      if (resumeTabProfileBtn) resumeTabProfileBtn.className = "btn btn-sm btn-outline-orange active";
      if (resumeSubViewProfile) resumeSubViewProfile.style.display = "block";
    }
  }

  if (resumeTabAuditBtn) resumeTabAuditBtn.addEventListener("click", () => setResumeSubTab("audit"));
  if (resumeTabRewritesBtn) resumeTabRewritesBtn.addEventListener("click", () => setResumeSubTab("rewrites"));
  if (resumeTabTechBtn) resumeTabTechBtn.addEventListener("click", () => setResumeSubTab("tech"));
  if (resumeTabProfileBtn) resumeTabProfileBtn.addEventListener("click", () => setResumeSubTab("profile"));

  function renderResumeAnalysis(ra) {
    if (!ra) return;

    const overallScore = ra.overall_score !== undefined ? Number(ra.overall_score).toFixed(1) : "8.5";
    const masterScoreEl = document.getElementById("resumeMasterScore");
    if (masterScoreEl) masterScoreEl.textContent = overallScore;

    const tierEl = document.getElementById("resumePercentileTier");
    if (tierEl && ra.percentile_tier) tierEl.textContent = ra.percentile_tier;

    // Sub-Scores
    const sub = ra.sub_scores || {};
    const setSub = (id, barId, val) => {
      const el = document.getElementById(id);
      const bar = document.getElementById(barId);
      const num = Number(val || 7.5);
      if (el) el.textContent = `${num.toFixed(1)} / 10`;
      if (bar) bar.style.width = `${Math.min(100, num * 10)}%`;
    };

    setSub("subScoreImpact", "barImpact", sub.impact);
    setSub("subScoreVerbs", "barVerbs", sub.verbs);
    setSub("subScoreSkills", "barSkills", sub.skills);
    setSub("subScoreAts", "barAts", sub.ats);
    setSub("subScoreLeadership", "barLeadership", sub.leadership);

    // Markdown Deep Insights
    const auditEl = document.getElementById("resumeAuditMarkdown");
    if (auditEl && ra.deep_insights && ra.deep_insights.markdown) {
      auditEl.innerHTML = marked.parse(ra.deep_insights.markdown);
    }

    // Google XYZ Rewrites
    const rewritesContainer = document.getElementById("resumeRewritesList");
    if (rewritesContainer && ra.weak_bullet_rewrites) {
      rewritesContainer.innerHTML = ra.weak_bullet_rewrites.map((r, i) => `
        <div class="glass-card" style="border: 1px solid var(--border-glass-orange); padding: 1.25rem;">
          <div style="font-size: 0.8rem; color: var(--rose); font-weight: 700; text-transform: uppercase; margin-bottom: 0.25rem;">Original Statement #${i + 1}</div>
          <div style="font-size: 0.95rem; color: var(--text-muted); margin-bottom: 0.5rem; text-decoration: line-through;">"${r.original}"</div>
          
          <div style="font-size: 0.8rem; color: var(--emerald); font-weight: 700; text-transform: uppercase; margin-bottom: 0.25rem;">Elite Google XYZ Formula Rewrite</div>
          <div style="font-size: 1rem; color: #FFFFFF; font-weight: 600; line-height: 1.5; margin-bottom: 0.5rem; background: rgba(16, 185, 129, 0.1); border-left: 3px solid var(--emerald); padding: 0.5rem 0.75rem; border-radius: 4px;">
            ${r.rewrite}
          </div>
          
          <div style="font-size: 0.8rem; color: var(--text-dim);">
            <strong>Weakness:</strong> ${r.weakness} &bull; <strong style="color: var(--orange-bright);">Advantage:</strong> ${r.advantage}
          </div>
        </div>
      `).join("");
    }

    // Skills Badges
    const matchedEl = document.getElementById("resumeMatchedSkillsBadges");
    if (matchedEl && ra.matched_skills) {
      matchedEl.innerHTML = Object.entries(ra.matched_skills).flatMap(([cat, skills]) => 
        skills.map(s => `<span class="badge badge-bool" style="padding: 0.4rem 0.6rem;">${s}</span>`)
      ).join("") || "<span style='color: var(--text-muted);'>No skills detected</span>";
    }

    const missingEl = document.getElementById("resumeMissingSkillsBadges");
    if (missingEl && ra.recommended_keywords) {
      missingEl.innerHTML = ra.recommended_keywords.map(k => `
        <span class="badge" style="background: rgba(255, 107, 0, 0.2); color: var(--orange-bright); border: 1px solid var(--border-glass-orange); cursor: pointer; padding: 0.4rem 0.6rem;" onclick="navigator.clipboard.writeText('${k}'); alert('Copied keyword: ${k}');">
          + ${k}
        </span>
      `).join("") || "<span style='color: var(--text-muted);'>Profile is comprehensive</span>";
    }

    // ATS Profile
    const prof = ra.profile || {};
    const atsName = document.getElementById("atsName");
    const atsEmail = document.getElementById("atsEmail");
    const atsPhone = document.getElementById("atsPhone");
    const atsLinks = document.getElementById("atsLinks");

    if (atsName) atsName.textContent = prof.name || "Candidate";
    if (atsEmail) atsEmail.textContent = prof.email || "Not Detected";
    if (atsPhone) atsPhone.textContent = prof.phone || "Not Detected";
    if (atsLinks) {
      const links = [];
      if (prof.linkedin) links.push(`<a href="${prof.linkedin}" target="_blank" style="color: var(--orange-bright);">LinkedIn</a>`);
      if (prof.github) links.push(`<a href="${prof.github}" target="_blank" style="color: var(--emerald);">GitHub</a>`);
      atsLinks.innerHTML = links.join(" &bull; ") || "None Detected";
    }

    const sectionsEl = document.getElementById("atsSectionsList");
    if (sectionsEl && prof.detected_sections) {
      sectionsEl.innerHTML = prof.detected_sections.map(s => `<span class="badge badge-id" style="padding: 0.35rem 0.65rem;">${s}</span>`).join("");
    }
  }

  // =========================================================
  // 6.2 Marksheet & Academic Intelligence Renderer
  // =========================================================
  let marksheetChartInstance = null;

  function renderMarksheetAnalysis(ma) {
    if (!ma) return;

    // Header & Meta
    const nameEl = document.getElementById("marksheetCandidateName");
    const tierBadge = document.getElementById("marksheetTierBadge");
    const examInfo = document.getElementById("marksheetExamInfo");

    if (nameEl) nameEl.textContent = ma.candidate_name || "Student Scorecard";
    const sm = ma.summary_metrics || {};
    if (tierBadge) {
      tierBadge.textContent = sm.academic_tier || "Distinction Standing";
      if (sm.tier_color) tierBadge.style.backgroundColor = sm.tier_color;
    }
    if (examInfo) {
      examInfo.textContent = `${ma.examination_name || "Examination"} • ${ma.institution_board || "Board"} • Roll: ${ma.roll_number || "N/A"} • Passing: ${ma.passing_year || "N/A"}`;
    }

    // Top Metric Cards
    const pctEl = document.getElementById("marksheetOverallPct");
    const totalEl = document.getElementById("marksheetTotalScore");
    const gpaEl = document.getElementById("marksheetGpa");
    const gpa4El = document.getElementById("marksheetGpa4");

    if (pctEl) pctEl.textContent = `${sm.overall_percentage || 0}%`;
    if (totalEl) totalEl.textContent = `${sm.total_marks_obtained || 0} / ${sm.total_max_marks || 0} Total Marks`;
    if (gpaEl) gpaEl.textContent = `${sm.gpa_out_of_10 || 0}`;
    if (gpa4El) gpa4El.textContent = `${sm.gpa_out_of_4 || 0} / 4.0 US Scale`;

    // Strongest & Weakest Subjects
    const strong = ma.strongest_subject || {};
    const weak = ma.weakest_subject || {};

    const strongSubEl = document.getElementById("marksheetStrongestSub");
    const strongMarksEl = document.getElementById("marksheetStrongestMarks");
    const weakSubEl = document.getElementById("marksheetWeakestSub");
    const weakMarksEl = document.getElementById("marksheetWeakestMarks");

    if (strongSubEl) strongSubEl.textContent = strong.subject || "N/A";
    if (strongMarksEl) strongMarksEl.textContent = `${strong.marks || 0} / ${strong.max || 100} (${strong.percentage || 0}%)`;

    if (weakSubEl) weakSubEl.textContent = weak.subject || "N/A";
    if (weakMarksEl) weakMarksEl.textContent = `${weak.marks || 0} / ${weak.max || 100} (${weak.percentage || 0}%)`;

    // Subject Breakdown Table
    const tableBody = document.getElementById("marksheetTableBody");
    const subjects = ma.subject_breakdown || [];
    if (tableBody) {
      tableBody.innerHTML = subjects.map(s => `
        <tr>
          <td style="font-weight: 600; color: #FFFFFF;">${s.subject}</td>
          <td style="color: var(--orange-bright); font-weight: 600;">${s.marks}</td>
          <td style="color: var(--text-muted);">${s.max}</td>
          <td>
            <span class="badge ${s.percentage >= 80 ? 'badge-num' : (s.percentage >= 60 ? 'badge-cat' : 'badge-date')}" style="padding: 0.25rem 0.5rem;">
              ${s.percentage}%
            </span>
          </td>
          <td style="font-size: 0.85rem; color: var(--emerald); font-weight: 600;">${s.grade}</td>
        </tr>
      `).join("");
    }

    // Interactive Bar Chart
    const canvas = document.getElementById("marksheetCanvas");
    if (canvas && typeof Chart !== "undefined") {
      if (marksheetChartInstance) marksheetChartInstance.destroy();
      const ctx = canvas.getContext("2d");

      marksheetChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
          labels: subjects.map(s => s.subject),
          datasets: [{
            label: "Percentage Score (%)",
            data: subjects.map(s => s.percentage),
            backgroundColor: subjects.map(s => s.percentage >= 90 ? "rgba(16, 185, 129, 0.85)" : (s.percentage >= 75 ? "rgba(255, 107, 0, 0.85)" : "rgba(239, 68, 68, 0.85)")),
            borderColor: subjects.map(s => s.percentage >= 90 ? "#10B981" : (s.percentage >= 75 ? "#FF6B00" : "#EF4444")),
            borderWidth: 1,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: "rgba(255,255,255,0.06)" },
              ticks: { color: "#94A3B8" }
            },
            x: {
              grid: { display: false },
              ticks: { color: "#FFFFFF", font: { weight: "600" } }
            }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }

    // Gemini Academic Guidance Markdown
    const guidanceEl = document.getElementById("marksheetGuidanceMarkdown");
    if (guidanceEl && ma.academic_guidance && ma.academic_guidance.markdown) {
      guidanceEl.innerHTML = marked.parse(ma.academic_guidance.markdown);
    }
  }

  // =========================================================
  // 7. Paginated Data Table
  // =========================================================
  function renderDataTable() {
    const d = appState.dataset;
    const container = document.getElementById("dataTableContainer");
    if (!container || !d) return;

    if (d.records && d.records.length > 0) {
      const cols = d.columns || Object.keys(d.records[0]);
      container.innerHTML = `
        <table>
          <thead>
            <tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>
          </thead>
          <tbody>
            ${d.records.map(row => `
              <tr>${cols.map(c => `<td>${row[c] !== undefined ? row[c] : ""}</td>`).join("")}</tr>
            `).join("")}
          </tbody>
        </table>
      `;
    } else {
      container.innerHTML = "<p style='color: var(--text-muted); padding: 1.5rem; text-align: center;'>No records available.</p>";
    }

    const pageIndicator = document.getElementById("pageIndicator");
    if (pageIndicator) pageIndicator.textContent = `Page ${appState.currentPage}`;
  }

  const prevPageBtn = document.getElementById("prevPageBtn");
  const nextPageBtn = document.getElementById("nextPageBtn");

  if (prevPageBtn) {
    prevPageBtn.addEventListener("click", () => {
      if (appState.currentPage > 1) {
        fetchDatasetState(appState.currentPage - 1);
      }
    });
  }

  if (nextPageBtn) {
    nextPageBtn.addEventListener("click", () => {
      fetchDatasetState(appState.currentPage + 1);
    });
  }

  // =========================================================
  // 8. Quality & Cleaning Hub
  // =========================================================
  function renderQualityHub() {
    const d = appState.dataset;
    if (!d || !d.quality) return;

    const q = d.quality;
    const healthEl = document.getElementById("qualityHealthScore");
    const summaryEl = document.getElementById("qualityHealthSummary");

    if (healthEl) {
      healthEl.textContent = `${q.health_score} / 100`;
      healthEl.style.color = q.health_score >= 80 ? "var(--emerald)" : q.health_score >= 60 ? "var(--amber)" : "var(--rose)";
    }

    if (summaryEl) {
      summaryEl.textContent = `Duplicates: ${q.duplicate_rows} (${q.duplicate_pct}%) • Missing Cells: ${q.total_missing_cells} • Outlier Columns: ${Object.keys(q.outliers || {}).length}`;
    }

    const outliersContainer = document.getElementById("outliersListContainer");
    if (outliersContainer && q.outliers) {
      const outlierEntries = Object.entries(q.outliers).filter(([_, o]) => o.iqr && o.iqr.outlier_count > 0);
      if (outlierEntries.length > 0) {
        outliersContainer.innerHTML = outlierEntries.map(([col, o]) => `
          <div style="margin-bottom: 0.75rem; padding: 0.6rem; background: rgba(0,0,0,0.3); border-radius: var(--radius-sm);">
            <strong style="color: #FFFFFF;">${col}</strong>: <span style="color: var(--rose); font-weight: 700;">${o.iqr.outlier_count} Outliers</span>
            <div style="font-size: 0.78rem; color: var(--text-dim);">Bounds: [${o.iqr.lower_bound}, ${o.iqr.upper_bound}]</div>
          </div>
        `).join("");
      } else {
        outliersContainer.innerHTML = "<p style='color: var(--emerald); font-size: 0.9rem;'>✓ Zero statistical outliers detected via IQR method.</p>";
      }
    }

    const missingContainer = document.getElementById("missingListContainer");
    if (missingContainer && q.missing_details) {
      if (q.missing_details.length > 0) {
        missingContainer.innerHTML = q.missing_details.map(m => `
          <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #FFFFFF;">${m.column}</span>
            <span class="badge badge-num">${m.missing_count} (${m.missing_pct}%)</span>
          </div>
        `).join("");
      } else {
        missingContainer.innerHTML = "<p style='color: var(--emerald); font-size: 0.9rem;'>✓ 100% complete dataset with 0 missing cells.</p>";
      }
    }
  }

  const runAutoCleanBtn = document.getElementById("runAutoCleanBtn");
  const restoreRawDataBtn = document.getElementById("restoreRawDataBtn");

  if (runAutoCleanBtn) {
    runAutoCleanBtn.addEventListener("click", async () => {
      runAutoCleanBtn.disabled = true;
      runAutoCleanBtn.textContent = "🧼 Cleaning Dataset...";
      try {
        const res = await fetch("/api/clean/auto", { method: "POST" });
        const data = await res.json();
        runAutoCleanBtn.disabled = false;
        runAutoCleanBtn.textContent = "🧼 Run Full Auto-Clean";
        if (res.ok) {
          await fetchDatasetState(1);
          alert("✓ Dataset cleaned successfully! Duplicates dropped and missing values imputed.");
        }
      } catch (e) {
        runAutoCleanBtn.disabled = false;
        runAutoCleanBtn.textContent = "🧼 Run Full Auto-Clean";
      }
    });
  }

  if (restoreRawDataBtn) {
    restoreRawDataBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/clean/reset", { method: "POST" });
        if (res.ok) {
          await fetchDatasetState(1);
          alert("Restored to original raw dataset.");
        }
      } catch (e) {}
    });
  }
  if (resetDataBtn) {
    resetDataBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/clean/reset", { method: "POST" });
        if (res.ok) {
          await fetchDatasetState(1);
        }
      } catch (e) {}
    });
  }

  // =========================================================
  // 9. AutoML Studio
  // =========================================================
  function populateMLDropdown() {
    const d = appState.dataset;
    const select = document.getElementById("mlTargetSelect");
    if (!select || !d || !d.columns) return;

    select.innerHTML = '<option value="">-- Select Target Column --</option>';
    d.columns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      select.appendChild(opt);
    });
  }

  const trainModelBtn = document.getElementById("trainModelBtn");
  if (trainModelBtn) {
    trainModelBtn.addEventListener("click", async () => {
      const select = document.getElementById("mlTargetSelect");
      const targetCol = select ? select.value : "";
      if (!targetCol) {
        alert("Please select a target column to predict.");
        return;
      }

      trainModelBtn.disabled = true;
      trainModelBtn.textContent = "🧠 Training ML Model...";

      try {
        const res = await fetch("/api/ml/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target_column: targetCol }),
        });
        const data = await res.json();
        trainModelBtn.disabled = false;
        trainModelBtn.textContent = "🚀 Train & Evaluate Model";

        if (res.ok) {
          const result = data.result;
          const container = document.getElementById("mlResultsContainer");
          if (container) container.style.display = "block";

          const metricsGrid = document.getElementById("mlMetricsGrid");
          if (metricsGrid && result.metrics) {
            metricsGrid.innerHTML = Object.entries(result.metrics).map(([k, v]) => `
              <div class="glass-card kpi-card">
                <div class="kpi-label">${k}</div>
                <div class="kpi-val orange">${v}</div>
              </div>
            `).join("");
          }

          // Feature Importance Chart
          if (result.feature_importance && result.feature_importance.length) {
            const chartCanvas = document.getElementById("mlImportanceChart");
            if (chartCanvas) {
              const ctx = chartCanvas.getContext("2d");
              if (appState.charts.importance) appState.charts.importance.destroy();

              appState.charts.importance = new Chart(ctx, {
                type: "bar",
                data: {
                  labels: result.feature_importance.map(f => f.Feature),
                  datasets: [{
                    label: "Predictive Weight",
                    data: result.feature_importance.map(f => f.Importance),
                    backgroundColor: "rgba(255, 107, 0, 0.85)",
                    borderColor: "#FF851B",
                    borderRadius: 6,
                  }],
                },
                options: {
                  indexAxis: "y",
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } },
                  scales: {
                    x: { grid: { color: "rgba(255,255,255,0.06)" } },
                    y: { grid: { display: false } },
                  },
                },
              });
            }
          }
        } else {
          alert(data.detail || "Model training failed.");
        }
      } catch (e) {
        trainModelBtn.disabled = false;
        trainModelBtn.textContent = "🚀 Train & Evaluate Model";
      }
    });
  }

  // =========================================================
  // 10. AI Analyst Chat & Strategic Briefing
  // =========================================================
  const refreshBriefingBtn = document.getElementById("refreshBriefingBtn");
  if (refreshBriefingBtn) {
    refreshBriefingBtn.addEventListener("click", async () => {
      const box = document.getElementById("briefingContentBox");
      if (box) box.innerHTML = "<p style='color: var(--orange-bright);'>⚡ Compiling multi-agent executive briefing...</p>";
      try {
        const res = await fetch("/api/ai/briefing");
        const data = await res.json();
        if (res.ok && box) {
          box.innerHTML = marked.parse(data.briefing);
        }
      } catch (e) {}
    });
  }

  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatMessages = document.getElementById("chatMessages");

  if (chatForm) {
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
            content += `
              <div class="table-glass-container" style="margin-top: 0.75rem;">
                <table>
                  <thead><tr>${keys.map(k => `<th>${k}</th>`).join("")}</tr></thead>
                  <tbody>${data.data.slice(0, 5).map(row => `<tr>${keys.map(k => `<td>${row[k]}</td>`).join("")}</tr>`).join("")}</tbody>
                </table>
              </div>
            `;
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
  }

  function appendChatBubble(role, htmlContent, isHtml = false) {
    if (!chatMessages) return null;
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

  // Initial Sync
  checkGeminiStatus();
  fetchDatasetState(1);
});
