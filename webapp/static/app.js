/* Contract Intelligence — web UI logic */

(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);

  const views = {
    upload: $("#view-upload"),
    loading: $("#view-loading"),
    results: $("#view-results"),
  };

  const fileInput = $("#file-input");
  const dropzone = $("#dropzone");
  const fileChip = $("#file-chip");
  const analyzeBtn = $("#analyze-btn");
  const uploadError = $("#upload-error");

  let selectedFile = null;
  let report = null;
  const state = { search: "", type: "", risk: "all" };

  function showView(name) {
    for (const k of Object.keys(views)) views[k].hidden = k !== name;
  }

  /* ---------- health ---------- */

  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      const pill = $("#status-pill");
      pill.classList.remove("mode-real", "mode-demo");
      if (h.mode === "real") {
        pill.classList.add("mode-real");
        $("#status-text").textContent = "Models ready";
      } else {
        pill.classList.add("mode-demo");
        $("#status-text").textContent = "Demo mode";
        $("#demo-note").hidden = false;
      }
    })
    .catch(() => {
      $("#status-text").textContent = "offline";
    });

  /* ---------- file selection ---------- */

  function setFile(file) {
    uploadError.hidden = true;
    if (!file) return;

    if (!/\.(pdf|docx|txt)$/i.test(file.name)) {
      uploadError.textContent = "Please choose a PDF, DOCX or TXT file.";
      uploadError.hidden = false;
      return;
    }

    selectedFile = file;
    $("#file-name").textContent = file.name;
    fileChip.hidden = false;
    analyzeBtn.disabled = false;
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    fileChip.hidden = true;
    analyzeBtn.disabled = true;
  }

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  $("#file-clear").addEventListener("click", clearFile);

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag");
    })
  );

  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag");
    })
  );

  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  /* ---------- analyse ---------- */

  const STEPS = [
    "Extracting document text…",
    "Detecting language…",
    "Segmenting clauses…",
    "Classifying clause types…",
    "Scoring clause risk…",
  ];

  let stepTimer = null;

  function startSteps() {
    let i = 0;
    $("#loading-step").textContent = STEPS[0];

    stepTimer = setInterval(() => {
      i = Math.min(i + 1, STEPS.length - 1);
      $("#loading-step").textContent = STEPS[i];
    }, 1400);
  }

  function stopSteps() {
    if (stepTimer) clearInterval(stepTimer);
    stepTimer = null;
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    showView("loading");
    startSteps();

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Filename": encodeURIComponent(selectedFile.name).replace(/%20/g, " "),
        },
        body: selectedFile,
      });

      const data = await res.json();
      stopSteps();

      if (data.status !== "success") {
        showView("upload");
        uploadError.textContent =
          data.message || data.error || "Analysis failed.";
        uploadError.hidden = false;
        return;
      }

      report = data;
      renderResults();
      showView("results");
      window.scrollTo({ top: 0 });
    } catch (err) {
      stopSteps();
      showView("upload");
      uploadError.textContent = "Could not reach the analysis server.";
      uploadError.hidden = false;
    }
  });

  /* ---------- helpers ---------- */

  const pct = (x) => Math.round(x * 100) + "%";

  const prettyType = (t) =>
    t
      .toLowerCase()
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

  /*
   * Convert the model's risk probability into a UI severity.
   *
   * risk_probability represents the model's estimated probability
   * of the UNFAIR class.
   *
   * This does not change the model's FAIR/UNFAIR decision.
   */
  function getRiskSeverity(probability) {
    if (probability >= 0.70) {
      return "high";
    }

    if (probability >= 0.40) {
      return "medium";
    }

    return "low";
  }

  function getRiskSeverityLabel(severity) {
    return severity.charAt(0).toUpperCase() + severity.slice(1);
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  /* ---------- rendering ---------- */

  function renderResults() {
    $("#res-doc-name").textContent = report.document_name;

    const meta = [
      report.mode === "real" ? "Full model pipeline" : "Demo pipeline",
      "Language: " + (report.language || "unknown"),
      report.page_count + (report.page_count === 1 ? " page" : " pages"),
      report.clause_count + " clauses",
    ];

    $("#res-meta").innerHTML = meta
      .map((m) => `<span>${esc(m)}</span>`)
      .join("");

    const counts = report.summary.risk_counts || {};
    const unfair = counts.UNFAIR || 0;
    const fair = counts.FAIR || 0;
    const total = unfair + fair || 1;

    $("#stat-clauses").textContent = report.clause_count;

    $("#stat-pages").textContent =
      report.page_count + (report.page_count === 1 ? " page" : " pages");

    const flaggedEl = $("#stat-flagged");
    flaggedEl.textContent = unfair;
    flaggedEl.classList.toggle("is-flag", unfair > 0);

    $("#stat-flagged-sub").textContent =
      pct(unfair / (report.clause_count || 1)) + " of clauses";

    $("#stat-conf").textContent =
      pct(report.summary.avg_clause_confidence || 0);

    $("#stat-lang").textContent =
      "language " + (report.language || "—");

    $("#risk-meter").innerHTML =
      `<div class="seg seg-unfair" style="width:${(unfair / total) * 100}%"></div>` +
      `<div class="seg seg-fair" style="width:${(fair / total) * 100}%"></div>`;

    $("#risk-legend").innerHTML =
      `<span><span class="lg-top"><span class="swatch fair"></span>Standard</span><b>${fair}</b></span>` +
      `<span><span class="lg-top"><span class="swatch unfair"></span>Flagged</span><b>${unfair}</b></span>`;

    const types = Object.entries(report.summary.clause_type_counts || {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6);

    const maxType = types.length ? types[0][1] : 1;

    $("#type-bars").innerHTML = types
      .map(
        ([t, n]) => `
        <div class="type-bar">
          <span class="tb-label">${esc(prettyType(t))}</span>
          <span class="tb-count">${n}</span>
          <span class="tb-track">
            <span class="tb-fill" style="width:${(n / maxType) * 100}%"></span>
          </span>
        </div>`
      )
      .join("");

    const sel = $("#filter-type");

    sel.innerHTML = '<option value="">All clause types</option>';

    Object.keys(report.summary.clause_type_counts || {})
      .sort()
      .forEach((t) => {
        const o = document.createElement("option");
        o.value = t;
        o.textContent =
          prettyType(t) +
          " (" +
          report.summary.clause_type_counts[t] +
          ")";

        sel.appendChild(o);
      });

    state.search = "";
    state.type = "";
    state.risk = "all";

    $("#search").value = "";

    document
      .querySelectorAll("#filter-risk button")
      .forEach((b) =>
        b.classList.toggle("active", b.dataset.risk === "all")
      );

    renderList();
  }

  function matches(c) {
    if (state.risk !== "all" && c.risk_label !== state.risk) {
      return false;
    }

    if (state.type && c.clause_type !== state.type) {
      return false;
    }

    if (state.search) {
      const q = state.search.toLowerCase();

      const hay = (
        c.clause_id +
        " " +
        c.section +
        " " +
        c.original_text +
        " " +
        prettyType(c.clause_type)
      ).toLowerCase();

      if (!hay.includes(q)) {
        return false;
      }
    }

    return true;
  }

  function renderList() {
    const list = $("#clause-list");
    const visible = report.clauses.filter(matches);

    $("#result-count").textContent =
      visible.length + " of " + report.clauses.length + " clauses";

    $("#empty-state").hidden = visible.length !== 0;

    list.innerHTML = visible
      .map((c) => {
        const unfair = c.risk_label === "UNFAIR";
        const text = c.original_text || c.processed_text || "";
        const long = text.length > 300;

        const probability = Number(c.risk_probability) || 0;
        const severity = getRiskSeverity(probability);
        const severityLabel = getRiskSeverityLabel(severity);

        const reasons =
          c.risk_reasons && c.risk_reasons.length
            ? `<ul class="risk-reasons">${c.risk_reasons
                .map((r) => `<li>${esc(r)}</li>`)
                .join("")}</ul>`
            : "";

        return `
        <li class="clause-card" data-risk="${c.risk_label}" data-severity="${severity}">
          <div class="cc-head">
            <span class="cc-id">${esc(c.clause_id)}</span>
            <span class="cc-type">${esc(prettyType(c.clause_type))}</span>
            ${c.section ? `<span class="cc-section">${esc(c.section)}</span>` : ""}
            <span class="cc-page">page ${c.page}</span>
          </div>

          <p class="cc-text ${long ? "clamp" : ""}">${esc(text)}</p>

          ${long ? '<button class="cc-toggle">Show more</button>' : ""}

          <div class="cc-meta">
            <span class="conf">
              type confidence
              <span class="conf-track">
                <span
                  class="conf-fill"
                  style="width:${c.clause_confidence * 100}%"
                ></span>
              </span>
              <b>${pct(c.clause_confidence)}</b>
            </span>

            <span class="risk-badge ${severity}">
              <span class="rb-dot"></span>
              ${severityLabel} Risk · ${pct(probability)} estimated unfairness
            </span>

            <span class="risk-status ${unfair ? "unfair" : "fair"}">
              ${unfair ? "Flagged" : "Standard"}
            </span>

            ${reasons}
          </div>
        </li>`;
      })
      .join("");

    list.querySelectorAll(".cc-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = btn.previousElementSibling;
        const clamped = p.classList.toggle("clamp");
        btn.textContent = clamped ? "Show more" : "Show less";
      });
    });
  }

  /* ---------- controls ---------- */

  $("#search").addEventListener("input", (e) => {
    state.search = e.target.value.trim();
    renderList();
  });

  $("#filter-type").addEventListener("change", (e) => {
    state.type = e.target.value;
    renderList();
  });

  $("#filter-risk").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;

    state.risk = btn.dataset.risk;

    document
      .querySelectorAll("#filter-risk button")
      .forEach((b) => b.classList.toggle("active", b === btn));

    renderList();
  });

  $("#btn-new").addEventListener("click", () => {
    clearFile();
    showView("upload");
  });

  $("#btn-print").addEventListener("click", () => window.print());

  $("#btn-export").addEventListener("click", () => {
    const blob = new Blob(
      [JSON.stringify(report, null, 2)],
      { type: "application/json" }
    );

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download =
      report.document_name.replace(/\.[^.]+$/, "") +
      "_analysis.json";

    document.body.appendChild(a);
    a.click();
    a.remove();

    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
})();
