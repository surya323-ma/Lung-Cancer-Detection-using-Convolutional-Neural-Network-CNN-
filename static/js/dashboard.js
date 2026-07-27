const CLASS_META = {
  lung_benign_tissue: { short: "Benign", badge: "benign", color: "#2FBF71" },
  lung_adenocarcinoma: { short: "Adenocarcinoma", badge: "adenocarcinoma", color: "#F4623A" },
  lung_squamous_cell_carcinoma: { short: "Squamous cell carcinoma", badge: "squamous", color: "#C7451F" },
};

function prettyClass(name) {
  return (CLASS_META[name] && CLASS_META[name].short) || name.replace(/_/g, " ");
}
function badgeClass(name) {
  return (CLASS_META[name] && CLASS_META[name].badge) || "benign";
}
function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

/* ---------------- Navigation ---------------- */
const VIEW_TITLES = {
  overview: ["Department Overview", "Pulmonary histopathology screening, at a glance"],
  analysis: ["Scan Analysis", "Upload a histopathology image for AI-assisted screening"],
  records: ["Patient Records", "Full history of processed scans in this session"],
  insights: ["Model Insights", "Architecture, serving mode, and validation notes"],
};

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
    document.getElementById(`view-${view}`).classList.remove("hidden");
    document.getElementById("view-title").textContent = VIEW_TITLES[view][0];
    document.getElementById("view-subtitle").textContent = VIEW_TITLES[view][1];
    if (view === "records") loadRecords();
    if (view === "insights") loadSystemStatus();
  });
});

/* ---------------- Clock ---------------- */
function tickClock() {
  const el = document.getElementById("clock");
  el.textContent = new Date().toLocaleString(undefined, {
    weekday: "short", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}
setInterval(tickClock, 1000);
tickClock();

/* ---------------- System status ---------------- */
async function loadSystemStatus() {
  try {
    const res = await fetch("/api/system-status");
    const data = await res.json();
    const isReal = data.model_mode === "trained_model";
    document.getElementById("model-pill-text").textContent = isReal
      ? "Trained model active"
      : "Demo heuristic mode";
    document.getElementById("insight-mode-text").textContent = isReal
      ? "A trained CNN checkpoint (model/lung_cnn.h5) is loaded and serving real inference."
      : "No trained checkpoint found. The API is generating image-statistics-based demo predictions, clearly labeled as non-diagnostic in every response.";
  } catch (e) {
    document.getElementById("model-pill-text").textContent = "Status unavailable";
  }
}

/* ---------------- Upload / Dropzone ---------------- */
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const dropzoneEmpty = document.getElementById("dropzone-empty");
const btnAnalyze = document.getElementById("btn-analyze");
const uploadStatus = document.getElementById("upload-status");
let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag-over"); })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    dropzoneEmpty.classList.add("hidden");
  };
  reader.readAsDataURL(file);
  btnAnalyze.disabled = false;
  uploadStatus.textContent = `${file.name} ready (${(file.size / 1024).toFixed(0)} KB)`;
}

btnAnalyze.addEventListener("click", async () => {
  if (!selectedFile) return;
  btnAnalyze.disabled = true;
  btnAnalyze.textContent = "Analyzing…";
  uploadStatus.textContent = "Running inference…";

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("patient_ref", document.getElementById("patient-ref").value);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    renderResult(data);
    uploadStatus.textContent = "Analysis complete.";
    loadOverview();
  } catch (err) {
    uploadStatus.textContent = `Error: ${err.message}`;
  } finally {
    btnAnalyze.disabled = false;
    btnAnalyze.textContent = "Run AI Analysis";
  }
});

function renderResult(data) {
  document.getElementById("result-empty").classList.add("hidden");
  document.getElementById("result-content").classList.remove("hidden");

  const modeTag = document.getElementById("result-mode-tag");
  modeTag.textContent = data.mode === "trained_model" ? "Trained model" : "Demo heuristic";

  const ring = document.getElementById("confidence-ring");
  const pct = data.malignant_probability;
  const deg = (pct / 100) * 360;
  ring.style.background = `conic-gradient(var(--coral) ${deg}deg, var(--line) ${deg}deg)`;
  document.getElementById("ring-value").textContent = `${pct.toFixed(1)}%`;

  document.getElementById("predicted-label").textContent = prettyClass(data.predicted_class);
  document.getElementById("predicted-conf").textContent = `${data.confidence.toFixed(1)}% model confidence`;

  const barsWrap = document.getElementById("prob-bars");
  barsWrap.innerHTML = "";
  data.class_probabilities.forEach((cp) => {
    const meta = CLASS_META[cp.label] || {};
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${prettyClass(cp.label)}</span>
      <span class="prob-track"><span class="prob-fill" style="width:${cp.confidence}%; background:${meta.color || '#14B8A6'}"></span></span>
      <span class="prob-pct">${cp.confidence.toFixed(1)}%</span>
    `;
    barsWrap.appendChild(row);
  });

  document.getElementById("mode-explainer").textContent =
    data.mode === "trained_model"
      ? "Generated by a trained CNN checkpoint. Still not a substitute for a licensed pathologist's diagnosis."
      : "No trained model checkpoint was found, so this result comes from a non-diagnostic image-statistics heuristic — for interface demonstration only.";
}

/* ---------------- Overview ---------------- */
let trendChart, distChart;

async function loadOverview() {
  const [statsRes, historyRes] = await Promise.all([
    fetch("/api/stats"), fetch("/api/history?limit=6"),
  ]);
  const stats = await statsRes.json();
  const history = await historyRes.json();

  document.getElementById("stat-total").textContent = stats.total_scans;
  document.getElementById("stat-benign").textContent = stats.benign_count;
  document.getElementById("stat-highrisk").textContent = stats.high_risk_flags;
  document.getElementById("stat-avgconf").textContent = `${stats.average_confidence}%`;

  const isReal = stats.model_mode === "trained_model";
  document.getElementById("model-pill-text").textContent = isReal ? "Trained model active" : "Demo heuristic mode";

  renderTrendChart(stats.scans_per_day);
  renderDistChart(stats);
  renderOverviewTable(history);
}

function renderOverviewTable(history) {
  const tbody = document.querySelector("#overview-table tbody");
  tbody.innerHTML = "";
  if (!history.length) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No scans yet — run one from Scan Analysis.</td></tr>`;
    return;
  }
  history.forEach((h) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${h.scan_id}</td>
      <td style="font-family: var(--font-body);">${h.patient_ref}</td>
      <td><span class="badge ${badgeClass(h.predicted_class)}">${prettyClass(h.predicted_class)}</span></td>
      <td>${h.confidence.toFixed(1)}%</td>
      <td>${fmtTime(h.timestamp)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadRecords() {
  const res = await fetch("/api/history?limit=200");
  const history = await res.json();
  document.getElementById("records-count").textContent = `${history.length} record${history.length === 1 ? "" : "s"}`;
  const tbody = document.querySelector("#records-table tbody");
  tbody.innerHTML = "";
  if (!history.length) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="6">No scans yet — run one from Scan Analysis.</td></tr>`;
    return;
  }
  history.forEach((h) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${h.scan_id}</td>
      <td style="font-family: var(--font-body);">${h.patient_ref}</td>
      <td><span class="badge ${badgeClass(h.predicted_class)}">${prettyClass(h.predicted_class)}</span></td>
      <td>${h.malignant_probability.toFixed(1)}%</td>
      <td>${h.confidence.toFixed(1)}%</td>
      <td>${h.timestamp.replace("T", " ").replace("Z", "")}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTrendChart(scansPerDay) {
  const ctx = document.getElementById("chart-trend");
  const labels = scansPerDay.map((d) => d.date.slice(5));
  const values = scansPerDay.map((d) => d.count);
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels.length ? labels : ["—"],
      datasets: [{
        data: values.length ? values : [0],
        borderColor: "#14B8A6",
        backgroundColor: "rgba(20,184,166,0.12)",
        fill: true, tension: 0.35, pointRadius: 3, pointBackgroundColor: "#14B8A6",
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: "#EEF1F5" } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderDistChart(stats) {
  const ctx = document.getElementById("chart-distribution");
  if (distChart) distChart.destroy();
  const values = [stats.benign_count, stats.adenocarcinoma_count, stats.squamous_count];
  const hasData = values.some((v) => v > 0);
  distChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Benign", "Adenocarcinoma", "Squamous cell carcinoma"],
      datasets: [{
        data: hasData ? values : [1, 0, 0],
        backgroundColor: ["#2FBF71", "#F4623A", "#C7451F"],
        borderWidth: 0,
      }],
    },
    options: {
      cutout: "68%",
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 11 } } } },
    },
  });
}

/* ---------------- Init ---------------- */
loadSystemStatus();
loadOverview();
