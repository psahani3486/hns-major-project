const predictBtn = document.getElementById("predictBtn");
const imageInput = document.getElementById("imageInput");
const apiBaseInput = document.getElementById("apiBase");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const dropZone = document.getElementById("dropZone");
const statusEl = document.getElementById("status");
const probList = document.getElementById("probList");
const historyList = document.getElementById("historyList");
const analysisMeta = document.getElementById("analysisMeta");
const modelDetailList = document.getElementById("modelDetailList");
const predictedLabel = document.getElementById("predictedLabel");
const predictedConfidence = document.getElementById("predictedConfidence");
const fileNameEl = document.getElementById("fileName");
const fileSizeEl = document.getElementById("fileSize");
const topKRange = document.getElementById("topKRange");
const topKValue = document.getElementById("topKValue");
const sortBy = document.getElementById("sortBy");
const copyJsonBtn = document.getElementById("copyJsonBtn");

const imgInput = document.getElementById("imgInput");
const imgHeatmap = document.getElementById("imgHeatmap");
const imgOverlay = document.getElementById("imgOverlay");
const downloadInputBtn = document.getElementById("downloadInputBtn");
const downloadHeatmapBtn = document.getElementById("downloadHeatmapBtn");
const downloadOverlayBtn = document.getElementById("downloadOverlayBtn");

const SETTINGS_KEY = "xai_frontend_settings";
const HISTORY_KEY = "xai_prediction_history";
const HISTORY_LIMIT = 6;

let latestResponse = null;
let latestProbs = [];

function readJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function toPct(x) {
  return `${(x * 100).toFixed(2)}%`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function applySettings() {
  const settings = readJson(SETTINGS_KEY, {});
  if (settings.apiBase) apiBaseInput.value = settings.apiBase;
  if (settings.topK) topKRange.value = String(settings.topK);
  if (settings.sortBy) sortBy.value = settings.sortBy;
  topKValue.textContent = topKRange.value;
}

function saveSettings() {
  writeJson(SETTINGS_KEY, {
    apiBase: apiBaseInput.value.trim(),
    topK: Number(topKRange.value),
    sortBy: sortBy.value,
  });
  setStatus("Settings saved.");
}

function renderHistory() {
  const history = readJson(HISTORY_KEY, []);
  historyList.innerHTML = "";
  if (!history.length) {
    const empty = document.createElement("li");
    empty.textContent = "No predictions yet.";
    empty.className = "history-empty";
    historyList.appendChild(empty);
    return;
  }

  for (const item of history) {
    const li = document.createElement("li");
    li.innerHTML = `<span><strong>${item.label}</strong> · ${item.confidence}</span><small>${item.fileName} · ${item.time}</small>`;
    historyList.appendChild(li);
  }
}

function pushHistory(item) {
  const history = readJson(HISTORY_KEY, []);
  history.unshift(item);
  writeJson(HISTORY_KEY, history.slice(0, HISTORY_LIMIT));
  renderHistory();
}

function renderProbabilities() {
  probList.innerHTML = "";
  if (!latestProbs.length) return;

  let list = [...latestProbs];
  if (sortBy.value === "name") {
    list.sort((a, b) => a.class_name.localeCompare(b.class_name));
  } else {
    list.sort((a, b) => b.probability - a.probability);
  }

  const topK = Number(topKRange.value);
  list = list.slice(0, topK);

  for (const item of list) {
    const li = document.createElement("li");
    const pct = Number(item.probability) * 100;
    li.innerHTML = `
      <div class="prob-head">
        <span>${item.class_name}</span>
        <strong>${toPct(item.probability)}</strong>
      </div>
      <div class="prob-track"><div class="prob-fill" style="width:${Math.max(1, Math.min(100, pct)).toFixed(2)}%"></div></div>
    `;
    probList.appendChild(li);
  }
}

function setDownloadButtons(enabled) {
  downloadInputBtn.disabled = !enabled;
  downloadHeatmapBtn.disabled = !enabled;
  downloadOverlayBtn.disabled = !enabled;
}

function toNumPct(v) {
  const n = Number(v);
  return Number.isFinite(n) ? toPct(n) : "-";
}

function renderDetailedResults(data) {
  analysisMeta.innerHTML = "";
  modelDetailList.innerHTML = "";

  const topMargin = data.analysis?.confidence_margin_top1_top2;
  const uncertainty = data.analysis?.normalized_entropy;
  const ensembleTop = data.ensemble_summary?.analysis?.predicted_label;

  const chips = [
    `Primary: ${data.predicted_label || "-"}`,
    `Top1-Top2 Margin: ${topMargin !== undefined ? toNumPct(topMargin) : "-"}`,
    `Uncertainty: ${uncertainty !== undefined ? `${(Number(uncertainty) * 100).toFixed(1)}%` : "-"}`,
  ];
  if (ensembleTop) chips.push(`Ensemble: ${ensembleTop}`);

  for (const text of chips) {
    const chip = document.createElement("span");
    chip.className = "analysis-chip";
    chip.textContent = text;
    analysisMeta.appendChild(chip);
  }

  const models = data.per_model_results || [];
  for (const model of models) {
    const li = document.createElement("li");
    const top2 = model.analysis?.top2 || [];
    const t1 = top2[0];
    const t2 = top2[1];

    li.innerHTML = `
      <div class="model-line">
        <strong>${model.model_name}</strong>
        <span>${model.predicted_label} (${toNumPct(model.top_confidence)})</span>
      </div>
      <div class="model-sub">
        Margin: ${toNumPct(model.analysis?.confidence_margin_top1_top2)}
        · Entropy: ${model.analysis?.normalized_entropy !== undefined ? `${(Number(model.analysis.normalized_entropy) * 100).toFixed(1)}%` : "-"}
      </div>
      <div class="model-top2">
        ${t1 ? `${t1.class_name}: ${toNumPct(t1.probability)}` : "-"}
        ${t2 ? ` | ${t2.class_name}: ${toNumPct(t2.probability)}` : ""}
      </div>
    `;
    modelDetailList.appendChild(li);
  }

  const ensemble = data.ensemble_summary;
  if (ensemble?.analysis) {
    const li = document.createElement("li");
    li.className = "ensemble-item";
    li.innerHTML = `
      <div class="model-line">
        <strong>Ensemble (${ensemble.method})</strong>
        <span>${ensemble.analysis.predicted_label} (${toNumPct(ensemble.analysis.top_confidence)})</span>
      </div>
      <div class="model-sub">Models used: ${ensemble.num_models}</div>
    `;
    modelDetailList.appendChild(li);
  }
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  predictBtn.textContent = isLoading ? "Running..." : "Run Prediction";
}

function setFileDetails(file) {
  fileNameEl.textContent = file ? file.name : "-";
  fileSizeEl.textContent = file ? formatBytes(file.size) : "-";
}

function updateInputPreview(file) {
  if (!file) {
    imgInput.removeAttribute("src");
    return;
  }
  imgInput.src = URL.createObjectURL(file);
}

function selectFile(file) {
  if (!file) return;

  const dt = new DataTransfer();
  dt.items.add(file);
  imageInput.files = dt.files;

  setFileDetails(file);
  updateInputPreview(file);
  setStatus("Image selected. Ready to run prediction.");
}

async function copyJsonToClipboard() {
  if (!latestResponse) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(latestResponse, null, 2));
    setStatus("Prediction JSON copied.");
  } catch {
    setStatus("Clipboard permission denied in this browser context.", true);
  }
}

function downloadImage(imgEl, fileName) {
  if (!imgEl.src) return;
  const a = document.createElement("a");
  a.href = imgEl.src;
  a.download = fileName;
  a.click();
}

function resetResults() {
  probList.innerHTML = "";
  analysisMeta.innerHTML = "";
  modelDetailList.innerHTML = "";
  predictedLabel.textContent = "-";
  predictedConfidence.textContent = "-";
  imgHeatmap.removeAttribute("src");
  imgOverlay.removeAttribute("src");
  latestResponse = null;
  latestProbs = [];
  copyJsonBtn.disabled = true;
  setDownloadButtons(false);
}

dropZone.addEventListener("click", () => imageInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    imageInput.click();
  }
});
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  const file = e.dataTransfer?.files?.[0];
  selectFile(file);
});

imageInput.addEventListener("change", () => {
  selectFile(imageInput.files?.[0]);
});

topKRange.addEventListener("input", () => {
  topKValue.textContent = topKRange.value;
  renderProbabilities();
});

sortBy.addEventListener("change", renderProbabilities);
saveSettingsBtn.addEventListener("click", saveSettings);
copyJsonBtn.addEventListener("click", copyJsonToClipboard);

downloadInputBtn.addEventListener("click", () => downloadImage(imgInput, "input.png"));
downloadHeatmapBtn.addEventListener("click", () => downloadImage(imgHeatmap, "heatmap.png"));
downloadOverlayBtn.addEventListener("click", () => downloadImage(imgOverlay, "overlay.png"));

predictBtn.addEventListener("click", async () => {
  const file = imageInput.files?.[0];
  if (!file) {
    setStatus("Please select an image first.", true);
    return;
  }

  const apiBase = apiBaseInput.value.trim().replace(/\/$/, "");
  if (!apiBase) {
    setStatus("Please enter backend URL.", true);
    return;
  }

  resetResults();
  setLoading(true);
  setStatus("Uploading image and running inference...");

  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch(`${apiBase}/predict`, {
      method: "POST",
      body: form,
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      const detail = errData.detail || `Request failed with status ${resp.status}`;
      throw new Error(detail);
    }

    const data = await resp.json();
    latestResponse = data;
    latestProbs = data.class_probabilities || [];
    predictedLabel.textContent = data.predicted_label;

    const top = latestProbs[0]?.probability;
    predictedConfidence.textContent = top !== undefined ? `Top confidence: ${toPct(top)}` : "-";

    renderProbabilities();
    renderDetailedResults(data);

    imgInput.src = `data:image/png;base64,${data.images.input_png_base64}`;
    imgHeatmap.src = `data:image/png;base64,${data.images.heatmap_png_base64}`;
    imgOverlay.src = `data:image/png;base64,${data.images.overlay_png_base64}`;

    copyJsonBtn.disabled = false;
    setDownloadButtons(true);

    pushHistory({
      label: data.predicted_label,
      confidence: top !== undefined ? toPct(top) : "-",
      fileName: file.name,
      time: new Date().toLocaleTimeString(),
    });

    setStatus("Inference completed.");
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  } finally {
    setLoading(false);
  }
});

applySettings();
renderHistory();
setFileDetails(null);
setDownloadButtons(false);
