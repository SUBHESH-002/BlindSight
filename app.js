const state = {
  file: null,
  objectUrl: null,
  lastPrediction: [],
};

const dropZone = document.getElementById("dropZone");
const videoInput = document.getElementById("videoInput");
const videoPreview = document.getElementById("videoPreview");

const predictBtn = document.getElementById("predictBtn");
const endpointInput = document.getElementById("endpointInput");
const queryInput = document.getElementById("queryInput");

const modelStatus = document.getElementById("modelStatus");
const frameState = document.getElementById("frameState");
const streamHealth = document.getElementById("streamHealth");

const timeline = document.getElementById("timeline");
const confidenceList = document.getElementById("confidenceList");

const primaryAction = document.getElementById("primaryAction");
const primaryScore = document.getElementById("primaryScore");

const motionScore = document.getElementById("motionScore");
const occlusionScore = document.getElementById("occlusionScore");
const gapScore = document.getElementById("gapScore");
const modeScore = document.getElementById("modeScore");


function setStatus(modelText, frameText, streamText = streamHealth.textContent) {
  modelStatus.textContent = modelText;
  frameState.textContent = frameText;
  streamHealth.textContent = streamText;
}


function setupTimeline() {
  timeline.innerHTML = "";

  for (let i = 0; i < 16; i++) {
    const tick = document.createElement("span");
    tick.className = "frame-tick";
    timeline.appendChild(tick);
  }
}


function setVideoFile(file) {
  if (!file || !file.type.startsWith("video/")) return;

  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
  }

  state.file = file;
  state.objectUrl = URL.createObjectURL(file);

  videoPreview.src = state.objectUrl;
  dropZone.classList.add("has-video");

  setStatus(modelStatus.textContent, "Video loaded");
}


async function requestBackendPrediction(endpoint) {
  const formData = new FormData();

  formData.append("video", state.file);
  formData.append("query", queryInput.value.trim() || "Predict action");

  const response = await fetch(endpoint || "/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const payload = await response.json();
      message = payload.detail || payload.message || message;
    } catch {
      const text = await response.text();
      if (text) message = text;
    }

    throw new Error(message);
  }

  const data = await response.json();
  return data.predictions || [];
}


function renderPrediction(predictions) {
  confidenceList.innerHTML = "";

  if (!predictions.length) {
    primaryAction.textContent = "No action detected";
    primaryScore.textContent = "--";
    return;
  }

  const top = predictions[0];
  primaryAction.textContent = top.label;
  primaryScore.textContent = `${Math.round(top.score * 100)}%`;
  modeScore.textContent = "Live";

  predictions.forEach((prediction) => {
    const row = document.createElement("div");
    row.className = "prediction-row";

    const percent = Math.round(prediction.score * 100);
    const timeLabel = prediction.time ? `${prediction.time} -> ` : "";

    row.innerHTML = `
  <div>
    <span>${timeLabel}${prediction.label}</span>
    <strong>${percent}%</strong>
  </div>
  <span class="bar"><span style="width: ${percent}%"></span></span>
`;

    confidenceList.appendChild(row);
  });

  motionScore.textContent = "Auto";
  occlusionScore.textContent = "Auto";
  gapScore.textContent = "Auto";
}


async function runPrediction() {
  if (!state.file) {
    setStatus(modelStatus.textContent, "Upload video first");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";
  setStatus("Running", "Sending video to backend", "Online");

  try {
    const endpoint = endpointInput.value.trim() || "/predict";
    const predictions = await requestBackendPrediction(endpoint);

    renderPrediction(predictions);
    setStatus("Done", "Prediction complete", "Online");
  } catch (err) {
    console.error(err);
    primaryAction.textContent = "Error";
    primaryScore.textContent = "--";
    modeScore.textContent = "Unavailable";
    setStatus("Model missing", err.message || "Backend error", "Offline");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Run prediction";
  }
}


async function refreshHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(`Health check failed (${response.status})`);
    }

    const data = await response.json();
    if (data.model_ready) {
      setStatus("Ready", "Waiting for video", "Online");
      modeScore.textContent = "Live";
      return;
    }

    setStatus("Model missing", data.message || "Model not available", "Offline");
    modeScore.textContent = "Unavailable";
  } catch (err) {
    setStatus("Backend offline", "Start FastAPI to enable prediction", "Offline");
    modeScore.textContent = "Offline";
  }
}


videoInput.addEventListener("change", (e) => {
  setVideoFile(e.target.files[0]);
});


dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  setVideoFile(e.dataTransfer.files[0]);
});


predictBtn.addEventListener("click", runPrediction);


setupTimeline();
renderPrediction([]);
refreshHealth();
