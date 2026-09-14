// ===========================================================================
// Real-Time Telemetry HUD Client (WebSockets, Canvas Waterfall & Charts)
// ===========================================================================

let ws = null;
let audioMuted = false;
let audioCtx = null;
let sirenOsc = null;

// History for height timeline (30 samples)
const heightHistory = [];
const MAX_HEIGHT_POINTS = 60;

// Waterfall spectrogram buffers (one canvas per node)
const canvases = {
  1: document.getElementById("specNode1"),
  2: document.getElementById("specNode2"),
  3: document.getElementById("specNode3"),
};
const ctxs = {
  1: canvases[1] ? canvases[1].getContext("2d") : null,
  2: canvases[2] ? canvases[2].getContext("2d") : null,
  3: canvases[3] ? canvases[3].getContext("2d") : null,
};

// Heatmap palette (0.0 to 1.0) -> RGB
function getHeatColor(val) {
  val = Math.max(0, Math.min(1, val));
  if (val < 0.25) {
    const t = val / 0.25;
    return [0, Math.floor(t * 128), Math.floor(t * 255)]; // Dark blue -> light blue
  } else if (val < 0.5) {
    const t = (val - 0.25) / 0.25;
    return [0, Math.floor(128 + t * 127), 255 - Math.floor(t * 128)]; // Blue -> Cyan
  } else if (val < 0.75) {
    const t = (val - 0.5) / 0.25;
    return [Math.floor(t * 255), 255, 0]; // Cyan -> Yellow
  } else {
    const t = (val - 0.75) / 0.25;
    return [255, Math.floor(255 * (1 - t)), 0]; // Yellow -> Red
  }
}

// Draw a single slice on the spectrogram canvas (scrolls left)
function pushSpectrogramSlice(nodeId, psd) {
  const c = canvases[nodeId];
  const ctx = ctxs[nodeId];
  if (!c || !ctx) return;

  const width = c.width;
  const height = c.height;

  // Scroll canvas left by 2 pixels
  const imgData = ctx.getImageData(2, 0, width - 2, height);
  ctx.putImageData(imgData, 0, 0);

  // Clear right edge
  ctx.fillStyle = "#050811";
  ctx.fillRect(width - 2, 0, 2, height);

  // If psd provided, draw colored pixels on the rightmost 2 columns
  if (!psd || psd.length === 0) {
    // Generate subtle noise floor slice
    psd = Array.from({ length: 32 }, () => Math.random() * 0.05);
  }

  const numBins = psd.length;
  const binHeight = height / numBins;

  // Normalize PSD
  const maxVal = Math.max(...psd, 0.1);
  for (let i = 0; i < numBins; i++) {
    const norm = psd[i] / maxVal;
    const [r, g, b] = getHeatColor(norm);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    // Draw inverted so low freq is center/bottom
    const y = height - (i + 1) * binHeight;
    ctx.fillRect(width - 2, Math.floor(y), 2, Math.ceil(binHeight));
  }
}

// Update Height Timeline Canvas
function drawHeightChart() {
  const c = document.getElementById("heightChart");
  if (!c) return;
  const ctx = c.getContext("2d");
  const width = c.width;
  const height = c.height;

  ctx.clearRect(0, 0, width, height);

  // Draw background grid lines (0m, 0.35m floor threshold, 1.0m, 1.7m)
  const maxH = 2.0; // 2 meters scale
  const toY = (h) => height - (h / maxH) * (height - 30) - 15;

  ctx.strokeStyle = "#1e2c4c";
  ctx.lineWidth = 1;
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#64748b";

  [0.0, 0.35, 1.0, 1.7].forEach((val) => {
    const y = toY(val);
    ctx.beginPath();
    ctx.moveTo(35, y);
    ctx.lineTo(width - 10, y);
    ctx.stroke();
    ctx.fillText(`${val.toFixed(2)}m`, 5, y + 3);
  });

  // Highlight floor threshold zone (< 0.35m) in faint red
  const floorY = toY(0.35);
  ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
  ctx.fillRect(35, floorY, width - 45, height - floorY);

  if (heightHistory.length < 2) return;

  // Draw altitude line
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.beginPath();

  const stepX = (width - 50) / (MAX_HEIGHT_POINTS - 1);
  heightHistory.forEach((pt, idx) => {
    const x = 40 + idx * stepX;
    const y = toY(pt);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw dot on latest point
  const lastX = 40 + (heightHistory.length - 1) * stepX;
  const lastY = toY(heightHistory[heightHistory.length - 1]);
  ctx.fillStyle = heightHistory[heightHistory.length - 1] < 0.35 ? "#ef4444" : "#10b981";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4.5, 0, Math.PI * 2);
  ctx.fill();
}

// Emergency Sound Synthesizer via Web Audio API
function playAlertSiren() {
  if (audioMuted) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(800, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.3);
    osc.frequency.exponentialRampToValueAtTime(800, audioCtx.currentTime + 0.6);

    gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.6);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.6);
  } catch (e) {
    console.warn("Audio error:", e);
  }
}

// Update State Badge
function updateStateBadge(state) {
  const badge = document.getElementById("systemStateBadge");
  if (!badge) return;

  badge.className = "status-badge";
  if (state.toLowerCase().includes("fall") && !state.toLowerCase().includes("suspected")) {
    badge.classList.add("badge-fall");
    badge.textContent = "FALL DETECTED";
    playAlertSiren();
  } else if (state.toLowerCase().includes("suspected")) {
    badge.classList.add("badge-suspected");
    badge.textContent = "FALL SUSPECTED";
  } else {
    badge.classList.add("badge-normal");
    badge.textContent = "SYSTEM NORMAL";
  }
}

// Add incident row to table
function addIncidentRow(inc) {
  const tbody = document.getElementById("incidentTableBody");
  if (!tbody) return;

  const tr = document.createElement("tr");
  const modTag = inc.modality.toLowerCase();
  const tagClass = modTag.includes("csi")
    ? "badge-tag-csi"
    : modTag.includes("radar")
    ? "badge-tag-radar"
    : "badge-tag-fusion";

  tr.innerHTML = `
    <td>${inc.timestamp || new Date().toLocaleTimeString()}</td>
    <td><span class="badge-tag ${tagClass}">${inc.modality.toUpperCase()}</span></td>
    <td><strong style="color:#f87171">${inc.event}</strong></td>
    <td>${inc.details || "-"}</td>
  `;

  tbody.insertBefore(tr, tbody.firstChild);

  // Keep max 25 rows
  while (tbody.children.length > 25) {
    tbody.removeChild(tbody.lastChild);
  }
}

// Connect WebSocket with Reconnect
function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[HUD] Connected to Telemetry Stream");
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      if (msg.type === "csi") {
        updateStateBadge(msg.state);

        // Update node metric displays
        const nodeElem = document.getElementById(`nodeStat${msg.node_id}`);
        if (nodeElem) {
          nodeElem.textContent = `${msg.velocity.toFixed(2)} m/s (surge ${msg.surge.toFixed(1)})`;
        }

        // Push to spectrogram waterfall
        pushSpectrogramSlice(msg.node_id, msg.doppler_psd);
      } else if (msg.type === "radar") {
        updateStateBadge(msg.state);

        // Update height display
        const heightElem = document.getElementById("statHeight");
        if (heightElem) {
          heightElem.textContent = `${msg.height.toFixed(2)} m`;
        }
        const postureElem = document.getElementById("statPosture");
        if (postureElem) {
          postureElem.textContent = msg.posture.toUpperCase();
        }

        // Add to height timeline
        heightHistory.push(msg.height);
        if (heightHistory.length > MAX_HEIGHT_POINTS) {
          heightHistory.shift();
        }
        drawHeightChart();
      } else if (msg.type === "alert") {
        addIncidentRow(msg);
      }
    } catch (e) {
      console.error("[HUD] JSON parse error:", e);
    }
  };

  ws.onclose = () => {
    console.warn("[HUD] WebSocket disconnected, retrying in 2s...");
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("[HUD] WebSocket error:", err);
    ws.close();
  };
}

// Fetch historical incidents on load
async function loadIncidents() {
  try {
    const res = await fetch("/api/incidents");
    const data = await res.json();
    if (data.incidents) {
      data.incidents.forEach((inc) => addIncidentRow(inc));
    }
  } catch (e) {
    console.warn("Could not load incidents:", e);
  }
}

// Initialize on DOM Ready
window.addEventListener("DOMContentLoaded", () => {
  // Size canvases to parent width
  Object.values(canvases).forEach((c) => {
    if (c) {
      c.width = c.parentElement.clientWidth || 240;
      c.height = 130;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#050811";
      ctx.fillRect(0, 0, c.width, c.height);
    }
  });

  const heightCanvas = document.getElementById("heightChart");
  if (heightCanvas) {
    heightCanvas.width = heightCanvas.parentElement.clientWidth || 360;
    heightCanvas.height = 180;
    // Initial dummy baseline points
    for (let i = 0; i < 30; i++) heightHistory.push(1.65);
    drawHeightChart();
  }

  // Mute audio button
  const muteBtn = document.getElementById("muteBtn");
  if (muteBtn) {
    muteBtn.addEventListener("click", () => {
      audioMuted = !audioMuted;
      muteBtn.textContent = audioMuted ? "🔇 Unmute Audio" : "🔊 Mute Audio";
    });
  }

  // Periodic dummy slice so waterfalls smoothly scroll even when quiet
  setInterval(() => {
    [1, 2, 3].forEach((nid) => pushSpectrogramSlice(nid, null));
  }, 100);

  loadIncidents();
  connectWebSocket();
});
