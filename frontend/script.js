// =========================================================================
// Linux Copilot XAI — Dashboard
// =========================================================================
// One master session id is shared across the AI Chat and AI One-Click Fix
// panels so their activity feeds into a single "Recent Activities" list.
// =========================================================================

// API base URL: was hardcoded to "http://localhost:8000", which silently
// broke every request whenever the frontend was opened from anywhere other
// than that exact host. Resolution order:
//   1. localStorage override ("linuxCopilotApiBaseUrl") - for pointing this
//      static frontend at a non-default host/port without editing source
//      (e.g. a judge's VM IP, or the API on a different port).
//   2. Same-origin + configured API port - works when the frontend is
//      served from the same machine as the API (the common case, e.g.
//      `python3 -m http.server` on a laptop also running the API on 8000).
const DEFAULT_API_PORT = "8000";
const API_BASE_URL = (() => {
  try {
    const override = window.localStorage.getItem("linuxCopilotApiBaseUrl");
    if (override) return override.replace(/\/+$/, "");
  } catch {
    // localStorage unavailable (privacy mode, etc.) - fall through to the default.
  }
  const { protocol, hostname } = window.location;
  const safeProtocol = protocol === "https:" || protocol === "http:" ? protocol : "http:";
  const safeHostname = hostname || "localhost";
  return `${safeProtocol}//${safeHostname}:${DEFAULT_API_PORT}`;
})();

// Demo mode: keep dashboard polling responsive without letting background
// panels compete with a live AI Assistant chat request for the same local
// LLM. Only the AI One-Click Fix panel's auto-refresh calls the LLM on its
// own schedule (it runs an AI diagnosis per detected issue) - everything
// else it gates (system/services/logs) is plain telemetry with no
// LLM call, so it stays on. Flip this to `false` (or set
// localStorage.linuxCopilotDemoMode = "false") to restore continuous
// auto-refresh of the fix panel outside of a live demo.
const DEMO_MODE = (() => {
  try {
    const stored = window.localStorage.getItem("linuxCopilotDemoMode");
    if (stored !== null) return stored !== "false";
  } catch {
    // localStorage unavailable - fall through to the default.
  }
  return true;
})();

const POLL_INTERVAL_MS = 5000;
const HISTORY_POINTS = 24;

const SESSION_ID = (() => {
  try {
    return crypto.randomUUID();
  } catch {
    return "session-" + Date.now() + "-" + Math.random().toString(16).slice(2);
  }
})();

document.getElementById("chat-session-chip").textContent = SESSION_ID.slice(0, 8);

// =========================================================================
// Linux Health Score
// =========================================================================
// Judge-facing single number (0-100) computed entirely from data the app
// already fetches — no new services, no new APIs, no duplicate monitoring:
//   CPU 25% · RAM 25% · Disk 25% · Failed Services 15% · Active Fix Engine
//   Alerts 10%
// Each fetcher below (refreshSystemInfo / refreshServices /
// refreshAifixIssues) updates its slice of `healthState` and calls
// renderHealthScore(); the score recomputes with whatever is freshest.

const healthState = {
  cpuPercent: null,
  ramPercent: null,
  diskPercent: null,
  failedServicesCount: null,
  activeIssues: null,
};

function clampScore(value) {
  return Math.max(0, Math.min(100, value));
}

function computeHealthScore() {
  const { cpuPercent, ramPercent, diskPercent, failedServicesCount, activeIssues } = healthState;
  // CPU/RAM/Disk are the core 75% of the score - wait for all three before
  // showing a number rather than rendering a misleading partial score.
  if (cpuPercent === null || ramPercent === null || diskPercent === null) return null;

  const cpuScore = clampScore(100 - cpuPercent);
  const ramScore = clampScore(100 - ramPercent);
  const diskScore = clampScore(100 - diskPercent);

  // Failed services and active alerts default to "no penalty" until their
  // panels have loaded once, so the score is available immediately and
  // only gets stricter as more data arrives.
  const failedPenalty = Math.min(100, (failedServicesCount ?? 0) * 20);
  const failedScore = 100 - failedPenalty;

  const issues = activeIssues ?? [];
  const alertPenalty = Math.min(
    100,
    issues.reduce((sum, issue) => sum + (issue.severity === "critical" ? 25 : 12), 0)
  );
  const alertScore = 100 - alertPenalty;

  const weighted =
    cpuScore * 0.25 +
    ramScore * 0.25 +
    diskScore * 0.25 +
    failedScore * 0.15 +
    alertScore * 0.1;

  return Math.round(clampScore(weighted));
}

function renderHealthScore() {
  const chip = document.getElementById("health-score-chip");
  const valueEl = document.getElementById("health-score-value");
  const score = computeHealthScore();

  if (score === null) {
    chip.dataset.level = "unknown";
    valueEl.textContent = "--";
    return;
  }

  valueEl.textContent = score;
  let level = "bad";
  if (score >= 80) level = "ok";
  else if (score >= 50) level = "warn";
  chip.dataset.level = level;
  chip.title = `Linux Health Score: ${score} / 100 — CPU 25% · RAM 25% · Disk 25% · Failed Services 15% · Active Fix Engine Alerts 10%`;
}

// ---------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const detail = data && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return data;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function formatUptime(seconds) {
  if (!seconds || seconds < 0) return "—";
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

const RISK_LABELS = {
  safe: "Safe",
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
  blocked: "Blocked",
};

function riskLabel(riskLevel) {
  return RISK_LABELS[riskLevel] || "Unknown risk";
}

function timeAgo(isoString) {
  if (!isoString) return "";
  const then = new Date(isoString.includes("Z") || isoString.includes("+") ? isoString : isoString + "Z");
  const diffSec = Math.max(0, Math.floor((Date.now() - then.getTime()) / 1000));
  if (diffSec < 5) return "just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86400)}d ago`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ---------------------------------------------------------------------
// Sparkline chart factory (Chart.js)
// ---------------------------------------------------------------------

function makeSparkline(canvasId, color) {
  if (typeof Chart === "undefined") return null;
  try {
    const ctx = document.getElementById(canvasId).getContext("2d");
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            data: [],
            borderColor: color,
            backgroundColor: color + "22",
            borderWidth: 2,
            fill: true,
            tension: 0.35,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          x: { display: false },
          y: { display: false, min: 0, max: 100 },
        },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        elements: { line: { capBezierPoints: true } },
      },
    });
  } catch (error) {
    console.error(`Could not initialize chart ${canvasId}:`, error.message);
    return null;
  }
}

function pushSparklinePoint(chart, value) {
  if (!chart) return;
  chart.data.labels.push("");
  chart.data.datasets[0].data.push(value);
  if (chart.data.labels.length > HISTORY_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update("none");
}

const cpuChart = makeSparkline("cpu-chart", "#f5a623");
const ramChart = makeSparkline("ram-chart", "#22d3ee");
const diskChart = makeSparkline("disk-chart", "#a78bfa");

let servicesDonut = null;

function renderServicesDonut(activeCount, inactiveCount) {
  if (typeof Chart === "undefined") return;
  try {
    const ctx = document.getElementById("services-donut").getContext("2d");
    if (servicesDonut) {
      servicesDonut.data.datasets[0].data = [activeCount, inactiveCount];
      servicesDonut.update();
      return;
    }
    servicesDonut = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Active", "Inactive"],
        datasets: [
          {
            data: [activeCount, inactiveCount],
            backgroundColor: ["#34d399", "#232b38"],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: false,
        cutout: "72%",
        plugins: { legend: { display: false }, tooltip: { enabled: true } },
      },
    });
  } catch (error) {
    console.error("Could not initialize services donut chart:", error.message);
  }
}

// =========================================================================
// System Health Cards (CPU / RAM / Disk) + status pills + services + logs
// =========================================================================

function setHealthLevel(card, percent) {
  card.removeAttribute("data-level");
  if (percent >= 90) card.setAttribute("data-level", "bad");
  else if (percent >= 75) card.setAttribute("data-level", "warn");
}

async function refreshStatusPills() {
  const apiPill = document.getElementById("status-api");
  const ollamaPill = document.getElementById("status-ollama");

  try {
    await apiFetch("/api/health");
    apiPill.classList.remove("status-unknown", "status-bad");
    apiPill.classList.add("status-ok");
  } catch {
    apiPill.classList.remove("status-unknown", "status-ok");
    apiPill.classList.add("status-bad");
  }

  try {
    const health = await apiFetch("/api/assistant/health");
    ollamaPill.classList.remove("status-unknown", "status-bad", "status-ok");
    ollamaPill.classList.add(health.reachable ? "status-ok" : "status-bad");
  } catch {
    ollamaPill.classList.remove("status-unknown", "status-ok");
    ollamaPill.classList.add("status-bad");
  }
}

async function refreshSystemInfo() {
  let info;
  try {
    info = await apiFetch("/api/system-info");
  } catch (error) {
    console.error("system-info fetch failed:", error.message);
    return;
  }

  document.getElementById("host-name").textContent = info.version?.hostname || "—";
  document.getElementById("host-uptime").textContent = formatUptime(info.uptime_seconds);

  // CPU
  const cpuPercent = info.cpu?.usage_percent ?? 0;
  document.getElementById("cpu-value").textContent = cpuPercent.toFixed(0);
  document.getElementById("cpu-sub").textContent = `${info.cpu?.logical_cores ?? "—"} cores`;
  const loadAvg = info.cpu?.load_avg_1m;
  document.getElementById("cpu-foot").textContent =
    loadAvg !== null && loadAvg !== undefined ? `load avg ${loadAvg.toFixed(2)}` : "—";
  setHealthLevel(document.querySelector('[data-metric="cpu"]'), cpuPercent);
  pushSparklinePoint(cpuChart, cpuPercent);

  // RAM
  const ramPercent = info.memory?.usage_percent ?? 0;
  document.getElementById("ram-value").textContent = ramPercent.toFixed(0);
  document.getElementById("ram-sub").textContent = formatBytes(info.memory?.total_bytes);
  document.getElementById("ram-foot").textContent = `${formatBytes(info.memory?.used_bytes)} used`;
  setHealthLevel(document.querySelector('[data-metric="ram"]'), ramPercent);
  pushSparklinePoint(ramChart, ramPercent);

  // Disk (primary "/" partition, or first available)
  const partitions = info.disk?.partitions || [];
  const rootPart = partitions.find((p) => p.mountpoint === "/") || partitions[0];
  const diskPercent = rootPart?.usage_percent ?? 0;
  document.getElementById("disk-value").textContent = diskPercent.toFixed(0);
  document.getElementById("disk-sub").textContent = rootPart ? formatBytes(rootPart.total_bytes) : "—";
  document.getElementById("disk-foot").textContent = rootPart
    ? `${formatBytes(rootPart.free_bytes)} free on ${rootPart.mountpoint}`
    : "no partitions found";
  setHealthLevel(document.querySelector('[data-metric="disk"]'), diskPercent);
  pushSparklinePoint(diskChart, diskPercent);

  healthState.cpuPercent = cpuPercent;
  healthState.ramPercent = ramPercent;
  healthState.diskPercent = diskPercent;
  renderHealthScore();

  return info;
}

async function refreshServices() {
  try {
    const data = await apiFetch("/api/services?limit=200");
    const list = document.getElementById("services-list");
    list.innerHTML = "";

    if (!data.services || data.services.length === 0) {
      list.innerHTML = `<li class="empty-note">${
        data.errors && data.errors.length ? escapeHtml(data.errors[0]) : "No services found."
      }</li>`;
      document.getElementById("services-summary").textContent = "—";
      renderServicesDonut(0, 0);
      healthState.failedServicesCount = 0;
      renderHealthScore();
      return;
    }

    let activeCount = 0;
    let failedCount = 0;
    data.services
      .slice()
      .sort((a, b) => (a.active_state === "active" ? -1 : 1))
      .forEach((svc) => {
        const isActive = svc.active_state === "active";
        const isFailed = svc.active_state === "failed" || svc.sub_state === "failed";
        if (isActive) activeCount += 1;
        if (isFailed) failedCount += 1;

        const li = document.createElement("li");
        li.className = `service-row ${isFailed ? "failed" : isActive ? "active" : "inactive"}`;
        li.innerHTML = `
          <span class="svc-dot"></span>
          <span class="svc-name" title="${escapeHtml(svc.name)}">${escapeHtml(svc.name)}</span>
          <span class="svc-state">${escapeHtml(svc.active_state || "?")}</span>
        `;
        list.appendChild(li);
      });

    document.getElementById("services-summary").textContent = `${activeCount}/${data.total_services} active`;
    renderServicesDonut(activeCount, data.total_services - activeCount);

    healthState.failedServicesCount = failedCount;
    renderHealthScore();
  } catch (error) {
    document.getElementById("services-list").innerHTML =
      `<li class="empty-note">Could not load services: ${escapeHtml(error.message)}</li>`;
  }
}

function priorityClass(priority) {
  const map = {
    "0": "priority-emerg", "1": "priority-alert", "2": "priority-crit", "3": "priority-err",
    "4": "priority-warning", "5": "priority-notice", "6": "priority-info", "7": "priority-debug",
  };
  return map[String(priority)] || "priority-unknown";
}

async function refreshLogs() {
  try {
    const data = await apiFetch("/api/logs?lines=60");
    const list = document.getElementById("logs-list");
    list.innerHTML = "";

    if (!data.entries || data.entries.length === 0) {
      list.innerHTML = `<li class="empty-note">${
        data.errors && data.errors.length
          ? escapeHtml(data.errors[0])
          : "No recent journal entries available on this host."
      }</li>`;
      document.getElementById("logs-summary").textContent = "—";
      return;
    }

    data.entries
      .slice()
      .reverse()
      .forEach((entry) => {
        const li = document.createElement("li");
        li.className = `log-row ${priorityClass(entry.priority)}`;
        const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : "";
        li.innerHTML = `
          <div class="log-top"><span>${escapeHtml(entry.unit || "system")}</span><span>${ts}</span></div>
          <div class="log-msg">${escapeHtml(entry.message)}</div>
        `;
        list.appendChild(li);
      });

    document.getElementById("logs-summary").textContent = `${data.total_entries} entries`;
  } catch (error) {
    document.getElementById("logs-list").innerHTML =
      `<li class="empty-note">Could not load logs: ${escapeHtml(error.message)}</li>`;
  }
}

document.getElementById("refresh-services-btn").addEventListener("click", refreshServices);
document.getElementById("refresh-logs-btn").addEventListener("click", refreshLogs);

// =========================================================================
// AI Chat Window
// =========================================================================

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");

function clearChatEmptyState() {
  const empty = chatWindow.querySelector(".chat-empty");
  if (empty) empty.remove();
}

function appendUserBubble(text) {
  clearChatEmptyState();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble user";
  bubble.textContent = text;
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendAssistantBubble(data) {
  clearChatEmptyState();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble assistant";

  const commandsHtml = (data.recommended_commands || [])
    .map(
      (c) => `
      <div class="fix-command-row">
        <code>${escapeHtml(c.command)}</code>
      </div>`
    )
    .join("");

  bubble.innerHTML = `
    <div class="chat-intent">${escapeHtml(data.intent || "general")}</div>
    <div>${escapeHtml(data.explanation)}</div>
    ${data.recommended_commands && data.recommended_commands.length ? `<div class="chat-commands">${commandsHtml}</div>` : ""}
    <div class="chat-confidence">Confidence: ${(data.confidence_score * 100).toFixed(0)}%</div>
    ${data.reasoning ? `<div class="chat-reasoning">${escapeHtml(data.reasoning)}</div>` : ""}
  `;

  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendErrorBubble(message) {
  clearChatEmptyState();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble assistant error";
  bubble.textContent = message;
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendChatMessage(message) {
  appendUserBubble(message);
  chatSendBtn.disabled = true;

  try {
    const data = await apiFetch("/api/assistant/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: SESSION_ID }),
    });
    appendAssistantBubble(data);
  } catch (error) {
    appendErrorBubble(`Could not reach the Ops Assistant: ${error.message}`);
  } finally {
    chatSendBtn.disabled = false;
    refreshActivity();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  sendChatMessage(message);
});

// =========================================================================
// Recent Activities (assistant chat interactions)
// =========================================================================

function truncate(text, max = 70) {
  if (!text) return "";
  return text.length > max ? text.slice(0, max - 1) + "…" : text;
}

async function refreshActivity() {
  const list = document.getElementById("activity-list");
  const entries = [];

  try {
    const chat = await apiFetch(`/api/assistant/history/${SESSION_ID}`);
    (chat.messages || [])
      .filter((m) => m.role === "user")
      .forEach((m) => {
        entries.push({
          type: "chat",
          icon: "💬",
          title: truncate(m.content),
          time: m.created_at,
        });
      });
  } catch {
    /* no chat history yet for this session — fine */
  }

  entries.sort((a, b) => new Date(b.time) - new Date(a.time));

  if (entries.length === 0) {
    list.innerHTML = '<li class="empty-note">Nothing yet — try asking the AI Assistant something.</li>';
    return;
  }

  list.innerHTML = "";
  entries.slice(0, 25).forEach((entry) => {
    const li = document.createElement("li");
    li.className = `activity-row type-${entry.type}`;
    li.innerHTML = `
      <span class="activity-icon">${entry.icon}</span>
      <div class="activity-body">
        <div class="activity-title">${escapeHtml(entry.title)}</div>
        <div class="activity-time">${timeAgo(entry.time)}</div>
      </div>
    `;
    list.appendChild(li);
  });
}

document.getElementById("refresh-activity-btn").addEventListener("click", refreshActivity);

// =========================================================================
// Safety rules modal
// =========================================================================

const safetyModal = document.getElementById("safety-modal");

async function openSafetyModal() {
  const content = document.getElementById("safety-rules-content");
  content.innerHTML = '<div class="loading">Loading…</div>';
  safetyModal.classList.remove("hidden");

  try {
    const data = await apiFetch("/api/commands/safety-rules");
    const renderGroup = (title, rules) =>
      `<div class="rule-group"><h4>${title}</h4>${rules
        .map(
          (r) =>
            `<div class="rule-item"><span class="rule-name">${escapeHtml(r.name)}</span> — ${escapeHtml(r.description)}</div>`
        )
        .join("")}</div>`;
    content.innerHTML =
      renderGroup(`Always Blocked (${data.block_rules.length})`, data.block_rules) +
      renderGroup(`Warnings (${data.warning_rules.length})`, data.warning_rules);
  } catch (error) {
    content.innerHTML = `<div class="error-box">Could not load safety rules: ${escapeHtml(error.message)}</div>`;
  }
}

document.getElementById("safety-rules-link").addEventListener("click", (event) => {
  event.preventDefault();
  openSafetyModal();
});
document.getElementById("close-safety-modal").addEventListener("click", () => safetyModal.classList.add("hidden"));
safetyModal.addEventListener("click", (event) => {
  if (event.target === safetyModal) safetyModal.classList.add("hidden");
});

// =========================================================================
// AI One-Click Fix Engine (Sprint 8)
// =========================================================================
// GET  /api/fixes/detect   -> Problem / Reason / Evidence / Confidence Score
//                              / Recommended Command for every issue found
// POST /api/fixes/generate -> prepares the recommended command through the
//                              EXISTING Safe Command Execution pipeline
// Confirmation/execution then reuses the EXISTING, unmodified
//   POST /api/commands/{execution_id}/confirm
// endpoint and the same safety-rule risk badges used across the app —
// nothing here runs automatically.

function aifixRenderEvidence(evidence) {
  return Object.entries(evidence || {})
    .map(([key, value]) => {
      const display = value && typeof value === "object" ? JSON.stringify(value) : String(value);
      return `<li><code>${escapeHtml(key)}</code>: ${escapeHtml(display)}</li>`;
    })
    .join("");
}

function confidenceLevel(score) {
  if (score >= 0.75) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

function aifixIssueCardHtml(issue) {
  const sev = issue.severity === "critical" ? "critical" : "warning";
  const confidencePercent = Math.round((issue.confidence_score || 0) * 100);
  const confLevel = confidenceLevel(issue.confidence_score || 0);
  return `
    <div class="aifix-issue aifix-severity-${sev}" data-issue-id="${escapeHtml(issue.issue_id)}">
      <div class="aifix-issue-head">
        <span class="aifix-severity-badge aifix-severity-${sev}">${sev === "critical" ? "CRITICAL" : "WARNING"}</span>
        <h3>${escapeHtml(issue.title)}</h3>
      </div>

      <div class="result-block">
        <h3>Problem</h3>
        <p>${escapeHtml(issue.problem)}</p>
      </div>

      <div class="result-block">
        <h3>Reason</h3>
        <p>${escapeHtml(issue.reason)}</p>
      </div>

      <div class="result-block">
        <h3>Evidence</h3>
        <ul class="aifix-evidence-list">${aifixRenderEvidence(issue.evidence)}</ul>
      </div>

      <div class="result-block confidence-block">
        <div class="confidence-block-head">
          <h3 style="margin:0;">Confidence Score</h3>
          <span class="confidence-label-tag" data-level="${confLevel}">${confLevel}</span>
        </div>
        <div class="confidence-bar-track">
          <div class="confidence-bar-fill" style="width:${confidencePercent}%"></div>
        </div>
        <span class="aifix-confidence-value" data-level="${confLevel}">${confidencePercent}%</span>
      </div>

      <div class="result-block">
        <h3>Recommended Fix</h3>
        <p class="aifix-recommended-action">${escapeHtml(issue.recommended_action)}</p>
        <div class="fix-command-row"><code>${escapeHtml(issue.recommended_command)}</code></div>
      </div>

      <div class="button-row">
        <button class="aifix-generate-btn">⚡ One-Click Fix</button>
      </div>

      <div class="aifix-preview inline-result hidden">
        <div class="blocked-banner aifix-blocked-banner hidden">
          🚫 <strong>Blocked</strong> — this command can never be executed here, confirmed or not.
        </div>

        <div class="result-block">
          <div class="result-header">
            <h3>Command <span class="editable-tag">editable</span></h3>
            <span class="risk-badge aifix-risk-badge"></span>
          </div>
          <textarea class="aifix-command-edit" rows="2" spellcheck="false"></textarea>
        </div>

        <div class="result-block">
          <h3>Explanation</h3>
          <p class="aifix-explanation"></p>
        </div>

        <ul class="risks-list aifix-findings"></ul>

        <div class="confirm-block aifix-confirm-block">
          <label class="confirm-checkbox-line">
            <input type="checkbox" class="aifix-confirm-checkbox" />
            I have reviewed this command and want to run it on this machine.
          </label>
          <div class="button-row">
            <button class="aifix-run-btn" disabled>▶ Run Fix</button>
            <button class="aifix-cancel-btn secondary">Cancel</button>
          </div>
        </div>
      </div>

      <div class="execution-result aifix-execution-result hidden">
        <h3 class="aifix-exec-heading"></h3>
        <div class="terminal-block">
          <div class="terminal-line"><span class="prompt">$</span> <span class="aifix-executed-command"></span></div>
          <pre class="aifix-exec-stdout"></pre>
          <pre class="aifix-exec-stderr stderr-text"></pre>
        </div>
        <p class="exec-meta">
          Exit code: <code class="aifix-exit-code"></code> &middot;
          Duration: <code class="aifix-duration"></code>s
        </p>
      </div>
    </div>
  `;
}

function aifixRenderFindings(findingsEl, matchedRules) {
  findingsEl.innerHTML = "";
  if (!matchedRules || matchedRules.length === 0) {
    findingsEl.innerHTML = '<li class="no-risks">No safety concerns were detected for this command.</li>';
    return;
  }
  matchedRules.forEach((rule) => {
    const severity = ["low", "medium", "high", "blocked"].includes(rule.severity) ? rule.severity : "low";
    const li = document.createElement("li");
    li.className = `risk-${severity}`;
    li.innerHTML = `<span class="risk-badge risk-${severity}">${severity}</span><span>${escapeHtml(rule.description)}</span>`;
    findingsEl.appendChild(li);
  });
}

function aifixRenderExecutionResult(card, data) {
  const executionResult = card.querySelector(".aifix-execution-result");
  executionResult.classList.remove("hidden");
  const headingText =
    {
      executed: "✅ Fix executed",
      rejected: "🚫 Cancelled — nothing was run",
      blocked: "🚫 Blocked — nothing was run",
    }[data.status] || data.status;

  card.querySelector(".aifix-exec-heading").textContent = headingText;
  card.querySelector(".aifix-executed-command").textContent = data.command;
  card.querySelector(".aifix-exec-stdout").textContent =
    data.stdout || (data.status === "executed" ? "(no output)" : "");
  card.querySelector(".aifix-exec-stderr").textContent = data.stderr || "";
  card.querySelector(".aifix-exit-code").textContent =
    data.exit_code === null || data.exit_code === undefined ? "—" : data.exit_code;
  card.querySelector(".aifix-duration").textContent = data.duration_seconds ?? 0;
}

function wireAifixCard(card, issue) {
  const generateBtn = card.querySelector(".aifix-generate-btn");
  const preview = card.querySelector(".aifix-preview");
  const blockedBanner = card.querySelector(".aifix-blocked-banner");
  const riskBadgeEl = card.querySelector(".aifix-risk-badge");
  const commandEdit = card.querySelector(".aifix-command-edit");
  const explanationEl = card.querySelector(".aifix-explanation");
  const findingsEl = card.querySelector(".aifix-findings");
  const confirmBlock = card.querySelector(".aifix-confirm-block");
  const confirmCheckbox = card.querySelector(".aifix-confirm-checkbox");
  const runBtn = card.querySelector(".aifix-run-btn");
  const cancelBtn = card.querySelector(".aifix-cancel-btn");
  const executionResult = card.querySelector(".aifix-execution-result");

  generateBtn.addEventListener("click", async () => {
    generateBtn.disabled = true;
    generateBtn.textContent = "Preparing…";
    try {
      const data = await apiFetch("/api/fixes/generate", {
        method: "POST",
        body: JSON.stringify({
          issue_id: issue.issue_id,
          issue_title: issue.title,
          command: issue.recommended_command,
          explanation: `Problem: ${issue.problem}\n\nReason: ${issue.reason}`,
          confidence_score: issue.confidence_score,
          session_id: SESSION_ID,
        }),
      });

      card.dataset.executionId = data.execution_id;
      commandEdit.value = data.command || "";
      explanationEl.textContent = data.explanation || "";
      aifixRenderFindings(findingsEl, data.matched_rules);

      riskBadgeEl.textContent = riskLabel(data.risk_level);
      riskBadgeEl.className = `risk-badge aifix-risk-badge risk-${data.risk_level}`;
      executionResult.classList.add("hidden");

      if (data.blocked) {
        blockedBanner.classList.remove("hidden");
        confirmBlock.classList.add("hidden");
      } else {
        blockedBanner.classList.add("hidden");
        confirmBlock.classList.remove("hidden");
        confirmCheckbox.checked = false;
        runBtn.disabled = true;
      }

      preview.classList.remove("hidden");
      preview.scrollIntoView({ behavior: "smooth", block: "nearest" });
      refreshActivity();
    } catch (error) {
      preview.classList.remove("hidden");
      blockedBanner.classList.add("hidden");
      confirmBlock.classList.add("hidden");
      explanationEl.textContent = `Could not prepare fix: ${error.message}`;
    } finally {
      generateBtn.disabled = false;
      generateBtn.textContent = "⚡ One-Click Fix";
    }
  });

  confirmCheckbox.addEventListener("change", () => {
    runBtn.disabled = !confirmCheckbox.checked;
  });

  runBtn.addEventListener("click", async () => {
    if (!card.dataset.executionId || !confirmCheckbox.checked) return;
    runBtn.disabled = true;
    runBtn.textContent = "Running...";
    try {
      const data = await apiFetch(`/api/commands/${card.dataset.executionId}/confirm`, {
        method: "POST",
        body: JSON.stringify({ confirm: true, edited_command: commandEdit.value }),
      });
      aifixRenderExecutionResult(card, data);
    } catch (error) {
      explanationEl.textContent = `Could not execute fix: ${error.message}`;
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "▶ Run Fix";
      refreshActivity();
    }
  });

  cancelBtn.addEventListener("click", async () => {
    if (!card.dataset.executionId) {
      preview.classList.add("hidden");
      return;
    }
    try {
      const data = await apiFetch(`/api/commands/${card.dataset.executionId}/confirm`, {
        method: "POST",
        body: JSON.stringify({ confirm: false }),
      });
      aifixRenderExecutionResult(card, data);
    } catch (error) {
      console.error("Could not cancel fix:", error.message);
    } finally {
      refreshActivity();
    }
  });
}

function renderAifixIssues(issues) {
  const container = document.getElementById("aifix-issues");
  const emptyNote = document.getElementById("aifix-empty");
  container.innerHTML = "";

  if (!issues || issues.length === 0) {
    emptyNote.classList.remove("hidden");
    return;
  }
  emptyNote.classList.add("hidden");

  issues.forEach((issue) => {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = aifixIssueCardHtml(issue);
    const card = wrapper.firstElementChild;
    wireAifixCard(card, issue);
    container.appendChild(card);
  });
}

async function refreshAifixIssues() {
  const statusChip = document.getElementById("aifix-status-chip");
  const statusText = document.getElementById("aifix-status-text");
  const loading = document.getElementById("aifix-loading");
  const errorBox = document.getElementById("aifix-error");

  loading.classList.remove("hidden");
  errorBox.classList.add("hidden");

  try {
    const data = await apiFetch("/api/fixes/detect");
    renderAifixIssues(data.issues);

    healthState.activeIssues = data.issues || [];
    renderHealthScore();

    statusChip.classList.remove("status-unknown", "status-ok", "status-bad");
    if (data.total_issues > 0) {
      statusChip.classList.add("status-bad");
      statusText.textContent = `${data.total_issues} issue${data.total_issues === 1 ? "" : "s"} found`;
    } else {
      statusChip.classList.add("status-ok");
      statusText.textContent = "all clear";
    }

    if (data.errors && data.errors.length > 0) {
      errorBox.classList.remove("hidden");
      errorBox.textContent = `Some checks could not run: ${data.errors.join(" ")}`;
    }
  } catch (error) {
    statusChip.classList.remove("status-unknown", "status-ok");
    statusChip.classList.add("status-bad");
    statusText.textContent = "scan failed";
    errorBox.classList.remove("hidden");
    errorBox.textContent = `Could not scan for issues: ${error.message}`;
  } finally {
    loading.classList.add("hidden");
  }
}

document.getElementById("aifix-refresh-btn").addEventListener("click", refreshAifixIssues);

// =========================================================================
// Bootstrap + polling
// =========================================================================

async function init() {
  refreshStatusPills();
  refreshSystemInfo();
  refreshServices();
  refreshLogs();
  refreshActivity();
  refreshAifixIssues();

  setInterval(refreshStatusPills, POLL_INTERVAL_MS * 2);
  setInterval(refreshSystemInfo, POLL_INTERVAL_MS);
  setInterval(refreshServices, POLL_INTERVAL_MS * 4);
  setInterval(refreshLogs, POLL_INTERVAL_MS * 4);

  // The AI One-Click Fix panel's refresh runs an LLM diagnosis call per
  // detected issue (unlike the other panels above, which are plain
  // telemetry reads). Auto-repeating that on a timer competes with a live
  // AI Assistant chat request for the same local model. Gated behind
  // DEMO_MODE so a live demo isn't fighting its own background polling for
  // Ollama; the initial refreshAifixIssues() call above and the manual
  // "Refresh" button both still work regardless of this setting.
  if (!DEMO_MODE) {
    setInterval(refreshAifixIssues, POLL_INTERVAL_MS * 6);
  }
}

init();