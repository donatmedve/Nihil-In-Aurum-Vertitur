/* =============================================================
   FX SYSTEM · CONTROL ROOM — app.js
   ---------------------------------------------------------------
   ► CONFIGURATION
       Set API_BASE to the URL of your running Flask backend.
       e.g. if your VPS is at 1.2.3.4:
           const API_BASE = "http://1.2.3.4:5050";
       For local development:
           const API_BASE = "http://localhost:5050";
   ============================================================= */

const API_BASE = "http://localhost:5050";   // ← CHANGE THIS

const POLL_MS  = 5_000;   // refresh interval

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let pnlChart        = null;
let prevOrderIds    = new Set();
let prevSignalHash  = "";
let isOnline        = false;

// ─────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function fmtPrice(v, dec = 5) {
  if (v == null || v === "") return "—";
  return Number(v).toFixed(dec);
}
function fmtUSD(v, dec = 2) {
  if (v == null) return "—";
  const n = Number(v);
  return (n >= 0 ? "+" : "") + "$" + Math.abs(n).toFixed(dec);
}
function fmtPct(v) {
  if (v == null) return "—";
  return (Number(v) * 100).toFixed(1) + "%";
}
function fmtTime(ts) {
  if (!ts) return "—";
  try {
    const d = new Date(ts);
    const pad = n => String(n).padStart(2, "0");
    return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  } catch { return String(ts).slice(0, 8); }
}
function fmtDate(ts) {
  if (!ts) return "—";
  try { return new Date(ts).toISOString().slice(0, 10); } catch { return ts; }
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ─────────────────────────────────────────────────────────────
// API fetch wrapper
// ─────────────────────────────────────────────────────────────
async function api(path) {
  const r = await fetch(API_BASE + path, {
    headers: { "Accept": "application/json" },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ─────────────────────────────────────────────────────────────
// Clock
// ─────────────────────────────────────────────────────────────
function tickClock() {
  const n = new Date();
  const pad = x => String(x).padStart(2, "0");
  $("clock").textContent =
    `${pad(n.getUTCHours())}:${pad(n.getUTCMinutes())}:${pad(n.getUTCSeconds())} UTC`;
}
setInterval(tickClock, 1000);
tickClock();

// ─────────────────────────────────────────────────────────────
// Connection state
// ─────────────────────────────────────────────────────────────
function setOnline(online) {
  if (isOnline === online) return;
  isOnline = online;
  const ring  = $("status-ring");
  const label = $("status-label");
  $("offline-banner").classList.toggle("hidden", online);
  if (online) {
    ring.className  = "status-ring live";
    label.textContent = "LIVE";
  } else {
    ring.className  = "status-ring dead";
    label.textContent = "OFFLINE";
  }
}

// ─────────────────────────────────────────────────────────────
// STATUS endpoint → kill switch, HWM, heartbeat
// ─────────────────────────────────────────────────────────────
async function loadStatus() {
  const d = await api("/api/status");

  setOnline(true);

  if (d.kill_switch) {
    $("status-ring").className = "status-ring dead";
    $("status-label").textContent = "KILLED";
    $("ks-badge").classList.remove("hidden");
  } else {
    $("ks-badge").classList.add("hidden");
  }

  if (d.high_water_mark != null) {
    const el = $("kpi-hwm").querySelector(".kpi-val");
    el.textContent = "$" + Number(d.high_water_mark).toFixed(0);
    el.className   = "kpi-val val-blue";
  }
  if (d.last_heartbeat) {
    const el = $("kpi-hb").querySelector(".kpi-val");
    el.textContent = fmtTime(d.last_heartbeat);
    el.className   = "kpi-val";
  }
  if (d.equity_usd != null) {
    const el = $("kpi-equity").querySelector(".kpi-val");
    el.textContent = "$" + Number(d.equity_usd).toFixed(0);
    el.className   = "kpi-val";
  }
  if (d.drawdown != null) {
    const el = $("kpi-drawdown").querySelector(".kpi-val");
    const pct = (Number(d.drawdown) * 100).toFixed(1) + "%";
    el.textContent = "-" + pct;
    el.className   = "kpi-val " + (Number(d.drawdown) > 0.03 ? "val-neg" : "val-warn");
  }
}

// ─────────────────────────────────────────────────────────────
// POSITIONS
// ─────────────────────────────────────────────────────────────
async function loadPositions() {
  const positions = await api("/api/positions");
  const open = positions.filter(p =>
    ["OPEN","open"].includes(p.status)
  );

  const kpiEl = $("kpi-positions").querySelector(".kpi-val");
  kpiEl.textContent = open.length;
  kpiEl.className   = "kpi-val " + (open.length > 0 ? "val-blue" : "");
  $("pos-badge").textContent = open.length;

  const wrap = $("positions-wrap");
  if (!open.length) {
    wrap.innerHTML = '<div class="empty-state">NO OPEN POSITIONS</div>';
    return;
  }
  wrap.innerHTML = `<div class="pos-list">${open.map(posCard).join("")}</div>`;
}

function posCard(p) {
  const side = (p.side || "").toLowerCase();
  const pnl  = p.pnl_usd != null ? Number(p.pnl_usd) : null;
  const pnlClass = pnl == null ? "" : pnl >= 0 ? "val-pos" : "val-neg";
  const pnlText  = pnl == null ? "—"
    : (pnl >= 0 ? "+" : "") + "$" + Math.abs(pnl).toFixed(2);

  return `
  <div class="pos-card ${side}">
    <div class="pos-top">
      <span class="pos-pair">${p.pair || "—"}</span>
      <span class="pos-side-badge ${side}">${(p.side || "").toUpperCase()}</span>
    </div>
    <div class="pos-row">
      <span class="lbl">ENTRY</span>
      <span class="val">${fmtPrice(p.entry_price)}</span>
    </div>
    <div class="pos-row">
      <span class="lbl">STOP LOSS</span>
      <span class="val">${fmtPrice(p.stop_loss_price)}</span>
    </div>
    <div class="pos-row">
      <span class="lbl">UNITS</span>
      <span class="val">${(p.units || 0).toLocaleString()}</span>
    </div>
    <div class="pos-row">
      <span class="lbl">OPENED</span>
      <span class="val">${fmtTime(p.opened_at_utc)} UTC</span>
    </div>
    <div class="pos-pnl ${pnlClass}">${pnlText}</div>
  </div>`;
}

// ─────────────────────────────────────────────────────────────
// P&L CHART
// ─────────────────────────────────────────────────────────────
async function loadPnl() {
  const data   = await api("/api/pnl");
  const sorted = [...data].sort((a, b) => a.date > b.date ? 1 : -1);
  const labels = sorted.map(d => d.date.slice(5));  // MM-DD
  const values = sorted.map(d => d.pnl_usd);

  // Today P&L KPI
  if (sorted.length) {
    const todayVal = sorted[sorted.length - 1]?.pnl_usd;
    const el = $("kpi-pnl").querySelector(".kpi-val");
    el.textContent = fmtUSD(todayVal);
    el.className   = "kpi-val " + (todayVal >= 0 ? "val-pos" : "val-neg");
  }

  const ctx = $("pnl-chart").getContext("2d");
  if (pnlChart) pnlChart.destroy();

  pnlChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(v =>
          v >= 0 ? "rgba(0,212,170,0.55)" : "rgba(255,45,85,0.55)"
        ),
        borderColor: values.map(v =>
          v >= 0 ? "#00d4aa" : "#ff2d55"
        ),
        borderWidth: 1,
        borderRadius: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#09090d",
          titleColor: "#3a3d52",
          bodyColor: "#b8bcd8",
          borderColor: "#1a1c26",
          borderWidth: 1,
          callbacks: {
            label: ctx =>
              (ctx.raw >= 0 ? "+" : "") + "$" + Number(ctx.raw).toFixed(2)
          }
        }
      },
      scales: {
        x: {
          grid: { color: "#1a1c26" },
          ticks: {
            color: "#3a3d52",
            font: { family: "'Courier Prime'", size: 9 },
            maxTicksLimit: 10,
          }
        },
        y: {
          grid: { color: "#1a1c26" },
          ticks: {
            color: "#3a3d52",
            font: { family: "'Courier Prime'", size: 9 },
            callback: v => (v >= 0 ? "+" : "") + "$" + v.toFixed(0)
          }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// ORDERS TABLE
// ─────────────────────────────────────────────────────────────
async function loadOrders() {
  const orders = await api("/api/orders");
  const today  = new Date().toISOString().slice(0, 10);
  const todayOrders = orders.filter(o =>
    (o.submitted_at || "").startsWith(today)
  );

  const el = $("kpi-orders").querySelector(".kpi-val");
  el.textContent = todayOrders.length;
  el.className   = "kpi-val val-blue";
  $("ord-badge").textContent = orders.length;

  // Win rate
  const filled = orders.filter(o => o.status?.toUpperCase() === "FILLED");
  const wins   = filled.filter(o => (o.pnl_usd || 0) > 0);
  if (filled.length) {
    const wr = $("kpi-win").querySelector(".kpi-val");
    wr.textContent = ((wins.length / filled.length) * 100).toFixed(0) + "%";
    wr.className   = "kpi-val " + (wins.length / filled.length >= 0.5 ? "val-pos" : "val-neg");
  }

  const tbody = $("orders-tbody");
  if (!orders.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">AWAITING DATA</td></tr>';
    return;
  }

  const newIds = new Set(orders.map(o => o.client_order_id));
  tbody.innerHTML = orders.map(o => {
    const isNew  = !prevOrderIds.has(o.client_order_id);
    const side   = (o.side || "").toUpperCase();
    const conf   = o.signal_confidence;
    const confPct = conf != null ? (conf * 100).toFixed(0) : null;
    const model  = (o.model_version || "—").slice(0, 12);

    return `<tr class="${isNew ? "new-row" : ""}">
      <td>${fmtTime(o.submitted_at)}</td>
      <td style="font-family:var(--font-display);font-weight:700;font-size:13px">${o.pair || "—"}</td>
      <td class="side-${side.toLowerCase()}">${side}</td>
      <td>${(o.units || 0).toLocaleString()}</td>
      <td>${o.filled_price ? fmtPrice(o.filled_price) : "—"}</td>
      <td>${o.stop_loss_price ? fmtPrice(o.stop_loss_price) : "—"}</td>
      <td>
        <div class="conf-wrap">
          <div class="conf-track">
            <div class="conf-fill" style="width:${confPct != null ? clamp(confPct,0,100) : 0}%"></div>
          </div>
          <span class="conf-num">${confPct != null ? confPct + "%" : "—"}</span>
        </div>
      </td>
      <td style="color:var(--text-dim);font-size:10px">${model}</td>
      <td>${statusBadge(o.status)}</td>
    </tr>`;
  }).join("");
  prevOrderIds = newIds;
}

function statusBadge(s) {
  const u = (s || "").toUpperCase();
  const map = {
    FILLED:       "st-filled",
    SUBMITTED:    "st-submitted",
    PENDING_SUBMIT:"st-pending",
    REJECTED:     "st-rejected",
    CANCELLED:    "st-cancelled",
    DEAD_LETTERED:"st-dead",
    UNCONFIRMED:  "st-pending",
  };
  const cls = map[u] || "st-pending";
  return `<span class="st ${cls}">${u || "UNKNOWN"}</span>`;
}

// ─────────────────────────────────────────────────────────────
// AI SIGNAL FEED
// ─────────────────────────────────────────────────────────────
async function loadSignals() {
  const [signals, decisions] = await Promise.all([
    api("/api/signals"),
    api("/api/decisions"),
  ]);

  // Merge & deduplicate
  const seen = new Set();
  const all  = [...signals, ...decisions].filter(e => {
    const key = (e.timestamp || e.time || "") + "|" + (e.message || e.event || "");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  all.sort((a, b) => {
    const ta = a.timestamp || a.time || "";
    const tb = b.timestamp || b.time || "";
    return tb > ta ? 1 : -1;
  });

  $("kpi-signals").querySelector(".kpi-val").textContent = signals.length;
  $("sig-badge").textContent = all.length;

  const hash = all.slice(0, 1).map(e =>
    (e.timestamp || e.time || "") + (e.message || "")
  ).join("|");

  const feed = $("ai-feed");
  if (hash === prevSignalHash) return;   // nothing new
  prevSignalHash = hash;

  if (!all.length) {
    feed.innerHTML = '<div class="empty-state" style="padding:40px 0">WAITING FOR SIGNALS</div>';
    return;
  }

  feed.innerHTML = all.slice(0, 80).map(sigCard).join("");
}

function sigClass(ev) {
  const msg = JSON.stringify(ev).toLowerCase();
  const sig = String(ev.signal ?? "");
  if (sig === "1"  || msg.includes('"buy"'))    return "buy";
  if (sig === "-1" || msg.includes('"sell"'))   return "sell";
  if (msg.includes("block") || msg.includes("denied") ||
      (msg.includes("trade_permitted") && msg.includes("false"))) return "block";
  return "abstain";
}

function sigTypeLabel(ev, cls) {
  if (cls === "buy")    return "BUY SIGNAL";
  if (cls === "sell")   return "SELL SIGNAL";
  if (cls === "block")  return "TRADE BLOCKED";
  const msg = ev.message || ev.event || "";
  if (msg) return msg.toUpperCase().slice(0, 30);
  return "ABSTAIN";
}

function sigCard(ev) {
  const cls     = sigClass(ev);
  const ts      = ev.timestamp || ev.time || "";
  const label   = sigTypeLabel(ev, cls);
  const conf    = ev.confidence != null ? Number(ev.confidence) : null;
  const confPct = conf != null ? (conf * 100).toFixed(1) : null;
  const confColor = cls === "buy" ? "var(--teal)" : cls === "sell" ? "var(--crimson)" : "var(--amber)";

  const chips = [
    ev.pair          && `PAIR: ${ev.pair}`,
    ev.model_version && `MODEL: ${String(ev.model_version).slice(0, 10)}`,
    ev.bar_utc       && `BAR: ${fmtTime(ev.bar_utc)}`,
  ].filter(Boolean);

  const reason = ev.reason || ev.rejection_reason || null;

  // Extra fields for transparency — show what the model "saw"
  const skip = new Set(["timestamp","time","message","event","signal",
    "confidence","pair","model_version","bar_utc","reason",
    "rejection_reason","level","levelname","name","pathname","lineno"]);
  const extras = Object.entries(ev)
    .filter(([k]) => !skip.has(k))
    .slice(0, 8)
    .map(([k, v]) => `<strong>${k}</strong>: ${JSON.stringify(v)}`)
    .join("<br>");

  return `
  <div class="sig-card ${cls}">
    <div class="sig-header">
      <span class="sig-type ${cls}">${label}</span>
      <span class="sig-time">${fmtTime(ts)} UTC</span>
    </div>

    ${ev.pair ? `<div class="sig-pair">${ev.pair}</div>` : ""}

    ${confPct != null ? `
    <div class="sig-conf-bar">
      <div class="sig-conf-track">
        <div class="sig-conf-fill" style="width:${clamp(confPct,0,100)}%;background:${confColor}"></div>
      </div>
      <span class="sig-conf-pct">${confPct}% conf</span>
    </div>` : ""}

    ${chips.length ? `
    <div class="sig-chips">
      ${chips.map(c => `<span class="chip">${c}</span>`).join("")}
    </div>` : ""}

    ${reason ? `
    <div class="sig-reason">⚠ <strong>BLOCKED:</strong> ${reason}</div>` : ""}

    ${extras ? `
    <div class="sig-extra">${extras}</div>` : ""}
  </div>`;
}

// ─────────────────────────────────────────────────────────────
// Refresh indicator
// ─────────────────────────────────────────────────────────────
function flashRefresh() {
  const dot = $("refresh-dot");
  dot.classList.add("active");
  setTimeout(() => dot.classList.remove("active"), 500);
}

// ─────────────────────────────────────────────────────────────
// Main refresh loop
// ─────────────────────────────────────────────────────────────
async function refresh() {
  flashRefresh();
  try {
    await Promise.allSettled([
      loadStatus(),
      loadPositions(),
      loadPnl(),
      loadOrders(),
      loadSignals(),
    ]);
  } catch (err) {
    setOnline(false);
    console.warn("[FX dashboard] refresh error:", err);
  }
}

refresh();
setInterval(refresh, POLL_MS);
