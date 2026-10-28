// Feature browser. Reads JSON written by `gpt2-sae export` from ./data/.
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { index: [], summary: null, selected: null, cache: new Map() };

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const showTok = (t) => esc(t.replace(/\n/g, "↵"));
const fmt = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));
const pct = (x) => (x == null ? "–" : `${(100 * x).toFixed(x < 0.001 ? 4 : 2)}%`);

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

// ---- tooltip -------------------------------------------------------------------------------

const tip = $("#tooltip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 12;
  const w = tip.offsetWidth;
  const x = Math.min(evt.clientX + pad, window.innerWidth - w - pad);
  tip.style.left = `${x}px`;
  tip.style.top = `${evt.clientY + pad}px`;
}
const hideTip = () => (tip.hidden = true);

// ---- charts --------------------------------------------------------------------------------

const SVG = "http://www.w3.org/2000/svg";
function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

// Vertical bars from a list of {value, tip}. Bars start at the baseline with rounded tops.
function barChart(bars, { width = 360, height = 120, xLabels = [] } = {}) {
  const m = { t: 6, r: 4, b: 18, l: 4 };
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, role: "img" });
  const max = Math.max(...bars.map((b) => b.value), 1e-9);
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const step = iw / bars.length;
  const gap = Math.min(2, step * 0.2);
  el("line", { class: "grid", x1: m.l, x2: width - m.r, y1: m.t + ih, y2: m.t + ih }, svg);
  bars.forEach((b, i) => {
    const h = (b.value / max) * ih;
    const x = m.l + i * step + gap / 2;
    const w = Math.max(step - gap, 0.5);
    if (h > 0) {
      const r = Math.min(4, w / 2, h);
      const y = m.t + ih - h;
      el("path", {
        class: "bar",
        d: `M${x},${m.t + ih} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${m.t + ih} Z`,
      }, svg);
    }
    const hit = el("rect", { class: "bar-hit", x: m.l + i * step, y: m.t, width: step, height: ih }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, b.tip));
    hit.addEventListener("mouseleave", hideTip);
  });
  for (const { i, text } of xLabels) {
    el("text", { class: "axis-label", x: m.l + (i + 0.5) * step, y: height - 4, "text-anchor": "middle" }, svg).textContent = text;
  }
  return svg;
}

// Single line over x, log-scaled y, with a crosshair tooltip.
function lineChart(points, { width = 520, height = 200, yLog = true, xFmt, yFmt } = {}) {
  const m = { t: 10, r: 10, b: 24, l: 44 };
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, role: "img" });
  const ys = points.map((p) => (yLog ? Math.log10(p.y) : p.y));
  const x0 = points[0].x, x1 = points[points.length - 1].x;
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (width - m.l - m.r);
  const sy = (y) => m.t + (1 - (y - y0) / (y1 - y0 || 1)) * (height - m.t - m.b);
  for (let k = 0; k <= 3; k++) {
    const yv = y0 + ((y1 - y0) * k) / 3;
    el("line", { class: "grid", x1: m.l, x2: width - m.r, y1: sy(yv), y2: sy(yv) }, svg);
    el("text", { class: "axis-label", x: m.l - 6, y: sy(yv) + 3, "text-anchor": "end" }, svg).textContent =
      yFmt(yLog ? 10 ** yv : yv);
  }
  for (const xv of [x0, (x0 + x1) / 2, x1]) {
    el("text", { class: "axis-label", x: sx(xv), y: height - 6, "text-anchor": "middle" }, svg).textContent = xFmt(xv);
  }
  el("path", { class: "line", d: points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(ys[i])}`).join(" ") }, svg);
  const cross = el("line", { class: "crosshair", y1: m.t, y2: height - m.b, visibility: "hidden" }, svg);
  const dot = el("circle", { class: "dot", r: 4, visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * width;
    let best = 0;
    for (let i = 1; i < points.length; i++) if (Math.abs(sx(points[i].x) - px) < Math.abs(sx(points[best].x) - px)) best = i;
    const p = points[best];
    cross.setAttribute("x1", sx(p.x));
    cross.setAttribute("x2", sx(p.x));
    dot.setAttribute("cx", sx(p.x));
    dot.setAttribute("cy", sy(ys[best]));
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("visibility", "visible");
    showTip(e, `${xFmt(p.x)} tokens<br>MSE <b>${yFmt(p.y)}</b>`);
  });
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
    hideTip();
  });
  return svg;
}

// ---- feature list ----------------------------------------------------------------------------

function sortedFiltered() {
  const q = $("#search").value.trim().toLowerCase();
  const key = $("#sort").value;
  let rows = state.index;
  if (q) {
    rows = rows.filter(
      (r) =>
        String(r.id) === q ||
        (r.label || "").toLowerCase().includes(q) ||
        (r.top_token || "").toLowerCase().includes(q),
    );
  }
  const by = {
    labeled: (a, b) => (b.label ? 1 : 0) - (a.label ? 1 : 0) || (b.confidence || 0) - (a.confidence || 0) || a.id - b.id,
    density: (a, b) => b.density - a.density,
    max_act: (a, b) => b.max_act - a.max_act,
    match: (a, b) => (b.match_cos || 0) - (a.match_cos || 0),
    id: (a, b) => a.id - b.id,
  }[key];
  return [...rows].sort(by);
}

function renderList() {
  const rows = sortedFiltered();
  $("#count").textContent = `${rows.length} of ${state.index.length} features`;
  const ul = $("#feature-list");
  ul.innerHTML = rows
    .slice(0, 600)
    .map(
      (r) => `<li data-id="${r.id}" aria-selected="${r.id === state.selected}">
        <span class="fid mono">#${r.id}</span>
        <span class="flabel ${r.label ? "" : "unlabeled"}">${esc(r.label || `top token ${showTok(r.top_token || "")}`)}</span>
      </li>`,
    )
    .join("");
}

// ---- feature detail --------------------------------------------------------------------------

async function loadFeature(id) {
  if (!state.cache.has(id)) state.cache.set(id, await getJSON(`data/features/${id}.json`));
  return state.cache.get(id);
}

function contextHTML(ctx, max) {
  const top = Math.max(...ctx.acts);
  const at = ctx.acts.indexOf(top);
  const toks = ctx.tokens
    .map((t, i) => {
      const a = ctx.acts[i];
      const alpha = a > 0 ? 0.12 + 0.68 * Math.min(a / max, 1) : 0;
      const style = alpha ? ` style="background: rgba(var(--hl-rgb), ${alpha.toFixed(2)})"` : "";
      const title = a > 0 ? ` title="${fmt(a, 2)}"` : "";
      return `<span class="tok${i === at ? " max" : ""}"${style}${title}>${showTok(t)}</span>`;
    })
    .join("");
  return `<div class="ctx"><span class="val">${fmt(top, 2)}</span>${toks}</div>`;
}

async function showFeature(id) {
  state.selected = id;
  history.replaceState(null, "", `#f=${id}`);
  for (const li of document.querySelectorAll("#feature-list li")) li.setAttribute("aria-selected", Number(li.dataset.id) === id);
  const detail = $("#detail");
  let f;
  try {
    f = await loadFeature(id);
  } catch {
    detail.innerHTML = `<p class="muted">Feature #${id} is not in the exported set.</p>`;
    return;
  }
  const label = f.label
    ? `${esc(f.label)}`
    : `<span class="muted">Unlabeled</span>`;
  const src = f.label_source === "hand" ? "hand label" : f.label_source ? `label by ${esc(f.label_source)}` : "";
  const match = f.match
    ? `<li>closest jb latent <a href="${f.match.url}" target="_blank" rel="noopener">#${f.match.jb_id}</a>, cosine <b>${fmt(f.match.cos, 2)}</b></li>`
    : "";
  detail.innerHTML = `
    <h2><span class="muted mono">#${f.id}</span> ${label}</h2>
    ${f.description ? `<p>${esc(f.description)}</p>` : ""}
    <ul class="stats">
      <li>fires on <b>${pct(f.density)}</b> of tokens</li>
      <li>max activation <b>${fmt(f.max_act, 2)}</b></li>
      ${f.confidence != null ? `<li>${src}, confidence <b>${fmt(f.confidence, 2)}</b></li>` : ""}
      ${match}
    </ul>
    <div class="row2">
      <div><h3>Activation histogram (log10 scale, nonzero only)</h3><div id="hist"></div></div>
      <div>
        <h3>Promoted output tokens</h3><div class="chips">${f.logit_up.map(([t]) => `<span class="chip">${showTok(t)}</span>`).join("")}</div>
        <h3>Suppressed output tokens</h3><div class="chips">${f.logit_down.map(([t]) => `<span class="chip">${showTok(t)}</span>`).join("")}</div>
      </div>
    </div>
    <h3>Top activating contexts</h3>
    <div class="contexts">${f.contexts.map((c) => contextHTML(c, f.max_act)).join("")}</div>`;

  const edges = f.hist_edges;
  const bars = f.hist.map((v, i) => ({
    value: v,
    tip: `${(10 ** edges[i]).toPrecision(2)}–${(10 ** edges[i + 1]).toPrecision(2)}: <b>${v.toLocaleString()}</b> tokens`,
  }));
  const ticks = [];
  edges.forEach((e, i) => {
    if (i < f.hist.length && Number.isInteger(Math.round(e * 1000) / 1000)) ticks.push({ i, text: `1e${Math.round(e)}` });
  });
  $("#hist").appendChild(barChart(bars, { xLabels: ticks }));
}

// ---- other tabs ------------------------------------------------------------------------------

function renderResults() {
  const s = state.summary;
  const box = $("#results");
  if (!s) return;
  const ev = s.eval || {};
  const rows = [
    ["CE loss recovered", "ce_recovered", 3],
    ["Fraction of variance unexplained", "fvu", 3],
    ["L0 (active latents per token)", "l0", 1],
    ["Dead latents on eval text", "dead_frac", 3],
    ["CE loss, clean", "ce_clean", 3],
    ["CE loss, SAE spliced in", "ce_spliced", 3],
    ["CE loss, activation zeroed", "ce_zero", 3],
  ];
  const cols = ["ours", "jb"].filter((k) => ev[k]);
  box.innerHTML = `
    <div class="cards">
      <div class="card">
        <h2>Eval metrics (${(ev.ours?.n_tokens || 0).toLocaleString()} held-out tokens)</h2>
        <table>
          <thead><tr><th>Metric</th>${cols.map((c) => `<th class="num">${c === "ours" ? "This SAE" : "gpt2-small-res-jb"}</th>`).join("")}</tr></thead>
          <tbody>${rows.map(([name, k, d]) => `<tr><td>${name}</td>${cols.map((c) => `<td class="num">${fmt(ev[c][k], d)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
        <p class="muted">Width ${s.d_sae.toLocaleString()}, k = ${s.k}, trained on ${(s.total_tokens / 1e6).toFixed(0)}M tokens. ${s.n_labeled} features labeled.</p>
      </div>
      <div class="card"><h2>Training: normalized MSE</h2><div id="train-chart"></div></div>
      ${s.match_hist ? `<div class="card"><h2>Best decoder cosine to a jb latent</h2><div id="match-chart"></div>
        <p class="muted">For each of our latents, the highest cosine similarity between its decoder direction and any jb latent's.
        Median ${fmt(ev.match?.median_cos, 2)}. ${pct(ev.match?.["frac_above_0.9"])} above 0.9.</p></div>` : ""}
    </div>`;
  if (s.training?.length > 1) {
    const pts = s.training.map((r) => ({ x: r.tokens, y: r.mse }));
    $("#train-chart").appendChild(
      lineChart(pts, { xFmt: (x) => `${(x / 1e6).toFixed(0)}M`, yFmt: (y) => y.toFixed(3) }),
    );
  }
  if (s.match_hist) {
    const n = s.match_hist.length;
    const bars = s.match_hist.map((v, i) => ({
      value: v,
      tip: `cosine ${(i / n).toFixed(3)}–${((i + 1) / n).toFixed(3)}: <b>${v.toLocaleString()}</b> latents`,
    }));
    const ticks = [0, 0.25, 0.5, 0.75].map((x) => ({ i: Math.round(x * n), text: x.toFixed(2) }));
    $("#match-chart").appendChild(barChart(bars, { width: 420, height: 140, xLabels: ticks }));
  }
}

function renderSteering() {
  const groups = state.summary?.steering || [];
  if (!groups.length) {
    $("#steering").innerHTML = `<p class="muted">No steering samples exported yet.</p>`;
    return;
  }
  $("#steering").innerHTML = groups
    .map((g) => {
      const byPrompt = {};
      for (const s of g.samples) (byPrompt[s.prompt] ||= []).push(s);
      return `<div class="card steer-group">
        <h2><a href="#f=${g.feature}" data-goto="${g.feature}">#${g.feature}</a> ${esc(g.label || "")}</h2>
        <table class="steer-sample"><tbody>
        ${Object.entries(byPrompt)
          .map(([p, ss]) =>
            ss.map((s) => `<tr><td>×${s.coef}</td><td><span class="prompt">${esc(p)}</span>${esc(s.text)}</td></tr>`).join(""),
          )
          .join("")}
        </tbody></table></div>`;
    })
    .join("");
}

function setTab(name) {
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", b.dataset.tab === name);
  for (const p of document.querySelectorAll(".tab-panel")) p.hidden = p.id !== `tab-${name}`;
}

// ---- wiring ----------------------------------------------------------------------------------

async function init() {
  try {
    [state.summary, state.index] = await Promise.all([getJSON("data/summary.json"), getJSON("data/index.json")]);
  } catch (err) {
    $("#detail").innerHTML = `<p class="muted">Could not load data (${esc(err.message)}). Run <code>gpt2-sae export</code> first.</p>`;
    return;
  }
  const s = state.summary;
  $("#run-line").textContent =
    `TopK sparse autoencoder on GPT-2 small, ${s.hook}. ${s.d_sae.toLocaleString()} latents, k = ${s.k}.`;
  renderList();
  renderResults();
  renderSteering();

  $("#search").addEventListener("input", renderList);
  $("#sort").addEventListener("change", renderList);
  $("#feature-list").addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (li) showFeature(Number(li.dataset.id));
  });
  for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => setTab(b.dataset.tab));
  $("#steering").addEventListener("click", (e) => {
    const a = e.target.closest("[data-goto]");
    if (!a) return;
    e.preventDefault();
    setTab("features");
    showFeature(Number(a.dataset.goto));
  });

  window.addEventListener("hashchange", () => {
    const id = Number(new URLSearchParams(location.hash.slice(1)).get("f"));
    if (Number.isFinite(id) && id !== state.selected) {
      setTab("features");
      showFeature(id);
    }
  });

  const fromHash = Number(new URLSearchParams(location.hash.slice(1)).get("f"));
  const first = sortedFiltered()[0];
  if (Number.isFinite(fromHash) && location.hash) showFeature(fromHash);
  else if (first) showFeature(first.id);
}

init();
