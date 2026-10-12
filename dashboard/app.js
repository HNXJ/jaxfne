"use strict";

var PALETTE = ["#1565c0", "#ff9800", "#00acc1", "#e53935", "#4caf50", "#8e24aa"];
var DASHES = ["solid", "dash", "dot", "dashdot"];
var GOLD = "#cfb87c";
var STAGES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"];

function el(tag, text) {
  var node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function link(href, text, cls) {
  var a = el("a", text);
  a.href = href;
  if (cls) a.className = cls;
  return a;
}

function shortSha(sha) {
  return sha ? String(sha).slice(0, 7) : "";
}

/* Verdict chips: text plus symbol, never color alone. */
function chipClass(verdict) {
  if (verdict === "pass" || verdict === "reproduced") return "chip-pass";
  if (verdict === "fail" || verdict === "not reproduced") return "chip-fail";
  if (verdict === "pending") return "chip-pending";
  return "chip-incomplete";
}

function chipSymbol(verdict) {
  if (verdict === "pass" || verdict === "reproduced") return "✓";
  if (verdict === "fail" || verdict === "not reproduced") return "✗";
  if (verdict === "pending") return "…";
  return "◌";
}

function verdictChip(verdict) {
  var span = el("span", chipSymbol(verdict) + " " + verdict);
  span.className = "chip " + chipClass(verdict);
  return span;
}

function statusChip(status) {
  var symbol = "?";
  if (status === "agrees") symbol = "✓";
  else if (status === "differs") symbol = "✗";
  else if (status === "invalid") symbol = "∅";
  else if (status === "pending") symbol = "…";
  var span = el("span", symbol + " " + status);
  var cls = status === "agrees" ? "chip-pass" : status === "pending" ? "chip-pending"
    : status === "differs" ? "chip-fail" : "chip-incomplete";
  span.className = "chip " + cls;
  return span;
}

function passCount(paper) {
  var criteria = paper.criteria || [];
  var k = criteria.filter(function (c) { return c.verdict === "pass"; }).length;
  return k + " of " + criteria.length + " criteria pass";
}

function stageTrack(stage) {
  var current = STAGES.indexOf(stage);
  var list = el("ol");
  list.className = "stage-track";
  list.setAttribute("aria-label", "Stage " + stage + " of P0 to P7");
  STAGES.forEach(function (name, i) {
    var item = el("li", name);
    item.className = i <= current ? "stage-fill" : "stage-empty";
    item.setAttribute("aria-hidden", "true");
    if (i === current) item.title = name + " (current)";
    list.appendChild(item);
  });
  return list;
}

// Plotly places annotations on a log axis in log10 units; shapes take data units.
function annotY(chart, v) {
  return chart.y && chart.y.log ? Math.log10(v) : v;
}

function axisTitle(axis) {
  var label = (axis && axis.label) || "";
  var unit = (axis && axis.unit) || "";
  return unit ? label + " (" + unit + ")" : label;
}

function groupColors(items) {
  var map = {};
  var next = 0;
  items.forEach(function (item) {
    var g = item.group === undefined || item.group === null ? item.name : item.group;
    if (!(g in map)) map[g] = PALETTE[next++ % PALETTE.length];
  });
  return map;
}

function plotColors() {
  var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return {
    fg: dark ? "#f2f2f2" : "#1a1a1a",
    grid: dark ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.15)"
  };
}

function renderLinesChart(div, chart) {
  var colors = groupColors(chart.traces || []);
  var seen = {};
  var data = (chart.traces || []).map(function (t) {
    var g = t.group;
    var n = seen[g] === undefined ? 0 : seen[g] + 1;
    seen[g] = n;
    var color = colors[g];
    return {
      x: t.x, y: t.y, name: t.name,
      mode: t.markers_only ? "markers" : "lines+markers",
      marker: { color: color },
      line: { color: color, dash: DASHES[n % DASHES.length] }
    };
  });
  var shapes = [];
  var annotations = [];
  (chart.refs || []).forEach(function (ref) {
    if (ref.kind === "hline") {
      shapes.push({
        type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: ref.y, y1: ref.y,
        line: { color: GOLD, dash: "dash", width: 2 }
      });
      annotations.push({ xref: "paper", x: 1, xanchor: "right", yref: "y", y: annotY(chart, ref.y), text: ref.label, showarrow: false });
    } else if (ref.kind === "band") {
      shapes.push({
        type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: ref.y0, y1: ref.y1,
        fillcolor: "rgba(207, 184, 124, 0.3)", line: { width: 0 }
      });
      annotations.push({ xref: "paper", x: 1, xanchor: "right", yref: "y", y: annotY(chart, ref.y1), text: ref.label, showarrow: false });
    }
  });
  var palette = plotColors();
  var w = div.clientWidth || 640;
  var layout = {
    title: { text: chart.title },
    autosize: true,
    height: Math.max(320, Math.round(w * 0.75)) + 100,
    xaxis: { title: { text: axisTitle(chart.x) }, mirror: false, showline: true, zeroline: false, gridcolor: palette.grid },
    yaxis: {
      title: { text: axisTitle(chart.y) }, mirror: false, showline: true, zeroline: false,
      gridcolor: palette.grid, type: chart.y && chart.y.log ? "log" : "linear",
      dtick: chart.y && chart.y.log ? "D2" : undefined
    },
    legend: { bgcolor: "rgba(0,0,0,0)", borderwidth: 0, orientation: "h", x: 0, xanchor: "left", y: -0.28, yanchor: "top" },
    shapes: shapes,
    annotations: annotations,
    font: { color: palette.fg },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 40, r: 12, b: 150, l: 60 }
  };
  window.Plotly.newPlot(div, data, layout, { responsive: true, displaylogo: false });
}

function renderBarsChart(div, chart) {
  var colors = groupColors(chart.traces || []);
  var data = (chart.traces || []).map(function (t, i) {
    return {
      x: chart.categories, y: t.y, name: t.name, type: "bar",
      marker: { color: colors[t.group === undefined ? t.name : t.group] || PALETTE[i % PALETTE.length] }
    };
  });
  var shapes = [];
  var annotations = [];
  (chart.refs || []).forEach(function (ref) {
    if (ref.kind === "hline") {
      shapes.push({
        type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: ref.y, y1: ref.y,
        line: { color: GOLD, dash: "dash", width: 2 }
      });
      annotations.push({ xref: "paper", x: 1, xanchor: "right", yref: "y", y: annotY(chart, ref.y), text: ref.label, showarrow: false });
    } else if (ref.kind === "band") {
      shapes.push({
        type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: ref.y0, y1: ref.y1,
        fillcolor: "rgba(207, 184, 124, 0.3)", line: { width: 0 }
      });
      annotations.push({ xref: "paper", x: 1, xanchor: "right", yref: "y", y: annotY(chart, ref.y1), text: ref.label, showarrow: false });
    }
  });
  var palette = plotColors();
  var w = div.clientWidth || 640;
  var layout = {
    title: { text: chart.title },
    autosize: true,
    height: Math.max(320, Math.round(w * 0.75)) + 100,
    barmode: "group",
    xaxis: { title: { text: axisTitle(chart.x) }, mirror: false, showline: true, zeroline: false, gridcolor: palette.grid },
    yaxis: {
      title: { text: axisTitle(chart.y) }, mirror: false, showline: true, zeroline: false,
      gridcolor: palette.grid, type: chart.y && chart.y.log ? "log" : "linear",
      dtick: chart.y && chart.y.log ? "D2" : undefined
    },
    legend: { bgcolor: "rgba(0,0,0,0)", borderwidth: 0, orientation: "h", x: 0, xanchor: "left", y: -0.28, yanchor: "top" },
    shapes: shapes,
    annotations: annotations,
    font: { color: palette.fg },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 40, r: 12, b: 150, l: 60 }
  };
  window.Plotly.newPlot(div, data, layout, { responsive: true, displaylogo: false });
}

function evidenceNode(evidence) {
  if (!evidence) return null;
  var path = evidence.path || "";
  if (evidence.url) return link(evidence.url, path, "ev-link");
  return el("span", path);
}

function valueCell(criterion) {
  var cell = document.createElement("td");
  var label = el("div", (criterion.value_label || "") + (criterion.unit ? " [" + criterion.unit + "]" : ""));
  label.className = "small";
  cell.appendChild(label);
  var valueText = criterion.value_text !== undefined && criterion.value_text !== null
    ? String(criterion.value_text) : "—";
  var ev = criterion.evidence || {};
  if (ev.url) cell.appendChild(link(ev.url, valueText, "ev-link"));
  else cell.appendChild(el("span", valueText));
  (criterion.extra || []).forEach(function (x) {
    var line = el("span", (x.label || "") + ": " + (x.value_text !== undefined && x.value_text !== null ? x.value_text : "—"));
    line.className = "extra small";
    cell.appendChild(line);
  });
  return cell;
}

function criteriaTable(paper) {
  var criteria = paper.criteria || [];
  if (!criteria.length) return el("p", "No criteria scored yet.");
  var wrap = document.createElement("div");
  var table = document.createElement("table");
  table.className = "criteria";
  var head = document.createElement("thead");
  var headRow = document.createElement("tr");
  var labels = ["ID", "Claim", "Criterion", "Paper", "Ours", "Verdict", "Reason"];
  labels.forEach(function (h) {
    headRow.appendChild(el("th", h));
  });
  head.appendChild(headRow);
  table.appendChild(head);
  var body = document.createElement("tbody");
  var lastGroup = null;
  criteria.forEach(function (c) {
    var group = c.group === undefined || c.group === null ? "" : String(c.group);
    if (group && group !== lastGroup) {
      var grow = document.createElement("tr");
      grow.className = "group-row";
      var gth = el("th", group);
      gth.colSpan = 7;
      grow.appendChild(gth);
      body.appendChild(grow);
      lastGroup = group;
    }
    var row = document.createElement("tr");
    var idCell = document.createElement("td");
    idCell.appendChild(el("span", c.id || ""));
    row.appendChild(idCell);
    var claimCell = document.createElement("td");
    claimCell.appendChild(el("div", c.claim || ""));
    var est = el("div", "Estimand: " + (c.estimand || ""));
    est.className = "small";
    claimCell.appendChild(est);
    var pre = el("div", "Prereg: " + (c.prereg || ""));
    pre.className = "small";
    claimCell.appendChild(pre);
    row.appendChild(claimCell);
    var critCell = document.createElement("td");
    critCell.appendChild(el("span", c.criterion || ""));
    row.appendChild(critCell);
    row.appendChild(el("td", c.paper_value || "—"));
    var valCell = valueCell(c);
    row.appendChild(valCell);
    var verdictCell = document.createElement("td");
    verdictCell.appendChild(verdictChip(c.verdict));
    row.appendChild(verdictCell);
    var reasonCell = document.createElement("td");
    reasonCell.appendChild(el("span", c.reason || ""));
    if (!c.reason) reasonCell.classList.add("empty-cell");
    row.appendChild(reasonCell);
    Array.prototype.forEach.call(row.children, function (td, i) {
      var lab = el("span", labels[i] + ": ");
      lab.className = "cell-label";
      td.insertBefore(lab, td.firstChild);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

/* Static PNG rendered from the same data by scripts/build_dashboard_figures.py. */
function staticFigure(paper, chart) {
  var box = el("details");
  box.className = "static-fig";
  box.appendChild(el("summary", "Static figure"));
  var img = el("img");
  img.src = "figures/" + paper.id + "-" + chart.id + ".png";
  img.alt = chart.title + " (static)";
  img.loading = "lazy";
  img.addEventListener("error", function () { box.hidden = true; });
  box.appendChild(img);
  return box;
}

function dataTable(chart) {
  var box = el("details");
  box.appendChild(el("summary", "Data table"));
  var table = el("table");
  table.className = "data-table";
  var hr = el("tr");
  ["Series", chart.kind === "bars" ? "Category" : axisTitle(chart.x), axisTitle(chart.y) || "value"].forEach(function (h) {
    hr.appendChild(el("th", h));
  });
  table.appendChild(hr);
  (chart.traces || []).forEach(function (t) {
    var xs = chart.kind === "bars" ? chart.categories : t.x;
    (t.y || []).forEach(function (y, i) {
      var row = el("tr");
      row.appendChild(el("td", t.name));
      row.appendChild(el("td", xs && xs[i] !== undefined ? String(xs[i]) : ""));
      row.appendChild(el("td", y === null || y === undefined ? "—" : String(y)));
      table.appendChild(row);
    });
  });
  box.appendChild(table);
  return box;
}

function chartsSection(paper) {
  var charts = paper.charts || [];
  var section = document.createElement("div");
  if (!charts.length) {
    section.appendChild(el("p", "No charts."));
    return section;
  }
  var grid = el("div");
  grid.className = "chart-grid";
  charts.forEach(function (chart) {
    var fig = el("figure");
    fig.className = "chart-fig";
    var div = el("div");
    div.className = "chart-div";
    div.id = "chart-" + paper.id + "-" + chart.id;
    fig.appendChild(div);
    var caption = el("figcaption", (chart.caption || ""));
    fig.appendChild(caption);
    fig.appendChild(staticFigure(paper, chart));
    fig.appendChild(dataTable(chart));
    var ev = evidenceNode(chart.evidence);
    if (ev) {
      var evLine = el("div");
      evLine.className = "small";
      evLine.appendChild(el("span", "Evidence: "));
      evLine.appendChild(ev);
      fig.appendChild(evLine);
    }
    grid.appendChild(fig);
  });
  section.appendChild(grid);
  return section;
}

function sensScale(values) {
  var nums = values.filter(function (v) { return typeof v === "number" && isFinite(v); });
  if (!nums.length) return null;
  var lo = Math.min.apply(null, nums);
  var hi = Math.max.apply(null, nums);
  if (lo === hi) { lo -= 1; hi += 1; }
  var pad = (hi - lo) * 0.08;
  return { lo: lo - pad, hi: hi + pad };
}

function sensRowViz(run, row) {
  var wrap = el("div");
  wrap.className = "sens-metric";
  var head = el("div", (row.metric || "") + (row.unit ? " [" + row.unit + "]" : ""));
  wrap.appendChild(head);
  var refRange = row.ref_range || null;
  var scale = sensScale([row.value, row.ref].concat(refRange || []));
  if (row.value === null || row.value === undefined || scale === null) {
    wrap.appendChild(el("div", "value: —"));
    var nums0 = el("div", "ref " + String(row.ref) +
      (refRange ? ", range " + refRange[0] + "–" + refRange[1] : ""));
    nums0.className = "sens-nums";
    wrap.appendChild(nums0);
    return wrap;
  }
  var pos = function (v) {
    return String(Math.max(0, Math.min(100, (v - scale.lo) / (scale.hi - scale.lo) * 100))) + "%";
  };
  var track = el("div");
  track.className = "sens-track";
  track.setAttribute("role", "img");
  track.setAttribute("aria-label", (row.metric || "") + ": value " + row.value +
    ", ref " + row.ref + (refRange ? ", range " + refRange[0] + " to " + refRange[1] : ""));
  if (refRange) {
    var band = el("div");
    band.className = "sens-band";
    band.style.left = pos(Math.min(refRange[0], refRange[1]));
    band.style.width = "calc(" + pos(Math.max(refRange[0], refRange[1])) + " - " + pos(Math.min(refRange[0], refRange[1])) + ")";
    track.appendChild(band);
  }
  if (typeof row.ref === "number" && isFinite(row.ref)) {
    var tick = el("div");
    tick.className = "sens-tick";
    tick.style.left = pos(row.ref);
    track.appendChild(tick);
  }
  var dot = el("div");
  dot.className = "sens-dot";
  dot.style.left = pos(row.value);
  track.appendChild(dot);
  wrap.appendChild(track);
  var nums = el("div", "value " + row.value + "; ref " + row.ref +
    (refRange ? "; range " + refRange[0] + "–" + refRange[1] : ""));
  nums.className = "sens-nums";
  wrap.appendChild(nums);
  return wrap;
}

function sensitivitySection(paper) {
  var runs = paper.sensitivity || [];
  var section = document.createElement("div");
  if (!runs.length) {
    section.appendChild(el("p", "No sensitivity runs."));
    return section;
  }
  runs.forEach(function (run) {
    var box = el("div");
    box.className = "sens-run";
    var head = el("div");
    var title = el("strong", (run.id || "") + ": " + (run.label || ""));
    head.appendChild(title);
    head.appendChild(el("span", " "));
    head.appendChild(statusChip(run.status));
    box.appendChild(head);
    if (run.note) {
      var note = el("div", run.note);
      note.className = "small";
      box.appendChild(note);
    }
    (run.rows || []).forEach(function (row) {
      box.appendChild(sensRowViz(run, row));
    });
    section.appendChild(box);
  });
  return section;
}

function preregsSection(paper) {
  var preregs = paper.preregs || [];
  if (!preregs.length) return el("p", "No preregistrations yet.");
  var list = el("ul");
  list.className = "preregs";
  preregs.forEach(function (p) {
    var item = document.createElement("li");
    item.appendChild(el("span", (p.id || "") + " — signed " + (p.signed || "") + " — "));
    if (p.url) item.appendChild(link(p.url, p.path || p.url, "ev-link"));
    else item.appendChild(el("span", p.path || ""));
    list.appendChild(item);
  });
  return list;
}

function openSection(paper) {
  var open = paper.open || [];
  if (!open.length) return el("p", "None.");
  var list = el("ul");
  list.className = "open-list";
  open.forEach(function (text) {
    list.appendChild(el("li", text));
  });
  return list;
}

/* ---- v2: classes, tabs, model pages ---- */

var CLASSES = [
  { id: "verified", label: "Verified replication", chip: "chip-pass", symbol: "✓",
    def: "Every primary criterion of a signed preregistration passes, run through the jaxfne library (B1)." },
  { id: "partial", label: "Partial replication", chip: "chip-partial", symbol: "◐",
    def: "A signed preregistration is scored and some criteria pass, or all pass on a standalone port only." },
  { id: "not reproduced", label: "Not reproduced", chip: "chip-fail", symbol: "✗",
    def: "A signed preregistration is scored and its verdict is a fail, shown with the same prominence as a pass (B2)." },
  { id: "demonstration", label: "Demonstration", chip: "chip-incomplete", symbol: "◌",
    def: "A published model runs in jaxfne without a scored numerical comparison to the paper. It carries no verdict." }
];
var TABS = ["overview", "models", "validation", "roadmap", "method"];

function classInfo(id) {
  var hit = null;
  CLASSES.forEach(function (c) { if (c.id === id) hit = c; });
  return hit || { id: id, label: String(id), chip: "chip-incomplete", symbol: "?", def: "" };
}

function classChip(id) {
  var info = classInfo(id);
  var span = el("span", info.symbol + " " + info.label);
  span.className = "chip chip-class " + info.chip;
  span.title = info.def;
  return span;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function scored(papers) {
  return papers.filter(function (p) { return p.class !== "demonstration"; });
}

function countVerdicts(papers) {
  var n = { pass: 0, fail: 0, pending: 0 };
  papers.forEach(function (p) {
    (p.criteria || []).forEach(function (c) { if (c.verdict in n) n[c.verdict] += 1; });
  });
  return n;
}

function kpi(value, label) {
  var box = el("div");
  box.className = "kpi";
  box.appendChild(el("div", String(value))).className = "kpi-value";
  box.appendChild(el("div", label)).className = "kpi-label";
  return box;
}

/* Markdown with TeX: math is cut out before marked runs and rendered by KaTeX afterwards. */
function renderMarkdown(text) {
  var box = el("div");
  box.className = "model-md";
  if (!window.marked) {
    var pre = el("pre", text);
    box.appendChild(pre);
    return box;
  }
  var math = [];
  var cut = text.replace(/\$\$([\s\S]+?)\$\$/g, function (_, tex) {
    math.push({ tex: tex, display: true });
    return "@@MATH" + (math.length - 1) + "@@";
  }).replace(/\$([^$\n]+?)\$/g, function (_, tex) {
    math.push({ tex: tex, display: false });
    return "@@MATH" + (math.length - 1) + "@@";
  });
  var html = window.marked.parse(cut);
  html = html.replace(/@@MATH(\d+)@@/g, function (_, i) {
    var m = math[Number(i)];
    if (window.katex) {
      return window.katex.renderToString(m.tex, { displayMode: m.display, throwOnError: false });
    }
    var esc = m.tex.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    return m.display ? "<pre>" + esc + "</pre>" : "<code>" + esc + "</code>";
  });
  box.innerHTML = html; // committed, reviewed files from this repository only
  return box;
}

function paperCard(paper) {
  var card = el("a");
  card.className = "card";
  card.href = "#model/" + paper.id;
  card.appendChild(el("h3", paper.title));
  card.appendChild(el("p", (paper.citation || "") + " · release " + (paper.release || ""))).className = "small";
  var chips = el("p");
  chips.appendChild(classChip(paper["class"]));
  if (paper.verdict && paper.verdict !== paper["class"]) {
    chips.appendChild(el("span", " "));
    chips.appendChild(verdictChip(paper.verdict));
    if (paper.verdict_note) chips.appendChild(el("span", " " + paper.verdict_note));
  }
  card.appendChild(chips);
  if (paper.question) card.appendChild(el("p", paper.question));
  if ((paper.criteria || []).length) card.appendChild(el("p", passCount(paper))).className = "pass-count";
  return card;
}

function renderOverview(view, data) {
  var papers = data.papers || [];
  view.appendChild(el("h2", "Overview"));
  var kpis = el("div");
  kpis.className = "kpis";
  CLASSES.forEach(function (c) {
    var n = papers.filter(function (p) { return p["class"] === c.id; }).length;
    kpis.appendChild(kpi(n, c.label));
  });
  var v = countVerdicts(scored(papers));
  kpis.appendChild(kpi(v.pass + " / " + v.fail + " / " + v.pending, "preregistered criteria pass / fail / pending"));
  view.appendChild(kpis);
  CLASSES.forEach(function (c) {
    var group = papers.filter(function (p) { return p["class"] === c.id; });
    if (!group.length) return;
    var h = el("h3", c.label);
    view.appendChild(h);
    view.appendChild(el("p", c.def)).className = "small";
    var cards = el("div");
    cards.className = "cards";
    group.forEach(function (p) { cards.appendChild(paperCard(p)); });
    view.appendChild(cards);
  });
}

function renderModels(view, data) {
  view.appendChild(el("h2", "Models"));
  var cards = el("div");
  cards.className = "cards";
  (data.papers || []).forEach(function (p) { cards.appendChild(paperCard(p)); });
  view.appendChild(cards);
}

function renderModel(view, data, paper) {
  var card = el("article");
  card.className = "detail-card";
  card.appendChild(el("h2", paper.title));
  var sub = el("p", (paper.citation || "") + " · release " + (paper.release || "") + " · " + (paper.tier || "") +
    (paper.stage ? " · stage " + paper.stage : ""));
  sub.className = "sub";
  card.appendChild(sub);
  var chips = el("p");
  chips.appendChild(classChip(paper["class"]));
  if (paper.verdict) {
    chips.appendChild(el("span", " Preregistered verdict: "));
    chips.appendChild(verdictChip(paper.verdict));
    if (paper.verdict_note) chips.appendChild(el("span", " " + paper.verdict_note));
  }
  card.appendChild(chips);
  if (paper.stage) card.appendChild(stageTrack(paper.stage));
  var lane = paper.lane || {};
  var laneLine = el("p");
  laneLine.className = "small";
  laneLine.appendChild(el("span", "Code and results at " + (lane.branch || "") + " @ "));
  if (lane.sha) laneLine.appendChild(link("https://github.com/hnxj/jaxfne/tree/" + lane.sha, shortSha(lane.sha)));
  card.appendChild(laneLine);
  card.appendChild(el("p", paper.summary || ""));

  var md = el("div");
  md.className = "model-page";
  card.appendChild(md);
  if (paper.page) {
    md.appendChild(el("p", "Loading the model page…"));
    fetch(paper.page).then(function (r) {
      if (!r.ok) throw new Error("missing");
      return r.text();
    }).then(function (text) {
      clear(md);
      md.appendChild(renderMarkdown(text));
    }).catch(function () {
      clear(md);
      md.appendChild(el("p", "The model page did not load."));
    });
  }

  card.appendChild(el("h3", paper["class"] === "demonstration" ? "Declared checks" : "Original and replicated"));
  card.appendChild(criteriaTable(paper));
  if ((paper.charts || []).length) {
    card.appendChild(el("h3", "Figures"));
    card.appendChild(chartsSection(paper));
  }
  if ((paper.sensitivity || []).length) {
    card.appendChild(el("h3", "Sensitivity"));
    card.appendChild(sensitivitySection(paper));
  }
  card.appendChild(el("h3", "Preregistrations"));
  card.appendChild(preregsSection(paper));
  card.appendChild(el("h3", "Open items"));
  card.appendChild(openSection(paper));
  view.appendChild(card);
  drawCharts(paper);
}

function drawCharts(paper) {
  (paper.charts || []).forEach(function (chart) {
    var div = document.getElementById("chart-" + paper.id + "-" + chart.id);
    if (!div) return;
    if (!window.Plotly) {
      div.appendChild(el("p", "The chart library did not load; the static figure and the data table below hold the same values."));
      return;
    }
    try {
      if (chart.kind === "bars") renderBarsChart(div, chart);
      else renderLinesChart(div, chart);
    } catch (e) {
      div.appendChild(el("p", "Chart failed to render; see the data table below."));
    }
  });
}

function renderValidation(view, data) {
  view.appendChild(el("h2", "Validation"));
  view.appendChild(el("p", "Every scored criterion across models. Values are copied from the scorer files at the commits shown; the page computes no verdict.")).className = "small";
  var label = el("label", "Show ");
  var select = el("select");
  ["all", "pass", "fail", "pending"].forEach(function (v) {
    var o = el("option", v);
    o.value = v;
    select.appendChild(o);
  });
  label.appendChild(select);
  view.appendChild(label);
  var table = el("table");
  table.className = "criteria";
  var head = el("thead");
  var hr = el("tr");
  var labels = ["Model", "Class", "ID", "Claim", "Paper", "Ours", "Verdict"];
  labels.forEach(function (h) { hr.appendChild(el("th", h)); });
  head.appendChild(hr);
  table.appendChild(head);
  var body = el("tbody");
  table.appendChild(body);
  view.appendChild(table);
  function fill() {
    clear(body);
    (data.papers || []).forEach(function (p) {
      (p.criteria || []).forEach(function (c) {
        if (select.value !== "all" && c.verdict !== select.value) return;
        var row = el("tr");
        var m = el("td");
        m.appendChild(link("#model/" + p.id, p.title));
        row.appendChild(m);
        row.appendChild(el("td", classInfo(p["class"]).label));
        row.appendChild(el("td", c.id + (c.group ? " (" + c.group + ")" : "")));
        row.appendChild(el("td", c.claim || ""));
        row.appendChild(el("td", c.paper_value || "—"));
        row.appendChild(valueCell(c));
        var vc = el("td");
        vc.appendChild(verdictChip(c.verdict));
        row.appendChild(vc);
        Array.prototype.forEach.call(row.children, function (td, i) {
          var lab = el("span", labels[i] + ": ");
          lab.className = "cell-label";
          td.insertBefore(lab, td.firstChild);
        });
        body.appendChild(row);
      });
    });
  }
  select.addEventListener("change", fill);
  fill();
}

function renderRoadmap(view, data) {
  view.appendChild(el("h2", "Roadmap"));
  view.appendChild(el("p", "Planned models by release. None has a scored run yet.")).className = "small";
  var table = el("table");
  table.className = "criteria";
  var hr = el("tr");
  ["Release", "Model", "Needs"].forEach(function (h) { hr.appendChild(el("th", h)); });
  var head = el("thead");
  head.appendChild(hr);
  table.appendChild(head);
  var body = el("tbody");
  (data.roadmap || []).forEach(function (r) {
    var row = el("tr");
    row.appendChild(el("td", r.release || ""));
    row.appendChild(el("td", r.model || ""));
    row.appendChild(el("td", r.needs || ""));
    body.appendChild(row);
  });
  table.appendChild(body);
  view.appendChild(table);
}

function renderMethod(view) {
  view.appendChild(el("h2", "Method"));
  var dl = el("dl");
  dl.className = "classes";
  CLASSES.forEach(function (c) {
    var dt = el("dt");
    dt.appendChild(classChip(c.id));
    dl.appendChild(dt);
    dl.appendChild(el("dd", c.def));
  });
  view.appendChild(dl);
  var rules = el("ul");
  [
    "The class sits beside the preregistered verdict, which always stays visible.",
    "The page computes no verdict: values and verdicts are copied by scripts/build_dashboard_data.py from scorer files at pinned commits, and its --check runs in CI.",
    "Failures carry the same prominence as passes (B2).",
    "Agreement with a paper is claimed only where a paper value sits beside ours in the table.",
    "A replication counts as done with jaxfne only when it runs through the library (B1); a standalone port is labelled as one."
  ].forEach(function (t) { rules.appendChild(el("li", t)); });
  view.appendChild(rules);
}

function route() {
  var h = location.hash ? decodeURIComponent(location.hash.slice(1)) : "overview";
  if (h.indexOf("model/") === 0) return { tab: "models", id: h.slice(6) };
  if (TABS.indexOf(h) >= 0) return { tab: h, id: null };
  return { tab: "models", id: h }; // v1 links: #<paper id>
}

function render(data) {
  var view = document.getElementById("view");
  purgePlots(view);
  clear(view);
  var r = route();
  Array.prototype.forEach.call(document.querySelectorAll("#tabs a"), function (a) {
    if (a.getAttribute("data-tab") === r.tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  var papers = data.papers || [];
  if (r.tab === "models" && r.id) {
    var paper = null;
    papers.forEach(function (p) { if (p.id === r.id) paper = p; });
    if (paper) renderModel(view, data, paper);
    else view.appendChild(el("p", "No model with id " + r.id + "."));
  } else if (r.tab === "models") renderModels(view, data);
  else if (r.tab === "validation") renderValidation(view, data);
  else if (r.tab === "roadmap") renderRoadmap(view, data);
  else if (r.tab === "method") renderMethod(view);
  else renderOverview(view, data);
  view.focus();
}

function purgePlots(root) {
  if (!window.Plotly) return;
  Array.prototype.forEach.call(root.querySelectorAll(".chart-div"), function (d) {
    try { window.Plotly.purge(d); } catch (e) { /* already empty */ }
  });
}

function renderHeader(data) {
  var generated = document.getElementById("generated");
  clear(generated);
  generated.appendChild(el("span", "Generated " + ((data.generated && data.generated.utc) || "unknown")));
  var sha = (data.generated && data.generated.dev_sha) || "";
  var devLink = document.getElementById("devsha");
  clear(devLink);
  devLink.appendChild(el("span", "build dev@" + shortSha(sha)));
  devLink.href = "https://github.com/hnxj/jaxfne/commit/" + sha;
  var doc = document.getElementById("replications-doc");
  doc.href = "https://github.com/hnxj/jaxfne/blob/" + sha + "/artifacts/etudes/REPLICATIONS.md";
  document.getElementById("banner").hidden = !data.sample;
}

function loadData() {
  return fetch("data/replications.json").then(function (r) {
    if (!r.ok) throw new Error("primary missing");
    return r.json();
  }).catch(function () {
    return fetch("data/replications.sample.json").then(function (r) {
      if (!r.ok) throw new Error("no data file");
      return r.json();
    });
  });
}

window.addEventListener("hashchange", function () {
  if (window.__data) render(window.__data);
});

loadData().then(function (data) {
  window.__data = data;
  renderHeader(data);
  render(data);
}).catch(function () {
  document.getElementById("view").appendChild(el("p", "Could not load replication data."));
});
