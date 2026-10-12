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
  ["ID", "Claim", "Criterion", "Value", "Verdict", "Reason"].forEach(function (h) {
    headRow.appendChild(el("th", h));
  });
  head.appendChild(headRow);
  table.appendChild(head);
  var body = document.createElement("tbody");
  var lastGroup = null;
  var labels = ["ID", "Claim", "Criterion", "Value", "Verdict", "Reason"];
  criteria.forEach(function (c) {
    var group = c.group === undefined || c.group === null ? "" : String(c.group);
    if (group && group !== lastGroup) {
      var grow = document.createElement("tr");
      grow.className = "group-row";
      var gth = el("th", group);
      gth.colSpan = 6;
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

function renderCards(data, selectedId) {
  var cards = document.getElementById("cards");
  while (cards.firstChild) cards.removeChild(cards.firstChild);
  (data.papers || []).forEach(function (paper) {
    var card = el("button");
    card.type = "button";
    card.className = "card";
    if (paper.id === selectedId) card.setAttribute("aria-current", "true");
    card.setAttribute("aria-label", paper.title + ", " + paper.verdict + ", " + passCount(paper));
    card.appendChild(el("h2", paper.title));
    card.appendChild(el("p", "Release " + (paper.release || "") + " · " + (paper.tier || "") + " · " + (paper.citation || "")));
    card.appendChild(stageTrack(paper.stage));
    var verdictRow = el("p");
    verdictRow.appendChild(verdictChip(paper.verdict));
    card.appendChild(verdictRow);
    var count = el("p", passCount(paper));
    count.className = "pass-count";
    card.appendChild(count);
    card.addEventListener("click", function () {
      if (location.hash !== "#" + paper.id) location.hash = "#" + paper.id;
      else renderDetail(window.__data);
    });
    card.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        location.hash = "#" + paper.id;
      }
    });
    cards.appendChild(card);
  });
}

function purgePlots(root) {
  if (!window.Plotly) return;
  var plots = root.querySelectorAll(".chart-div");
  Array.prototype.forEach.call(plots, function (d) {
    try { window.Plotly.purge(d); } catch (e) { /* already empty */ }
  });
}

function renderDetail(data) {
  var detail = document.getElementById("detail");
  purgePlots(detail);
  while (detail.firstChild) detail.removeChild(detail.firstChild);
  var papers = data.papers || [];
  if (!papers.length) {
    detail.appendChild(el("p", "No papers in this data file."));
    return;
  }
  var hash = location.hash ? location.hash.slice(1) : "";
  var paper = null;
  papers.forEach(function (p) { if (p.id === hash) paper = p; });
  if (!paper) paper = papers[0];
  renderCards(data, paper.id);

  var card = el("div");
  card.className = "detail-card";
  card.appendChild(el("h2", paper.title));
  card.appendChild(el("p", paper.summary || ""));
  var sub = el("p", (paper.citation || "") + " · Release " + (paper.release || "") +
    " · " + (paper.tier || "") + " · Stage " + (paper.stage || ""));
  sub.className = "sub";
  card.appendChild(sub);
  var lane = paper.lane || {};
  var laneLine = el("p");
  laneLine.className = "small";
  laneLine.appendChild(el("span", "Lane " + (lane.branch || "") + " @ "));
  if (lane.sha) laneLine.appendChild(link("https://github.com/hnxj/jaxfne/commit/" + lane.sha, shortSha(lane.sha)));
  card.appendChild(laneLine);
  var verdictRow = el("p");
  verdictRow.appendChild(verdictChip(paper.verdict));
  verdictRow.appendChild(el("span", " " + passCount(paper)));
  card.appendChild(verdictRow);

  card.appendChild(el("h3", "Criteria"));
  card.appendChild(criteriaTable(paper));
  card.appendChild(el("h3", "Charts"));
  card.appendChild(chartsSection(paper));
  card.appendChild(el("h3", "Sensitivity"));
  card.appendChild(sensitivitySection(paper));
  card.appendChild(el("h3", "Preregistrations"));
  card.appendChild(preregsSection(paper));
  card.appendChild(el("h3", "Open items"));
  card.appendChild(openSection(paper));
  detail.appendChild(card);
  (paper.charts || []).forEach(function (chart) {
    var div = document.getElementById("chart-" + paper.id + "-" + chart.id);
    if (!div) return;
    if (!window.Plotly) {
      div.appendChild(el("p", "Chart library did not load."));
      return;
    }
    try {
      if (chart.kind === "bars") renderBarsChart(div, chart);
      else renderLinesChart(div, chart);
    } catch (e) {
      div.appendChild(el("p", "Chart failed to render."));
    }
  });
}

function renderHeader(data) {
  var generated = document.getElementById("generated");
  while (generated.firstChild) generated.removeChild(generated.firstChild);
  generated.appendChild(el("span", "Generated " + ((data.generated && data.generated.utc) || "unknown")));
  var sha = (data.generated && data.generated.dev_sha) || "";
  var devLink = document.getElementById("devsha");
  while (devLink.firstChild) devLink.removeChild(devLink.firstChild);
  devLink.appendChild(el("span", "dev " + shortSha(sha)));
  devLink.href = "https://github.com/hnxj/jaxfne/commit/" + sha;
  var doc = document.getElementById("replications-doc");
  doc.href = "https://github.com/hnxj/jaxfne/blob/" + sha + "/artifacts/etudes/REPLICATIONS.md";
  var banner = document.getElementById("banner");
  banner.hidden = !data.sample;
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
  if (window.__data) renderDetail(window.__data);
});

loadData().then(function (data) {
  window.__data = data;
  renderHeader(data);
  renderDetail(data);
}).catch(function () {
  var detail = document.getElementById("detail");
  detail.appendChild(el("p", "Could not load replication data."));
});
