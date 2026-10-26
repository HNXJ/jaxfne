// Render <div class="mermaid"> flowcharts in the site palette of the active color scheme, and
// re-render them when the reader toggles between the light and dark schemes.
(function () {
  var PALETTES = {
    slate: {
      background: "#0d1117",
      primaryColor: "#1f2a37",
      primaryBorderColor: "#c3aa5f",
      primaryTextColor: "#e6edf3",
      lineColor: "#9aa4b2",
      edgeLabelBackground: "#0d1117",
    },
    default: {
      background: "#ffffff",
      primaryColor: "#f3f1ea",
      primaryBorderColor: "#c3aa5f",
      primaryTextColor: "#1b1b1b",
      lineColor: "#5f5f5f",
      edgeLabelBackground: "#ffffff",
    },
  };
  var sources = [];

  function scheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "slate" : "default";
  }

  function render() {
    var s = scheme();
    var vars = Object.assign({ fontFamily: "Roboto, Helvetica, Arial, sans-serif" }, PALETTES[s]);
    mermaid.initialize({
      startOnLoad: false,
      theme: s === "slate" ? "dark" : "base",
      themeVariables: vars,
      // SVG labels: HTML labels are measured before the page font loads and clip.
      flowchart: { curve: "basis", htmlLabels: false, padding: 12 },
    });
    var nodes = document.querySelectorAll("div.mermaid");
    nodes.forEach(function (node, i) {
      // innerHTML keeps <br> line breaks inside node labels.
      if (sources[i] === undefined) sources[i] = node.innerHTML;
      node.removeAttribute("data-processed");
      node.innerHTML = sources[i];
    });
    mermaid.run({ nodes: nodes });
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (typeof mermaid === "undefined") return;
    render();
    new MutationObserver(render).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  });
})();
