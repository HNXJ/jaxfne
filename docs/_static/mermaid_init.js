// Render <div class="mermaid"> flowcharts in the site's dark palette.
document.addEventListener("DOMContentLoaded", function () {
  if (typeof mermaid === "undefined") return;
  mermaid.initialize({
    startOnLoad: true,
    theme: "dark",
    themeVariables: {
      background: "#0d1117",
      primaryColor: "#1f2a37",
      primaryBorderColor: "#c3aa5f",
      primaryTextColor: "#e6edf3",
      lineColor: "#9aa4b2",
      edgeLabelBackground: "#0d1117",
      fontFamily: "Roboto, Helvetica, Arial, sans-serif",
    },
    // SVG labels: HTML labels are measured before the page font loads and clip.
    flowchart: { curve: "basis", htmlLabels: false, padding: 12 },
  });
});
