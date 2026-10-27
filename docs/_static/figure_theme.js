// Posts the page's palette to each figure-site iframe. The figure's HTML switches its palette on the
// message and follows the OS scheme until the first one arrives. The message names the figure site's
// origin as its target, so a page from any other origin never receives it.
(function () {
  var FIGURES = "https://hnxj.github.io/jaxfne/figures/";
  var ORIGIN = "https://hnxj.github.io";

  function scheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
  }

  function post() {
    var frames = document.querySelectorAll('iframe[src^="' + FIGURES + '"]');
    frames.forEach(function (frame) {
      if (frame.contentWindow) {
        frame.contentWindow.postMessage({ type: "jaxfne-scheme", scheme: scheme() }, ORIGIN);
      }
    });
  }

  document.addEventListener("DOMContentLoaded", post);
  // A figure loads after the page; the palette toggle changes the body's attribute.
  document.addEventListener("load", post, true);
  new MutationObserver(post).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"]
  });
  if (window.document$) {
    window.document$.subscribe(post);
  }
})();
