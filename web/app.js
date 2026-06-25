// Tiny hash-router that builds the component nav and renders the selected widget.
// Widgets self-register into window.NanoGPTWidgets (see widgets/*.js).

(function () {
  "use strict";

  // Full component catalog (matches docs/ build order). `built: true` means a widget
  // is registered in window.NanoGPTWidgets under this id; others show "coming soon".
  const CATALOG = [
    { id: "03_tokenization",        num: "03", title: "Tokenization",        built: true },
    { id: "04_embeddings",          num: "04", title: "Embeddings",          built: true },
    { id: "05_positional_encoding", num: "05", title: "Positional Encoding", built: true },
    { id: "06_self_attention",      num: "06", title: "Self-Attention",      built: true },
    { id: "07_multi_head_attention",num: "07", title: "Multi-Head Attention",built: true },
    { id: "08_feed_forward",        num: "08", title: "Feed-Forward",        built: true },
    { id: "09_layernorm_residual",  num: "09", title: "LayerNorm + Residual",built: true },
    { id: "10_transformer_block",   num: "10", title: "Transformer Block",   built: true },
    { id: "11_gpt_model",           num: "11", title: "GPT Model",           built: true },
  ];

  const registry = window.NanoGPTWidgets || {};
  const navEl = document.getElementById("nav");
  const rootEl = document.getElementById("widget-root");

  function buildNav(activeId) {
    navEl.innerHTML = "";
    CATALOG.forEach((c) => {
      const a = document.createElement("a");
      a.className = "nav-item" + (c.id === activeId ? " active" : "");
      a.href = "#" + c.id;
      a.innerHTML =
        `<span><span class="nav-num">${c.num}</span>${c.title}</span>` +
        `<span class="badge ${c.built ? "done" : "todo"}">${c.built ? "live" : "soon"}</span>`;
      navEl.appendChild(a);
    });
  }

  function renderComingSoon(c) {
    rootEl.innerHTML =
      `<div class="widget-head"><h2>${c.title}</h2>` +
      `<p class="links"><a href="../docs/${c.id}.md">math doc</a> &middot; ` +
      `<a href="../docs/assets/${c.id}.png">drawing</a></p></div>` +
      `<div class="panel"><p class="placeholder">Widget coming soon. ` +
      `Read the math in <a href="../docs/${c.id}.md">docs/${c.id}.md</a> for now.</p></div>`;
  }

  function render() {
    const id = (location.hash || "#05_positional_encoding").slice(1);
    const c = CATALOG.find((x) => x.id === id) || CATALOG[2];
    buildNav(c.id);
    rootEl.innerHTML = "";
    const widget = registry[c.id];
    if (widget && typeof widget.render === "function") {
      widget.render(rootEl);
    } else {
      renderComingSoon(c);
    }
  }

  window.addEventListener("hashchange", render);
  render();
})();
