// Transformer block widget — interactive pre-LN schematic with residual paths.
// See docs/10_transformer_block.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;

  const STEPS = [
    { id: "in", label: "input x", shape: "(B, T, C)", desc: "A token vector per position, carrying info gathered so far.", link: null },
    { id: "ln1", label: "LayerNorm", shape: "(B, T, C)", desc: "Pre-LN: normalize the input to the attention sublayer.", link: "09_layernorm_residual" },
    { id: "mha", label: "Multi-Head Attention", shape: "(B, T, C)", desc: "Tokens communicate: gather context from earlier positions.", link: "07_multi_head_attention" },
    { id: "add1", label: "+ residual", shape: "(B, T, C)", desc: "Add the attention output back to x (gradient highway).", link: null },
    { id: "ln2", label: "LayerNorm", shape: "(B, T, C)", desc: "Pre-LN: normalize the input to the feed-forward sublayer.", link: "09_layernorm_residual" },
    { id: "ffn", label: "Feed-Forward", shape: "(B, T, C)", desc: "Each token thinks: position-wise non-linear MLP.", link: "08_feed_forward" },
    { id: "add2", label: "+ residual", shape: "(B, T, C)", desc: "Add the FFN output back (residual). Shape preserved end-to-end.", link: null },
    { id: "out", label: "output", shape: "(B, T, C)", desc: "Ready to feed into the next block. Stack n_layer of these.", link: null },
  ];

  function render(root) {
    root.innerHTML =
      U.header("10_transformer_block", "Transformer Block",
        "The repeated unit (the N&times; box). Pre-LN + residual around <b>attention</b> " +
        "(communicate) and <b>feed-forward</b> (compute). Click a stage to learn more or jump " +
        "to its widget. Notice the shape stays (B, T, C) throughout &mdash; that's why blocks stack.") +
      `<div style="display:flex;gap:24px;flex-wrap:wrap">
        <div class="panel" style="flex:1;min-width:280px"><h3>x = x + Sublayer(LayerNorm(x))</h3>
          <div id="tb-flow" style="display:flex;flex-direction:column;gap:0;align-items:stretch"></div></div>
        <div class="panel" style="flex:1;min-width:260px"><h3>Details</h3>
          <div id="tb-info"><p class="blurb">Hover or click a stage on the left.</p></div></div>
      </div>`;

    const flow = root.querySelector("#tb-flow");
    const info = root.querySelector("#tb-info");

    STEPS.forEach((s, i) => {
      if (i > 0) {
        const arrow = document.createElement("div");
        arrow.style.cssText = "text-align:center;color:var(--muted);font-size:1.1rem;line-height:1";
        arrow.textContent = "↓";
        flow.appendChild(arrow);
      }
      const box = document.createElement("div");
      const isResidual = s.id.startsWith("add");
      const isSub = s.id === "mha" || s.id === "ffn";
      box.style.cssText =
        "padding:10px 12px;border-radius:8px;border:1px solid var(--border);cursor:pointer;" +
        "text-align:center;transition:background .12s;" +
        (isSub ? "background:#fef3c7;font-weight:600;" : isResidual ? "background:#dcfce7;" : "background:var(--panel);");
      box.innerHTML = `${s.label} <span class="badge" style="margin-left:6px">${s.shape}</span>`;
      const show = () => {
        info.innerHTML =
          `<h3 style="margin-top:0">${s.label}</h3><p class="blurb">${s.desc}</p>` +
          `<p class="links">shape ${s.shape}` +
          (s.link ? ` &middot; <a href="#${s.link}">open ${s.link} widget →</a>` : "") + `</p>`;
      };
      box.addEventListener("mouseenter", show);
      box.addEventListener("click", () => { if (s.link) location.hash = s.link; else show(); });
      flow.appendChild(box);
    });
  }

  window.NanoGPTWidgets["10_transformer_block"] = { title: "Transformer Block", render };
})();
