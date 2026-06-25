// GPT model widget — the full decoder-only stack as a clickable map, with a shape tracker.
// See docs/11_gpt_model.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;

  const STACK = [
    { label: "Token ids", shape: "(B, T)", desc: "Input characters mapped to integer ids.", link: "03_tokenization" },
    { label: "Token Embedding", shape: "(B, T, C)", desc: "Look up a learnable vector per token id.", link: "04_embeddings" },
    { label: "+ Positional Encoding", shape: "(B, T, C)", desc: "Add order information so attention knows positions.", link: "05_positional_encoding" },
    { label: "Dropout", shape: "(B, T, C)", desc: "Regularization during training.", link: null },
    { label: "Transformer Block × n_layer", shape: "(B, T, C)", desc: "The stacked core: attention + feed-forward with residuals.", link: "10_transformer_block" },
    { label: "Final LayerNorm", shape: "(B, T, C)", desc: "Normalize before projecting to vocabulary.", link: "09_layernorm_residual" },
    { label: "LM Head (Linear)", shape: "(B, T, V)", desc: "Project each token to a score per vocabulary entry. Weight-tied with the embedding.", link: null },
    { label: "Softmax", shape: "(B, T, V)", desc: "Turn scores into next-token probabilities.", link: null },
  ];

  function render(root) {
    root.innerHTML =
      U.header("11_gpt_model", "The GPT Model",
        "Everything assembled into a decoder-only language model. Data flows bottom&rarr;top; " +
        "the LM head produces a probability distribution over the next token at every position. " +
        "Click any stage to open its widget. Loss at init &asymp; ln(V) &asymp; 4.17 for V=65.") +
      `<div style="display:flex;gap:24px;flex-wrap:wrap">
        <div class="panel" style="flex:1;min-width:300px"><h3>Forward pass (bottom &rarr; top)</h3>
          <div id="gm-stack" style="display:flex;flex-direction:column-reverse;gap:0"></div></div>
        <div class="panel" style="flex:1;min-width:260px"><h3>Details</h3>
          <div id="gm-info"><p class="blurb">Hover or click a stage.</p>
          <div class="formula" style="margin-top:12px">logits = LayerNorm(blocks(emb + pos)) &times; W_head<br>loss = cross_entropy(logits, targets)</div></div></div>
      </div>`;

    const stack = root.querySelector("#gm-stack");
    const info = root.querySelector("#gm-info");

    STACK.forEach((s, i) => {
      if (i > 0) {
        const arrow = document.createElement("div");
        arrow.style.cssText = "text-align:center;color:var(--muted);font-size:1.1rem;line-height:1";
        arrow.textContent = "↑";
        stack.appendChild(arrow);
      }
      const box = document.createElement("div");
      const isCore = s.label.startsWith("Transformer Block");
      box.style.cssText =
        "padding:10px 12px;border-radius:8px;border:1px solid var(--border);cursor:pointer;text-align:center;" +
        (isCore ? "background:#e0e7ff;font-weight:600;" : "background:var(--panel);");
      box.innerHTML = `${s.label} <span class="badge" style="margin-left:6px">${s.shape}</span>`;
      const show = () => {
        info.innerHTML =
          `<h3 style="margin-top:0">${s.label}</h3><p class="blurb">${s.desc}</p>` +
          `<p class="links">shape ${s.shape}` +
          (s.link ? ` &middot; <a href="#${s.link}">open ${s.link} widget →</a>` : "") + `</p>`;
      };
      box.addEventListener("mouseenter", show);
      box.addEventListener("click", () => { if (s.link) location.hash = s.link; else show(); });
      stack.appendChild(box);
    });
  }

  window.NanoGPTWidgets["11_gpt_model"] = { title: "GPT Model", render };
})();
