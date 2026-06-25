// Self-attention widget — scaled dot-product with a causal-mask toggle. Illustrative Q/K/V.
// See docs/06_self_attention.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;

  const TOKENS = ["The", "cat", "sat", "on", "the", "mat"];

  function render(root) {
    const state = { causal: true, d: 8, seed: 7, query: 5 };

    root.innerHTML =
      U.header("06_self_attention", "Self-Attention",
        "Each token's <b>query</b> is compared with every token's <b>key</b> to get scores " +
        "QK&#7488;/&radic;d. A causal mask hides the future, then softmax turns scores into " +
        "weights that sum to 1. The output is the weighted average of <b>values</b>.") +
      `<div class="panel"><h3>Controls</h3><div class="controls">
        <div class="control"><label>Causal mask</label>
          <button id="sa-mask" class="nav-item" style="justify-content:center"></button></div>
        <div class="control"><label>Head size d = <span class="val" id="sa-d"></span></label>
          <input id="sa-d-r" type="range" min="2" max="32" step="1"></div>
        <div class="control"><label>Inspect query = <span class="val" id="sa-q"></span></label>
          <input id="sa-q-r" type="range" min="0" max="5" step="1"></div>
        <div class="control"><label>&nbsp;</label>
          <button id="sa-reseed" class="nav-item" style="justify-content:center">Reseed Q/K</button></div>
      </div></div>
      <div class="panel"><h3>1 &middot; Scores = QK&#7488; / &radic;d &nbsp;<span class="blurb">(rows = queries, cols = keys)</span></h3>
        <p class="blurb" id="sa-mask-note"></p><div id="sa-scores"></div></div>
      <div class="panel"><h3>2 &middot; Attention weights = softmax(scores)</h3>
        <p class="blurb">Each row sums to 1. Greyed cells are masked (zero weight).</p><div id="sa-weights"></div></div>
      <div class="panel"><h3>3 &middot; Where query "<span id="sa-qtok"></span>" looks</h3>
        <p class="blurb">Attention distribution of the selected query over all keys.</p><div id="sa-dist"></div></div>`;

    function compute() {
      const T = TOKENS.length, d = state.d;
      const rng = U.mulberry32(state.seed);
      const Q = U.randnMatrix(rng, T, d), K = U.randnMatrix(rng, T, d);
      const scale = 1 / Math.sqrt(d);
      const raw = U.matmul(Q, U.transpose(K)).map((row) => row.map((v) => v * scale));
      const masked = raw.map((row, i) =>
        row.map((v, j) => (state.causal && j > i ? -Infinity : v)));
      const weights = masked.map((row) => U.softmaxRow(row));
      return { T, raw, weights };
    }

    function draw() {
      const { T, raw, weights } = compute();
      const maskFn = (i, j) => state.causal && j > i;

      root.querySelector("#sa-d").textContent = state.d;
      root.querySelector("#sa-q").textContent = state.query;
      root.querySelector("#sa-qtok").textContent = TOKENS[state.query];
      root.querySelector("#sa-mask").textContent = state.causal ? "ON (decoder)" : "OFF (encoder)";
      root.querySelector("#sa-mask").classList.toggle("active", state.causal);
      root.querySelector("#sa-mask-note").textContent = state.causal
        ? "Masked: token t attends only to positions ≤ t (no peeking ahead)."
        : "Unmasked: every token attends to every token (bidirectional).";

      U.heatmap(root.querySelector("#sa-scores"), raw.map((r) => r.map((v) => (isFinite(v) ? v : 0))), {
        rowLabels: TOKENS, colLabels: TOKENS, mask: maskFn, max: 2.5, cell: 46,
      });
      U.heatmap(root.querySelector("#sa-weights"), weights, {
        rowLabels: TOKENS, colLabels: TOKENS, mask: maskFn, cell: 46,
        color: d3.scaleSequential(d3.interpolateBlues).domain([0, 1]), fmt: (v) => v.toFixed(2),
      });
      U.barChart(root.querySelector("#sa-dist"), weights[state.query], {
        color: d3.scaleSequential(d3.interpolateBlues).domain([0, 1]),
      });
    }

    root.querySelector("#sa-d-r").value = state.d;
    root.querySelector("#sa-q-r").value = state.query;
    root.querySelector("#sa-mask").addEventListener("click", () => { state.causal = !state.causal; draw(); });
    root.querySelector("#sa-d-r").addEventListener("input", (e) => { state.d = +e.target.value; draw(); });
    root.querySelector("#sa-q-r").addEventListener("input", (e) => { state.query = +e.target.value; draw(); });
    root.querySelector("#sa-reseed").addEventListener("click", () => { state.seed = (state.seed * 7 + 1) % 9973; draw(); });
    draw();
  }

  window.NanoGPTWidgets["06_self_attention"] = { title: "Self-Attention", render };
})();
