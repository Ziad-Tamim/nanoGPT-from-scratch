// Multi-head attention widget — several heads attend differently, in parallel.
// See docs/07_multi_head_attention.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;
  const TOKENS = ["The", "cat", "sat", "on", "the", "mat"];

  function render(root) {
    const state = { nHead: 4, C: 32 };

    root.innerHTML =
      U.header("07_multi_head_attention", "Multi-Head Attention",
        "One head learns one relationship. Multi-head attention splits the width C into " +
        "<b>n_head</b> subspaces of size C / n_head, runs an independent causal head in each " +
        "(below), concatenates the results, and projects back to C. Each head attends differently.") +
      `<div class="panel"><h3>Controls</h3><div class="controls">
        <div class="control"><label>Model width C = <span class="val" id="mh-C"></span></label>
          <input id="mh-C-r" type="range" min="8" max="64" step="8"></div>
        <div class="control"><label>Number of heads = <span class="val" id="mh-h"></span></label>
          <input id="mh-h-r" type="range" min="1" max="8" step="1"></div>
        <div class="control"><label>Per-head size = <span class="val" id="mh-d" style="color:var(--accent)"></span></label>
          <span class="blurb">C / n_head</span></div>
      </div></div>
      <div class="panel"><h3>Per-head attention weights (each causal, but different)</h3>
        <div id="mh-grid" style="display:flex;flex-wrap:wrap;gap:18px"></div></div>
      <div class="panel"><h3>Then: concat &rarr; project</h3>
        <div class="formula">concat(head&#8321; … head&#8341;)  &rarr;  &times; W&#8338;  &rarr;  output (B, T, C)</div>
        <p class="blurb" id="mh-note" style="margin-top:10px"></p></div>`;

    function headWeights(seed) {
      const T = TOKENS.length, d = Math.max(2, state.C / state.nHead);
      const rng = U.mulberry32(seed);
      const Q = U.randnMatrix(rng, T, d), K = U.randnMatrix(rng, T, d);
      const scale = 1 / Math.sqrt(d);
      const raw = U.matmul(Q, U.transpose(K)).map((r, i) =>
        r.map((v, j) => (j > i ? -Infinity : v * scale)));
      return raw.map((r) => U.softmaxRow(r));
    }

    function draw() {
      const d = Math.max(2, Math.round(state.C / state.nHead));
      root.querySelector("#mh-C").textContent = state.C;
      root.querySelector("#mh-h").textContent = state.nHead;
      root.querySelector("#mh-d").textContent = d;
      root.querySelector("#mh-note").textContent =
        `${state.nHead} heads × ${d} dims = ${state.nHead * d} ≈ C (${state.C}). ` +
        "Splitting costs no extra parameters — total projection size stays ~3C².";

      const grid = root.querySelector("#mh-grid");
      grid.innerHTML = "";
      for (let h = 0; h < state.nHead; h++) {
        const cell = document.createElement("div");
        cell.style.cssText = "text-align:center";
        const title = document.createElement("div");
        title.className = "blurb";
        title.style.marginBottom = "4px";
        title.textContent = `head ${h + 1}`;
        const plot = document.createElement("div");
        plot.style.width = "190px";
        cell.appendChild(title); cell.appendChild(plot);
        grid.appendChild(cell);
        U.heatmap(plot, headWeights(100 + h * 13), {
          rowLabels: TOKENS, colLabels: TOKENS, mask: (i, j) => j > i, cell: 26,
          color: d3.scaleSequential(d3.interpolateBlues).domain([0, 1]),
        });
      }
    }

    root.querySelector("#mh-C-r").value = state.C;
    root.querySelector("#mh-h-r").value = state.nHead;
    root.querySelector("#mh-C-r").addEventListener("input", (e) => { state.C = +e.target.value; draw(); });
    root.querySelector("#mh-h-r").addEventListener("input", (e) => { state.nHead = +e.target.value; draw(); });
    draw();
  }

  window.NanoGPTWidgets["07_multi_head_attention"] = { title: "Multi-Head Attention", render };
})();
