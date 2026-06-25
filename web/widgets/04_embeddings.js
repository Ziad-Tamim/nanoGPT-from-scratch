// Embeddings widget — token id indexes a learnable table to get a dense vector.
// See docs/04_embeddings.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;

  function render(root) {
    const state = { V: 8, C: 12, sel: 3 };
    const tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"];

    root.innerHTML =
      U.header("04_embeddings", "Token Embeddings",
        "Each token id selects one <b>row</b> of a learnable table E (V&times;C). That row is " +
        "the token's vector. Lookup is identical to multiplying a one-hot vector by E. Pick a token:") +
      `<div class="panel"><h3>Controls</h3><div class="controls">
        <div class="control"><label>Vocabulary size V = <span class="val" id="em-V"></span></label>
          <input id="em-V-r" type="range" min="4" max="12" step="1"></div>
        <div class="control"><label>Embedding dim C = <span class="val" id="em-C"></span></label>
          <input id="em-C-r" type="range" min="4" max="24" step="1"></div>
        <div class="control"><label>Selected token id = <span class="val" id="em-sel"></span></label>
          <input id="em-sel-r" type="range" min="0" max="7" step="1"></div>
      </div></div>
      <div class="panel"><h3>Embedding table E (rows = tokens, cols = dimensions)</h3>
        <p class="blurb">The highlighted row is the selected token's embedding vector.</p>
        <div id="em-table"></div></div>
      <div class="panel"><h3>Embedding vector for token "<span id="em-tok"></span>" (id <span id="em-sel2"></span>)</h3>
        <div id="em-vec"></div></div>`;

    let E = null;
    function rebuildTable() {
      const rng = U.mulberry32(42);
      E = U.randnMatrix(rng, state.V, state.C, 0.6); // seeded illustrative weights
    }

    function draw() {
      root.querySelector("#em-V").textContent = state.V;
      root.querySelector("#em-C").textContent = state.C;
      root.querySelector("#em-sel").textContent = state.sel;
      root.querySelector("#em-sel2").textContent = state.sel;
      root.querySelector("#em-tok").textContent = tokens[state.sel];

      U.heatmap(root.querySelector("#em-table"), E, {
        max: 1.5,
        rowLabels: tokens.slice(0, state.V),
        highlight: { row: state.sel },
      });
      U.barChart(root.querySelector("#em-vec"), E[state.sel], { tickEvery: 2 });
    }

    function refresh() {
      const selR = root.querySelector("#em-sel-r");
      state.sel = Math.min(state.sel, state.V - 1);
      selR.max = state.V - 1; selR.value = state.sel;
      rebuildTable();
      draw();
    }

    root.querySelector("#em-V-r").value = state.V;
    root.querySelector("#em-C-r").value = state.C;
    root.querySelector("#em-sel-r").value = state.sel;
    root.querySelector("#em-V-r").addEventListener("input", (e) => { state.V = +e.target.value; refresh(); });
    root.querySelector("#em-C-r").addEventListener("input", (e) => { state.C = +e.target.value; refresh(); });
    root.querySelector("#em-sel-r").addEventListener("input", (e) => { state.sel = +e.target.value; draw(); });
    refresh();
  }

  window.NanoGPTWidgets["04_embeddings"] = { title: "Embeddings", render };
})();
