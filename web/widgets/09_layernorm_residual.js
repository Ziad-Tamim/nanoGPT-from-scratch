// LayerNorm + residual widget — normalize a token vector, then rescale; explain residuals.
// See docs/09_layernorm_residual.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;
  const C = 12;

  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };

  function render(root) {
    const state = { shift: 3, scale: 2, gamma: 1, beta: 0, seed: 5 };

    root.innerHTML =
      U.header("09_layernorm_residual", "LayerNorm & Residuals",
        "LayerNorm rescales each token's vector to zero mean / unit variance (over C), then " +
        "applies learnable &gamma; (scale) and &beta; (shift). Residual connections add the input " +
        "back so gradients flow freely. Drag the sliders:") +
      `<div class="panel"><h3>Controls</h3><div class="controls">
        <div class="control"><label>Input mean shift = <span class="val" id="ln-shift"></span></label>
          <input id="ln-shift-r" type="range" min="-5" max="5" step="0.5"></div>
        <div class="control"><label>Input scale = <span class="val" id="ln-scale"></span></label>
          <input id="ln-scale-r" type="range" min="0.2" max="5" step="0.2"></div>
        <div class="control"><label>&gamma; (learnable scale) = <span class="val" id="ln-g"></span></label>
          <input id="ln-g-r" type="range" min="0" max="3" step="0.1"></div>
        <div class="control"><label>&beta; (learnable shift) = <span class="val" id="ln-b"></span></label>
          <input id="ln-b-r" type="range" min="-2" max="2" step="0.1"></div>
      </div></div>
      <div class="panel"><h3>1 &middot; Raw input x</h3><p class="blurb" id="ln-s1"></p><div id="ln-raw"></div></div>
      <div class="panel"><h3>2 &middot; Normalized x&#770; = (x &minus; &mu;) / &radic;(&sigma;&sup2;+&epsilon;)</h3>
        <p class="blurb" id="ln-s2"></p><div id="ln-norm"></div></div>
      <div class="panel"><h3>3 &middot; Output = &gamma; &middot; x&#770; + &beta;</h3><p class="blurb" id="ln-s3"></p><div id="ln-out"></div></div>
      <div class="panel"><h3>Residual connection</h3>
        <div class="formula">x &larr; x + Sublayer(LayerNorm(x))</div>
        <p class="blurb" style="margin-top:10px">The <code>+ x</code> creates a gradient highway: &part;(x+f(x))/&part;x = I + &part;f/&part;x, so the gradient never vanishes through depth.</p></div>`;

    function draw() {
      root.querySelector("#ln-shift").textContent = state.shift;
      root.querySelector("#ln-scale").textContent = state.scale;
      root.querySelector("#ln-g").textContent = state.gamma.toFixed(1);
      root.querySelector("#ln-b").textContent = state.beta.toFixed(1);

      const rng = U.mulberry32(state.seed);
      const base = Array.from({ length: C }, () => U.randn(rng));
      const x = base.map((v) => v * state.scale + state.shift);
      const m = mean(x), s = std(x);
      const xhat = x.map((v) => (v - m) / Math.sqrt(s * s + 1e-5));
      const out = xhat.map((v) => state.gamma * v + state.beta);

      root.querySelector("#ln-s1").textContent = `mean = ${m.toFixed(2)},  std = ${s.toFixed(2)}  (drifts with the sliders)`;
      root.querySelector("#ln-s2").textContent = `mean ≈ ${mean(xhat).toFixed(2)},  std ≈ ${std(xhat).toFixed(2)}  (always ~0 and ~1, whatever the input)`;
      root.querySelector("#ln-s3").textContent = `mean ≈ ${mean(out).toFixed(2)},  std ≈ ${std(out).toFixed(2)}  (γ, β let the model undo/rescale if useful)`;
      U.barChart(root.querySelector("#ln-raw"), x, { height: 120 });
      U.barChart(root.querySelector("#ln-norm"), xhat, { height: 120 });
      U.barChart(root.querySelector("#ln-out"), out, { height: 120 });
    }

    const bind = (id, key, fn = parseFloat) =>
      root.querySelector(id).addEventListener("input", (e) => { state[key] = fn(e.target.value); draw(); });
    root.querySelector("#ln-shift-r").value = state.shift;
    root.querySelector("#ln-scale-r").value = state.scale;
    root.querySelector("#ln-g-r").value = state.gamma;
    root.querySelector("#ln-b-r").value = state.beta;
    bind("#ln-shift-r", "shift"); bind("#ln-scale-r", "scale");
    bind("#ln-g-r", "gamma"); bind("#ln-b-r", "beta");
    draw();
  }

  window.NanoGPTWidgets["09_layernorm_residual"] = { title: "LayerNorm + Residual", render };
})();
