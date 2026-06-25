// Feed-forward widget — GELU vs ReLU, and a vector flowing expand -> activate -> project.
// See docs/08_feed_forward.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;

  // erf approximation (Abramowitz & Stegun 7.1.26) for an exact-ish GELU.
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
  const relu = (x) => Math.max(0, x);

  function curve(host, fns) {
    host.innerHTML = "";
    const W = 620, H = 220, m = { t: 10, r: 14, b: 26, l: 34 };
    const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([-4, 4]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([-1, 4]).range([H - m.b, m.t]);
    svg.append("g").attr("transform", `translate(0,${y(0)})`).attr("class", "axis").call(d3.axisBottom(x).ticks(8));
    svg.append("g").attr("transform", `translate(${m.l},0)`).attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    const xs = d3.range(-4, 4.02, 0.05);
    fns.forEach((f) => {
      const line = d3.line().x((d) => x(d)).y((d) => y(f.fn(d)));
      svg.append("path").datum(xs).attr("fill", "none").attr("stroke", f.color).attr("stroke-width", 2.2).attr("d", line);
      svg.append("text").attr("x", W - m.r - 4).attr("y", y(f.fn(3.6))).attr("text-anchor", "end")
        .attr("fill", f.color).style("font-size", "12px").style("font-weight", 600).text(f.name);
    });
  }

  function render(root) {
    const C = 8;
    root.innerHTML =
      U.header("08_feed_forward", "Feed-Forward Network",
        "After attention mixes tokens, a position-wise MLP lets each token <b>think</b>: " +
        "expand C&rarr;4C, apply a non-linearity (GELU), project 4C&rarr;C. Same MLP at every position.") +
      `<div class="panel"><h3>Activation: GELU vs ReLU</h3>
        <p class="blurb">GELU is a smooth version of ReLU that lets small negatives leak through.</p>
        <div id="ff-curve"></div></div>
      <div class="panel"><h3>A vector flowing through the FFN</h3>
        <p class="blurb">Illustrative weights. Notice the hidden layer is 4&times; wider.</p>
        <div style="margin:6px 0"><b>input x</b> (C=${C})</div><div id="ff-in"></div>
        <div style="margin:6px 0"><b>hidden</b> = GELU(x W&#8321;) &nbsp;(4C=${4 * C})</div><div id="ff-hid"></div>
        <div style="margin:6px 0"><b>output</b> = hidden W&#8322; &nbsp;(C=${C})</div><div id="ff-out"></div>
        <div style="margin-top:10px"><button id="ff-reseed" class="nav-item" style="justify-content:center;display:inline-flex">Resample x</button></div>
      </div>`;

    let seed = 3;
    function draw() {
      curve(root.querySelector("#ff-curve"), [
        { fn: gelu, name: "GELU", color: "#4f46e5" },
        { fn: relu, name: "ReLU", color: "#9ca3af" },
      ]);
      const rng = U.mulberry32(seed);
      const x = Array.from({ length: C }, () => U.randn(rng));
      const W1 = U.randnMatrix(rng, C, 4 * C, 0.4);
      const W2 = U.randnMatrix(rng, 4 * C, C, 0.4);
      const hiddenPre = U.matmul([x], W1)[0];
      const hidden = hiddenPre.map(gelu);
      const out = U.matmul([hidden], W2)[0];
      U.barChart(root.querySelector("#ff-in"), x, { height: 110 });
      U.barChart(root.querySelector("#ff-hid"), hidden, { height: 110, tickEvery: 4 });
      U.barChart(root.querySelector("#ff-out"), out, { height: 110 });
    }
    root.querySelector("#ff-reseed").addEventListener("click", () => { seed = (seed * 7 + 1) % 9973; draw(); });
    draw();
  }

  window.NanoGPTWidgets["08_feed_forward"] = { title: "Feed-Forward", render };
})();
