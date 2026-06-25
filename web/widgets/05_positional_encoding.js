// Positional Encoding widget — sinusoidal PE, fully illustrative (no model needed).
// See docs/05_positional_encoding.md and docs/assets/05_positional_encoding.png.
//
//   PE(pos, 2k)   = sin( pos / base^(2k / C) )
//   PE(pos, 2k+1) = cos( pos / base^(2k / C) )

(function () {
  "use strict";

  window.NanoGPTWidgets = window.NanoGPTWidgets || {};

  // One PE value for a (position, dimension) pair.
  function peValue(pos, i, C, base) {
    const k2 = 2 * Math.floor(i / 2); // pair index -> 2k
    const denom = Math.pow(base, k2 / C);
    const angle = pos / denom;
    return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  }

  function render(root) {
    // --- state ---
    const state = { T: 32, C: 32, base: 10000, selPos: 8, selDim: 4 };

    // --- scaffold ---
    root.innerHTML = `
      <div class="widget-head">
        <h2>Positional Encoding</h2>
        <p class="links">
          <a href="../docs/05_positional_encoding.md">math doc</a> &middot;
          <a href="../docs/assets/05_positional_encoding.png">drawing</a> &middot;
          <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need &sect;3.5</a>
        </p>
        <p class="blurb">Self-attention has no built-in sense of order. We <em>add</em> a
          position vector to each token embedding. Each dimension is a sinusoid of a
          different frequency &mdash; drag the sliders and watch the pattern change.</p>
      </div>

      <div class="panel">
        <h3>Controls</h3>
        <div class="controls">
          <div class="control"><label>Sequence length T = <span class="val" id="pe-T"></span></label>
            <input type="range" id="pe-T-r" min="8" max="64" step="1"></div>
          <div class="control"><label>Model dim C = <span class="val" id="pe-C"></span></label>
            <input type="range" id="pe-C-r" min="8" max="64" step="2"></div>
          <div class="control"><label>Highlight position = <span class="val" id="pe-pos"></span></label>
            <input type="range" id="pe-pos-r" min="0" max="31" step="1"></div>
          <div class="control"><label>Highlight dimension = <span class="val" id="pe-dim"></span></label>
            <input type="range" id="pe-dim-r" min="0" max="31" step="1"></div>
        </div>
      </div>

      <div class="panel">
        <h3>Heatmap &mdash; PE(position, dimension)</h3>
        <p class="blurb">Rows = positions, columns = dimensions. Low dims wiggle fast,
          high dims wiggle slowly. Hover a cell for its value.</p>
        <div id="pe-heatmap"></div>
      </div>

      <div class="panel">
        <h3>The wave for dimension <span class="val" id="pe-dim2"></span> across positions</h3>
        <div id="pe-wave"></div>
      </div>

      <div class="panel">
        <h3>Encoding vector at position <span class="val" id="pe-pos2"></span></h3>
        <p class="blurb">This whole vector gets added to that token's embedding.</p>
        <div id="pe-vector"></div>
      </div>

      <div class="panel">
        <h3>Formula</h3>
        <div class="formula">
PE(pos, 2k)&nbsp;&nbsp;= <span class="hl">sin</span>( pos / base<sup>2k / C</sup> )<br>
PE(pos, 2k+1) = <span class="hl">cos</span>( pos / base<sup>2k / C</sup> )
&nbsp;&nbsp;&nbsp;&nbsp;base = ${state.base}, C = model dim
        </div>
      </div>
    `;

    // tooltip
    let tip = document.querySelector(".cell-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "cell-tip";
      document.body.appendChild(tip);
    }

    const color = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]); // +1 red, -1 blue

    function clampSelectors() {
      state.selPos = Math.min(state.selPos, state.T - 1);
      state.selDim = Math.min(state.selDim, state.C - 1);
      const posR = root.querySelector("#pe-pos-r");
      const dimR = root.querySelector("#pe-dim-r");
      posR.max = state.T - 1; posR.value = state.selPos;
      dimR.max = state.C - 1; dimR.value = state.selDim;
    }

    function drawHeatmap() {
      const host = root.querySelector("#pe-heatmap");
      host.innerHTML = "";
      const cell = Math.max(6, Math.floor(560 / state.C));
      const W = state.C * cell, H = state.T * cell;
      const m = { top: 18, right: 10, bottom: 28, left: 34 };
      const svg = d3.select(host).append("svg")
        .attr("viewBox", `0 0 ${W + m.left + m.right} ${H + m.top + m.bottom}`);
      const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

      for (let pos = 0; pos < state.T; pos++) {
        for (let i = 0; i < state.C; i++) {
          const v = peValue(pos, i, state.C, state.base);
          g.append("rect")
            .attr("x", i * cell).attr("y", pos * cell)
            .attr("width", cell).attr("height", cell)
            .attr("fill", color(v))
            .attr("stroke", (pos === state.selPos || i === state.selDim) ? "#111827" : "none")
            .attr("stroke-width", (pos === state.selPos || i === state.selDim) ? 1 : 0)
            .on("mousemove", (e) => {
              tip.style.opacity = 1;
              tip.style.left = e.clientX + 12 + "px";
              tip.style.top = e.clientY + 12 + "px";
              tip.textContent = `pos ${pos}, dim ${i} = ${v.toFixed(3)}`;
            })
            .on("mouseleave", () => (tip.style.opacity = 0));
        }
      }
      g.append("text").attr("x", W / 2).attr("y", -6).attr("text-anchor", "middle")
        .attr("class", "axis").text("dimension →");
      g.append("text").attr("transform", `translate(-22,${H / 2}) rotate(-90)`)
        .attr("text-anchor", "middle").attr("class", "axis").text("position →");
    }

    function lineChart(hostSel, points, highlightX) {
      const host = root.querySelector(hostSel);
      host.innerHTML = "";
      const W = 620, H = 150, m = { top: 10, right: 12, bottom: 26, left: 34 };
      const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      const x = d3.scaleLinear().domain([0, points.length - 1]).range([m.left, W - m.right]);
      const y = d3.scaleLinear().domain([-1.05, 1.05]).range([H - m.bottom, m.top]);
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y(0)})`)
        .call(d3.axisBottom(x).ticks(8).tickSize(0)).select(".domain").attr("stroke", "#cbd5e1");
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`)
        .call(d3.axisLeft(y).ticks(3));
      const line = d3.line().x((d, i) => x(i)).y((d) => y(d));
      svg.append("path").datum(points).attr("fill", "none")
        .attr("stroke", "#4f46e5").attr("stroke-width", 2).attr("d", line);
      if (highlightX != null) {
        svg.append("circle").attr("cx", x(highlightX)).attr("cy", y(points[highlightX]))
          .attr("r", 4).attr("fill", "#ef4444");
      }
    }

    function barChart(hostSel, values) {
      const host = root.querySelector(hostSel);
      host.innerHTML = "";
      const W = 620, H = 150, m = { top: 10, right: 12, bottom: 26, left: 34 };
      const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
      const x = d3.scaleBand().domain(d3.range(values.length)).range([m.left, W - m.right]).padding(0.15);
      const y = d3.scaleLinear().domain([-1.05, 1.05]).range([H - m.bottom, m.top]);
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y(0)})`)
        .call(d3.axisBottom(x).tickValues(x.domain().filter((d) => d % 4 === 0)).tickSize(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`)
        .call(d3.axisLeft(y).ticks(3));
      svg.selectAll("rect").data(values).enter().append("rect")
        .attr("x", (d, i) => x(i)).attr("width", x.bandwidth())
        .attr("y", (d) => Math.min(y(d), y(0)))
        .attr("height", (d) => Math.abs(y(d) - y(0)))
        .attr("fill", (d) => color(d));
    }

    function redrawDerived() {
      const wave = d3.range(state.T).map((pos) => peValue(pos, state.selDim, state.C, state.base));
      lineChart("#pe-wave", wave, state.selPos);
      const vec = d3.range(state.C).map((i) => peValue(state.selPos, i, state.C, state.base));
      barChart("#pe-vector", vec);
      root.querySelector("#pe-dim2").textContent = state.selDim;
      root.querySelector("#pe-pos2").textContent = state.selPos;
    }

    function syncLabels() {
      root.querySelector("#pe-T").textContent = state.T;
      root.querySelector("#pe-C").textContent = state.C;
      root.querySelector("#pe-pos").textContent = state.selPos;
      root.querySelector("#pe-dim").textContent = state.selDim;
    }

    function fullRedraw() {
      clampSelectors();
      syncLabels();
      drawHeatmap();
      redrawDerived();
    }

    // --- wire controls ---
    const Tr = root.querySelector("#pe-T-r");
    const Cr = root.querySelector("#pe-C-r");
    const posR = root.querySelector("#pe-pos-r");
    const dimR = root.querySelector("#pe-dim-r");
    Tr.value = state.T; Cr.value = state.C;

    Tr.addEventListener("input", (e) => { state.T = +e.target.value; fullRedraw(); });
    Cr.addEventListener("input", (e) => { state.C = +e.target.value; fullRedraw(); });
    posR.addEventListener("input", (e) => { state.selPos = +e.target.value; syncLabels(); drawHeatmap(); redrawDerived(); });
    dimR.addEventListener("input", (e) => { state.selDim = +e.target.value; syncLabels(); drawHeatmap(); redrawDerived(); });

    fullRedraw();
  }

  window.NanoGPTWidgets["05_positional_encoding"] = {
    title: "Positional Encoding",
    doc: "../docs/05_positional_encoding.md",
    drawing: "../docs/assets/05_positional_encoding.png",
    render,
  };
})();
