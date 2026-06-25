// Shared helpers for the widgets: seeded RNG, tiny linear algebra, and reusable D3 charts.
// Loaded before the individual widget scripts. Everything illustrative (no real model).

(function () {
  "use strict";

  // --- deterministic RNG so visuals are stable across reloads ---
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(rng) {
    // Box-Muller standard normal
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function randnMatrix(rng, rows, cols, scale = 1) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => randn(rng) * scale)
    );
  }

  // --- tiny linear algebra on plain arrays ---
  function matmul(A, B) {
    const m = A.length, k = A[0].length, n = B[0].length;
    const out = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        for (let j = 0; j < n; j++) out[i][j] += a * B[p][j];
      }
    return out;
  }
  const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  function softmaxRow(arr) {
    const max = Math.max(...arr.filter((x) => isFinite(x)));
    const exps = arr.map((x) => (isFinite(x) ? Math.exp(x - max) : 0));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((e) => e / sum);
  }

  // --- shared tooltip ---
  function tooltip() {
    let el = document.querySelector(".cell-tip");
    if (!el) {
      el = document.createElement("div");
      el.className = "cell-tip";
      document.body.appendChild(el);
    }
    return {
      show(e, text) {
        el.style.opacity = 1;
        el.style.left = e.clientX + 12 + "px";
        el.style.top = e.clientY + 12 + "px";
        el.textContent = text;
      },
      hide() { el.style.opacity = 0; },
    };
  }

  // --- reusable heatmap ---
  // host: DOM element, M: 2D array. opts: {cell, max, rowLabels, colLabels, mask, highlight, fmt}
  function heatmap(host, M, opts = {}) {
    host.innerHTML = "";
    const rows = M.length, cols = M[0].length;
    const cell = opts.cell || Math.max(14, Math.min(40, Math.floor(420 / cols)));
    const max = opts.max || Math.max(1e-6, d3.max(M.flat().map(Math.abs)));
    const color = opts.color || d3.scaleSequential(d3.interpolateRdBu).domain([max, -max]);
    const fmt = opts.fmt || ((v) => v.toFixed(2));
    const tip = tooltip();
    const mL = opts.rowLabels ? 26 : 6, mT = opts.colLabels ? 20 : 6;
    const W = cols * cell + mL + 8, H = rows * cell + mT + 8;
    const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const g = svg.append("g").attr("transform", `translate(${mL},${mT})`);

    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        const masked = opts.mask && opts.mask(i, j);
        const hl = opts.highlight && (opts.highlight.row === i || opts.highlight.col === j);
        g.append("rect")
          .attr("x", j * cell).attr("y", i * cell)
          .attr("width", cell - 1).attr("height", cell - 1)
          .attr("fill", masked ? "#e5e7eb" : color(M[i][j]))
          .attr("stroke", hl ? "#111827" : "none")
          .attr("stroke-width", hl ? 1.5 : 0)
          .on("mousemove", (e) =>
            tip.show(e, `[${i},${j}] = ${masked ? "masked" : fmt(M[i][j])}`))
          .on("mouseleave", tip.hide);
      }
    if (opts.colLabels)
      opts.colLabels.forEach((t, j) =>
        g.append("text").attr("x", j * cell + cell / 2).attr("y", -6)
          .attr("text-anchor", "middle").attr("class", "axis").style("font-size", "10px").text(t));
    if (opts.rowLabels)
      opts.rowLabels.forEach((t, i) =>
        g.append("text").attr("x", -6).attr("y", i * cell + cell / 2 + 3)
          .attr("text-anchor", "end").attr("class", "axis").style("font-size", "10px").text(t));
    return svg;
  }

  // --- reusable signed bar chart over a 1-D vector ---
  function barChart(host, values, opts = {}) {
    host.innerHTML = "";
    const W = opts.width || 620, H = opts.height || 140, m = { t: 8, r: 10, b: 22, l: 30 };
    const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleBand().domain(d3.range(values.length)).range([m.l, W - m.r]).padding(0.12);
    const ext = d3.max(values, (v) => Math.abs(v)) || 1;
    const y = d3.scaleLinear().domain([-ext * 1.1, ext * 1.1]).range([H - m.b, m.t]);
    const color = opts.color || d3.scaleSequential(d3.interpolateRdBu).domain([ext, -ext]);
    svg.append("g").attr("transform", `translate(0,${y(0)})`).attr("class", "axis")
      .call(d3.axisBottom(x).tickValues(x.domain().filter((d) => d % (opts.tickEvery || 1) === 0)).tickSize(0));
    svg.append("g").attr("transform", `translate(${m.l},0)`).attr("class", "axis").call(d3.axisLeft(y).ticks(3));
    svg.selectAll("rect.bar").data(values).enter().append("rect")
      .attr("x", (d, i) => x(i)).attr("width", x.bandwidth())
      .attr("y", (d) => Math.min(y(d), y(0))).attr("height", (d) => Math.abs(y(d) - y(0)))
      .attr("fill", (d) => color(d));
    return svg;
  }

  // --- standard widget header (title + links + blurb) ---
  function header(id, title, blurb, extraLinks = "") {
    return `<div class="widget-head"><h2>${title}</h2>
      <p class="links"><a href="../docs/${id}.md">math doc</a> &middot;
      <a href="../docs/assets/${id}.png">drawing</a>${extraLinks}</p>
      <p class="blurb">${blurb}</p></div>`;
  }

  window.NanoGPTUtil = {
    mulberry32, randn, randnMatrix, matmul, transpose, softmaxRow,
    tooltip, heatmap, barChart, header,
  };
})();
