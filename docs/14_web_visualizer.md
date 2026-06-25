# 14 — Interactive Web Visualizer

> **Code:** `web/` · **Status:** scaffold + first widget done (positional encoding).

An interactive site that lets you *see* each Transformer component. Built as a **static
site** (vanilla JS + [D3](https://d3js.org/)) so it hosts free on **GitHub Pages** with no
server. Each component gets its own widget that mirrors `src/nanogpt/` and the `docs/` pages.

## Goals

- One interactive widget per component, matching the build order in [00_plan.md](00_plan.md).
- Each widget links to its math doc (`docs/NN_*.md`) and its Excalidraw drawing
  (`docs/assets/NN_*.png`). So the learning path is **drawing → math → live interactive**.
- **Illustrative numbers first** (works before the model is trained), with an optional
  "use real model" toggle later that loads exported weights.

## Structure

```
web/
├── index.html              # landing + component nav; loads a widget into the main panel
├── styles.css
├── app.js                  # tiny hash-router: nav click → render the widget
├── widgets/
│   ├── _util.js                    ✅ shared RNG / matmul / D3 heatmap + bar chart
│   ├── 03_tokenization.js          ✅ built
│   ├── 04_embeddings.js            ✅ built
│   ├── 05_positional_encoding.js   ✅ built
│   ├── 06_self_attention.js        ✅ built
│   ├── 07_multi_head_attention.js  ✅ built
│   ├── 08_feed_forward.js          ✅ built
│   ├── 09_layernorm_residual.js    ✅ built
│   ├── 10_transformer_block.js     ✅ built
│   └── 11_gpt_model.js             ✅ built
├── data/
│   └── sample_weights.json         # (later) exported from the trained model
└── README.md
```

## How a widget is wired

Each widget file self-registers a render function:

```js
window.NanoGPTWidgets = window.NanoGPTWidgets || {};
window.NanoGPTWidgets["05_positional_encoding"] = {
  title: "Positional Encoding",
  doc: "../docs/05_positional_encoding.md",
  drawing: "../docs/assets/05_positional_encoding.png",
  render(root) { /* build controls + D3 visuals into `root` */ },
};
```

`app.js` reads the registry, builds the left-hand nav, and renders the selected widget into
the main panel (component id lives in the URL hash, e.g. `#05_positional_encoding`).

## Widget status

All nine component widgets (03–11) are built and use illustrative seeded data, plus a shared
`_util.js`. The remaining enhancement is swapping in **real model weights** (below).

## "Real model later"

When a checkpoint exists, `scripts/export_for_web.py` will run one sample input through the
model and dump attention matrices + embeddings to `web/data/sample_weights.json`. Widgets get
a toggle that swaps illustrative numbers for those — same render code, no rewrite.

## Hosting on GitHub Pages

1. Push the repo to GitHub.
2. Settings → Pages → Build and deployment → Deploy from a branch.
3. Branch `main`, folder `/web` (or move `web/` contents to `/docs` if Pages requires `/docs`).
4. Site goes live at `https://<user>.github.io/<repo>/`.

D3 is loaded from a CDN, so there's nothing to build or bundle.
