# Interactive Transformer Visualizer

A static site (vanilla JS + [D3](https://d3js.org/)) that visualizes each component of the
nanoGPT model. No build step, no server. See [docs/14_web_visualizer.md](../docs/14_web_visualizer.md)
for the full design.

## Run locally

It's just static files, but browsers block `fetch`/module loading from `file://`, so serve it:

```bash
# from the repo root
python -m http.server 8000
# then open http://localhost:8000/web/
```

(Or any static server / the VS Code "Live Server" extension.)

## What's here

| File | Role |
|---|---|
| `index.html` | Landing page + component nav |
| `styles.css` | Styling |
| `app.js` | Hash-router: builds nav, renders the selected widget |
| `widgets/_util.js` | Shared helpers: seeded RNG, matmul/softmax, reusable D3 heatmap + bar chart |
| `widgets/03_tokenization.js` | ✅ Type text → token ids + vocab |
| `widgets/04_embeddings.js` | ✅ Token id → embedding row (lookup table heatmap) |
| `widgets/05_positional_encoding.js` | ✅ Sinusoidal sin/cos heatmap + waves |
| `widgets/06_self_attention.js` | ✅ QKᵀ scores → causal mask toggle → softmax weights |
| `widgets/07_multi_head_attention.js` | ✅ Parallel per-head attention patterns |
| `widgets/08_feed_forward.js` | ✅ GELU vs ReLU + a vector through expand→activate→project |
| `widgets/09_layernorm_residual.js` | ✅ Normalize a vector live; γ/β; residual highway |
| `widgets/10_transformer_block.js` | ✅ Clickable pre-LN block schematic |
| `widgets/11_gpt_model.js` | ✅ Clickable full-stack map with shape tracker |
| `data/` | Exported real-model weights (added once a checkpoint exists) |

All widgets use **illustrative** (seeded synthetic) numbers so they work without a trained
model. A later `scripts/export_for_web.py` can swap in real attention weights.

## Adding a widget

Create `widgets/NN_name.js` that self-registers:

```js
window.NanoGPTWidgets = window.NanoGPTWidgets || {};
window.NanoGPTWidgets["NN_name"] = {
  title: "...", doc: "../docs/NN_name.md", drawing: "../docs/assets/NN_name.png",
  render(root) { /* build controls + D3 visuals into `root` */ },
};
```

Then: add a `<script src="widgets/NN_name.js"></script>` to `index.html` (before `app.js`),
and flip `built: true` for that id in the `CATALOG` array in `app.js`.

## Deploy (GitHub Pages)

Settings → Pages → Deploy from branch → `main` → `/web`. Live at
`https://<user>.github.io/<repo>/`. D3 loads from a CDN, so there's nothing to bundle.
