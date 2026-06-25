// Tokenization widget — character-level encode/decode, fully illustrative.
// See docs/03_tokenization.md.

(function () {
  "use strict";
  window.NanoGPTWidgets = window.NanoGPTWidgets || {};
  const U = window.NanoGPTUtil;

  function render(root) {
    root.innerHTML =
      U.header("03_tokenization", "Tokenization",
        "A character-level tokenizer maps each character to an integer id. The vocabulary is " +
        "just the sorted set of characters. Type below and watch text become token ids.") +
      `<div class="panel">
        <h3>Type some text</h3>
        <div class="control"><input id="tk-in" type="text" value="Hello, World!"
          style="width:100%;padding:8px;border:1px solid var(--border);border-radius:8px;font-size:1rem"></div>
      </div>
      <div class="panel"><h3>Encoded token ids</h3><div id="tk-chips"></div>
        <p class="blurb" style="margin-top:12px">Decoded back: <code id="tk-dec"></code> (lossless)</p>
      </div>
      <div class="panel"><h3>Vocabulary (stoi: char &rarr; id)</h3>
        <p class="blurb">Built from the characters in your text, sorted.</p>
        <div id="tk-vocab"></div>
      </div>`;

    const input = root.querySelector("#tk-in");

    function update() {
      const text = input.value;
      const chars = [...new Set(text)].sort();
      const stoi = new Map(chars.map((c, i) => [c, i]));

      // chips: each character with its id
      const chipHost = root.querySelector("#tk-chips");
      chipHost.innerHTML = "";
      [...text].forEach((c) => {
        const chip = document.createElement("span");
        chip.style.cssText =
          "display:inline-flex;flex-direction:column;align-items:center;margin:3px;" +
          "padding:5px 9px;border:1px solid var(--border);border-radius:8px;background:var(--accent-soft)";
        const disp = c === " " ? "␣" : c === "\n" ? "⏎" : c;
        chip.innerHTML =
          `<span style="font-weight:600">${disp}</span>` +
          `<span style="color:var(--accent);font-size:0.8rem;font-variant-numeric:tabular-nums">${stoi.get(c)}</span>`;
        chipHost.appendChild(chip);
      });

      root.querySelector("#tk-dec").textContent =
        JSON.stringify([...text].map((c) => stoi.get(c)));

      // vocab table
      const vHost = root.querySelector("#tk-vocab");
      vHost.innerHTML = chars
        .map((c) => {
          const disp = c === " " ? "␣ (space)" : c === "\n" ? "⏎ (newline)" : c;
          return `<span style="display:inline-block;margin:2px 8px;font-variant-numeric:tabular-nums">` +
            `<b style="color:var(--accent)">${stoi.get(c)}</b> &rarr; ${disp}</span>`;
        })
        .join("");
    }

    input.addEventListener("input", update);
    update();
  }

  window.NanoGPTWidgets["03_tokenization"] = { title: "Tokenization", render };
})();
