(function () {
  const parts = location.pathname.split("/").filter(Boolean);
  let here = parts.pop() || "index.html";
  if (!here.endsWith(".html")) here = "index.html";
  const items = [
    ["RL 背景", [
      ["index.html", "总览"],
      ["foundations.html", "RL 基础"],
      ["deep.html", "深度 RL"],
      ["llm.html", "RLHF / DPO / GRPO"],
    ]],
    ["本仓库架构", [
      ["systems.html", "系统问题"],
      ["architecture.html", "仓库怎么拆"],
    ]],
    ["未来方向", [
      ["directions.html", "能做的 RL 方向"],
    ]],
  ];
  let html = `<div class="brand">RL Handbook</div>
    <div class="sub">背景 · 架构 · 方向</div>`;
  for (const [sec, links] of items) {
    html += `<div class="sec">${sec}</div>`;
    for (const [href, label] of links) {
      const cur = href === here || (here === "" && href === "index.html") ? ' style="background:#1e252d"' : "";
      html += `<a href="${href}"${cur}>${label}</a>`;
    }
  }
  const nav = document.getElementById("nav");
  if (nav) nav.innerHTML = html;

  if (document.querySelector("link[data-katex]")) return;
  const css = document.createElement("link");
  css.rel = "stylesheet";
  css.href = "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css";
  css.dataset.katex = "1";
  document.head.appendChild(css);
  const js = document.createElement("script");
  js.src = "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js";
  js.onload = function () {
    const auto = document.createElement("script");
    auto.src = "https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js";
    auto.onload = function () {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "code"],
      });
    };
    document.head.appendChild(auto);
  };
  document.head.appendChild(js);
})();
