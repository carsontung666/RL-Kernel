(function () {
  const parts = location.pathname.split("/").filter(Boolean);
  let here = parts.pop() || "index.html";
  if (!here.endsWith(".html")) here = "index.html";
  const items = [
    ["入门", [
      ["index.html", "总览"],
    ]],
    ["知识", [
      ["foundations.html", "RL 基础"],
      ["deep.html", "深度 RL"],
      ["llm.html", "RLHF / DPO / GRPO"],
      ["systems.html", "系统与 RL-Kernel"],
    ]],
    ["资料", [
      ["repos.html", "仓库与课程"],
    ]],
  ];
  let html = `<div class="brand">RL Handbook</div>
    <div class="sub">入门 · 知识</div>`;
  for (const [sec, links] of items) {
    html += `<div class="sec">${sec}</div>`;
    for (const [href, label] of links) {
      const cur = href === here || (here === "" && href === "index.html") ? ' style="background:#1e252d"' : "";
      html += `<a href="${href}"${cur}>${label}</a>`;
    }
  }
  const nav = document.getElementById("nav");
  if (nav) nav.innerHTML = html;
})();
