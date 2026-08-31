# RL Handbook

挂在 [carsontung666/RL-Kernel](https://github.com/carsontung666/RL-Kernel) 的 **`gh-pages`** 分支上，不碰 fork 的 `main` / 代码分支。

学习站：

**https://carsontung666.github.io/RL-Kernel/learn/learn/**

顺序：总览 → 基础 → 深度 RL → RLHF/DPO/GRPO → 系统层 → 仓库与课程 → 论文 → 四周计划。

## 打开 Pages（只需一次）

1. 打开 https://github.com/carsontung666/RL-Kernel/settings/pages
2. **Source** 选 **Deploy from a branch**
3. Branch：`gh-pages`，folder：`/ (root)`
4. Save，等一两分钟

如果仓库名大小写被 GitHub Pages 规范化，URL 也可能是 `carsontung666.github.io/rl-kernel/`。两种都试一下。

## 本地预览

```bash
cd /home/junyao/code/rl-handbook/learn
python3 -m http.server 4100
```

浏览器打开 http://127.0.0.1:4100

本 fork 的 `gh-pages` 根目录仍是 MkDocs 项目文档；学习手册在 `/learn/`，不覆盖它。
