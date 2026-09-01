#!/usr/bin/env python3
"""Fail if shipped handbook HTML is missing required topic clusters."""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def load(name: str) -> str:
    path = ROOT / name
    if not path.is_file():
        raise SystemExit(f"missing shipped page: {name}")
    return path.read_text(encoding="utf-8")


def must_contain(page: str, haystack: str, needles: list[str]) -> None:
    missing = [n for n in needles if n.lower() not in haystack.lower()]
    if missing:
        raise SystemExit(f"{page} missing {missing}")


def headings(html: str) -> list[str]:
    return re.findall(r"<h[12][^>]*>(.*?)</h[12]>", html, flags=re.I | re.S)


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def main() -> int:
    nav = load("nav.js")
    must_contain("nav.js", nav, ["RL 背景", "本仓库架构", "未来方向"])

    index = load("index.html")
    must_contain(
        "index.html",
        index,
        ["RL 背景", "本仓库架构", "未来方向", "architecture.html", "directions.html"],
    )

    foundations = load("foundations.html")
    deep = load("deep.html")
    llm = load("llm.html")
    background = foundations + deep + llm
    must_contain("foundations.html", foundations, ["MDP", "Bellman"])
    must_contain("deep.html", deep, ["importance ratio", "PPO", r"A_t"])
    must_contain("llm.html", llm, ["GRPO", "DPO", "RLHF"])
    if "\\[" not in background and "\\(" not in background:
        raise SystemExit("background pages have no KaTeX delimiters")

    systems = load("systems.html")
    architecture = load("architecture.html")
    arch = systems + architecture
    must_contain("architecture.html", architecture, ["rl_engine", "csrc", "KernelRegistry"])
    must_contain("architecture cluster", arch, ["CUDA", "Triton", "ROCm", "executors"])

    directions = load("directions.html")
    must_contain("directions.html", directions, ["一致性", "跨配置"])
    if not any(k in directions for k in ("融合", "grpo_loss", "ratio_kl", "多模态")):
        raise SystemExit("directions.html missing fused-loss or multimodal operators")

    print("ok pages", ["index", "foundations", "deep", "llm", "systems", "architecture", "directions"])
    print("ok nav clusters: RL 背景 / 本仓库架构 / 未来方向")
    print("ok math delimiters on background pages")
    print("--- headings ---")
    for name in (
        "index.html",
        "foundations.html",
        "deep.html",
        "llm.html",
        "systems.html",
        "architecture.html",
        "directions.html",
    ):
        print(name)
        for h in headings(load(name)):
            print(" ", strip_tags(h).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
