#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_all_aris_skills.py
===========================
一键下载 ARIS（Auto-claude-code-research-in-sleep）的**全部技能 + 工具脚本 +
共享依赖**到当前项目，供 Claude Code 使用。

为什么需要它：在 Cowork 沙箱里 github.com 被代理屏蔽（git clone 403），且只能
逐个文件抓取、也拿不到 tools/ 脚本。这个脚本在**你自己的电脑**上运行，能一次把
全部内容装好（含手动复制拿不到的 research_wiki.py / evidence_check.py /
verify_paper_audits.sh / 各种 fetcher 等）。

用法（在 D:\\detr_Q3 目录下）：
    python download_all_aris_skills.py            # 下载全部技能 + tools（跳过已存在）
    python download_all_aris_skills.py --force    # 覆盖已存在文件
    python download_all_aris_skills.py --codex    # 连 Codex 镜像(skills-codex/)也一起下
    python download_all_aris_skills.py --dest .   # 指定目标项目根（默认当前目录）

映射规则：
    repo  skills/<...>   ->  <project>/.claude/skills/<...>
    repo  tools/<...>    ->  <project>/.aris/tools/<...>
这样 Claude Code 能发现技能，技能里的 helper 解析链（.aris/tools/ 优先）也能命中。
"""

import argparse
import json
import os
import stat
import sys
import urllib.request
import urllib.error

OWNER = "wanshuiyin"
REPO = "Auto-claude-code-research-in-sleep"
BRANCH = "main"
TREE_API = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/"

UA = {"User-Agent": "aris-skill-downloader"}


def http_get(url, binary=False, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if binary else data.decode("utf-8")


def list_repo_files(include_codex=False):
    """返回 repo 里所有需要下载的 blob 路径。"""
    print("• 拉取仓库文件清单 (git tree API) ...")
    tree = json.loads(http_get(TREE_API))
    if tree.get("truncated"):
        print("  ⚠ 警告: GitHub 返回的树被截断，可能漏文件。建议改用 git clone。")
    paths = []
    for node in tree.get("tree", []):
        if node.get("type") != "blob":
            continue
        p = node["path"]
        if not (p.startswith("skills/") or p.startswith("tools/")):
            continue
        if not include_codex and "skills-codex" in p:
            continue
        paths.append(p)
    return paths


def repo_path_to_local(repo_path, dest):
    """skills/* -> .claude/skills/* ; tools/* -> .aris/tools/* """
    if repo_path.startswith("skills/"):
        rel = os.path.join(".claude", "skills", repo_path[len("skills/"):])
    elif repo_path.startswith("tools/"):
        rel = os.path.join(".aris", "tools", repo_path[len("tools/"):])
    else:
        rel = repo_path
    return os.path.join(dest, *rel.split("/"))


def main():
    ap = argparse.ArgumentParser(description="下载全部 ARIS 技能 + tools 到当前项目")
    ap.add_argument("--dest", default=".", help="目标项目根目录（默认当前目录）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    ap.add_argument("--codex", action="store_true", help="同时下载 Codex 镜像 skills-codex/")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    print(f"• 目标项目: {dest}")

    try:
        files = list_repo_files(include_codex=args.codex)
    except urllib.error.URLError as e:
        print(f"✗ 无法访问 GitHub: {e}\n  请检查网络，或改用: git clone {RAW_BASE.split('/raw')[0]}")
        sys.exit(1)

    n_skills = len({p.split("/")[1] for p in files if p.startswith("skills/") and "/" in p[len("skills/"):]})
    print(f"• 待下载 {len(files)} 个文件（约 {n_skills} 个技能目录）\n")

    downloaded = skipped = failed = 0
    for i, rp in enumerate(sorted(files), 1):
        local = repo_path_to_local(rp, dest)
        if os.path.exists(local) and not args.force:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        try:
            content = http_get(RAW_BASE + rp, binary=True)
            with open(local, "wb") as f:
                f.write(content)
            # 给 .sh / .py 脚本加可执行位（非 Windows）
            if rp.endswith((".sh", ".py")) and os.name != "nt":
                st = os.stat(local)
                os.chmod(local, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            downloaded += 1
            print(f"  [{i}/{len(files)}] ✓ {rp}")
        except Exception as e:  # noqa
            failed += 1
            print(f"  [{i}/{len(files)}] ✗ {rp}  ({e})")

    print(f"\n完成：下载 {downloaded}，跳过 {skipped}（已存在），失败 {failed}。")
    print(f"技能装在: {os.path.join(dest, '.claude', 'skills')}")
    print(f"工具装在: {os.path.join(dest, '.aris', 'tools')}")
    if failed:
        print("有文件失败，可重跑（带 --force 覆盖）或改用 git clone + tools/install_aris.sh。")


if __name__ == "__main__":
    main()
