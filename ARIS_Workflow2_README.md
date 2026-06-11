# ARIS 工作流 2（auto-review-loop）— 安装说明

来源：[wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)（ARIS，"睡一觉醒来看结果"自动科研循环）。
工作流 2 = **自动评审循环**：外部模型评审 → 自动修复 → 再评审，直到给出正面结论或达到轮数上限（默认 4 轮）。

## 已安装内容

装在你的项目 `D:\detr_Q3` 下，给 **Claude Code** 用：

```
.claude/skills/
  # 工作流 2（auto-review-loop 全家桶）
  auto-review-loop/SKILL.md            # 主技能（Codex MCP 评审）
  auto-review-loop-llm/SKILL.md        # 变体：任意 OpenAI 兼容 LLM（经 llm-chat MCP）
  auto-review-loop-minimax/SKILL.md    # 变体：MiniMax API
  result-to-claim/SKILL.md             # 子技能：结果转结构化 claim
  training-check/SKILL.md              # 子技能：训练中轮询 WandB 健康度
  # 编排器（工作流入口）
  research-pipeline/SKILL.md           # 全流程 W1→1.5→2→3
  idea-discovery/SKILL.md              # 工作流 1：找 idea
  experiment-bridge/SKILL.md           # 工作流 1.5：实现并部署实验
  shared-references/                   # 10 个协议文档（运行时读取）
.aris/tools/
  save_trace.sh                        # 评审留痕助手（review tracing）
```

共 **8 个技能**已手动装好。其余 68 个 + 全部 `tools/` 脚本，用下面的下载脚本一次补齐。

安装时已校验：5 个 SKILL.md 齐全，所有 `../shared-references/*.md` 引用都能解析，`save_trace.sh` 语法通过。

## 怎么用

这些是 **Claude Code 的 slash 技能**，不是当前 Cowork 会话的技能。用法：在 Claude Code 里打开 `D:\detr_Q3` 这个项目，然后：

```
/auto-review-loop "对我的 fire/smoke 检测报告做提交前评审"
```

它会读你的项目文档与实验结果，多轮评审并自动修复，过程写入 `review-stage/AUTO_REVIEW.md`，状态存 `review-stage/REVIEW_STATE.json`（支持断点续跑）。

常用开关（追加在参数里）：

```
/auto-review-loop "..." — difficulty: hard          # 加 reviewer memory + 辩论环节
/auto-review-loop "..." — human checkpoint: true     # 每轮停下来等你确认
/auto-review-loop "..." — difficulty: nightmare      # GPT 直接读仓库、独立核对代码与claim
```

## 评审后端（必须有一个）

循环的核心是"换一个模型来挑刺"，所以需要一个**外部评审后端**——本 Cowork 会话里没有，要在 Claude Code 里自己配：

| 技能 | 后端 | 配置 |
|---|---|---|
| `auto-review-loop`（默认/推荐） | Codex MCP（GPT，xhigh） | 在 Claude Code 装 Codex MCP 后即可 |
| `auto-review-loop-llm` | 任意 OpenAI 兼容 API（DeepSeek / Kimi / GLM / 通义…） | 配 `llm-chat` MCP，填 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` |
| `auto-review-loop-minimax` | MiniMax API | 设 `MINIMAX_API_KEY` |

没配后端时技能仍会启动，但评审那一步会失败——先配好再跑。

## 下载全部 76 个技能（用脚本，已为你准备好）

在 Cowork 沙箱里 github.com 被代理屏蔽（`git clone` 返回 403），且逐个手动抓取
拿不到 `tools/` 脚本——而很多技能要靠它们才能完整工作。所以给了你一键脚本
**`download_all_aris_skills.py`**。在你自己电脑上（能访问 GitHub）于项目根目录运行：

```bash
cd D:\detr_Q3
python download_all_aris_skills.py            # 全部技能 + tools/ + 依赖，跳过已存在
# python download_all_aris_skills.py --force  # 覆盖重下
# python download_all_aris_skills.py --codex  # 连 Codex 镜像 skills-codex/ 也下
```

它按 `skills/* → .claude/skills/*`、`tools/* → .aris/tools/*` 装好全套（含
`research_wiki.py` / `evidence_check.py` / `verify_paper_audits.sh` / 各 fetcher 等
helper 脚本），已装好的 8 个会自动跳过。

**或者用官方 git 方式**（同样在你电脑上跑）：

```bash
git clone https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git
bash Auto-claude-code-research-in-sleep/tools/install_aris.sh D:/detr_Q3
```

`install_aris.sh` 会建好 `.aris/tools/` 软链，让所有 helper 脚本就位。两种方式都会让
`.claude/skills/` 下出现全部 76 个技能，在 Claude Code 里即可用 `/<技能名>` 调用。

## 说明

- 各 SKILL.md 直接引用的 shared-references 都已装齐、可解析。
- 少数**更深层**的引用（如 `acceptance-gate.md`、`reviewer-independence.md` 以及 `evidence_check.py` / `research_wiki.py` 等 helper 脚本）属于全仓库其它部分，这里没装；相关技能都是**优雅降级**的——缺了只会跳过对应副作用（如 research-wiki 写入），主流程照常出结果。要补齐就按上面的 `install_aris.sh` 走一遍。
- 想让这些技能在**当前 Cowork 会话**里也能直接调用，需在 设置 → Capabilities 里管理技能；本会话只能把文件装进你的项目供 Claude Code 使用。
