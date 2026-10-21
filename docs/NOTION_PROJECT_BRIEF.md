# Setwise Reranker Baselines：版本与实验设计

整理日期：2026-10-21

版本：v0.1.0（本地工程与离线验证阶段）

GitHub 目标仓库：https://github.com/xiao11zhang/Rubric-ranker-experiment

版本发布目标：v0.1.0，本地工程与离线验证阶段。远程提交记录以 GitHub 历史为准。

Notion 私有草稿：https://app.notion.com/p/400e4101b9378165991ce3c8c1cf8306

## 实验目标

训练 Reranker 从固定候选文档集合中选择互补、覆盖充分且低冗余的文档子集，输出文档 ID。
Retriever、Reranker 与下游 Generator 分离。先在固定候选池的单轮 RAG 中比较训练机制，
后续再考虑 Deep Research 与新方法。

| 组别 | 初始化 | 第二阶段 | 研究问题 |
|---|---|---|---|
| B0 | 共同 Rubric-guided SFT checkpoint | 不继续训练 | SFT 的基础水平 |
| B1 | 与 B0 相同的 checkpoint | RR-style GRPO | 集合级奖励的增益 |
| B2 | 与 B0/B1 相同的 checkpoint | 同一 GRPO 主干加 ATD/ATO | 自适应辅导与 token 级信号的额外增益 |

每个模型规模只做一次共同 SFT；不同模型规模不能共享同一权重文件。
这是统一协议下的机制比较，不是照搬不同论文的完整实验条件。

## 当前版本实现

- 模型、数据、算法、Judge 与运行环境分开配置。
- B1/B2 共同参数自动比较，只允许组别、实验名与输出目录不同。
- 候选 ID、顺序、Rubric、标签和缓存清单校验。
- 数据转换与共同 SFT 数据导出；正式 Student prompt 原样保留。
- 直接执行固定版本 RR 源码中的解析、评分 prompt 构建、重试与聚合函数。
- Judge API transport、响应缓存和评分诊断；密钥只通过环境变量提供。
- 同一 ATO/verl GPU 启动适配层，默认只预览命令。
- 官方 selector/reflector 模板原样保留；hint 模板仅做两处协议适配。
- 离线数值参考、官方源码差分测试与来源哈希校验。

## 当前配置与待锁定事项

正式目标模型为 Qwen3-8B；当前模板面向 Qwen3-1.7B 单卡 pilot。
候选协议建议使用 BGE-base-en-v1.5、固定 Wikipedia 2018-12 和 Top-30，
但语料快照、模型 revision、真实数据集、Judge 模型、预算与下游 Generator 尚未最终锁定。

当前 pilot 模板：学习率 1e-6、query batch 4、rollout group 4、temperature 0.7、
top-p 0.8、top-k 20、clip 0.2、KL 系数 0.001、最多 50 steps。
这些数值尚未通过 A100 实测，不能作为正式实验配置或显存承诺。

## 上游复用与改编

| 项目 | 固定来源 |
|---|---|
| RubricRanker | https://github.com/8421BCD/RubricRanker/tree/f464f083bd54cf6b1464a4347d7e9811a2c6de50 |
| AdaTutoRank | https://github.com/AdaTutoRank/AdaTutoRank/tree/7df929c42482dccd0181b29a22983356762a7b92 |

官方源文件、模板和许可证保存于 `src/setwise/_upstream/`，文件清单包含 Git blob 哈希。
RR 有效奖励保持 0-1。ATO hint 阈值对应为 0.3/0.6，保持原论文边界含义。
两组统一使用 RR 奖励与 KL 设置，因此 B2 属于受控改编。

ATO 采用：

`A_ATO = 0.7 * A_GRPO + 0.3 * m * (log p_teacher - log p_old)`

Teacher 固定；selector/reflector 使用当前策略；sibling 仅在评分严格更优时提供相关 hint。
恢复纯 GRPO 的一致性检查需把 GRPO 权重设为 1，而非只清空 ATD 信号。

Judge 持续失败时，当前共同配置为中止该批次。这是明确记录的工程改编；
`upstream_drop` 可恢复 RR 官方丢弃失败维度、重新加权的行为。B1/B2 必须一致。

## 版本变更记录

### v0.1.0

1. 建立模块化配置、数据合同、训练适配与离线评测入口。
2. 将早期自写评分实现替换为固定快照中的官方 RR 函数与官方评分 prompt。
3. 将简化 ATO 提示替换为官方模板，并仅保留评分尺度与 Rubric 维度的必要适配。
4. 禁止正式数据自动使用自写 Student prompt。
5. 增加上游源码、许可证、commit/blob 哈希及逐项对齐说明。

## 验证证据与限制

上一轮本地验证记录：27 项离线测试通过；包含 40 组与完整官方 RR 文件的
reward、完整评分 prompt 和重试行为差分检查。上游 7 个快照文件哈希验证通过，
B1/B2 共同配置比较通过，离线 mock 样例评测入口通过。

上述记录不代表真实 GPU、API 或训练效果已验证。
尚未完成：Hydra/GPU 集成、Ray 环境传播、真实 Teacher token 对齐与截断检查、
真实训练 reward 曲线、Reranker 批量生成、统一 Generator 的 EM/F1 和多 seed 正式比较。

## 后续验收路径

1. 锁定数据、模型、Judge、候选缓存与共同 SFT 来源。
2. Linux A100 上验证依赖、配置解析、单步训练与保存恢复。
3. 从同一小模型 SFT checkpoint 比较 B1/B2，检查固定验证 reward 与输出行为。
4. 同时观察格式合法率、选集大小、组内 reward 方差、KL、entropy、clip fraction、
   hint/fallback 比例、ATD/GRPO 量级及实际成本。
5. 通过 pilot 后，迁移至 Qwen3-8B 和正式预算，再进行下游与多 seed 评测。

代码与细节依据：README.md、docs/UPSTREAM_ALIGNMENT.md、configs/common.toml。
