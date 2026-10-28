# 当前流程：RR RAG LoRA SFT（r=16）→ 原 OPSD step-10/20 格式对比

按用户更新的方案：先用 RR 官方 SFT 数据对 Qwen3-4B 做 LoRA SFT；再从同一 SFT checkpoint 初始化 Teacher 和 Student，使用现有 OPSD 算法运行 20 step，保存 step-10 和 step-20，并和此前的格式正确率比较。

## 数据已下载并核查

- 官方原文件：`data/raw/rubricranker_sft-data.json`，285,407,867 bytes。
- 官方入口：[RR SFT data](https://modelscope.cn/datasets/lwhlwh/rubricranker_sft_data/files)。
- SHA-256：`ae75a131d68826ffc36a76de6e29b302ea900c6db281637113568ec3ba5cdd40`。
- 原文件共 10,602 条，RAG 3,889 条，Deep Research 6,713 条；本阶段只使用 RAG。
- 19 条与当前非空 ID 输出合同不兼容：17 个 `[]`、1 个字符串 `None`、1 个 `[0]`，记录在 `data/rr_rag_v1/excluded.json`，没有修补或重写标签。
- 实际训练 3,870 条；与当前 RR RAG val 160 个问题的归一化精确交集为 0。未声称完成语义近重复检查。
- 官方原 system/instruction/output 子集另存在 `rr_rag_official_prompt.json`。实际训练文件为 `rr_rag_opsd_prompt.json`，使用原 `opsd.common.messages` 的 Student 输入；候选正文、顺序和官方 silver output 保持不变。

官方原 prompt 包含通用五维标准和最多选择 10 篇的说明。为继续使用当前 OPSD 输入，本版本保留官方标签（实际均为 1–10 篇），将输入模板对齐到当前 Student；没有给 Student 添加 Query-specific Rubric。这是任务适配，不宣称逐字复现 RR 的完整混合场景 SFT。

## 本次配置

| 参数 | 当前值 |
|---|---|
| 模型 | Qwen3-4B 原始基础权重，未使用此前 OPSD adapter |
| SFT | LoRA，r=16、alpha=32、dropout=0、all linear targets |
| 学习率 | `1e-4`，LoRA 适配值；RR 官方全参数配置为 `5e-6` |
| Epochs / Scheduler / Warmup | 1 / cosine / 0.1（本次运行保留已启动的两轮调度轨迹，见下文） |
| 单卡 batch / 梯度累积 | 1 / 8，有效 batch=8 |
| 精度 / 上下文上限 | BF16 / 16,384 tokens |
| 模板 | qwen3，enable_thinking=false |
| Loss | 官方 LLaMA-Factory SFT 交叉熵，仅 assistant 标签参与 |
| 优化器 | AdamW fused 内核 |
| 保存 | 每 epoch 保存模型，保存最终 adapter；按 RR 设置不保存 optimizer 状态 |

训练后端从 RR 固定 commit `f464f083bd54cf6b1464a4347d7e9811a2c6de50` 的 LLaMA-Factory 导出，位于服务器 `backend/`；可移植源码归档是本地 `upstream/rr-lmf-f464f083.tar`。官方原 YAML 和启动脚本保存在 `upstream/`。当前采用单 GPU，不使用官方多 GPU DeepSpeed 启动方式。

`preflight.json` 已验证全部 3,870 条 SFT prompt 与原 OPSD 输入前缀逐 token 相同，标签与 EOS 完整；最长 prompt 8,101、最长目标 42、最长总序列 8,109 tokens，无需裁剪。

## 已通过的检查

最长 8 条真实样本完成 r=16 的两步 LoRA SFT，adapter 正常保存；峰值分配显存约 23.4 GiB。LoRA 合并也已使用 smoke adapter 在 CPU 上验证：词表保持一致、导出后全部 160 条验证 prompt 的 token IDs 与原模型一致。

早先全参数显存测试失败的记录保留作历史，不属于当前运行流程；当前正式训练已切换为 LoRA。

## 服务器上的自动顺序流程

项目：`/ai/yaodong/opsd-20261026/`。运行时：`.venv-opsd/bin/python`。

1. SFT 改为一轮。当前已运行的训练由 `limit_epoch.py` 接管，在首轮 checkpoint（预计 step-484）完整保存后停止，并选择该 checkpoint。训练器启动时计划为两轮、968 次更新；本次保留已有优化器和学习率调度轨迹，不从头重训。磁盘 `configs/lora.json` 和配置生成器均已改为一轮；未来全新启动会采用真正的一轮调度，与本次中途缩短的调度轨迹不同。
2. `merge.py` 将选中的第一轮 SFT adapter 合并为独立 BF16 4B 模型。验证词表不变，导出原始 Qwen chat template，保持 OPSD 输入编码不变。
3. 原封不动调用 `opsd.train`，读取 `configs/opsd_after_sft.toml`：20 steps，每 10 steps 保存。相对此前配置只更改 `model` 和 `output`。
4. `compare_formats.py` 比较 SFT 起点和新 OPSD step-10/20，再与旧原始 4B、旧 OPSD step-10/20 汇总对照。

Teacher 与 Student 都从合并后的同一 SFT 模型初始化。Teacher 冻结，Student 新建 LoRA；不叠加旧 OPSD adapter。不加入 GRPO/ATO，不比较 Teacher 有/无 Rubric 打分。

验证条件固定：同一 160 条 RR RAG val、同一候选顺序、同一 Student prompt、batch=16、seed=20261026；贪心与采样分开统计。推理输出上限保持 128 tokens，采样为 temperature=1、top-p=1、top-k=0，**不混入 256-token 截断对照**。

任何阶段报错会中止后续阶段并记录状态，不会自动换超参重新训练。当前没有提供优化器精确断点恢复；不要将模型 checkpoint 初始化误称为完整续训。

## 查看状态和结果

```bash
ssh yaodong-server
cd /ai/yaodong/opsd-20261026
cat opsd/sft/pipeline-status.json
tail -f server-logs/sft-pipeline-lora_sft-001.log
```

关键路径：

- SFT adapter：`opsd/sft/runs/rr-qwen3-4b-lora-r16-001/checkpoint-484/`（以 `selected-epoch-one.json` 中实际选择为准）
- 合并模型：`opsd/sft/models/rr-qwen3-4b-sft-r16-merged-001/`
- 新 OPSD checkpoint：`opsd/runs/post-sft-opsd-pilot-001/step-10/`、`step-20/`
- 最终对比：`opsd/sft/results/post-sft-format-001/REPORT.md`、`summary.json`
- 各阶段日志：`server-logs/sft-pipeline-*.log`

本地 TensorBoard：<http://127.0.0.1:16006/>，SFT run 为 `rr-sft-lora-r16-001/tensorboard`。隧道断开后：

```powershell
ssh -N -o ServerAliveInterval=30 -L 127.0.0.1:16006:127.0.0.1:6006 yaodong-server
```

SFT 的交叉熵与 OPSD surrogate loss 含义不同，不直接拼接比较。最终以固定验证集的格式合法率、正常 EOS 比例、重复率、输出长度为依据；这些不等于相关性排序准确率。

当前流程由 `launch_epoch_limit.py` 启动的边界控制器接管；原流程控制器已停止，SFT 子进程保留。**不要重复运行任何启动脚本**。状态文件与输出目录拒绝覆盖既有记录。服务器后台进程继续执行，关闭本地 TensorBoard 不会终止训练。
