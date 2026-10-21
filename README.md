# Setwise Baselines

模块化的 RR-SFT（B0）、GRPO（B1）、GRPO+ATD/ATO（B2）实验代码。
本仓库维护任务适配层；评分直接执行固定快照中的 RR 原函数，GPU 训练使用固定的 ATO/verl checkout。
官方源码、prompt、许可证与 commit/blob 哈希保存在 `src/setwise/_upstream/`。
原件不修改，必要改编逐项列在 [上游对齐说明](docs/UPSTREAM_ALIGNMENT.md)。
**当前状态：本地协议/数值检查已实现；GPU、Hydra 配置兼容性与真实模型训练尚未验证。**
默认配置是待校准的单卡 pilot，不能直接称为论文正式实验配置。

## 本地检查

Python 3.11+，核心模块无第三方依赖：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m setwise.cli check configs/b1.toml
python -m setwise.cli compare configs/b1.toml configs/b2.toml
python -m setwise.cli verify-upstream
python -m setwise.cli train configs/b2.toml --ato-root /opt/AdaTutoRank/ATO
```

`train` 默认只输出命令计划，不加载模型，不访问 Judge。增加 `--execute` 才启动训练。
目录内所有相对数据/输出路径，以调用命令时的工作目录为准。建议始终从项目根目录调用。

## 数据合同

每行一个 JSON 对象：

```json
{"id":"train-001","query":"Which city ...?","split":"train","docs":[{"position":1,"document_id":"wiki:123","content":"A frozen passage"}],"rubrics":{"set_level_rubrics":{"Relevance":[{"description":"Contains supporting evidence","weight":3}]},"doc_level_rubrics":{}},"silver_ids":[1]}
```

`position` 连续从 1 开始；全局 ID 唯一；正文提前冻结。启用 Source Authority/Timeliness
时，每个候选须包含同名的 [0,1] 预评分。`silver_ids` 为已核查的共同 SFT 标签。
正式数据还须提供 `prompt` 字段（从已核查的官方数据/统一预处理结果保留的 system/user 消息列表）。
示例只有结构示意，不能直接作为正式训练数据。除 `split=debug` fixture 外，
转换器不再自动生成自写 Student prompt；`prepare` 和 `export-sft` 会原样保留输入 prompt。
数据审计时须确认这些 prompt 没有答案、Rubric、gold evidence 泄漏。
代码不下载语料，不自动生成付费标签，也不自动生成正式 query-specific Rubric。

```bash
pip install -e '.[data]'
setwise prepare data/train.jsonl data/train.parquet --expected-k 30
setwise prepare data/val.jsonl data/val.parquet --expected-k 30
setwise export-sft data/sft.jsonl data/shared-sft.json
```

转换器输出各自的 `.manifest.json`。正式运行使用总 manifest：
`{"train": <train.manifest.json 内容>, "val": <val.manifest.json 内容>}`。
将总 manifest 文件的 SHA256 写入配置；启动时核对实际 parquet hash 与 query ID 不相交。
相似 query/答案重叠需要额外数据审计，本检查只识别相同 query ID。

## 共同 SFT

导出的 `messages` 数据可注册为 LLaMA-Factory ShareGPT 数据：

```json
{"shared_sft":{"file_name":"shared-sft.json","formatting":"sharegpt","columns":{"messages":"messages"},"tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}}
```

参考 RR 设置：full SFT、qwen3 template、enable_thinking=false、BF16、学习率 5e-6、
cosine、warmup_ratio=0.1、2 epochs、train_on_prompt=false。这些是起始建议，
并非已经在此环境验证。每个模型规模各训练一次共同 SFT；B0 直接评测它，B1/B2 从它分叉。

## GPU 训练接入

在 Linux A100 环境准备并固定 AdaTutoRank/ATO（含 SGLang）环境、安装本包。
本仓库不自动克隆或安装它。`runtime.backend_commit` 已锁为
`7df929c42482dccd0181b29a22983356762a7b92`；启动前要求该上游 checkout 无本地修改。
目前适配基于固定 commit 的静态审计，没有经过真实 GPU 集成测试；先执行上游 Hydra
配置解析检查与单步 smoke test，发现字段不兼容应更新适配层并重新锁定 commit。

1. 填写模型 revision、SFT checkpoint 路径和目录 hash；目录 hash 使用
   `setwise.backend.checkpoint_hash`，所有文件均计入。
2. 固定 train/val 数据及总 manifest hash，锁定真实 Judge model。
3. 配置 `OPENAI_API_KEY` 等环境变量。凭证不写进配置或日志。
4. B1/B2 `compare` 必须通过；仅允许 method、实验名、输出目录不同。
5. `setwise train configs/b1.toml --ato-root /opt/AdaTutoRank/ATO --execute`。
6. 从相同 SFT checkpoint 启动 B2。正式训练前先用极少 step 验证。

两组使用同一 GRPO、KL、采样和 rollout correction；B2 使用发布形式：
`0.7 * A_GRPO + 0.3 * mask * (logp_teacher - logp_old)`。
Teacher 固定，selector/reflector 使用当前策略；不使用完整词表 KL 蒸馏。
RR reward 统一为 [0,1]，hint 阈值为 0.3/0.6；适配层使用 0.3 的下一个浮点数作为上游
通用分档的下阈值，使恰好 0.3 保持低档，并通过配置传递到所有 Ray worker。
关闭 ATD 的一致性检查应设 GRPO 权重为 **1**，不是只将蒸馏信号清零。

selector/reflector YAML 与官方原件逐字节相同；hint YAML 仅替换两处尺度/维度描述，
其余内容保留官方模板。原始九维版本另外保存，可以逐项核对。
GPU 前仍需检查实际 Teacher prompt 和截断行为。当前实现不声称完整复现论文。
`base_model`/revision 是 SFT 来源记录；RL 实际加载 `sft_checkpoint`，不会自动下载 base model。

## Reward 与观察

默认 `rr_compatible` 解析：去除越界 ID、去重；无 ID=-1；清理后为空=0。
严格合法率另记，非法格式排除 ATD；正式 baseline 保留 RR 宽松解析，不再提供 strict 替代奖励。
评分 prompt 原样使用 RR 官方版本，Judge 返回空格分隔的整数，解析和聚合均直接调用 RR 原函数。
包括按官方顺序展开 Rubric、局部重编号选中文档、整数解析、文档预评分忽略 None 等行为。
真实训练数据的预评分仍要求完整；reward 适配器本身保持上游缺失值行为，以便差分验证。

`judge_failure_policy="abort"` 是明确记录的工程改编：RR 重试耗尽后中止该批次。
切换为 `"upstream_drop"` 时保留官方的失败维度丢弃/重新加权行为，并记录 `judge_incomplete`。
两组必须使用相同策略。请求次数默认与 RR 一致为 4，API 请求不额外设置 temperature。
API cache 仅保存通过官方解析的原始响应，不保存密钥；不保证并发相同请求只收费一次。

TensorBoard 由 verl 记录训练指标，上游 ATO 还记录 joint advantage、hint 和 rollout。
自定义 reward 返回 score、format_valid、num_selected、集合 rubric_scores、Judge 调用/缓存状态、
解析失败次数和 judge_incomplete。reward 本身为 RR 原函数的结果。
需在 GPU smoke test 确认对应版本对这些自定义字段的聚合和展示。
重点观察固定验证 reward、格式合法率、选集大小、组内 reward 方差、KL、entropy、clip fraction，
以及 B2 hint/fallback 比例和 ATD/GRPO 量级。训练 reward 上涨不等于下游回答效果提高。

已有 checkpoint 的预测保存为 JSONL `{"id":"...","output":"[1] [3]"}`：

```bash
setwise evaluate configs/b0.toml data/val.jsonl predictions.jsonl results.jsonl --allow-api
```

此命令评估预测的选集 reward；不负责加载 Reranker 或调用下游 Generator。
Generator 推理、EM/F1、多卡成本分析是后续模块，目前没有实现，不能声称已有端到端评测。
`configs/debug.toml` 的 mock reward 仅用于离线 fixture；不代表真实质量，GPU 正式入口禁止它。

## 来源

- RR：https://github.com/8421BCD/RubricRanker
- ATO：https://github.com/AdaTutoRank/AdaTutoRank

本仓库仅保存必要的上游源文件与模板快照，没有复制完整训练项目。
源码原有版权头保留，上游 Apache-2.0 许可证随快照附带；来源与改动见对齐说明。
