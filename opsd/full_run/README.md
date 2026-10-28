# RR RAG OPSD：完整一轮与开发验证

用户已授权执行本方案。新运行目录：`opsd/runs/rag-full-epoch-001/`。原20步预实验与原训练源码保留，不覆盖。

## 固定方案

- 起点：现有LoRA r=16 SFT一轮、step-484的合并Qwen3-4B模型。Teacher与Student从同一模型初始化；Teacher冻结，Student新建LoRA r=16、alpha=32。
- 数据：RR已准备的4,000条RAG训练记录；seed=42打乱，无放回遍历一次，每条产生一个on-policy response。
- 累积4条/次更新，共1,000次更新。学习率保持1e-5常数，AdamW与梯度裁剪保持原设置。
- 原 `opsd/train.py`、`objective.py`、`common.py` 均不修改；新入口只增加遍历、工程保存恢复和验证。单条response的采样、Teacher打分、反向传播，以及LoRA、AdamW、梯度裁剪调用通过AST与原实现逐项一致性测试。
- 保持正确Rubric PI；Student只见Query和候选。无新reward、ATO、GRPO、Teacher无Rubric对比、输出修复、数据过滤或静默截断。

## 验证及模型选择

- step0及每100steps：固定既有200题validation，HotpotQA与MuSiQue各100题。
- greedy、batch8、最多128输出tokens，原Student prompt与候选顺序；答案/支持标签/Rubric不进入模型。
- 主指标：两个数据集的balanced macro All-support@5；并记录support Recall/nDCG、集合P/R/F1、选择数、格式合法率、EOS及答案命中代理。
- 最佳checkpoint固定选择规则：先比较balanced macro All-support@5，再比较集合F1，仍相同选较早step。SFT step0也参与，旧20步预实验不参与新运行的最佳选择。
- 训练结束后，在原已审计的全量validation（HotpotQA 7,404 + MuSiQue 2,410，共9,814题）上比较SFT、step500、step1000及小验证集最佳checkpoint；重复步骤只评一次。
- 全量validation包含选模型的200题，因此仍是开发诊断，不称独立最终测试。最终测试集不读取、不推理、不参与模型或超参数选择。后续最终测试清单必须排除已使用开发问题及近重复。
- 验证后恢复Python、Torch CPU、全部CUDA随机状态及Student train/eval模式，避免验证消耗训练采样随机流。
- 不根据验证结果自动改学习率、提前停止或重新训练；异常会使流程停止并记录。

## 保存与恢复

step10、20及每100steps保存adapter、tokenizer、优化器状态、Python/Torch CPU/CUDA RNG、遍历顺序、游标和配置hash。写入临时目录并完成hash后才原子改名为完整checkpoint；不将部分保存当作可恢复点。

需要恢复时，先检查故障原因与运行进程，明确选择最新完整checkpoint：

```bash
cd /ai/yaodong/opsd-20261026
.venv-opsd/bin/python -m opsd.full_run.train --resume opsd/runs/rag-full-epoch-001/step-100
```

该命令仅为恢复示例，当前后台流程在运行时不要执行。恢复会归档checkpoint之后的未提交日志，回到该checkpoint的优化器、随机状态和数据游标；不承诺跨设备/软件版本的GPU逐bit一致。后续全量验证可独立运行 `python -m opsd.full_run.full_validation`，不要重复启动总流程。

## 查看进度

```bash
ssh yaodong-server
cd /ai/yaodong/opsd-20261026
cat opsd/full_run/pipeline-status.json
cat opsd/runs/rag-full-epoch-001/status.json
tail -f server-logs/rag-full-epoch-001-training.log
```

TensorBoard：本地隧道可用时 <http://127.0.0.1:16006/>，选择 `rag-full-epoch-001/tensorboard`。查看 `train/*` 和 `validation/{hotpot,musique,balanced_macro}/*`。surrogate loss不是结果reward，也不直接表示检索性能。

关键文件：

- `opsd/full_run/plan.json`：预检后冻结的完整计划、数据排列、输入/代码/模型hash与长度上限。
- `opsd/runs/rag-full-epoch-001/train.jsonl`：训练输出与数据覆盖轨迹。
- `small-validation-history.json`、`best-small-validation.json`：每100步的验证结果与最佳选择。
- `coverage.json`：训练结束时核验4,000条记录恰好覆盖一次。
- `full-validation-selection.json`：全量验证开始前固定的checkpoint清单。
- `full-validation-summary.json`、`REPORT.md`：全量开发评测完成后的结果。

已通过11项检查：原OPSD逐response更新及关键优化调用AST一致性、全量排列、最佳选择规则、验证随机状态隔离、CPU toy模型优化器/RNG恢复，以及6项证据映射/评分测试。它们不代替正式训练的稳定性与质量监测。

## 后续变更
用户已取消 step-1000 全量验证。当前由 skip_final_checkpoint.py 接管，在 step-500 完整保存后终止旧推理进程并汇总 SFT、step-300、step-500；step-1000 权重保留，不删除。原冻结训练计划保留为历史，执行变更记录在 full-validation-amendment.json。不要重复启动旧 full_validation 入口。
