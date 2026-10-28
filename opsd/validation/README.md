# 只读验证

此目录仅新增推理和统计工具，不修改 `train.py`、`objective.py`、`common.py`、`predict.py`、训练配置或 adapter 权重。

验证协议：官方 RR RAG val 全部 160 条；原始 4B、step-10、step-20；同一候选顺序、Student prompt 和 128-token 上限。分别运行贪心解码与 temperature=1/top-p=1/top-k=0 的采样。采样种子固定；采样结果仍只代表一组随机种子，不能据小幅变化宣称统计显著。

首个 batch=4 的验证因吞吐较低停止，仅保留部分结果于服务器 `results/server-001/`。用于最终比较的完整运行是 `results/server-002/`，batch=16；所有 checkpoint 使用同一批次划分。没有合并两轮不同 batch 的结果。

`run_checks.py` 不构造 optimizer，不调用 backward；使用 inference_mode。验证前后比较核心源码、训练配置、验证数据与两个 adapter 文件的 SHA-256。`summary.json` 完成后才表示全流程结束；`generation-summary.json` 是阶段性文件。

指标：格式合法率、正常 EOS、合法且有 EOS、重复 ID、语法错误、越界、选取全部候选、输出 token 数。`selected` 为所有 query 的平均选择数，无效输出计 0；不是有效输出条件下的均值。

用户已取消 Teacher 有/无 Rubric 的打分比较。本轮在进入该阶段前停止并保存生成结果，随后以 `--resume --skip-pi` 只续跑 checkpoint 输出对照；没有执行 PI 打分。

RR val 缺标准 qrels，本验证不报告相关性排序准确率、nDCG 或 RAG EM/F1。

服务器启动（新 run-id、新输出目录；不覆盖）：

```bash
cd /ai/yaodong/opsd-20261026
.venv-opsd/bin/python opsd/validation/launch.py --run-id 003 --batch-size 16 --skip-pi
```

完整结果可用 `export_tensorboard.py` 单独导出固定验证曲线，不写入旧训练事件目录。
