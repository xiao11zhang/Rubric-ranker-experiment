# 当前服务器部署

2026-10-26，SSH 别名 `yaodong-server`。

- 持久化项目：`/ai/yaodong/opsd-20261026/`
- Python：`/ai/yaodong/opsd-20261026/.venv-opsd/bin/python`
- 模型：`/ai/yaodong/selector/models/Qwen3-4B`
- GPU：单张 A100 80GB，Teacher/Student 都使用 `cuda:0`
- 环境复用旧 SFT 包目录的只读路径，新增 TensorBoard/tomli 安装在本项目虚拟环境，不修改旧环境。
- 服务器外部 DNS 暂不可用；当前使用本地模型和离线依赖运行。

本地曲线：<http://127.0.0.1:16006/>。TensorBoard 在服务器只监听 `127.0.0.1:6006`。

SSH 隧道关闭后，可在 Windows 终端重新启动并保持窗口打开：

```powershell
ssh -N -L 127.0.0.1:16006:127.0.0.1:6006 yaodong-server
```

查看服务器训练日志：

```bash
ssh yaodong-server
cd /ai/yaodong/opsd-20261026
tail -f server-logs/pilot.log
```

限步启动器会将日志和 PID 存在 `server-logs/`；重复执行时拒绝覆盖已有日志。`server-smoke.toml` 是 2-step 验证；`server-pilot.toml` 是从相同基础模型重新初始化的 20-step pilot，不接续 smoke 权重。没有启动全量 epoch 或多条件实验。

当前没有 reward 曲线；请查看 `train/loss`、`train/teacher_student_logp_gap`、合法率、EOS、输出长度和梯度范数。loss 可以为负，不是交叉熵，不能把其下降直接解释为性能提升。

20-step pilot 已于 2026-10-26 正常完成并自动停止。共采样 80 个 rollout，格式合法率 82.5%，EOS 结束率 87.5%，loss/梯度均为有限值；训练循环耗时约 405 秒（不含模型加载和全数据长度预检）。最终 adapter 位于 `opsd/runs/server-pilot-001/step-20/`。这些数字不代表检索质量评估。

本地已同步日志到 `opsd/dist/server-pilot-train.jsonl`，汇总到 `opsd/dist/server-pilot-summary.json`。

重载验证只用两条真实 val query 检查能否推理，不是完整质量评测。RR val 缺标准 qrels，指标工具会明确输出 `ranking_metrics_available=false`。
