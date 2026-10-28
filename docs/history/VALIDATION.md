# 本地验证记录

日期：2026-10-26。

- 从完整官方 RR Parquet 成功导出 RAG-only train=4,000、val=160。
- 源文件与导出 JSONL SHA-256 保存在 `data/rr_rag/manifest.json`。
- 训练入口不带 `--execute` 完成配置、数据完整性及 train/val query 精确交集检查。
- 5 项 unittest 全部通过，无 skip：PI 输入隔离、输出格式、候选顺序 hash、划分交集、response token 对齐与 loss 梯度方向/Teacher 无梯度。
- 已用 CPU PyTorch 执行数学测试；Python 源文件编译检查通过。
- 尚未下载 Qwen3-4B 权重或运行 GPU 模型；真实 Transformers/PEFT 集成、训练显存、checkpoint 重载和检索性能仍需服务器 smoke 验证。

以上是初次本地检查。随后服务器验证：单张 A100 80GB，PyTorch 2.6.0+cu124、Transformers 4.52.4、PEFT 0.15.2；复用服务器已有 Qwen3-4B，未重新下载。2-step smoke 已完成，输出均合法且有 EOS，第一个 step 的 grad norm 为 149.44；保存的 252 个 LoRA B 张量中存在非零更新（最大范数约 0.006586）。完整数据 Student/Teacher 最大 prompt 长度分别为 8,181/8,470 tokens。TensorBoard 真实标量已通过本地 SSH 隧道读取。

本地临时 CPU 依赖在 `.test-deps/`，不包含在上传包中。

服务器后续验证：smoke adapter 重载后在两条真实 val query 上可正常生成合法 ID 序列。随后独立 20-step pilot 完成，80 个 rollout，格式合法率 82.5%、EOS 结束率 87.5%，全部 loss/梯度范数有限，最终 step-20 adapter 已保存。尚未进行独立检索质量评测。
