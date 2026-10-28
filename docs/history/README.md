# RR RAG → Qwen3-4B OPSD

> 2026-10-26 当前流程已更新为：先使用 RR RAG 官方标签做 **r=16 LoRA SFT**，再接原 OPSD 的 10/20-step 对照。运行与数据说明见 [sft/README.md](sft/README.md)。下文原始 4B 直接 OPSD 流程保留作为旧对照。

独立于旧 `src/setwise`、GRPO 和 ATO 的最小实验目录。Teacher 和 Student 从同一 Qwen3-4B checkpoint 初始化；Teacher 全程冻结并额外看到 RR 的 Query-specific Rubric；Student 只看到 Query、候选文档和输出格式要求。

本版本实现代码与数据准备，不代表已经通过服务器 GPU 训练。默认 LoRA 更新 Student、双 GPU 单进程；不是多进程分布式训练，也不自动启动长期实验。配置中的 20 steps 只供流程检查，不是最终实验方案。

## 文件

```text
opsd/
  common.py          数据合同、独立 Student/Teacher prompt、输出校验
  prepare.py         RR Parquet → RAG-only JSONL + manifest
  objective.py       response token 对齐、sampled-token 蒸馏目标
  train.py           Student rollout → Teacher 重打分 → Student 更新
  predict.py         原始 Student / LoRA Student 批量预测
  evaluate.py        格式统计；有外部 qrels 时才计算排序指标
  configs/pilot.toml 可编辑的最小运行配置
  tests/             输入隔离、格式、划分、梯度与 token 对齐检查
  data/rr_rag/       本地已准备的 4,000 train + 160 val（Git 忽略）
```

## 方法边界

对每个 micro-batch（当前为一个 Query），Student 在无 Rubric 输入 x 下生成 y，然后冻结 Teacher 在 (x, Rubric) 下对**完全相同的 response token IDs 与前缀**计算概率。Teacher 和 Student 使用同一 tokenizer。

```text
A_t = stop_gradient(log p_teacher(y_t | x, Rubric, y_<t)
                    - log p_student_old(y_t | x, y_<t))
loss = -mean_t(A_t * log p_student(y_t | x, y_<t))
```

采用 sampled-token reverse-KL 的 policy-gradient surrogate，类似 ADA 的 token log-prob gap 蒸馏分支；没有 outcome reward、GRPO 组优势、奖励路由、selector 或 reflector。不是全词表 KL，也不宣称完整复现 ADA。使用当前 Student 生成，参数在梯度累积期间不更新；重打分的 Student 概率 detach 后作为 old probability，因此不需要另一份 old model。

为让采样分布与重打分概率一致，第一版固定 temperature=1、top_p=1、top_k=0，不对采样分布截断；两模型都关闭 dropout、关闭 thinking。EOS 参与 loss；单序列没有 padding loss。达到生成上限的序列按截断轨迹训练，日志记录 `ended_eos=false`。无效格式仍参与蒸馏，日志单独记录；没有生成标签修复、格式 reward 或偷偷补齐文档。

默认 `use_lora=true` 是节省显存的工程起点，并非用户最终选定训练策略；可设 false 做全参数更新，但显存需求会显著提高。Teacher 不更新。未来研究 moving teacher 时应另建明确配置。

默认 `top_k=0` 沿用 RR 可变长度有序子集任务。若确定固定 Top-k，将其设为正整数，并在预测/评测传相同 `--top-k`；这一变更改变任务，不能混在同一结果表里当作仅改变 Rubric 内容。

`pi_mode=correct` 注入原始 Rubric；`none` 为无 PI 对照。跨 Query、无关内容的 donor 生成尚未实现，避免在基本流程通过前锁定替换方案。

## 上传与环境

可以用 `python -m opsd.package --include-data --output opsd/dist/opsd-server.zip` 打包上传。压缩包只含代码、配置、说明、测试和导出数据，排除本地测试依赖、模型与运行日志。服务器解压后从包含 `opsd/` 的目录执行命令。

把整个 `opsd/` 文件夹上传到服务器某个项目目录，例如 `/workspace/reranker/opsd/`，**从其父目录 `/workspace/reranker/` 执行以下命令**。不需要上传旧 ATO 工程或 `docs/research/python_deps`。

本地 `opsd/data/rr_rag/` 已导出真实 RR RAG 数据；它被 Git 忽略，所以 `git clone` 不会携带数据。直接上传文件夹时显式包含 `data/rr_rag/train.jsonl`、`val.jsonl`、`manifest.json`。不必上传原始混合场景 Parquet。

服务器使用 Python 3.10+（3.10 通过 tomli 读取配置），先按服务器 CUDA/驱动安装兼容的 PyTorch，再安装其余依赖。这里的版本范围不是经过 GPU 验证的锁文件；服务器验证成功后保存实际版本。

```bash
python -m venv .venv
source .venv/bin/activate
# 先安装与你的 CUDA 环境兼容的 PyTorch，参见 https://pytorch.org/get-started/locally/
python -m pip install -r opsd/requirements.txt
python -m pip freeze > opsd/environment.lock.txt
python -m unittest discover -s opsd/tests -v
nvidia-smi
```

两张 GPU 时默认 Student `cuda:0`、Teacher `cuda:1`。只有一张 GPU 可把两个 device 都设为 `cuda:0`，但本实现不提供自动 offload 或显存保证。长候选输入和完整 logits 会占用较多显存；先做 1–2 steps 的独立 smoke 配置。不要使用 `torchrun` 或 `accelerate launch`，本入口是单进程。

编辑 `configs/pilot.toml`：优先将 `model` 改为服务器同一个本地 Qwen3-4B 基础 checkpoint 路径，或使用模型 ID 并把 revision 固定到 commit。不要把已经训练过 ATO/RL 的模型偷偷当作基础起点。Teacher/Student 初始权重相同，LoRA 初始化为不改变基础模型输出。

[Qwen3-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B)提供模型使用及 `enable_thinking=False` 的说明。本代码显式使用该模板选项。

## 数据准备（已包含导出数据时可跳过）

原始文件来自 [RR 官方 RL 数据](https://modelscope.cn/datasets/lwhlwh/rubricranker_rl_data/files)。分别下载 train.parquet 与 val.parquet，不重新随机划分。

```bash
python -m opsd.prepare \
  --train /path/to/rr/train.parquet \
  --val /path/to/rr/val.parquet \
  --output opsd/data/rr_rag
```

只筛选 `scenario=rag`，保留原始候选顺序、正文、Rubric 描述和权重，不传递原始 prompt、thinking 或 ground_truth 给 Student。数据包含 source index、candidate hash、rubric hash。输出目录必须不存在。manifest 记录源文件与导出文件 SHA-256。只做规范化 query 精确去重，不能替代语义近重复与原始 benchmark ID 泄漏审计。

## 训练

```bash
# 只核对配置、数据和划分；不加载模型
python -m opsd.train --config opsd/configs/pilot.toml

# 检查设备/路径后实际执行
python -m opsd.train --config opsd/configs/pilot.toml --execute
```

执行时先用 tokenizer 对所有 train/val 的 Student/Teacher 输入做长度检查，再加载模型。超过 `max_prompt_tokens` 会报错，不静默过滤或截断候选/Rubric。这个数不是任何 GPU 的可运行保证；如失败，明确调整长度预算或另行制定文档裁剪协议。总序列长度还必须落在模型 context limit 内。

输出包括 `config.json`、`metadata.json`（模型 commit、数据 hash、依赖版本、最大 prompt 长度）、`train.jsonl`（sample ID、原始输出、合法性、EOS、loss/log-prob gap、梯度范数）、`step-N/` checkpoint。LoRA 模式保存 adapter；全参数模式保存模型。

输出目录必须是新的。**暂不支持 optimizer/RNG 精确断点续训**；checkpoint 用于预测或作为后续初始化，不能宣称已恢复完全相同训练轨迹。验证集目前用于数据/长度检查，不自动根据验证分数选模型。

## 预测与比较

先记录未蒸馏 4B 的基线，再在同一验证集、同一输出协议下预测蒸馏模型。

```bash
python -m opsd.predict --model /path/to/Qwen3-4B \
  --data opsd/data/rr_rag/val.jsonl --output opsd/runs/base-val.jsonl

python -m opsd.predict --model /path/to/Qwen3-4B \
  --adapter opsd/runs/pilot-correct/step-20 \
  --data opsd/data/rr_rag/val.jsonl --output opsd/runs/distilled-val.jsonl

python -m opsd.evaluate --data opsd/data/rr_rag/val.jsonl \
  --predictions opsd/runs/distilled-val.jsonl --output opsd/runs/format-metrics.json
```

预测对 Student 不注入任何 Rubric。全参数训练时将 `--model` 指向 step-N，不传 `--adapter`。原始模型/adapter 的基础权重、revision 必须对应；使用模型 ID 时传相同 `--revision`。

RR val 没有标准 qrels，因此默认评测只报告格式合法率、选择数量，**不把这些指标称为排序质量提升**。后续得到可对齐的 qrels 后，使用独立 JSONL：

```json
{"id":"rr:val:SOURCE_INDEX","candidate_hash":"MATCHING_HASH","grades":{"1":2,"3":1}}
```

grade 必须是真实检索相关性标注，位置从 1 开始，与当前候选池一致；不能从 Rubric 或 Teacher 选择结果冒充人工标签。`--qrels` 会启用 `--metric-k` 对应的 nDCG 和候选池内 Recall，未列文档按 supplied-qrels 的 0 相关约定处理；无正例 query 不参与这两个均值，无效输出按空列表计零。候选池 Recall 不是整个 corpus 的 Recall。代码不调用在线 Judge，不生成 RAG 答案；独立排序/RAG 质量评测仍是后续资源准备项。

## 本地查看实时曲线

训练会同时写入 JSONL 和 TensorBoard events。服务器启动：

```bash
tensorboard --logdir opsd/runs --host 127.0.0.1 --port 6006
```

本地另开终端（SERVER 替换为 SSH 配置别名）：

```bash
ssh -N -L 16006:127.0.0.1:6006 SERVER
```

然后浏览器访问 `http://127.0.0.1:16006`。只在回环地址提供服务。曲线包含 loss、Teacher–Student log-prob gap、格式合法率、EOS 比例、输出长度、梯度范数；当前纯 OPSD 无 reward，不能把 log-prob gap 或 loss 叫作 reward，也不能从 loss 下降直接推断排序性能改善。

## 本地验证与服务器验收

已使用真实 RR Parquet 导出 4,000 train / 160 val，并通过不加载模型的训练计划检查。单元测试会在有 PyTorch 时检查 loss 梯度方向、Teacher 无梯度和 response-token 对齐；缺 PyTorch 时明确 skip，不能当作全部训练检查通过。

服务器 smoke 必须进一步核查：实际 tokenizer 长度；Teacher 无梯度且参数不更新；Student 有非零梯度并更新；无 PI 初始条件下两模型 log-prob 接近；保存的 adapter 能被 `predict.py` 重载；GPU 峰值显存及 step 时间。未通过前不扩大步数或将 loss 下降当作检索性能提升。
