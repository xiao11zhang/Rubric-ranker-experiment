# RAG OPSD：三组 Rubric 内容实验

本仓库保存已执行的三组实验代码：**正确 Rubric、部分同领域整份错配、去命名实体混合条件**。Teacher与Student均为Qwen3-4B，从同一轮LoRA SFT合并模型初始化；Teacher冻结、Student新建LoRA r=16。仅改变Teacher额外接收的Rubric，Student只看Query与候选文档。

此前仓库的RR/GRPO/ATO框架已从当前文件树移除，Git历史保留。本次三组实验仅做OPSD，不加入GRPO/ATO。代码、数据处理与已完成结果详见[完整实验总结](docs/EXPERIMENT_SUMMARY.md)。

## 实验分类

| 组别 | Teacher条件 | 训练覆盖 | 已完成评估 |
|---|---|---|---|
| 第一组 | 当前Query的正确原Rubric | 4000条，1000步，完整一轮 | dev200；全量开发SFT/300/500，取消1000全量评估 |
| 第二组 | 同领域整份错配，未知领域保留自己的Rubric | 同顺序前2000条，500步；1702错配/298回退 | dev200；未做全量开发评估 |
| 第五组 | description内四类命名实体替换，异常整份回退 | 同顺序前2000条，500步 | dev200；仅step200完成全量开发评估 |

第三、四组尚未实现，未补造结果。第五组实际1729条发生替换、120条未标注到目标实体而未替换、151条异常回退；剩余57条抽查经用户明确取消，不能称全量语义审核通过。

## 目录

```text
opsd/
  common.py, objective.py, train.py    原OPSD公共输入与目标
  sft/                               RR RAG LoRA SFT及合并
  full_run/                          第一组训练、保存、验证
  rubric_swap/                       第二组分类、映射、训练、报告
  rubric_swap_control/               第二组历史调度接管代码
  group05_deentity/                   第五组实体转换、回退、来源校验、训练
  dev_eval/                          支持证据开发数据与离线指标
  tests/, validation/                公共单测和早期格式诊断
docs/
  REPRODUCE.md                       数据、模型、运行顺序与迁移边界
  EXPERIMENT_SUMMARY.md               完整方案和实际结果
  history/                           原工作区历史说明
SOURCE_MANIFEST.json                 发布文件大小及SHA256
```

## 环境与快速检查

建议Linux、Python 3.11+。单卡运行需同时容纳两个4B模型和Student训练状态；第五组实际使用BF16。先单独安装适配本机CUDA的PyTorch，再安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r opsd/requirements.txt
python -m unittest opsd.tests.test_core opsd.dev_eval.test_metrics opsd.rubric_swap.test_rubrics opsd.rubric_swap.test_mixed opsd.group05_deentity.test_group05 opsd.group05_deentity.test_original_fallback -v
```

测试只验证实现，不替代真实数据审核或GPU训练验证。SFT后端LLaMA-Factory需另外从RR固定commit准备，**未包含后端源码归档、数据或模型**。

## 数据与模型准备

- [RR官方SFT数据](https://modelscope.cn/datasets/lwhlwh/rubricranker_sft_data/files)：RAG3889条，排除19条不兼容标签，实际3870条。
- [RR官方RL数据](https://modelscope.cn/datasets/lwhlwh/rubricranker_rl_data/files)：按官方划分筛选RAG，4000训练/160格式验证。
- [HotpotQA](https://hotpotqa.github.io/)和[MuSiQue](https://github.com/StonyBrookNLP/musique)validation：去重/证据映射检查后9814题，其中固定200题用于过程验证。
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)原模型及共同SFT checkpoint-484合并模型需自行准备。
- 第二组的正式领域标签/错配映射、第五组派生数据/来源manifest仍保留在原服务器，**未上传GitHub**。重新生成会形成新数据实例，不能宣称逐条复现原实验。

具体转换命令和文件清单见[复现说明](docs/REPRODUCE.md)。不要只下载代码就直接运行旧后台调度器。

## 共同训练与评测协议

纯OPSD采样token log-prob gap蒸馏，Teacher冻结，Student LoRA r16/alpha32，学习率1e-5，seed42，梯度累计4，最大输出128。模型自然选择可变长有序文档子集，不强制固定Top-k。没有Generator/Judge或outcome reward。

每100步在固定dev200验证。选点依次比较balanced-macro All-support@5、集合F1、较早step。9814题全量开发集包含dev200，不是独立最终测试集；最终测试未使用。

## 实际结果

下表为**相同dev200**、各组0–500步范围内选出的最佳点：

| 条件 | Step | All-support@5 | Recall@5 | 集合F1 |
|---|---:|---:|---:|---:|
| 共同SFT | 0 | 65.00% | 83.25% | 82.47% |
| 第一组正确Rubric | 300 | 75.50% | 89.04% | 83.76% |
| 第二组同领域错配混合 | 400 | 67.00% | 83.54% | 82.10% |
| 第五组去实体混合 | 200 | 68.00% | 85.25% | 82.95% |

全量开发9814题：SFT All-support@5为62.19%；第一组step300为72.22%、step500为71.55%；第五组step200为66.31%。第二组没有全量结果；第一组没有step200全量结果，因此不能声称已完成相同步数的三组全量对照。单种子、回退比例不同且第五组未完成抽查，限制见完整总结。

## 可移植性与安全边界

这些文件是已运行工程的源代码快照，配置保留原服务器绝对模型路径。新机器需改为自己的资源路径，并在修改全部配置/代码后重新生成冻结计划；不能把旧计划中的hash手改为“通过”。原共同SFT在首轮停止，但沿用先前两轮初始化的学习率调度，重新从头一轮SFT不保证完全相同轨迹。

各子目录README含历史调度记录，以本README、[复现说明](docs/REPRODUCE.md)和用户最终停止约束为准。特别是旧pipeline/full_validation可能默认评估已取消的checkpoint，勿直接执行。上传不启动训练、不修改OPSD算法，也不修改原服务器任何实验产物。

本仓库不包含.env、密钥、SSH配置、原始数据、标签/映射本体、权重、优化器、预测逐条日志或大型压缩包。只包含代码、配置、文档与聚合结果。原服务器资源需另行备份。

参考：[RubricRanker](https://github.com/8421BCD/RubricRanker)（f464f083bd54cf6b1464a4347d7e9811a2c6de50）、[AdaTutoRank](https://github.com/AdaTutoRank/AdaTutoRank)（7df929c42482dccd0181b29a22983356762a7b92）。本实现借鉴蒸馏思路，不宣称完整复现两篇论文。数据与模型遵循各自上游许可；未将其重新授权。
