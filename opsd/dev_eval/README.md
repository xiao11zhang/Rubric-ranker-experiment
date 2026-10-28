# 独立 RAG 开发评测（不依赖 Generator）

当前实验固定 SFT 一轮的 checkpoint-484 合并模型，比较它与已有 OPSD step-10、step-20。仅推理与离线评分，不训练、不修改现有 OPSD 算法，也不根据结果自动更改模型或超参数。

## 可用资源与用途

| 数据 | 当前核查 | 本阶段处理 |
|---|---|---|
| RR RL RAG val | 160条，Query/候选/Rubric；无答案及人工相关性/证据标签 | 保留为原格式诊断，不据此声称排序提升 |
| HotpotQA distractor validation | 7,405条；答案、候选标题/句子、supporting_facts 的标题与句号 | 人工支持证据开发评测 |
| MuSiQue validation | 2,417条；答案/别名、候选段落、is_supporting、分解问题的paragraph_support_idx | 人工支持证据开发评测 |
| SetwiseEvalKit short | 已有2,061条，答案与候选，但未确认原始官方split和完整人工证据映射 | 本轮不推理、不作为独立开发集 |
| ADA BM25候选包 | 既有审计记录含多种test与2Wiki dev；本轮未读取文件本体 | test保留；2Wiki需后续核查证据映射，不混入本轮 |
| BEIR等检索基准 | 可提供qrels，但需选择独立dev及固定候选，不带现成同Query Rubric | 本轮不下载、不使用test做调参 |

来源：

- [HotpotQA 官方项目](https://hotpotqa.github.io/)；[官方维护的 Hugging Face 数据](https://huggingface.co/datasets/hotpotqa/hotpot_qa)，配置distractor，split validation。
- [MuSiQue 作者仓库](https://github.com/StonyBrookNLP/musique)；服务器下载脚本实际使用的[数据镜像](https://huggingface.co/datasets/dgslibisey/MuSiQue)，split validation，答案可用部分。
- 既有服务器下载脚本：`/ai/yaodong/selector/scripts/dl_closed_pool.sh`。本次复用原有Parquet，按字节SHA256固定本地快照；未声称恢复了下载时的远端revision。

## 数据审计与冻结

`prepare.py`读取完整两份validation，逐题核查证据映射，并针对RR OPSD train 4,000题、RR val 160题、实际SFT 3,870题及保守加入的全部官方RAG SFT 3,889题排查问题重叠。

规范化：Unicode NFKC、casefold、单词/数字token与空白统一。排除规范化精确交集，以及最少5个不同token时Jaccard>=0.8的词面近重复。不能保证排除语义改写、相关实体问题或基础模型预训练污染。

本次全量审计：

- HotpotQA：1题支持句号超出候选句子范围，排除；剩7,404题。
- MuSiQue：7题与前面开发问题规范化重复，排除；剩2,410题。
- 未发现与上述SFT/RR训练/RR验证数据达到排除阈值的问题重叠。

首轮固定从每个数据集的合格问题按 `SHA256("rag-dev-v1-20261026" + source_id)` 排序取100题，共200题。抽样在任何模型预测前完成，不依赖模型表现；不是全量开发集。全量合格数据保存在独立文件，未将剩余开发数据改称最终测试集。

`data/dev-v1/protocol.json`记录源文件hash、冻结问题hash、排除数、候选数、问题类型、协议和原训练源码hash。抽样内HotpotQA为79 bridge/21 comparison；MuSiQue为54两跳、29三跳、17四跳。不能将每个子类型很小样本的差异视为稳定结论。

## 评测协议

- 使用原 `opsd.common.encode_prompt` 的Student输入；只提供Query及带标题文档正文。答案、别名、支持标签、问题分解和Rubric均不进入输入。
- 原官方封闭候选池顺序保留，不额外检索或扩增。HotpotQA抽样99题各10篇、1题4篇；MuSiQue抽样99题各20篇、1题19篇。
- 固定现有SFT模型及两个adapter，冻结参数；BF16、greedy、batch=8、max_new_tokens=128。不采样，不根据结果调整上限。
- 保留模型自然输出的可变长子集，不强迫Top-k、不补足未输出文档。格式非法整条计空，不去重或修复。
- 主指标：**All-support@5**，前至多5篇所选文档是否包含全部人工标注支持段落。
- 辅指标：support Recall/nDCG@2/5/10，所选完整集合的证据P/R/F1、完整证据覆盖率、选择数量、格式合法率。
- support-nDCG基于人工支持段落二值标注；非支持候选只按“未标支持”计0，不据此声称它们是穷尽判定的真正不相关文档。
- 答案别名完整token序列命中只作辅助代理，不跨文档拼接；排除yes/no及单字符别名，单独报告有效问题数。不替代中间桥接证据覆盖。
- 固定基线：候选原顺序、池内BM25（k1=1.2、b=.75），以及支持段落优先的oracle上限。BM25仅在当前候选池计算词频，不等同原始全库BM25检索。
- 同题配对差值和5,000次bootstrap 95%区间；合并结果按两个数据集分层重采样。区间不反映训练随机性。

这些指标衡量封闭候选池内证据选择，不等同开放检索的全库Recall，也不等同RR论文的Generator答案EM。最终测试集不读取、不推理、不参与模型/超参数选择。最终测试清单尚未在本轮锁定；后续确定最终测试时，必须排除本开发集的原始问题ID及近重复，不能仅因发布包名称不同就认为独立。

## 文件与运行

```bash
cd /ai/yaodong/opsd-20261026
.venv-opsd/bin/python -m unittest opsd.dev_eval.test_metrics
# prepare已执行，data/dev-v1已冻结，禁止覆盖
# launch已执行，禁止重复启动
cat opsd/dev_eval/results/dev-v1/status.json
tail -f server-logs/dev-eval-inference-001.log
```

`run.py`会在开始和结束核对原训练源码、配置、SFT模型与两个adapter的hash。各checkpoint预测逐题保存，便于复查证据漏选。报告与结果路径：`results/dev-v1/REPORT.md`、`summary.json`、`per-query-metrics.json`、`integrity.json`。
