# RAG Reranker：Rubric 内容对 OPSD 蒸馏的影响——实验总结

整理日期：2026-10-27（北京时间）。本报告汇总已完成实验，不代表论文最终结论；可直接导入 Notion。数值依据服务器保存的配置、实际训练覆盖、dev200历史与全量开发结果整理，没有新增训练或评测。

## 一、研究目标与当前结论

研究问题：在 Reranker 的 On-Policy Self-Distillation（OPSD）中，Teacher 额外看到的 Rubric 内容，如何影响只看 Query 和候选文档的 Student？本阶段仅研究标准 RAG 封闭候选池内的文档重排序/证据选择，不开展 Deep Search 或 Agent 实验。

当前 dev200 上，正确 Rubric 组的主要证据覆盖指标最好，去实体混合组其次，同领域错配混合组较低。第五组 step-200 在9814题全量开发集上的 All-support@5 为66.31%，相对共同SFT起点提高约4.13个百分点（按未舍入原始值计算）。这些结果尚不能证明 Rubric 实体内容的独立因果作用，也不能证明正确PI在所有任务上优于无关PI。

## 二、实验分类及完成情况

| 类别 | 条件与目的 | 实际完成 | 结果定位 |
|---|---|---|---|
| 前期诊断 | 原始4B直接OPSD，与LoRA SFT后OPSD比较输出格式 | SFT及10/20步试跑完成 | 格式稳定性诊断，不作为正式排序质量结论 |
| 第一组 | Teacher使用当前Query的正确原Rubric | 1000步、4000条，完整遍历一次；dev200至1000步 | 正确Rubric基线；跨组主比较限定0–500步 |
| 第二组 | 自动领域内整份Rubric一对一交换；不确定标签保留原Rubric | 500步、2000条、dev200完成 | 部分同领域错配混合条件 |
| 第三组 | embedding语义近邻Rubric交换 | 未实现、未训练 | 只有方案，不填结果 |
| 第四组 | 题型通用Rubric模板 | 未实现、未训练 | 只有方案，不填结果 |
| 第五组 | Rubric description去四类命名实体，异常整份回退 | 500步、2000条、dev200完成；step-200全量开发完成 | 去实体与原文保留混合条件 |

注意：第一组训练到1000步，与用户后来取消的“step-1000全量开发评估”是两件事。训练日志确认其4000条各使用一次；step-1000全量开发评估未进行。第二组和第五组只使用相同训练顺序的前2000条，不称为完整4000条的一轮。

## 三、数据集的选择与处理

### 3.1 RR官方SFT数据：用于共同起点

- 官方文件共10602条，其中RAG 3889条、Deep Research 6713条；仅选RAG。
- 排除19条与当前非空合法ID列表合同不兼容的标签：17条`[]`、1条字符串`None`、1条`[0]`。不修补标签、不改写答案，实际SFT为3870条。
- 使用官方silver文档选择标签，候选正文和顺序保留。输入模板对齐本项目Student prompt，不给Student添加Query专属Rubric；不是逐字复现RR原混合场景SFT。
- Qwen3-4B做LoRA SFT，r=16、alpha=32、dropout=0，学习率1e-4、有效batch=8、BF16、上下文上限16384；选取首轮checkpoint-484后合并。
- 复现注意：本次训练最初按两轮调度启动，随后在首轮checkpoint保存后停止，保留已运行的两轮cosine调度轨迹。不能简单等同于从头设置一轮调度的新训练；各组实际复用同一合并模型。
- SFT标签是silver，不当作人工相关性真值。实际SFT问题与RR val 160题归一化精确交集为0；后续独立开发准备又做更严格的问题去重检查。

### 3.2 RR官方RL/RAG数据：用于OPSD训练

- 保留官方train/val划分，按`scenario=rag`筛选：训练4000条、验证160条。
- 每条保留Query、候选文档列表、原始Rubric及候选/Rubric哈希，转换成JSONL；不重新生成Rubric、不重排候选。
- RR Rubric与原Query逐条对应。第一组直接使用该对应关系；第二、第五组只改变Teacher接收的Rubric内容。
- 此包没有可直接支撑本项目证据排序评分的答案/人工证据标注。160条RR val用于格式诊断，不以其合法率代替排序质量。
- 训练顺序以固定种子shuffle后冻结，实际日志校验。第二/第五组采用第一组实际前500步的同一2000条、同一顺序，而非另抽2000条。

### 3.3 独立开发评测：HotpotQA与MuSiQue

| 数据 | 原始validation | 处理 | 最终合格开发题数 |
|---|---:|---|---:|
| HotpotQA distractor | 7405 | 排除1题支持句号超出候选句范围 | 7404 |
| MuSiQue | 2417 | 排除7题与前面开发问题规范化重复 | 2410 |
| 合计 | 9822 | 固定问题与候选、证据映射 | 9814 |

HotpotQA使用`supporting_facts`的标题/句号映射到候选段落；MuSiQue使用支持段落标记及分解问题支持索引，答案/别名只用于辅助离线指标。答案、支持标签、问题分解均不进入Student输入。

开发数据针对RR train 4000、RR val 160、实际SFT 3870及保守加入的全部官方RAG SFT 3889题检查重叠：Unicode NFKC、casefold和token规范化，排除精确重合及满足最低token要求的Jaccard≥0.8词面近重复。未发现与这些训练/验证问题达到排除阈值的重叠；不能保证排除语义改写或预训练污染。

dev200在模型预测前固定：每个数据集按`SHA256("rag-dev-v1-20261026" + source_id)`排序取100题，共200题。全量开发集9814题包含这200题，因此不是独立最终测试集。正式各组未使用最终测试集进行选点或评估。

BEIR、ADA候选包及其他基准没有进入本次正式结果表；本次不是全库检索实验，也不是RR论文Generator答案EM的复现。

## 四、共同训练方案与算法边界

Teacher和Student均为Qwen3-4B，从相同LoRA SFT checkpoint-484合并模型初始化。Teacher冻结，Student重新建立LoRA，而非在其他实验组adapter上续训。Student只接收原Query与候选文档；Teacher额外接收当前条件下的Rubric，仍能从原Query看到实体。

Student先生成文档ID序列；Teacher对完全相同的response token IDs及前缀打分。训练使用采样token上的log概率差，而非额外生成“正确答案”后做普通SFT：

```text
A_t = stop_gradient(log p_teacher(y_t | Query, docs, Rubric, y_<t)
                    - log p_student_old(y_t | Query, docs, y_<t))
loss = -mean_t(A_t * log p_student(y_t | Query, docs, y_<t))
```

这是参考ADA蒸馏思路的sampled-token reverse-KL策略梯度替代目标，不是全词表KL，也不宣称完整复现AdaTutoRank。不加入ATO、GRPO、outcome reward或Teacher有/无Rubric打分对照；日志中的loss/log-prob gap不应直接称为下游reward或检索收益。

| 训练项 | 共同设置 |
|---|---|
| Student LoRA | r=16、alpha=32、dropout=0 |
| 学习率 / 优化器 | 1e-5 / AdamW |
| seed / 梯度累计 / 最大梯度范数 | 42 / 4 / 1.0 |
| 精度 / prompt上限 / 输出上限 | BF16 / 12000 / 128 tokens |
| rollout | temperature=1、top-p=1、top-k=0，关闭thinking/dropout |
| 保存与过程验证 | 每100步checkpoint与dev200；step0参与选点 |
| 文档输出任务 | 可变长度有序子集，未强制固定Top-k |

每条样本一个rollout，每4条累积一次更新；500步对应2000条。EOS参与蒸馏，非法输出也参与原OPSD更新，未新增格式奖励或输出修复机制。

## 五、各正式组的数据干预

### 第一组：正确Rubric

保留Query与官方原Rubric对应关系。完整训练4000条/1000步；跨组公平预算比较使用其前500步以内checkpoint。保护原结果目录，尤其step-300，不改算法和历史产物。

### 第二组：部分同领域整份错配

- 全部4000条自动领域分类：3383条有明确主领域；617条unknown/uncertain统一标为`original_fallback`，保留自己的整份原Rubric。
- 明确标签属于自动分类，不是人工真值。仅3383条在同领域内一对一错配：排除自身、相同Query、相同Rubric，donor恰好使用一次，不跨域补配；617条不当donor。
- 训练实际前2000条日志核验：1702条错配（85.10%）、298条原文回退（14.90%）。不能用全量617/4000代替此实际比例。
- Student输入、Query、候选、目标函数及训练顺序不变，只换Teacher整份Rubric。
- 500步与dev200完成后停止；未进行9814题全量开发评估。

### 第五组：去四类命名实体，异常整份回退

- 仅处理Rubric各description值中的人名/角色、机构、地点、作品名；确定性替换为带类型占位符，同一实体真别名尽量保持一致，不同实体分开。
- 保留实体跨度外文字、独立日期/数字、关系、否定、结构、列表顺序和权重。Teacher仍看原Query，故不是“Teacher完全不知道实体”。
- 复用980条此前agent复查记录；其余1020条使用`deepseek-v4.1-flash`单次API提取，由程序替换与重建校验。没有对每条再做第二次API语义复核。API模型名称仅为请求/服务返回标识，不声称核实其底层权重身份。
- 用户要求本轮151条异常全部回退，包括曾修正过的本轮异常；最终每条整份保留原Rubric，原响应和历史尝试留存。
- 用户随后取消剩余57条规定抽查，直接进入训练；manifest明确`sample_audit_complete=false`、`semantic_review_complete=false`，没有伪造审核通过。

| 第五组实际处理 | 数量 | 占2000条 |
|---|---:|---:|
| 发生实体替换 | 1729 | 86.45% |
| 未标注到目标实体、未替换 | 120 | 6.00% |
| 异常整份原Rubric回退 | 151 | 7.55% |

120条“未替换”不代表经逐条确认没有实体，也可能漏标；151条“回退”不是去实体成功。第五组是混合条件，不应称为全部2000条均已充分去实体。

训练至500步、2000条后停止并完成dev200。全量开发只完成step-200；用户取消step-300和step-500全量评估，权重保留。流程已结束，未继续训练。

## 六、统一评测口径

- 不使用Generator或LLM Judge；在官方封闭候选池内用人工支持证据标注评分，保留候选原顺序。
- 原Student prompt，BF16，greedy，batch=8，max_new_tokens=128。非法输出整条计空，不修复、不补足、不强迫选满5篇。
- All-support@5：输出前至多5篇是否覆盖全部标注支持段落；这是主要选点指标。
- Support Recall@5：前至多5篇覆盖的标注支持比例。Support nDCG@5：依据二值支持标注计算的排序指标。
- 集合F1：对完整可变长输出集合计算支持证据P/R/F1，不等同于F1@5。
- balanced macro：先分别汇总HotpotQA、MuSiQue，再等权平均；pooled按所有题目平均，两者不能混用。
- 最佳点规则：balanced macro All-support@5优先，其次集合F1，再相同取较早step；不按全量开发结果重新选点。
- 非支持候选只是“未被标为支持”，不保证真正无关。上述证据指标不是实际RAG最终答案正确率。

## 七、结果汇总

### 7.1 各组在0–500步范围内选出的最佳点：dev200

| 模型/条件 | Step | All-support@5 | Recall@5 | nDCG@5 | 集合F1 | 格式合法率 |
|---|---|---|---|---|---|---|
| 共同SFT起点 | 0 | 65.00% | 83.25% | 85.21% | 82.47% | 100.00% |
| 第一组：正确Rubric | 300 | 75.50% | 89.04% | 89.61% | 83.76% | 100.00% |
| 第二组：同领域错配混合 | 400 | 67.00% | 83.54% | 85.56% | 82.10% | 100.00% |
| 第五组：去实体混合 | 200 | 68.00% | 85.25% | 86.91% | 82.95% | 100.00% |

第一组相对SFT的All-support@5提高10.50个百分点，第二组提高2.00个百分点，第五组提高3.00个百分点。第五组比第一组低7.50个百分点、比第二组高1.00个百分点。不同最佳点是按相同规则在共同0–500步范围内选择，不代表这些checkpoint训练步数相同。

### 7.2 相同500步：dev200

| 组别 | All-support@5 | Recall@5 | nDCG@5 | 集合F1 | 格式合法率 |
|---|---|---|---|---|---|
| 第一组：正确Rubric | 73.50% | 88.46% | 89.33% | 83.89% | 100.00% |
| 第二组：同领域错配混合 | 64.00% | 82.75% | 85.01% | 81.91% | 100.00% |
| 第五组：去实体混合 | 68.00% | 85.04% | 86.87% | 82.82% | 100.00% |

第五组step-500的All-support@5比正确组低5.50个百分点，比错配组高4.00个百分点；三组格式合法率均为100%，因此此次主要差异不是格式是否合法。

### 7.3 dev200主要指标随步数变化

| Step | 第一组正确Rubric | 第二组同领域错配 | 第五组去实体混合 |
|---|---|---|---|
| 0 | 65.00% | 65.00% | 65.00% |
| 100 | 71.50% | 66.00% | 67.50% |
| 200 | 72.50% | 65.00% | 68.00% |
| 300 | 75.50% | 64.00% | 67.50% |
| 400 | 75.00% | 67.00% | 68.00% |
| 500 | 73.50% | 64.00% | 68.00% |

### 7.4 全量开发集9814题：仅列已完成结果

| 模型/条件 | Step | All-support@5 | Recall@5 | nDCG@5 | 集合F1 | 格式合法率 |
|---|---|---|---|---|---|---|
| 共同SFT起点 | 0 | 62.19% | 80.52% | 82.66% | 78.91% | 99.91% |
| 第一组：正确Rubric | 300 | 72.22% | 86.67% | 87.07% | 79.99% | 99.83% |
| 第一组：正确Rubric | 500 | 71.55% | 86.59% | 87.07% | 80.21% | 99.80% |
| 第五组：去实体混合 | 200 | 66.31% | 83.21% | 84.62% | 79.49% | 99.90% |

第五组step-200相对SFT的All-support@5提高约4.13个百分点、集合F1提高约0.58个百分点。显示值66.31%与62.19%直接相减为4.12个百分点，差异来自四舍五入；本报告提升量按未舍入值计算。

限制：第二组没有全量开发结果；第一组没有step-200全量结果，第五组没有step-300/500全量结果，因此不能据此声称已完成相同步数的全量跨组对照。第一组step-300和第五组step-200都是各自dev200选点，但步数不同。9814题包含dev200，不能称为最终测试集表现。

### 7.5 前期格式诊断：RR val 160题

| 模型 | greedy合法率 | sampled合法率 |
|---|---|---|
| 原始4B | 86.875% | 88.125% |
| 原始4B直接OPSD-10 | 83.125% | 77.500% |
| 原始4B直接OPSD-20 | 80.625% | 78.750% |
| 共同LoRA SFT起点 | 100.000% | 98.125% |
| SFT后OPSD-10 | 100.000% | 99.375% |
| SFT后OPSD-20 | 99.375% | 95.625% |

此表固定同160题、相同prompt与候选、batch16、输出上限128，greedy与sampled分开。不能与正式dev200或9814题格式率混为同一数据。它支持先SFT改善格式稳定性的工程选择，不证明排序质量一定提高。

## 八、结论与尚不能回答的问题

1. 当前相同dev200、相同0–500步范围内，正确Query专属Rubric保留最强证据覆盖收益；同领域整份错配的收益较弱，去实体混合介于两者之间。
2. 第五组仍保留Query专属Rubric的关系和证据筛选逻辑；第二组替换整份Rubric，不只是去掉名称。因此两者不能解释成唯一差别是“是否有实体”。
3. 第二组14.90%、第五组7.55%为原Rubric回退；第五组另6.00%没有替换。两组干预覆盖率不同，可能影响结果。
4. 第五组取消剩余抽查，可能有实体漏检、类型或别名错误；自动领域标签也不是人工真值。必须作为实验限制披露。
5. 当前是固定训练种子的单次运行，未汇总多seed训练方差，未对三组差值完成显著性检验。特别是dev200上的1个百分点差异，不宜直接视为稳定优势。
6. 所有结论限于封闭候选池的支持证据选择；不外推到全库检索、Generator答案质量、Deep Search或其他模型。

## 九、API数据处理Token用量

| 范围 | 输入Token | 输出Token | 总Token |
|---|---|---|---|
| 正式run-007-single | 1,040,987 | 54,301 | 1,095,288 |
| 含历史试跑累计 | 1,050,058 | 76,917 | 1,126,975 |

当前正式批次输入中缓存命中585984 Token，已包含在输入总数，不能重复相加。累计保留旧提取、旧双调用复核、错误/拒绝响应的已知usage，并按响应去重；未返回或早期未保存的失败usage是未知，不算0。统计为服务返回用量，未核实账单金额；后续本地回退、重建、汇总没有新增API调用。

## 十、复现文件与证据索引

服务器项目根目录：`/ai/yaodong/opsd-20261026/`。下表均为相对项目根目录路径。文件位于`/ai/yaodong`的GPFS挂载存储；正常关机预计保留，删除实例/存储卷的策略需以平台为准。

| 内容 | 路径/文件 |
|---|---|
| 共同SFT说明与数据审计 | `opsd/sft/README.md`、`opsd/sft/preflight.json`、`opsd/sft/selected-epoch-one.json` |
| 共同合并模型 | `opsd/sft/models/rr-qwen3-4b-sft-r16-merged-001/` |
| RR RAG训练和格式验证 | `opsd/data/rr_rag/train.jsonl`、`val.jsonl`、`manifest.json` |
| 开发数据协议 | `opsd/dev_eval/README.md`、`opsd/dev_eval/data/dev-v1/protocol.json` |
| 第一组冻结计划与结果 | `opsd/full_run/plan.json`、`opsd/runs/rag-full-epoch-001/` |
| 第二组映射/标签与结果 | `opsd/rubric_swap/`、`opsd/runs/rag-rubric-same-domain-001/` |
| 第二组实际比例 | 第二组run下`rubric-assignment-summary.json`、`training-dev200-summary.json`、`REPORT.md` |
| 第五组训练配置与计划 | `opsd/group05_deentity/config.training.toml`、`plans/deentity-001.json` |
| 第五组最终数据/来源/审核豁免 | `opsd/group05_deentity/data/agent/candidates.jsonl`、`review.jsonl`、`manifest.json` |
| 第五组结果 | `opsd/group05_deentity/runs/deentity-001/` |
| 第五组实际覆盖与报告 | 第五组run下`coverage.json`、`reports/TRAINING.md`、`reports/FULL_DEVELOPMENT.md` |
| 第五组取消评估记录 | 第五组run下`evaluation-schedule-amendment.json`、`full-validation-summary.json` |
| 各组逐步dev200 | 各run下`small-validation-history.json`、`best-small-validation.json` |
| 第五组API用量 | `opsd/group05_deentity/reports/API_TOKEN_USAGE.json`、`API_TOKEN_USAGE_CURRENT.json` |
| 前期格式结果 | `opsd/sft/results/post-sft-format-001/REPORT.md` |

本报告的结构化只读证据快照：本地`D:/reranker/opsd/reports/EXPERIMENT_SUMMARY_EVIDENCE_20261027.json`。报告未包含.env或API密钥，也不需要将凭据导入Notion。

## 十一、官方来源与参考代码

- RR代码：[RubricRanker](https://github.com/8421BCD/RubricRanker)。本项目记录的固定参考commit：`f464f083bd54cf6b1464a4347d7e9811a2c6de50`。
- [RR官方SFT数据](https://modelscope.cn/datasets/lwhlwh/rubricranker_sft_data/files)；[RR官方RL数据](https://modelscope.cn/datasets/lwhlwh/rubricranker_rl_data/files)。本次只选RAG。
- ADA参考：[AdaTutoRank](https://github.com/AdaTutoRank/AdaTutoRank)。记录的参考commit：`7df929c42482dccd0181b29a22983356762a7b92`。仅借鉴蒸馏思路，不宣称复现其完整训练组合。
- [Qwen3-4B模型](https://huggingface.co/Qwen/Qwen3-4B)。
- [HotpotQA官方项目](https://hotpotqa.github.io/)、[数据配置入口](https://huggingface.co/datasets/hotpotqa/hotpot_qa)：distractor validation。
- [MuSiQue作者仓库](https://github.com/StonyBrookNLP/musique)、[本项目复用的数据镜像](https://huggingface.co/datasets/dgslibisey/MuSiQue)：validation。源文件按本地SHA256冻结，不声称恢复了最初下载时远端revision。

上述链接是项目已记录的来源入口；最终实验数值以本报告列出的服务器实际产物为准。第三、四组未执行，最终测试未使用，本报告没有补造相应结果。
