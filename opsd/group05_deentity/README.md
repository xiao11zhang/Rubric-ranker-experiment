# 第五组：原始 Rubric 去实体化

## 最新执行约定（2026-10-27，覆盖下方历史流程）

用户明确取消剩余57条风险/固定抽查，并授权直接启动服务器训练。已汇总全部2000条：1729条发生实体替换，120条未标注到目标实体而原样保留，151条异常全部整份原Rubric回退（7.55%），包括此前已修正的本轮异常。回退不是去实体成功，未完成抽查不声明通过。manifest明确`sample_audit_complete=false`、`semantic_review_complete=false`并保存用户豁免依据，未伪造抽查记录。

训练配置改用`config.training.toml`（保留原提取配置和证据不变），数据来源声明`external_api_user_waived`；`user_waived.py`核验来源、顺序、哈希、程序转换及回退原文完整性。模型与算法不变：共同一轮SFT起点、全新LoRA r16、冻结Teacher、同基线实际前2000条顺序、原OPSD超参，500步停止并完成dev200；随后按原规则评估0/300/500及dev200最佳点的9814条全量开发集，最终测试不使用。服务器预检使用`python -m opsd.group05_deentity.prepare --config opsd/group05_deentity/config.training.toml`，训练入口仍为`opsd.group05_deentity.train --execute`。

服务器项目`/ai/yaodong/opsd-20261026`；本组输出`opsd/group05_deentity/runs/deentity-001`，控制状态`server-logs/group05-deentity-001-pipeline-status.json`。实际是否启动和步数以服务器状态为准。小时监测已删除；不得因下方历史审核要求再次等待审核。

**当前流程（2026-10-27）：** 使用`config.api.single.toml`，模型`deepseek-v4.1-flash`，当前运行`run-007-single`。保留已审核agent结果980条，剩余1020条每条仅一次API提取，程序替换与重建校验，再由agent检查异常、歧义和固定抽样；无第二次API复核。首5条已实际阅读，其中4条本地修正并保留历史，之后继续剩余1015条。实际完成数量以`data/agent/progress.json`为准，`validated`不是逐条语义审核通过。

密钥仅由程序加载，agent不读取或展示.env内容。Token报告覆盖正式运行与历史试跑，缺失usage为未知而非0；见`api_usage.py`及`reports/API_TOKEN_USAGE*.json`。当前请求关闭thinking并保留证书/主机名验证；旧-0731无渠道、双调用和输出预算诊断均为历史，不代表当前状态。详见[API_GUIDE.md](API_GUIDE.md)。数据与必需抽查全部完成后才正式汇总、上传和训练。

本组研究：保留 Query 专属的证据筛选逻辑时，Rubric 中显式出现的人名、机构、地点、作品名是否影响 OPSD 蒸馏。Teacher 仍看原 Query，因此不能解释为 Teacher 完全不知道实体。

> 后续数据处理新增API接替入口：先读 [API_GUIDE.md](API_GUIDE.md)。使用独立 `config.api.single.toml`（每条一次API提取，程序校验与agent异常/抽样审核），保留已完成的agent审核结果，在原 `data/agent/` 目录继续处理并统一汇总。下文的 `config.toml` / agent手工标注流程保留作为已有批次的历史与可选方式，不应与API流程同时写入。服务器训练算法和实验条件不变。

原agent阶段采用 `annotation_provider = "agent_annotations"`，已审核980条只读复用。当前API配置为`external_api`，必须全部2000条程序通过、异常已解决且agent必需抽查完成后，才汇总和上传；服务器随后执行必要预检、第五组500步OPSD/dev200及已授权全量开发评估。未抽查普通API记录明确`semantic_reviewed=false`，不称全量语义审核通过。

旧的服务器模型提取试跑仅作为历史诊断保留，不再运行 Qwen3-4B 或其他服务器模型提取，不增加推理预算或进行新的JSON试跑。第二组已经结束，未进行9814题全量开发评估；第五组流程不恢复第二组评估。

审核来源明确记录为 **agent语义复查与确定性程序校验**，不称人工审核或人工真值。仅记录可核实的模型身份；当前精确型号无法独立核实，`model_id = null`，不填写虚构的型号或模型权重哈希。

## 独立组织

```text
opsd/group05_deentity/
  config.toml               第五组专用配置
  entities.py               确定性名称替换、结构检查
  agent_annotations.py      当前本地分批标注、替换、复查及汇总入口
  extract.py                历史模型提取入口；当前provider禁止执行提取
  audit.py                  候选重建、审核记录及哈希核验工具
  protocol.py               核对基线配置、固定2000条、保护原文件
  prepare.py                审核及长度预检、冻结训练计划
  train.py                  第五组入口：500步后安全结束
  full_validation.py        同一9814条开发集评测与比较报告
  reporting.py              根据实际日志生成第五组报告
  test_group05.py           实体规则和训练边界合成测试
  test_agent_annotations.py agent流程与来源校验测试
  data/agent/               当前标注、预览、复查、来源清单及进度
    source-manifest.json    冻结输入、顺序、规则和代码哈希
    progress.json           本地标注/复查进度快照
    batches/                每批原文、实体标注、转换预览和复查证据
    candidates.jsonl        全部完成后汇总生成的派生Rubric
    review.jsonl            全部完成后汇总生成的agent复查记录
    manifest.json           全部完成后生成的正式来源清单
  data/source/              基线实际训练日志副本等来源证据
  plans/                    运行后生成：冻结计划
  reports/                  运行后生成：训练前实体审核报告
  runs/deentity-001/        运行后生成：第五组独立adapter、日志、TensorBoard、评测
    reports/TRAINING.md
    reports/actual-entities.json
    reports/FULL_DEVELOPMENT.md
```

代码、派生数据、计划、报告和模型输出都归在第五组目录，不挪动第二组或基线文件。`runs/`、`data/` 等运行产物默认被本目录 `.gitignore` 排除，但上传/备份时要显式包含所需数据和计划。

训练共享 `opsd/objective.py`、`opsd/common.py` 和 `opsd/full_run/` 的 checkpoint、评测、RNG 工具。第五组独立训练入口沿用基线微步逻辑，合成测试用 AST 检查：除 Teacher Rubric 输入和审计字段外，每条 response 的 OPSD 更新一致。不修改已冻结的基线或第二组代码。

## 固定实验条件

- 配置以原基线 `opsd/full_run/plan.json` 为准，Teacher 与 Student 从同一 SFT merged 模型初始化，新建 Student LoRA，Teacher 冻结。
- 保留基线 `opsd/runs/rag-full-epoch-001/step-300/`；它是比较对象，绝不能作为本组训练初始化。
- 基线仍配置1000步；本组明确在500个优化更新结束。复用基线 `order_indices/order_ids` 的前2000条，梯度累计4，不重新抽样、不筛选补位。
- 正式预检还会读取基线实际 `train.jsonl`，核对前500步每一步的输入ID，与计划前缀完全一致。
- 学习率、种子、LoRA参数、rollout、非法输出处理、EOS/loss、梯度裁剪均沿用基线。
- 相同dev200与选点规则；step0/100/200/300/400/500验证。step500 adapter、优化器/RNG checkpoint和验证完成后正常结束，不进入501。
- 全量开发集预先选择SFT、step300、step500和dev200最佳step，去重后评测。SFT结果在模型和题目ID一致时复用基线。若最佳点是100/200/400且基线缺该点全量结果，报告明确标出，后续再补齐相同步数对照。
- 9814条包含用于选点的dev200，称为全量开发集，不是独立最终测试集。

## 去实体规则

1. 只处理 Rubric 中 `description` 字段的命名实体：人名、命名机构/组织、命名地点、作品名。普通角色、关系词、抽象概念、国籍形容词等不自动去除。
2. agent标注阶段只阅读Rubric description及记录ID/路径等审计信息，不读取Query、答案或候选文档来识别实体；只标注实体和原文精确别名，不自由重写Rubric。训练阶段Teacher仍接收原Query。
3. 程序按实体组生成 `<PERSON_A>`、`<PERSON_B>`、`<ORG_A>`、`<PLACE_A>`、`<WORK_A>` 等占位符。编号在每条记录内部重置。同一实体的别名对应同一个占位符，不同实体可区分。
4. 保留日期、年份、数值条件、否定、比较维度及关系链。数值若属于完整命名实体则随名称替换，例如小说《1984》；独立的“1984年”“至少3个”等条件保留。
5. 列表顺序、条目数、字段名、层次、权重及替换跨度以外的文字都不变。候选输出可通过程序从原始 Rubric 和实体表重建验证，不能直接手工自由改写正文。
6. 数字形式的作品名必须附精确上下文 `contexts`，如 `{"path":["set_level_rubrics","Relevance",0,"description"],"quote":"the novel 1984"}`。只替换该唯一上下文中的名称，不能全局替换同形的独立年份；上下文缺失、不唯一或重复别名则报错，交给审核修正。其他有歧义的名称也可用contexts限定出现位置。
7. 同一实体全名和短别名嵌套时优先完整名称。跨实体的重叠、未知类型、不存在的名称或无合法JSON等应报错阻止提交或预检，不得训练。漏识别、误识别和别名合并是否合理仍需agent语义复查，不能仅靠程序通过来判定。
8. 没有命名实体的记录原样保留，记录为 `no_entities_detected`，审核也要确认是否漏检。不会因为无实体而剔除或补抽训练记录。

例：

```text
原始：Compare Alice and Bob, using evidence from Oxford University in 1990; do not use fewer than 3 documents.
转换：Compare <PERSON_A> and <PERSON_B>, using evidence from <ORG_A> in 1990; do not use fewer than 3 documents.
```

占位符改变了名称表达及token长度，不能声称完全控制了token长度。单靠原文替换也不能证明全部实体均已删除。报告同时展示有替换/原样保留比例与审核情况。

## 本地标注与复查

以下命令均从含 `opsd/` 的项目根目录运行。现有来源已初始化，固定为100批、每批20条，总计2000条；不要重新初始化或覆盖已提交文件。

### 1. 阅读原文并标注

阅读 `data/agent/batches/batch-0002.input.json` 这类批次输入的全部description，逐条辨别实体、别名和歧义上下文，按原ID顺序编写对应的 `.annotations.json`：

```json
{"batch_id":"batch-0002","records":[{"id":"实际记录ID","entities":[{"type":"person","mentions":["Alice Smith","Alice"]}]}]}
```

示例仅展示格式；正式文件必须覆盖该批全部20条，不能补造ID。无目标实体时填写 `entities: []`，仍须复查。歧义名称或纯数字作品名按前述规则附 `contexts`。

```bash
python -m opsd.group05_deentity.agent_annotations commit --batch batch-0002 --annotations opsd/group05_deentity/data/agent/batches/batch-0002.annotations.json
```

提交会核验来源和代码哈希、ID覆盖及顺序，生成 `.candidates.json`、`.preview.json`、`.commit.json`；这一步只完成程序转换，不代表语义审核通过。文件拒绝覆盖。

### 2. 逐条复查转换预览

agent阅读该批完整 `.preview.json`，逐条对照原文与转换结果，检查实体类型、漏检、别名合并、替换范围及日期/关系/否定条件。确认后编写 `.decisions.json`，每条填写具体复查依据：

```json
{"batch_id":"batch-0002","records":[{"id":"实际记录ID","decision":"approved","review_method":"agent_semantic_review_and_deterministic_program_validation","notes":"记录该条实际检查的实体、别名与保留条件"}]}
```

正式决策同样必须覆盖该批全部记录。不得自动批量批准未读记录。正常记录的notes使用简短、具体的检查结论；同名实体、别名归属、数字作品名等歧义保留必要说明。原文完整读取后，可用全部替换位置及其上下文核对转换，必要时回看完整字段，减少重复展示。完整原始/派生文件与审核证据保留，不为压缩文字重做已审核记录。

发现错误先修正实体标注，重新生成和复查受影响候选。对**已提交但尚未审核**的批次，使用独立修订文件及归档入口，例如：

```bash
python -m opsd.group05_deentity.revise_unreviewed --batch batch-0006 --annotations opsd/group05_deentity/data/agent/batches/batch-0006.annotations.revision-001.json --reason "说明实际修订原因"
```

入口在修改前核验旧产物哈希与新标注完整性，将旧候选/预览/提交记录归档到 `data/agent/amendments/`，保留旧实体标注与来源清单并绑定新旧哈希，生成新候选后仍须语义复查。当前入口拒绝修改已审核批次；已有同名修订目录也拒绝覆盖。不得用原Rubric静默回退、丢样本或重抽样来绕过错误。

```bash
python -m opsd.group05_deentity.agent_annotations review --batch batch-0002 --decisions opsd/group05_deentity/data/agent/batches/batch-0002.decisions.json
```

生成 `.review.json` 和 `.review-provenance.json`，绑定复查决策、转换预览及候选哈希，记录 `reviewer = "Codex agent"`、`human_review_claimed = false`。

### 3. 全量汇总与上传边界

```bash
python -m opsd.group05_deentity.agent_annotations progress
```

查看已提交标注数、agent复查数及下一待复查批次。`progress.json` 是本地标注阶段快照，其中上传/训练字段不能替代服务器真实状态。

只有全部100批、2000条均经过实际语义复查后，才执行：

```bash
python -m opsd.group05_deentity.agent_annotations finalize
```

汇总器核验覆盖、顺序、来源、批次标注与复查证据及代码哈希，生成 `data/agent/candidates.jsonl`、`review.jsonl`、`manifest.json`。这些路径与当前 `config.toml` 一致。来源清单保留规则版本、批次输入、实体标注、复查决策、实际基线顺序及代码哈希，不伪装成服务器模型提取结果。

标注初始化后，`agent_annotations.py`、`entities.py`、配置及来源输入已有哈希约束；修改会导致后续提交失败，不能绕过检查。全部数据完成后才将第五组代码、配置、完整 `data/agent/` 和来源清单引用的 `data/source/` 等证据按项目相对路径上传。不能只上传最终两个JSONL，服务器预检还会核对批次及来源证据。

已记录一次可追溯的预处理修订 `amendments/scoped-aliases-001/`：允许不同实体在**各自明确且互不重叠的上下文**中使用相同拼写，例如 Hamlet 作品与角色。任一实体未限定上下文或同一位置发生冲突仍报错。原代码、原来源清单、失败标注及修订证明已保留；前40条的转换结果经程序重建核验不变，未重新声称语义审核，未改变训练算法或参数。

## 服务器预检、训练与评估

以下为数据全部就绪并上传后的操作步骤，不表示这些步骤已经执行。使用原项目训练环境，保留配置中共同SFT模型的服务器路径。

### 1. 预检、冻结、训练

```bash
python -m opsd.group05_deentity.prepare
python -m opsd.group05_deentity.train
python -m opsd.group05_deentity.train --execute
```

prepare核验 `agent_annotations` 来源证据、全部2000条复查记录、基线完整性、实际输入顺序、Teacher长度及上下文预算；不截断、不丢样本。生成 `plans/deentity-001.json`，绑定派生数据、审核记录、代码、SFT模型和基线文件哈希。之后修改这些文件会使计划失效，需新计划/输出目录。

train默认只核验，`--execute`才训练。默认输出 `opsd/group05_deentity/runs/deentity-001/`，拒绝覆盖。训练日志逐条记录原始及实际Teacher Rubric哈希，结束后核验实际ID顺序和所有Rubric哈希，生成本组报告。

```bash
python -m opsd.group05_deentity.train --execute --resume opsd/group05_deentity/runs/deentity-001/step-100
```

只允许本组最新完整checkpoint恢复；保留adapter、optimizer/RNG及日志。冻结上限一直为500，不能跨组加载checkpoint继续训练。

### 2. 全量开发集比较

```bash
python -m opsd.group05_deentity.full_validation
python -m opsd.group05_deentity.full_validation --execute
```

默认只显示预选checkpoint；`--execute`执行评测。预测保存在本组run的 `full-validation/step-*`，比较报告在本组run的 `reports/FULL_DEVELOPMENT.md`。原始基线、第二组报告均不覆盖。

## 本地验证与边界

```bash
python -m unittest opsd.group05_deentity.test_group05 opsd.group05_deentity.test_agent_annotations opsd.group05_deentity.test_scoped_aliases opsd.group05_deentity.test_revision -v
```

已通过的本地合成测试涵盖实体/别名一致性、日期数值与权重保留、错误不回退、审核哈希绑定、agent来源与批次流程、断点前缀、实际训练日志审计、全量选点和原OPSD微步一致性。测试通过不能替代真实2000条的逐条语义复查或服务器预检/GPU训练验证。

截至2026-10-27本次推进，本地已完成前15批：标注300/2000条、agent复查300/2000条，下一批为 `batch-0016`。阶段核验记录在 `data/agent/checks/partial-0300.json` 和 `PARTIAL-0300.md`。后续实时进度以 `data/agent/progress.json` 和批次证据为准。数据尚未全量完成，未上传、未开始第五组训练或评估。第二组结果仍保留在其独立目录，其未执行全量开发评估的结论不变。

候选记录的 `status = needs_review` 是转换时写入的状态，不随审核改变；实际是否复查通过，应核对独立 `.review.json`、`.review-provenance.json` 和其哈希，不能仅据候选status统计待复查数量。程序转换及agent复查也不构成人工真值或无遗漏的保证。
