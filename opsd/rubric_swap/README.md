# 第二组：自动领域标签的部分同领域 Rubric 错配

用户已授权继续执行。代码已同步至 yaodong-server，37项服务器检查及真实数据预检通过；正式映射为3383条同领域错配、617条原Rubric回退。前2000条计划为1702条错配（85.1%）、298条回退（14.9%），须在训练结束后以实际日志核验。后台流程已启动：训练500步（含dev200）后进行原方案全量开发评估。不要重复运行启动命令。

## 最新执行调整：只完成训练和dev200报告

用户要求当前500步训练、step-500 checkpoint及dev200验证完成后，先不进入全量开发评估。原冻结计划和源码不改；独立的 `opsd.rubric_swap_control.training_only` 已停止旧调度器并接管，当前训练子进程保留。不要重启旧pipeline或手动运行full_validation。

报告 `opsd/runs/rag-rubric-same-domain-001/REPORT.md` 明确声明本组未进行9,814题全量开发评估、最终测试未使用。训练中版本标为阶段性；500步及验证完成后更新最终版，包含经核验的实际错配/回退比例、逐步dev200结果、同预算正确组对照及最佳checkpoint。执行变更记录在 `training-only-amendment.json`，最终摘要在 `training-dev200-summary.json`。下文原定全量评估命令仅保留作后续参考，未经新的执行授权不运行。
## 已确认的实验定义

- 第一组 `opsd/full_run/`、`opsd/train.py`、`opsd/common.py`、`opsd/objective.py` 以及原始训练数据不修改。服务器正确组 `opsd/runs/rag-full-epoch-001`，尤其step-300权重及配套文件保持原样。
- 第二组Teacher、Student都从第一组相同的一轮SFT step-484合并模型初始化。Teacher冻结；Student新建LoRA r=16、alpha=32，不从第一组step-300 adapter继续训练。
- 4,000条训练记录中，3,383条采用模型明确给出的13类主领域；它们不是人工审核真值。617条unknown/uncertain（包含6条分类失败）统一使用执行标签 `original_fallback`，不进行错配、不删除。
- 仅3,383条互相充当donor，按各自主领域生成固定一对一错配，排除自身、相同Query、相同Rubric内容。每个donor恰好使用一次。若某明确领域无法合法配对，报错；不跨领域补配，也不默默转为回退。
- 617条映射为自己→自己，Teacher继续使用自己的原Rubric。它们不能作为3,383条的donor，也不会在617条之间相互交换。
- 原始 `domain-labels.jsonl` 保留unknown及uncertain标记，不改写成已审核。特殊执行标签只进入新计划/映射：`primary_domain=original_fallback`、`assignment_type=original_fallback`，另保留原分类字段与回退原因。
- 只替换Teacher整份Rubric。Student输入、原Query、原候选文档、原损失、rollout、学习率、seed、梯度累计、训练顺序保持第一组口径。

## 配置与执行边界

`same-domain.toml` 保留 `steps=1000`、梯度累计4，新增：

```toml
uncertain_policy = "original_fallback"
expected_fallback_records = 617
expected_mismatch_records = 3383
stop_after_step = 500
```

准备阶段校验全量计数恰好617/3383，且配置、数据和共同SFT起点与第一组一致。计划单独冻结 `execution_stop_after_step=500`。

第二组本次仅执行前500个优化更新，即第一组相同排列的前2,000条记录。默认就使用计划冻结的500步边界；即使未传CLI停止参数也不会跑到1000。显式CLI停止参数必须与计划相同。step-500 checkpoint完整保存且该步dev200验证完成后退出，不进入501。不把本次称为完整遍历4,000条的一轮训练。

未知标签默认策略在函数层面仍为严格报错；只有显式启用 `original_fallback` 的新配置才允许上述混合条件，原正确模式无需领域标签。

## 实际训练比例报告

每条训练日志新增 `rubric_assignment_type`、`rubric_assignment_domain`，保留 `rubric_donor_id`。训练到500步后：

1. 校验实际日志恰好为step1..500，包含2,000条、无重复、与第一组数据顺序前缀一致。
2. 逐条核对实际donor、分配类型与执行标签是否符合冻结映射。
3. 写入 `rubric-assignment-summary.json`，输出实际错配/回退数量、占比、各执行领域分布，同时打印到日志和最终状态。

```json
{
  "source": "verified_actual_training_log",
  "completed_steps": 500,
  "records": 2000,
  "counts": {"same_domain_mismatch": "实际条数", "original_fallback": "实际条数", "correct_original": 0},
  "ratios": {"same_domain_mismatch": "实际条数/2000", "original_fallback": "实际条数/2000", "correct_original": 0},
  "full_epoch_completed": false
}
```

以上是字段示例，不是实际训练结果。prepare还会给出全量4,000条与计划前2,000条的预期统计，明确与完成训练后来自实际日志的报告区分。不能用617/4000直接代替实际前2,000条的回退占比。

## 验证、选点和训练后评估：沿用既定方案

- step0及每100steps在第一组相同dev200上验证，实际为0/100/200/300/400/500。
- 原Student prompt、候选顺序、greedy、batch8、max_new_tokens128及证据指标保持不变。不会新加领域筛选或根据模型结果调整验证问题。
- 最佳选择仍先比较balanced macro All-support@5，再比较集合F1，仍相同取较早step；SFT step0参与。
- 训练后评估使用原9,814题全量开发集：HotpotQA 7,404、MuSiQue 2,410。它包含dev200，不称为独立最终测试。此前保留的最终测试集不启用、不用于选点。
- 全量比较新组step-300、step-500及同规则选出的最佳checkpoint，去重；SFT共用已有结果。正确组在同样0..500预算内按同规则选最佳（既有step-300），同时报告相同步数结果。
- 如果新组最佳是100/200/400，补齐正确组该步的全量开发对照。新增的正确组推理输出也只写入第二组目录，不写第一组目录。
- `full_validation.py` 是第二组独立入口，复用第一组公共evaluation模块，不修改第一组。默认只校验，显式 `--execute` 才做推理。

## 后续获授权时使用的命令（当前不要执行）

服务器项目根目录 `/ai/yaodong/opsd-20261026`，使用原 `.venv-opsd/bin/python`：

```bash
# 先同步本目录源文件，保留第一组和原始分类产物；不要运行旧自动分类启动器。
python -m opsd.rubric_swap.prepare --config opsd/rubric_swap/same-domain.toml --plan opsd/rubric_swap/plans/same-domain-001.json
# 仅校验冻结计划，不训练
python -m opsd.rubric_swap.train --plan opsd/rubric_swap/plans/same-domain-001.json
# 获得执行授权后才训练；停止边界默认为冻结的500步
python -m opsd.rubric_swap.train --plan opsd/rubric_swap/plans/same-domain-001.json --execute
# 中断后只从同一新组的最新完整checkpoint恢复
python -m opsd.rubric_swap.train --plan opsd/rubric_swap/plans/same-domain-001.json --execute --resume opsd/runs/rag-rubric-same-domain-001/step-100
# 500步训练及验证完成后，先查看选点，再显式启动全量开发推理
python -m opsd.rubric_swap.full_validation --plan opsd/rubric_swap/plans/same-domain-001.json
python -m opsd.rubric_swap.full_validation --plan opsd/rubric_swap/plans/same-domain-001.json --execute
```

计划、映射和运行目录拒绝覆盖。模式、标签、映射、代码、SFT及第一组step-300的文件hash均绑定实验身份。新组不能从第一组checkpoint恢复训练，也不能在续训时改变停止边界。

## 文件与验证范围

- `rubrics.py`：严格/混合标签策略、同领域映射、回退自映射、全量约束检查、Teacher Rubric选择。
- `prepare.py`：共同起点和数据预检、617/3383计数检查、映射及500步执行计划冻结。
- `train.py`：保留原OPSD更新，增加分配日志、实际比例报告与500步安全停止。
- `reporting.py`：计划比例及经核验实际训练比例，支持恢复后从完整日志重算。
- `full_validation.py`：原规则选点、相同步数对照、只读复用正确组结果和全量开发评估。
- `classify.py` 和原自动分类产物：本次不改、不重跑。

本地26项测试通过：严格及混合映射、回退输入不变、donor隔离、篡改拒绝、4,000条合成映射/2,000条实际日志分母、500步默认边界、checkpoint选点与OPSD微步AST一致性等。没有真实数据映射或GPU训练/评估验证；这些须等执行获授权后在服务器完成。

## 结果解释边界

这是基于自动标签的部分同领域Rubric错配，不能称全量错配或人工审核通过。整份Rubric的条目、权重和长度会一起改变。映射构造覆盖全部4,000条，但只训练前2,000条时，两组实际用到的Rubric边际分布不保证一致。原始训练数据接口、Student输入和第一组复现入口不变。