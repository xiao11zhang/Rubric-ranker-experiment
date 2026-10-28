# 复现与迁移说明

## 1. 本发布的范围

包含三组代码、原服务器配置、离线测试和聚合结果；不是包含全部数据与模型的自足复现包。原路径根为`/ai/yaodong/opsd-20261026`。所有命令从仓库根目录执行。

不要将本次代码覆盖回原服务器运行目录：正式计划绑定源文件hash。本仓库README整理及新工作目录本身都会导致旧计划不适用。请在新目录使用副本；原服务器和历史模型保持不变。

## 2. RR数据与共同SFT

下载官方SFT JSON及RL的train/val Parquet后，例如：

```bash
python -m opsd.prepare --train /path/to/train.parquet --val /path/to/val.parquet --output opsd/data/rr_rag
python -m opsd.sft.prepare --source /path/to/rubricranker_sft-data.json --exclude-val opsd/data/rr_rag/val.jsonl --output opsd/sft/data/rr_rag_v1
```

输出目录必须不存在，避免覆盖。使用RR固定commit的LLaMA-Factory源码准备`opsd/sft/backend/`；相应归档未纳入Git。`opsd/sft/configs/lora.json`是当前一轮配置，原运行从两轮计划中途停在首轮，精确历史需共同checkpoint-484而非重新训练后声称等价。入口`python -m opsd.sft.entry --config <lora配置>`实际会训练，请只在资源路径、数据和输出目录明确后使用。合并脚本参数以`python -m opsd.sft.merge --help`为准。

## 3. 独立开发集

准备HotpotQA distractor与MuSiQue validation的原始Parquet；原默认路径在`opsd/dev_eval/prepare.py`的RAW。先修改到实际原文件位置，再生成开发协议，不能拿最终测试集替代。`python -m opsd.dev_eval.prepare`冻结9814题合格开发问题及dev200，包含训练/SFT问题词面去重及支持证据映射检查。具体所需源文件与SFT审计文件见该脚本。

## 4. 第一组

准备相同共同SFT合并模型、RR JSONL、开发协议及合并manifest。修改`opsd/full_run/config.toml`及`opsd/sft/configs/opsd_after_sft.toml`中的资源路径，二者参数受一致性检查约束。然后：

```bash
python -m opsd.full_run.prepare
python -m opsd.full_run.train
```

`full_run.train`本身会训练，不能把它当作dry-run。第一组1000步/4000条。旧全量评估默认包含step1000，与本次最终取消要求不同；需按自己明确选定的评估清单运行，不能直接重启历史pipeline。

## 5. 第二组

依赖第一组冻结计划、完整实际训练日志及共同SFT，用它核验相同前2000条。恢复原服务器`opsd/rubric_swap/data/`标签和映射证据，或用分类工具重新生成并标为新数据版本。全量3383错配+617原文回退；原实验前2000条实际1702/298。

```bash
python -m opsd.rubric_swap.prepare --config opsd/rubric_swap/same-domain.toml --plan opsd/rubric_swap/plans/same-domain-001.json
python -m opsd.rubric_swap.train --plan opsd/rubric_swap/plans/same-domain-001.json
python -m opsd.rubric_swap.train --plan opsd/rubric_swap/plans/same-domain-001.json --execute
```

第二组默认先核验，`--execute`才训练；500步保存与dev200后停止。本次没有全量开发评估，不执行旧自动pipeline或full_validation。

## 6. 第五组

实际配置`opsd/group05_deentity/config.training.toml`，provider为`external_api_user_waived`；原API配置与代码保留，不能把未完成抽查标为已完成。需恢复原服务器`opsd/group05_deentity/data/agent/`、`data/source/`及其中manifest引用的证据（含原agent批次/API records/reviews）。这些真实派生数据未上传Git。

```bash
python -m opsd.group05_deentity.prepare --config opsd/group05_deentity/config.training.toml
python -m opsd.group05_deentity.train
python -m opsd.group05_deentity.train --execute
```

`train`默认核验，`--execute`实际训练，500步停止。固定序列前2000条：1729替换、120未替换、151明确原文回退。数据生成入口`api_extract`、`original_fallback`、`user_waived`具有来源/审核策略约束；它们并非任意JSON都可直接通过的转换器。重新调用API需用户授权及程序环境凭据，普通训练不需要API密钥。

第五组原全量脚本默认选0/300/500/dev200最佳点；本次用户最后只保留SFT复用和step200全量结果。不要直接执行该旧默认清单。历史调度取消脚本可能包含旧PID/路径，仅保留作实现证据，不可在新服务器盲目执行。

## 7. 文件完整性与再生成

源代码发布清单`SOURCE_MANIFEST.json`只校验本次GitHub快照，不替代实际训练计划。新目录的所有路径/配置调整完成后重新运行预检，绑定新hash；不要伪造原标签/抽查/权重hash。相同算法和超参不保证跨设备、依赖版本或新SFT轨迹逐bit相同。

上传前会在独立发布目录执行合成单测与所有Python源码语法检查；不会启动分类、API或训练。结果和数据限制详见`EXPERIMENT_SUMMARY.md`。
