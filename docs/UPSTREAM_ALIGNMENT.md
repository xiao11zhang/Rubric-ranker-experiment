# 上游复用与必要改编

本文件描述当前实现的事实，不表示已完成真实 GPU 训练验证。

## 固定来源

| 来源 | Commit | 本地快照 |
|---|---|---|
| RubricRanker | `f464f083bd54cf6b1464a4347d7e9811a2c6de50` | `src/setwise/_upstream/rr/` |
| AdaTutoRank | `7df929c42482dccd0181b29a22983356762a7b92` | `src/setwise/_upstream/ato/` |

`_upstream/manifest.json` 保存每个文件的官方路径、commit、Git blob SHA1 和永久链接。
`setwise.upstream.verify_sources()` 按 Git blob 规则验证字节一致性。
原始源文件和 YAML 不编辑，许可证与版权头保留。

## RR reward：直接复用原函数

`rewards.py` 不再自行实现 Rubric 展开、评分 prompt 拼接、分数解析或奖励聚合。
`upstream.rr_module()` 从未修改的官方 `set_selection.py` 中选取下列原 AST 节点执行：

- `_parse_selected_ids`、`_filter_ids`、`_parse_rubric_scores`；
- `_flatten_rubrics`、`_format_rubrics_block`、`_format_doc_set`；
- `_build_set_level_prompt`、`_score_set_level`、`_doc_level_mean`、`compute_score`。

函数体不变，Rubric 枚举和正则常量也来自同一文件。每个请求创建独立命名空间，
不会在线程间修改共享的官方模块全局变量。

只替换以下 I/O 边界：

1. prompt 加载位置指向本地官方 YAML 快照。零依赖检查使用只支持这些已审计
   `|-` 字面量模板的读取器；不进行改写、折叠或补充。
2. `_call_gpt` 连接配置化 API transport；请求内容与上游相同，只有 model/messages，
   不再使用此前自写的 JSON scoring prompt，也不额外设置 temperature。
3. 上游 import-time `.env` 加载被隔离，凭证只从显式配置的环境变量读取。
4. 通过官方解析的响应可以缓存；缓存不改变 prompt 和有效奖励计算。

默认 `max_attempts=4` 与官方一致。`judge_failure_policy` 必须明确选择：

- `upstream_drop`：保持官方的重试耗尽后丢弃集合评分维度、重新计算权重行为。
- `abort`（当前共同配置默认）：原函数执行后，如本应评分但全部失败，抛异常中止批次。
  这是此前讨论过的工程改编，不声称与官方失败行为完全相同。

严格格式诊断和 ATD mask 是额外信息；它们不会代替官方的宽松 ID reward 解析。
样例 `debug_gold_ids` reward 仍只用于明确标记的 mock fixture。

## ATO prompt：官方模板加最小协议变更

`prompts/self_selector.yaml` 与 `prompts/self_reflector.yaml` 与官方快照逐字节相同。
`prompts/adaptive_hints.yaml` 只对官方模板执行两次文字替换：

| 官方文字 | 本实验文字 | 原因 |
|---|---|---|
| `three tiers / nine dimensions, scored 0-10` | `RR set-level and document-level rubrics, aggregate reward 0-1` | B1/B2 统一 RR 五类 Rubric 与 0-1 reward |
| `out of 10 by the judge` | `out of 1 by the judge` | sibling 分数使用 RR 的归一化输出 |

不再使用自行简化的提示正文。原始模板另外保存，测试限定差异只能是上面两项。

## ATO 算法与共同 GRPO

GPU 路径仍调用固定 ATO checkout 的实现，本地 `algorithms.py` 只作为数值参考，
不接管 Student 更新。B1 使用 vanilla policy loss；B2 使用上游 sdpo 内的 joint advantage 分支：

`A = 0.7 A_GRPO + 0.3 m (log p_teacher - log p_old)`。

共同设置与原始 ATO 的主要差异：使用 RR reward、统一 RR-style KL=0.001、统一采样与训练预算。
这是受控机制比较，不是复现 ATO 原论文整体实验。
hint 阈值从 3/6 转为 0.3/0.6；下阈值向上移动一个浮点数，适配上游通用分桶的边界语义。
selector/reflector 由当前策略生成，Teacher 冻结，低质量 hint 仍需 sibling 严格更优。

## 正式 Rubric 与 Student prompt 数据

本仓库没有自行生成正式 query-specific Rubric。它们必须来自经过审计的 RR 数据，
或之后明确授权的官方生成流程。`examples/debug.jsonl` 是自写测试数据，不可用于正式结论。

正式数据必须携带冻结 `prompt` 消息；数据转换器与 SFT 导出原样保留，不再默认为训练数据
生成自写 Student prompt。候选池变化时必须检查/重建 silver labels 和 prompt，
不能只把官方数据的文档内容替换掉。

## 验证边界

离线差分测试独立执行完整官方文件（仅 stub SDK/环境加载），与适配器比较
reward、完整 prompt、重试次数和失败行为，覆盖多维顺序、缺失预评分、重复/越界 ID。
来源哈希与 ATO 模板最小差异另有检查。

尚未验证：真实 API 响应、GPU Hydra 解析、Ray 环境传播、Teacher 截断、实际模型梯度、
reward 曲线与下游 Generator 指标。不能由离线检查推断实验效果。
