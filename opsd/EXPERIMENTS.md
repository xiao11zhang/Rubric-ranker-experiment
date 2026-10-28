# 实验目录索引

各组配置、专用数据处理和报告分别保管；共同SFT起点、原训练数据、OPSD目标及评测工具复用，历史基线文件不移动。

| 组别 | 条件 | 专用代码/说明 | 结果位置 |
|---|---|---|---|
| 第一组 | 正确Rubric基线 | `full_run/README.md` | 服务器 `runs/rag-full-epoch-001/`；本地可能只同步报告 |
| 第二组 | 同领域Rubric交换 | `rubric_swap/README.md` | 以该组冻结config/plan中的output为准 |
| 第三组 | embedding语义近邻交换 | 本次未实现，后续独立目录 | 尚未由第五组代码创建 |
| 第四组 | 题型通用Rubric模板 | 本次未实现，后续独立目录 | 尚未由第五组代码创建 |
| 第五组 | 原始Rubric去命名实体，保留日期数值条件 | `group05_deentity/README.md` | `group05_deentity/runs/deentity-001/`，其中`reports/`放报告 |

第五组准备阶段数据、审核清单、审核报告分别放在 `group05_deentity/data/` 与 `group05_deentity/reports/`，冻结计划放在 `group05_deentity/plans/`；这些目录在实际执行后生成。无需把基线或其他组输出复制到第五组。

本索引为代码组织说明，不表示任何尚未执行的组已经取得训练或评测结果。
