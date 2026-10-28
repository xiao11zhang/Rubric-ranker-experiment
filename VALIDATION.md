# 发布验证

2026-10-28：独立源码目录中46项单元测试，45项通过，1项跳过（依赖可用性条件）；所有Python源码AST解析通过。未执行GPU训练/评估或API请求。

命令：`python -m unittest opsd.tests.test_core opsd.dev_eval.test_metrics opsd.rubric_swap.test_rubrics opsd.rubric_swap.test_mixed opsd.group05_deentity.test_group05 opsd.group05_deentity.test_original_fallback -q`。

源码发布不包含训练数据/模型，因此不声称克隆后无需资源准备即可精确复现历史产物。
