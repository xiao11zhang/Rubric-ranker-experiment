# 第五组：单次 API 提取，程序校验与 agent 抽查

当前流程：**每条一次API识别实体 → 程序替换和结构校验 → agent检查异常/歧义并抽查正常记录 → 统一汇总**。没有第二次API复核请求。模型、接口和发送字段沿用用户已经明确授权的配置，不向API发送Query、候选文档或答案。

原agent通过审核的批次只读复用，原始RR数据不改动。结果仍归 `data/agent/`，最终生成同一套 candidates.jsonl、review.jsonl、manifest.json及整理报告，不另建实验。

## 配置与证据

- 使用 **config.api.single.toml**；旧config.toml/config.api.toml及run-001/002/003诊断保留，不直接修改历史冻结配置来续跑。
- 新运行名为 `run-004-single`，新增证据位于 `data/agent/api/run-004-single/`。
  此为初始单调用试跑名；**当前以config.api.single.toml为准，为run-007-single**。run004/005保留响应不完整、输出预算及usage诊断；run006已发现按类型误并实体，不可训练。run007已补一实体一组的明确示例，关闭thinking，每条一次请求，无自动重试。首5条实际复查发现别名/角色等问题，4条由agent本地修正，无额外API复核；后续异常和固定抽样仍必须真实检查。
- 新配置 `api_max_attempts=1`，每个处理尝试最多一个HTTP请求，不自动网络重试。只有用户/主窗口显式使用 `--retry-failed` 才会再次尝试失败记录，重试可能产生额外费用。
- 客户端继续验证TLS证书和主机名，拒绝重定向转发密钥。不要关闭证书验证来解决连接错误。
- 根目录 `.env` 保存 RUBRIC_API_KEY、RUBRIC_API_BASE_URL、RUBRIC_API_MODEL；环境中非空同名变量优先。密钥不进入来源清单/日志/上传包。
- 当前仅支持Bearer鉴权的JSON chat-completions协议（model/messages/max_tokens → choices[0].message.content），API接口/模型若改变要按授权范围处理。
- 用户最新明确模型为`deepseek-v4.1-flash`，接口仍为`https://api.shubiaobiao.cn/v1/chat/completions`；旧`-0731`不再调用。程序核对这两个目标值；agent不得读取.env内容。`api_thinking="disabled"`传入官方thinking关闭参数，实际返回已不含大量推理输出；仍按usage真实统计，不推断上游权重身份。

Token统计（只读）分别输出全部历史和当前运行；完成后重新生成并交付：

```bash
python -m opsd.group05_deentity.api_usage --output opsd/group05_deentity/reports/API_TOKEN_USAGE.json
python -m opsd.group05_deentity.api_usage --run run-007-single --output opsd/group05_deentity/reports/API_TOKEN_USAGE_CURRENT.json
```

保存服务返回的prompt/completion/total以及reasoning/cached细项。截断响应保留finish_reason和usage并标error；最早客户端未保留的失败usage无法事后补造。累计token不等同于已核实计费额。

## 提取与恢复

先停止旧双请求流程继续运行，保留其诊断。不要让旧agent汇总和API处理同时写同一最终数据。旧agent冻结配置/代码保持不变。

```bash
# 只读预览，不读密钥、不联网
python -m opsd.group05_deentity.api_extract

# 先5条真实兼容性与提取质量检查
python -m opsd.group05_deentity.api_extract --execute --limit 5 --workers 2

# 确认小批结果后继续剩余数据
python -m opsd.group05_deentity.api_extract --execute --resume --workers 4
```

程序通过的API记录标为 `validated`，**不是逐条语义审核通过**。每条保存实体列表、确定性候选、模型返回身份/usage、来源与哈希。正常通过的记录不再请求。中断前已保存的提取响应可直接用于本地转换，不补发复核请求。

网络失败、非法标注或转换错误标为error，不静默使用原Rubric，不剔除/补抽记录。确需网络重试时显式运行：

```bash
python -m opsd.group05_deentity.api_extract --execute --resume --retry-failed --limit 5
```

active.lock避免两个处理器重复请求。确认旧进程已经结束后才能处理异常遗留锁。网络未通时只诊断连接，不扩大批量。

## agent审核：仅本地操作，零API请求

```bash
python -m opsd.group05_deentity.api_extract --audit
```

在当前run的 `audit/` 下生成 plan.json、preview.json、decisions.template.json：

- 固定种子抽查API记录的5%，至少20条（少于20时全部），另外覆盖前5条试跑、少量无实体和各实体类型样本。
- 数字作品名、相同拼写对应不同实体等高风险项全部检查。
- preview同时列出已发现的程序错误；先由agent修正或显式重试，不能因为JSON合法就放行。
- 完成程序校验不声称语义正确。无抽查的普通记录明确记录 `semantic_reviewed=false`；旧agent审核和新抽查都有真实来源，不称人工真值。

agent实际检查preview之后，将模板另存为 decisions.json，逐条填写简短结论：

```json
[{"id":"实际ID","rubric_hash":"对应候选哈希","decision":"approved","notes":"已核对类型/别名，年份和数值保留。"}]
```

然后执行，无网络请求：

```bash
python -m opsd.group05_deentity.api_extract --audit-decisions opsd/group05_deentity/data/agent/api/run-004-single/audit/decisions.json
```

未审阅、缺notes或哈希不一致的记录不能批准。抽查可先做前5条，再处理全部后生成最新清单；已审核且哈希相同的记录复用。

对于错误/歧义需要agent更正实体时，制作修正JSON列表（必须由agent审核，不是机械批准），保留原API尝试：

```json
[{"id":"实际ID","entities":[{"type":"person","mentions":["原文名称"]}],"notes":"说明修正点与语义检查结论。"}]
```

```bash
python -m opsd.group05_deentity.api_extract --repairs opsd/group05_deentity/data/agent/api/run-004-single/corrections.json
```

修正调用原实体转换函数，检查原文精确匹配、结构、年份数字和别名约束；不请求API。修正记录注明agent_corrected，审核绑定最终候选哈希，旧尝试保存于attempts。

## 最新异常处理约定（2026-10-27）

用户明确要求本轮151条异常全部回退为整份原Rubric，不再逐条修正；此前已修正的本轮异常也纳入回退。API原响应与修正尝试保留，记录标记`processing_mode=original_fallback`、`deentity_success=false`，不能计作无实体或去实体成功。已核验151/2000条（7.55%）原文完整保留；在API处理的1020条中占14.80%。正常记录仍执行既定风险与固定抽样，不增加API复核请求。小时监测已按用户要求删除。

原提取代码和算法保持冻结。回退操作由独立`original_fallback.py`处理，决策证据位于当前run的`audit/fallback-all-anomalies-151.json`及`original-fallback-summary.json`；后者覆盖全部151条，包括此前2条回退。回退审核只核验原文保留和用户决策，不声称去实体语义审核通过。暂存的`pending-repairs-*`/`repair-spec-*`保留为历史诊断，不再应用。

完成其余必需抽查后，采用以下本地汇总入口，确保manifest、候选记录和报告均携带回退标记与统计：

```bash
python -m opsd.group05_deentity.original_fallback --finalize
```

不得使用下方旧的直接`api_extract --finalize`命令作为本次最终入口，避免遗漏回退分类。训练和模型输入接口不变，回退样本Teacher接收自己的完整原Rubric。

## 汇总和训练

API记录全部通过程序校验，异常已解决，必需抽查与歧义审核完成后：

```bash
python -m opsd.group05_deentity.api_extract --finalize
```

不联网；仍按基线实际前2000条顺序，统一生成：

- `data/agent/candidates.jsonl`：去实体Rubric及逐条来源。
- `data/agent/review.jsonl`：区分旧agent语义审核、程序校验和新agent抽查/修正。
- `data/agent/manifest.json`：绑定数据/规则/代码、API记录和抽查证据；明确 `semantic_review_complete=false`、`sample_audit_complete=true`，不冒充全量逐条复核。
- `reports/API_DATA_PREPARATION.md`：实际完成数量、agent/API来源及审核覆盖。

抽查缺失或任何error未解决会阻止汇总。已有最终文件拒绝覆盖。代码、配置、数据和manifest引用的证据保持相对路径上传；**服务器不需要.env**。

```bash
python -m opsd.group05_deentity.prepare --config opsd/group05_deentity/config.api.single.toml
python -m opsd.group05_deentity.train --execute
```

训练仍从基线共同SFT起点开始，固定同2000条/顺序，500step、accum4、dev200与9814全量开发评测规则保持不变。只有数据审核方式改为程序校验加agent风险/抽样审核，并如实报告该限制。基线step300及其他组不修改。
