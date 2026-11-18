# 双强度校准精简试点协议（v1.1-pilot）

> **发布时纯编辑投影，不是逐字冻结原件。** 已运行、已审计的冻结协议正文 SHA256 仍为 `241245163ced34ab10197b594f087e0da497791c38084662095b5ba476b18ff8`，那才是实验与选择锁的审计依据。本公开副本仅将内部绝对路径改为相对占位说明，因此本文件的 SHA256 与冻结 hash 不同。禁止把本公开文本称作冻结原件或与 plan hash 逐字等同。独立核算已通过（审计标识 `dual-pilot-test-audit-20261118-010451-KGjmJ9` 的 verification.json，exit 0；官方报告 SHA256 `bc9eaed3bdbc90a3793d69e0ad6dc5f3f160fad7a8c9df79c08ff9eff5ba08d3`）。核算只核对其文字与绑定，未重算选择，也未修改 STOP 阈值。

- 协议编号：`dual-kappa-pilot-v1.1`
- 版本：v1.1-pilot
- 状态：**实施准备协议**。本文件冻结算法、网格、五臂、平局、STOP、train-only 类型与输入闭环。批准本实现进入合成预检不等于批准真实 val 选参或 test 评分。真实六任务选择须在验证者集中预检通过、并由 SW 另行安排后执行；test 必须在六任务锁完成且验锁之后再统一评一次。
- 日期：2026-11-18（Asia/Shanghai）
- 收口修订（同日）：明确 STOP 异常拒绝与 MAE 改善公式 (refmean-dualmean)/refmean，锁文件独占创建，前 10 名单的正负方向与不凑数规则，以及 CV 使用 ddof=0。网格、五臂、阈值与拟合公式不变。
- 跳过项（本试点明确不做）：原规划 A 的四臂诊断、Huber/MAE 拟合、correction-energy、新门控、输出裁负、基模重训、多维区域轮廓、only-shift、调旧、rolling 选参。
- 评估性质：历史 test 已暴露。即使将来评分，也只是探索性，不是盲测；不得用 test 改候选、网格或 STOP 阈值。

本协议不修改已冻结的 `val-select-kappa-v3`、原选择锁、原 test 报告或 PR21/22/23 材料。新 worktree 中的 `src/val_select/` 是只读字节快照，仅复用 I/O、schema、证据闭环、指标与旧/单 kappa solver。

## 1. 继承而不改写的约定

- 拟合只用 train 72h。n=72，mu 为训练窗预测均值，x=p-mu（val/test 仍减训练窗 mu），r=y-p，S=sum x^2，T=sum x r，rbar 为训练窗平均残差。
- 新形式：alpha = T / (S + kappa_alpha * n * mu^2)，c = rbar / (1 + kappa_c)，a=1+alpha，b=c-alpha*mu，p'=a p + b = p + alpha x + c。
- 对角 kappa_alpha = kappa_c = kappa 必须与已冻结 `solve_centered` 逐点数值一致（含近常数与分母退化）。这保证 9 个对角点嵌套原单 kappa。
- 近常数：range(p) <= 1e-15 * max(1, max|p|)，单位固定为 `dc_clipped_count`。kappa_alpha=kappa_c=0 且近常数：alpha=0，c=rbar（与 kappa=0 中心化 OLS 约定相同）。identity 是独立分支，alpha=c=0，不是 inf。
- 非退化 mu=0 且 S>0：斜率惩罚为 0，alpha=T/S；S=0 或分母 <= 0 时 alpha=0，保留 c。不另加 epsilon 改目标。
- 门控：训练 72h 原始上车合计 >= 150，期望 207 有效格、50 过门格。参数只在过门格上求。**非门控格保持原 FT**（a=1, b=0），逐元素精确不改。
- 单位 `dc_clipped_count`，value_scale=234 的文档约定继承输入协议；全程 float64；预测不裁成非负。负预测只计数量与最小值。
- 时间与来源层继承 `documented_from_bundle`：train [5832,5904)、val [5904,7344)、test [7344,8784) 为 `documented_range_from_bundle`；FT commit 仍为 unknown；`7a04ca4` 仍为 documented_short_revision；权重哈希仍为 claimed。不得升级为 recorded。
- 输入协议仍为 `val-select-input-bundle-v1`。选择前重算 canonical evidence_hash，以已验收原 ZIP 与非 test 白名单为信任锚。test 成员只在验锁后、先比对文件字节 SHA 再 decode。

## 2. 候选网格与预算

- G = {0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16}：9 个对角 (kappa, kappa)。
- H = {0, 0.25, 1, 4, 16}：全部有序对去掉对角，得 20 个非对角。
- 显式 identity。
- `budget_nested10` = 9 对角 + identity = 10。这是单 kappa 的 val 选参基线。
- `budget_expand30` = 9 对角 + 20 非对角 + identity = 30。
- **30 与 10 搜索预算不等。** 10 是 30 的子集，必须同时完整记录两条曲线。禁止事后丢掉非对角来假装公平。禁止为旧法补无效候选凑 30。禁止无限加密网格。

选参指标：完整 val 1440h 整城 MAE（official city / valid mask）。禁止用 test 或 RMSE 改候选。块与 rolling 本轮不跑。

## 3. 五臂（仅此五臂）

| 臂 | 含义 | 是否搜索 |
|---|---|---|
| FT | identity，原微调预测 | 否 |
| OLD_FIXED | 旧法固定 m=1，lambda=72 | 否 |
| NEW_FIXED | 新法固定 kappa=1，即 (1,1) | 否 |
| SINGLE_VAL10 | 在 budget_nested10 上按 val 整城 MAE 选一对（对角或 identity） | 是，10 |
| DOUBLE_VAL30 | 在 budget_expand30 上按 val 整城 MAE 选一对 | 是，30 |

六任务：NY/CHI × seed 42/43/44。缺失种子记缺失，不能动态丢弃，不能产生 scorable 锁。

必要基线（FT、旧固定、新固定）在本协议内用同一对 train/val 数组复现，日志注明复用的是同配对数组，而不是复制旧锁里的数字。

## 4. 全局平局

对当前预算内的整条曲线取 Lmin = min MAE。容差 delta = 1e-10 * max(1, |Lmin|)。所有 MAE - Lmin <= delta 的候选进入并列集。先选 identity；否则按 (kappa_alpha + kappa_c, kappa_alpha, kappa_c) 降序取最大。容差不是可接受性能平台，不按结果改。全部候选损失与决胜依据入锁。

对角上该规则与原「更大有限强度」一致。

## 5. 预锁 STOP（test 上判定，不在 val 上改规则）

在正式 test 表写完后，用全精度比值判定；任一条不满足即普通 STOP，不扩网格、不换 loss、不继续本轮以外的实验。三条必须同时满足才标记「值得后续讨论」。即便通过也结束本轮，等待用户后续决定。

阈值是预先锁定的工程筛选，不是显著性检验，不事后调整。判定前必须先严格验证：全部所需任务（六任务 × DOUBLE_VAL30 / OLD_FIXED / SINGLE_VAL10）的 MAE 与 RMSE 均为有限且非负；按 source 三个 seed 算术平均后，参考均值必须大于 0，聚合与比值必须有限。聚合溢出、零参考（含零 RMSE 参考）、NaN、Inf 或负度量属于异常拒绝（抛出错误），既不是 PASS，也不是普通 STOP。

1. 按 source 对三个 seed 做算术平均 MAE / RMSE。DOUBLE_VAL30 相对 OLD_FIXED 以及相对 SINGLE_VAL10：每个 source 的相对 MAE 改善 (refmean - dualmean) / refmean >= 0.001。共四个比较全部满足。
2. 对应的有效 source 平均 RMSE 比值 dualmean/refmean <= 1.005。共四个比较全部满足。
3. 六任务中至少 4 个任务的 DOUBLE_VAL30 MAE <= 该任务 OLD_FIXED MAE。相等判定容差为 1e-10 + 1e-12 * |ref|，不用四舍五入。

Seed 共享同一 DC，不得把六任务或三 seed 当 iid 样本做显著性声明。

## 6. 同一次评分中的逐格分析（仅在 test 正式评分之后）

主对照 OLD_FIXED，另对照 SINGLE_VAL10。对每个格 j：Delta MAE_j 与它对整城 MAE 的精确贡献 Delta MAE_j / C（各格小时数相同）；Delta MSE_j 用于识别拉高整城 RMSE 的格。禁止把 RMSE_j 相加。Delta MAE_j = MAE_ref_j - MAE_new_j：大于 0 为改善（new 更好），小于 0 为退化，恰好为 0 的格不计。前 10 改善与前 10 退化各自只收录方向正确的格，最多 10 个，数量不足不凑数、不把另一方向或零 delta 填进名单。报告 seed 方向一致性与 |整城 MAE 贡献| 在非零格中前 10 的集中比例。负预测数量与最小值只描述，不裁负。

类型只用 train 真值 72h、过门格预先定义，并写入选择锁，不得用 test 定类型或挑阈值，不得在无 POI 时声称居住/商业功能区。三轴：平均流量、变异系数 CV（总体标准差，ddof=0；训练均值恰好为 0 的格单列为 mean_zero，不进入 CV 分位）、零流量比例（y=0 的小时占比）。每轴在过门格的 train-only 线性分位 1/3 与 2/3 上固定三档：x<=q1 低，q1<x<=q2 中，x>q2 高。分位边界重合允许空组。格 ID/坐标只使用已核映射，不猜测。

## 7. 输入闭环与失败处理

- 选择入口 role=select 不得打开 test 数组。全部六任务 train/val 加载、schema、共同 DC 真值与门控、以及 documented 证据闭环通过后，才进入候选或固定臂评分。共同标签被改则候选评分次数为 0，无有效锁，test 读取为 0。
- 评分入口先核锁 hash、完整性、train/val 复材与 binding，再（若 documented）重算 evidence，再对 test 文件字节 SHA 与声明/锁内 inventory 比较，不匹配则拒绝 decode。之后才 schema / 共同 truth / 写正式表。
- NaN、缺小时、数值失败、hash 不一致：阻断，保留错误，不静默丢候选或任务。负结果全部保留。
- 选择锁路径若已存在（含目录、损坏符号链接等不可用形式）必须在任何输入加载与候选评分之前拒绝；最终以 O_CREAT|O_EXCL 独占创建，不能用覆盖写入。晚到的竞态同样不得覆盖。不覆盖已有锁、旧运行目录或冻结 worktree。

## 8. 标签预算与产出

标签：72h 拟合 + 约 60 天 val 选参；若随后评分再使用已暴露的 1440h test。所有开发与合成试验须在审计日志中声明。

锁必须包含：协议与配置快照、实现指纹、全部 30 与 10 候选定义、每任务全候选 val 曲线与选择依据、五臂、train typologies、预锁 STOP 规则、输入 hash、来源证据、test 文件 inventory（仅字节，未 decode）。

真实执行仍须另写 git 外运行目录（命令、起止 CST、退出码、环境、前后 hash/mtime）。本文件对应的代码与测试不得纳入二进制缓存、权重或原 ZIP。

## 9. 明确不做

不根据已见 test 设计或改阈值；不自动发布或创建 GitHub PR；不修改已冻结的先前 val-select 实验工作树（发布文本以相对占位表示，不写出内部绝对路径）、旧锁 `val-select-lock-20261117-1646`、旧评分 `val-select-score-20261117-1657` 或 publish 工作区。
