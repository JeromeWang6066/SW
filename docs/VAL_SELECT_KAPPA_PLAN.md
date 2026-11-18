# 用官方 val 选强度的三阶段协议（中心化仿射）

冻结时间：2026-11-17 14:14:58 Asia/Shanghai（CST, UTC+8）  
本修订：2026-11-17 14:35 Asia/Shanghai（CST, UTC+8），纳入验证者 14:19 静态审查的 B1–B4 及必要项  
v3 修订：2026-11-17 14:40 Asia/Shanghai（CST, UTC+8），纳入验证者 14:32 定向复审收口；v1/v2 审计目录保留不改  
v4 修订：2026-11-17 14:45 Asia/Shanghai（CST, UTC+8），仅收口测试隔离与「先共同 truth，后候选评分」的执行顺序；算法/网格/研究协议不变；v1–v3 审计目录保留不改  
输入协议修订：2026-11-17 16:21 Asia/Shanghai（CST, UTC+8），新增 `val-select-input-bundle-v1` / 显式 opt-in 的 `documented_from_bundle` 适配；不改算法、候选网格、臂或平局准则；v1–v4 审计目录保留不改  
证据闭环修订：2026-11-17 16:32 Asia/Shanghai（CST, UTC+8），仅修 ZIP 信任锚重算与 test 字节核验（B1/B2）；算法/网格/strict/来源层不改；既有 input-bundle-v1 审计目录保留不改  
依赖的 PR20 明确 head：`a0a055f9bcb2cd9ffff7ceb04d9c68a0c4371a2c`  
本分支：`exp/val-select-kappa`  
工作树：`/workspace/sw-experimenter/SW-val-select`  
协议编号：`val-select-kappa-v3`  
输入协议编号：`val-select-input-bundle-v1`

本页是拟入库的执行协议，不是实验结果。合成单测可在本工作树运行；真实评分尚未开始，也尚未获审查放行。不改写 PR20 的 72h 留一天选参协议。不 push、不开 PR。v1 审计记录保留在 git 外目录，不被本修订覆盖。

## 0. 标签预算与已暴露评估（必须写进每份报告）

本协议在 72h 训练窗上拟合逐格参数，并用官方约 60 天 DC 验证集全城 MAE 选择强度（m 或 κ）。这不是「全流程只有 72h 标签」。既有 test 在先前工作中已经被看过，本轮 test 只做探索性复评，不得回改选择或协议。新锁箱需要未来未见数据。

## 1. 分支关系与建议的 PR stack（本轮不发布）

今天（2026-11-17 14:09 Asia/Shanghai）经 connector 与安全 fetch 核对（本修订未再向 GitHub 发请求；下列 SHA 仍是当时记录）：

| 对象 | 状态 | SHA | 时间（Asia/Shanghai） |
| --- | --- | --- | --- |
| GitHub `main` / 本地部署树 `/workspace/sw-experimenter/SW` | 干净，未改 | `c040b5b3c996381ff4f1b65684b7956041670fb3` | 提交 2026-11-11 16:53:06 |
| 开放 PR #20 `exp/cell-affine-v2` | open，未合并，mergeable clean | head `a0a055f9bcb2cd9ffff7ceb04d9c68a0c4371a2c` | 提交 2026-11-17 13:00:43；PR 创建 13:01:51 |
| 开放 PR #19 `docs/advisor-2026-11-16` | open，未合并，mergeable clean | head `b38caef1f9bd091dac21fb47dad70793a9c6f6d5` | 提交 2026-11-16 14:30:36 |
| 本工作分支 | 仅本地 | 由 PR20 head 分出 | 见上 |

PR19 与 PR20 都基于同一 `main`。文件交叉只有 `docs/profiles/README.md`。PR19 是导师四项文档，PR20 是中心化仿射对照。二者互不依赖。

建议（先审，本轮不 push、不开 PR）：

- 成果 PR1（阶段 1 代码/plan/单测及日后真实结果）以 PR20 head 为 base，而不是 `main`，也不叠在 PR19 上。
- 成果 PR2 叠在 PR1 上，PR3 叠在 PR2 上。
- 若 PR20 先合入 `main`，再把 PR1 retarget 到更新后的 `main`。
- 不 merge 任何现有或新 PR。未来真正开 PR 需要用户直接批准，或明确授权由 SW 代表发布；仅有 agent 转述不够。

## 2. 源码核查（待与真实输入对照的已核实代码事实）

下列来自 PR20 head 与当前 `src/`，不是新实验结果。

- 微调配置 `configs/finetune_ny2dc.yaml` / `finetune_chi2dc.yaml` 无 `calendar` 键。`train_finetune.py` 用 `early_stop_dataset(prepared)`，缺省即官方完整 val。**当前源码支持这条 early-stop 路径。** 六个既有 checkpoint 是否确实按该路径、用官方 val 训练，属于实际来源待核，不得断言每份权重已证实经过 val early stop。本协议若再用同一官方 val 选强度，在源码路径成立的前提下属于 val 标签的第二次使用，必须披露；这也带来对未来分布的风险，但不能用源码支持来代替对具体权重的来源核验。不强制重训；干净协议另线。
- `min_max_normalize` 在完整 `raw` 序列上算 99.9 分位与 `min=max(0, 最小值)`，再用于全部划分。**归一化统计包含 val/test 时段**，不得声称无泄漏。
- PR20 与训练评估都不把预测裁成非负。本协议不额外改裁负，也不把 a 限制在 [0,1]。κ 不是 a。
- 门控：72h 原始上车合计 ≥150，期望 207 个有效格、50 个过门格。参数只在训练 72h 上求，p̄ 固定后用于 val/test。train 缓存必须带原始 72h 合计，加载时按冻结门限重算并与 `eligible` 对照。
- 旧公式保持 λ=m·n。正 m 保持原惩罚目标。m=0 与 κ=0 走同一条中心化 OLS 路径；秩亏常数输入约定 α=0、c=r̄（a=1、b=r̄）。近常数阈值冻结为 `range(p) <= 1e-15 * max(1, max|p|)`，是在固定 DC 计数单位下预定的近秩亏数值约定，两 solver 共用，并在锁中记录触发格。不按结果改，不加额外 epsilon 改变适定目标，也不声称在任意单位缩放下严格不变。identity 仍是独立分支。
- PR20 预测缓存键为 `{source}_seed{seed}_{split}.npz`，train 形状 (72, 207)，val/test 期望 (1440, 207)，仅有效格；seed 42 权重名为 `finetune_{ny,chi}2dc.pt`，43/44 为 `finetune_{ny,chi}2dc_seed{43,44}.pt`。**PR20 缓存本身没有完整 provenance / 显式 time_index。本协议不再允许按 shape 重建小时，也不把文件名当作 source/seed 的证明。** 缺元数据的 legacy 文件必须明确阻断并要求补证，不得猜测重建。
- 默认时间范围：train 必须是 5832–5903。val 默认 [5904, 7344)、test 默认 [7344, 8784)。在 `documented_from_bundle` 模式下，这些小时来自 bundle README（train 另有脚本常量 `LABEL_START`/`N_HOURS` 的文本支持），状态标注为 `documented_range_from_bundle_pending_independent_raw_recompute`。捆绑的预测脚本只 *import* `prepare_city_windows`，该依赖未随包提供，因此 **不能声称 val 起点已由捆绑脚本独立代码证明**。跨 split 必须相邻连续，所有任务的 DC 标签与列顺序必须一致。
- PR20 选参是 72h 留一天，平局容差 1e-9 取更大强度，且基模已见完 72h，不是严格 end-to-end holdout。本协议不回改那套旧规则。

## 3. 统一冻结规则

- 研究单位：NY→DC 与 CHI→DC，种子预锁 {42, 43, 44}。不是不同目标城，也不在阶段中途按成绩丢种子。可用种子集合若要变更，只能先报告并做新版本冻结（`seed_freeze_id`），不能自动 drop。
- 六臂：FT（精确恒等）；固定旧仿射 m=1（λ=n）；固定新中心化 κ=1；val 调 m 的旧法；val 调 κ 的新中心化；真正仅平移 val 调 κ。
- 候选：`{0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16}` + 显式 `identity`。identity 不是很大的有限值，也不是用 `inf` 偷偷代替的同一路径标记以外的近似。三法同预算，不细化搜索。
- 主指标：已裁到有效格后的全城 MAE（与 PR20 `score_arrays` 相同约化）。辅报 RMSE、原 positive-only MAPE、门控子集。
- 选择：argmin val 全城 MAE。数值并列容差为 `1e-10 * max(1, |min_loss|)`，先 identity，再更大有限强度。这是数值容差，不是可接受性能损失平台，不按结果改 δ。逐候选损失与选择依据全部记录。
- κ=0 / m=0 且预测为常数（秩亏）：统一 α=0、c=r̄，即 a=1、b=r̄（OLS 截距解）。非常数时 m=0 与 κ=0 必须走同一中心化 OLS 路径，预测逐点一致。identity 独立。NaN、缺小时、数值失败阻断受影响的选择，不静默丢掉某个候选再在剩余里选。
- 全程 float64 累加误差。
- 各方法各自选择、各自报告，不用 test 挑选部署方法。
- 执行顺序：三阶段全部非 test 选参先做完并锁 hash，再按 1→2→3 做 test 评分汇报。不能用阶段 1 的 test 反馈改阶段 2/3。选择器与 test 评分入口分离。

## 3b. 锁绑定与评分入口（B1–B3）

可评分锁必须绑定：

- plan 文件内容 SHA256；
- 实现快照：所有实际用到的模块（`src/val_select/*`、runner `scripts/20_val_select_kappa.py`，以及 plan 本身）逐文件 SHA256 与合计 fingerprint；
- 完整协议配置快照及其 hash（候选、臂、种子、门控、时间默认值、近常数阈值、并列容差等）；
- 每个任务的 train/val 文件 hash、sidecar hash，以及从载荷重算的 manifest 内容（source/seed/split、DC clipped-count 单位、cell IDs 顺序、精确 time_index、归一化拟合范围、checkpoint/config/生成 commit、门控重算结果）。

`score-test` 在调用 test loader 之前必须：核验 lock hash、核验预锁任务/种子/方法/候选完整且 chosen 属于冻结候选、stage1/2/3 非空、stage2 三个 rolling 轮次完整、stage3 种子集合一致、无 blocked、`scorable=true`；再重新计算 plan/实现/配置快照，并重新读取 train/val、比对文件 hash 与 manifest，且按时间/列精确比较实际 y 与 train raw gate，而不是只比对 manifest。任一变化在打开 test 之前拒绝。打开 test 后先对全部 test 做统一预检（同一 DC 标签、列顺序、跨 split 身份/单位/归一化/checkpoint 一致；不同 seed 不要求同一权重 hash），全部通过后才正式评分并写表。

payload 与 sidecar 的关键重复字段先规范化再比较，冲突拒绝；合法等价双表示通过。recorded 来源拒绝 null/占位，并校验 SHA256/commit 格式与协议分位表示。格式合法只说明可校验，不能声称历史来源已实证。seed 与 time_index 在转 int 前必须是有限整数；raw gate 计数有限非负；eligible 只接受严格 bool 或 0/1。全部任务的 train/val 必须先完成加载、schema 与共同 DC truth/gate 核验，全部通过后才进行任何候选评分。默认精确比较，不见差放宽。

实现快照包含 `src/__init__.py` 与全部实际用到的模块。plan/solver 指纹反例只允许改临时镜像 root，不得改真实工作树后再 finally 恢复。v2 单测曾改真实 plan/solver 再恢复，其「仅临时文件」陈述不准确；v2 日志与 patch 保留不改，纠正说明写在 v3 审计目录。

缺任务、缺阶段或空 stage1 不能生成可评分锁（只可写 `*.blocked.json` 诊断文件）。此类锁或被改坏后重算 hash 的残缺锁，评分入口必须拒绝，且 test loader 调用次数为 0。

库存（`inventory`）只做 stat 与 sha256，不解压任何预测数组，包括 test。test 结构核验留在评分入口。锁文件不被评分流程改写；不存在持久的 `test_touched` 标志。重复评分不得覆盖已有报告或运行目录，必须使用新的排他 run dir，并在审计中如实记录命令、Asia/Shanghai 时间、退出码、实现 fingerprint，以及此次是否为合成夹具。

## 4. 三阶段

**阶段 1。** 每个 source×seed×可调方法：只在 train 72h 拟合各候选，在完整 val 上打全城 MAE，锁定一个强度。固定臂不调参。记录曲线、是否选中 identity、固定对照与调参对照。全部预锁种子齐报；缺失记缺失，不挑种子。

**阶段 2。** 使用同一 train 拟合与同一缓存。按 crop 后有效 label 时间戳切 4 个连续块；先核完整 1440h、默认起止 [5904, 7344)，有缺口或不足则失败，不硬凑 4×360，不静默补小时。完整时每块自然为 360h，这是完整窗口的推论不是填充。报告每候选每块 MAE 与块内最优强度，作为稳定性诊断。整段（完整 val）选择与阶段 1 相同，因此最终 test 必须与阶段 1 相同，不得称新提升。独立的 expanding rolling：前一块（约 15 天）选强度 → 下一块评分；前两块选 → 第三块评分；前三块选 → 最后一块评分。参数仍只在 train 72h 拟合，val 不进入拟合。同一 prefix 预算也用于固定基线，并在同一后续块上评分；后三块评分（可调臂与固定基线）都按有效观测数加权汇总。不用「先看完整 val 再选出的分数」当作公平的滚动对手，也不引入 minmax / mean+SD 等额外目标。不用 test 选择滚动策略。

**阶段 3。** 每个 source×方法，在事前锁定的可用种子上，对每个候选取各 seed 的完整 val 绝对 MAE 等权平均，选一个共享强度。各 seed 的 a/b 或 c 仍用该 seed 自己的 72h 训练窗拟合。独立选择与共享选择必须在同一预锁种子集合上比较。缺种子必须在阶段 1 前锁定集合并报告，不能看成绩后动态丢弃。多 seed 共用同一 DC，不是新的独立数据。报告每 seed 的 test 配对差及均值/范围，但不把「哪条路线更好」交给 test。

## 5. 输入缺口与可复用预测缓存最小规格

截至 2026-11-17 14:08–14:14 Asia/Shanghai，仅在 `/workspace/sw-experimenter` 内盘点（未搜用户主目录、未访注册电脑；本修订未再下载或补训）：

- `data/processed/{ny,chi,dc}/` 与 `checkpoints/` 仍只有 `.gitkeep`。
- 无 `*.pt` / `*.npy` / `*.npz` 预测缓存。
- `docs/profiles/*.json` 是已入库的聚合文档，**不能当作完整预测**。
- 因此真实评分被 block。不补训、不伪造、不下载用户电脑文件。真实 cache 格式在拿到文件前不得凭猜测当作已验证；本协议先给出规范 schema 与适配接口，文件到达后再核验。

优先请 Jerome 提供预测缓存（避免在助手电脑重放整段微调）。六个任务 × 三个独立文件，test 必须分开存放：

`{ny|chi}_seed{42|43|44}_{train|val|test}.npz`

载荷或 sidecar manifest 必须自证，不能靠文件名：

- `prediction`、`target`：float64，形状 `(T, C)`，单位必须声明为 `dc_clipped_count`（反归一化到 DC 截断计数）。C 为有效格（期望 207）。train T=72，val/test 期望 1440。
- `source`、`seed`、`split`：写在载荷/sidecar 内，并与请求参数一致。
- `cell_ids`：长度为 C 的有序唯一标识，跨任务必须同一顺序。
- `time_index`：int64，长度为 T，必须显式给出。不得按 shape 补小时。train 核 5832–5903；val/test 按真实生成链核准，协议默认分别为 5904–7343 与 7344–8783，且跨 split 连续。
- `value_scale`、`min_val`、`norm_percentile`、`norm_fitted_on`（须写明拟合时间范围，例如完整 raw 序列）。
- `checkpoint_path`、`checkpoint_sha256`、`config_id`、`generator_commit`、`provenance_kind`。真实评分只接受 `recorded` 且字段通过格式校验（64 位 hex SHA256、40 位 hex commit、协议分位 99.9、`norm_fitted_on=full_raw_series`，拒绝 null/占位）。这只说明格式可核，不能当成历史来源已实证。缺项必须补证。`synthetic` 仅用于合成单测并需显式 `--allow-synthetic`。
- train 另需：`eligible`、`raw_train_counts`、`gate_threshold=150`；加载时按原始 72h 合计重算门控。期望 50 个过门格。

不需要（若缓存完整且 hash 可核）：源城原始数组、完整 zip、为补缺而重训的新权重、路由缓存、逐格大图。若只能提供权重与 `data/processed/dc`，也可以在放行后于本机按 PR20 `stage_predict` 生成缓存，但仍须把 test 文件与选择器隔离，并补齐上述 provenance。公开数据直链见 `docs/DATA.md`（CrossTReS zip ≈19.6 MB / 解压 ≈370 MB，SHA256 已记录），本轮不自动下载。

## 6. 代码入口与合成测试（不是真实实验）

```bash
source /workspace/sw-experimenter/venv/bin/activate
cd /workspace/sw-experimenter/SW-val-select
# 历史记录：最初部署的 /workspace/sw-experimenter/venv 在 2026-11-17 16:13 Asia/Shanghai 已不存在。
# 恢复后的隔离解释器（不要假装从未中断）：
/workspace/sw-experimenter/venv-20261117-r2/bin/python tests/test_val_select_kappa.py
# 真实数据就绪且审查放行后才可：
# python scripts/20_val_select_kappa.py inventory --cache-dir PATH
# python scripts/20_val_select_kappa.py lock-selections --cache-dir PATH --lock-path PATH
# python scripts/20_val_select_kappa.py score-test --cache-dir PATH --lock-path PATH --report-path PATH \
#   --run-dir PATH --i-understand-this-is-exploratory
```

`lock-selections` 只读 train/val，仅在完整无 blocked 时写入可评分锁。`score-test` 在核过 lock hash、plan/实现/配置与 train/val manifest 之后才读 test，写入新的排他 run dir，按 1→2→3 写报告，不覆盖旧输出，不改锁文件。数据、权重、逐格大缓存、锁文件中的大数组都不入库；本协议文档与算法代码拟入库。

合成单测使用临时目录中的夹具文件，退出后删除；通过只说明算法、绑定与隔离可用，不能写成真实实验已完成，也不能当成已对真实 cache 格式完成核验。

## 7. 未做与阻塞

- 未真实读取 test 评分，未读真实数据或 checkpoint（目录为空，仅盘点）。
- 未 push、未开 PR、未在 GitHub 评论，本轮也不做 git commit。
- 合成测试通过只说明算法与隔离可用，不能写成真实实验已完成。
- 阻塞：预锁六任务的 train/val/test 预测缓存（及权重 hash 与完整 provenance）仍缺。等待负责人向 Jerome 收集，并等待审查放行后再评分。

## 8. 输入协议 `val-select-input-bundle-v1`（documented_from_bundle）

这是输入模式冻结，不是新的选参算法。`val-select-kappa-v3` 的臂、候选 `{0,.125,.25,.5,1,2,4,8,16}+identity`、平局规则与 solver 均不变。严格 `recorded` 路径保持不改，也没有从 recorded 失败回退到 documented 的通道。

- 显式 opt-in：`allow_documented_from_bundle=True`（CLI `--allow-documented-from-bundle`）。缺省拒绝。不得把 PR20 原生 NPZ 的缺失字段填成 recorded 证明。
- 三层来源：（1）已核 zip/成员 SHA256 与数组事实（p/y、raw gate 重算、同 split 目标字节一致）；（2）文档约定：任务身份、单位 `dc_clipped_count`、value_scale=234 与 min/max 来自 metadata/脚本文本而非值域猜测、时间 `documented_range_from_bundle`；（3）未知/声称：fine-tune git commit 为 `unknown`；`7a04ca4` 只是 `documented_short_revision`，不得补成 40 位或称为 recorded；文本权重 hash 为 `claimed`，因 `.pt` 不在包内未重哈希；全序列 raw 的 99.9 分位统计未独立重算。
- train 标签相对 `clip(raw, 0, 234)` 的最大绝对差冻结为 `7.62939453125e-06`，不是字节相等。六任务同一 split 的 y 则是字节一致。二者必须分开报告。
- 原数组只只读复制，不重排、不重归一化、不静默裁切。捆绑脚本只读文本、不执行，不联网下载用户数据。
- 选择锁绑定：原 zip SHA256、已核 train/val 成员、README/脚本/gate 元数据、字段映射、输入模式、适配实现指纹。test 校验和只是 SHA256SUMS/中央目录上的 declared，待评分阶段实核；适配与 select 不得打开真实 test 数组。
- 本轮如在审查放行后用该 bundle 做配对，只作为探索性同输入比较，并透明披露上述来源限制，不因缺少完美历史来源要求重训。
- 证据闭环（不改算法/网格/准则）：选择与评分前必须从**当前 ZIP 容器**重算信任锚（整体 sha256、非 test 白名单成员哈希、SHA256SUMS 文本中的声明），并与锁内证据做完整规范内容比对；`evidence_hash` 必须由内容重算，相同 hash 字符串不能豁免内容差异；不得把 `input_evidence` 与自身比较。已验收原包锚为 `99594eaa2b200fe4b8be51823e5716d6152de3186ccf05afb21b4bba81ef6845`，抽出目录加自改 SHA256SUMS 不能冒充该包。SHA256SUMS 缺少必需条目必须拒绝。评分在 NPZ 解码前先对 test 文件字节做 sha256 并与 declared 比较，两者不是同一步。

