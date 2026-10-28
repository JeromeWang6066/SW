# Agent / 本地接手交接文档（HANDOFF）

> **给下一位 coding agent 或本机同学的冷启动说明。**  
> 中文为主；论文 / 代码专名保留英文。  
> **请先读本文，再改代码。** 事实均摘自当前 `main` 上已合并文档（PR #1 / #2），勿凭记忆编造 URL、SHA、形状或指标。

**仓库 HEAD（文档撰写时 `main`）：** `1409c16` — `docs: verified NY/CHI/DC data URLs + UPSTREAM_MAP (prep, no training) (#2)`  
**公开仓：** https://github.com/JeromeWang6066/SW

---

## 1. 项目身份

| 项 | 内容 |
|---|---|
| 学生 | Tianyao Wang / Jerome（GitHub [@JeromeWang6066](https://github.com/JeromeWang6066)） |
| 项目 | Duke Kunshan University（DKU）CS **Signature Work（毕业设计）** |
| 仓库 | https://github.com/JeromeWang6066/SW （**public**） |
| 顾问认可主题 | **冻结源城教师** +（后续）**不可学习的工作日 24h Pearson ρ̃ 乘子** 调制 KD，用于跨城市网格流量 / 需求预测 |
| Phase-1 骨干 | **CrossTReS ST-Net**（先复现，不先造新网络） |
| 协议 | **STBaT 式 3-day few-shot**（目标城 train ≈ 72 小时） |
| 城市 | Source：**NY** / **CHI** → Target：**DC** |
| 实验优先级 | **Groups 1–2 先做**；Pearson / region matching = **Group 3+ hooks only**（勿超前实现最终公式） |

一句话研究问题（见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)）：在跨城网格少样本设定下，冻结源城教师 +（后续）工作日 Pearson ρ̃ 调制的 KD，能否相对标准 KD / 无迁移基线稳定提升 DC 预测？

---

## 2. 已在 `main` 上完成的事

### PR #1（已合并）— 脚手架

- 文档骨架：`README.md`、`docs/REPRODUCTION.md`、`docs/EXPERIMENT_PLAN.md`、`docs/PAPERS.md`、`docs/DATA.md`（初版）、顾问 HTML `docs/advisor/选题汇报.html`
- 配置占位：`configs/*.yaml`（`default`、`teacher_{ny,chi}`、`baseline_target_fewshot`、`student_kd_{ny,chi}2dc`）
- **诚实 stub**：`src/models/st_net.py`、`src/models/kd.py`、`src/train/train_{teacher,student_kd}.py`、`src/data/*`、`src/utils/pearson_weekday_profile.py`
- Smoke：`scripts/00_check_env.py`、`01_verify_data_layout.py`、`02_smoke_import.py`；`Makefile` 目标
- MIT `LICENSE`、`.gitignore`（zip / npy / ckpt 等不入库）

合并提交：`e09e5df` — *Scaffold prep repo: docs, configs, and honest KD/ST-Net stubs (#1)*

### PR #2（已合并）— 数据事实 + 上游映射

- **已核实** CrossTReS / STBaT zip **直链 + SHA256**、解压布局、`T×H×W` → 写入 [DATA.md](DATA.md)
- 新增 [UPSTREAM_MAP.md](UPSTREAM_MAP.md)（上游路径 ↔ 本仓 stub / 移植清单）
- 扩展 [REPRODUCTION.md](REPRODUCTION.md) Day 0–N；同步 README / PAPERS / `notes/progress.md`
- 合成形状：`scripts/03_synthetic_shapes.py`、`tests/test_synthetic_shapes.py`
- **仍无** Izadi 官方代码 URL（见 PAPERS.md TODO）

合并提交：`1409c16` — *docs: verified NY/CHI/DC data URLs + UPSTREAM_MAP (prep, no training) (#2)*

### 明确的非目标（已遵守，继续遵守）

| 非目标 | 说明 |
|---|---|
| ❌ 大文件进 git | 不提交 zip / npy / 原始 trip / checkpoint |
| ❌ 虚构训练结果 | 只记录真实评估。`--max-steps` 的输出不是实验结果。数字见 RESULTS.md |
| ❌ 虚构指标表 | 无假 MAE/RMSE；`notes/progress.md` 实跑前填 `—` |
| ❌ 多 GB 下载进 git 工作树再 commit | 本机下载到 `data/raw/`（已 gitignore）即可 |

---

## 3. 关键文档地图（先开哪个文件）

| 需求 | 打开 |
|---|---|
| **冷启动 / 本交接** | [`docs/HANDOFF.md`](HANDOFF.md)（本文） |
| 下载 URL、SHA256、形状、软链步骤 | [`docs/DATA.md`](DATA.md) |
| 上游文件 ↔ 本仓 stub、移植优先级 | [`docs/UPSTREAM_MAP.md`](UPSTREAM_MAP.md) |
| Day 0–N 操作清单（真实复现主手册） | [`docs/REPRODUCTION.md`](REPRODUCTION.md) |
| Groups 1–4 与“完成”定义 | [`docs/EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) |
| 论文 / 代码 / DOI | [`docs/PAPERS.md`](PAPERS.md) |
| 选题幻灯片 | [`docs/advisor/选题汇报.html`](advisor/选题汇报.html) |
| 实验笔记（按日追加） | [`notes/progress.md`](../notes/progress.md) |
| 超参占位 | `configs/*.yaml` |
| 模型 / 训环 / 数据 / 指标代码 | `src/**` |
| 环境 / 布局 / 导入 / 形状冒烟 | `scripts/00`–`03_*.py` |
| 无真实数据的形状断言 | `tests/test_synthetic_shapes.py` |

读序建议：**HANDOFF → DATA → UPSTREAM_MAP → REPRODUCTION → EXPERIMENT_PLAN → PAPERS**，再改 `src/`。

---

## 4. 已核实数据事实（摘自 DATA.md — 勿改数字除非重新核验）

> 下列链接于 **2026-10-27** 下载并解压核验。本仓库**不** redistrib zip。

### CrossTReS（推荐）

| 项 | 值 |
|---|---|
| 仓 | https://github.com/KL4805/CrossTReS |
| **直链** | https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip |
| 备用 raw | https://raw.githubusercontent.com/KL4805/CrossTReS/main/data/crosstres_data.zip |
| 归档 | `crosstres_data.zip`（≈ 19.6 MB 压缩；解压后 ≈ 370 MB） |
| **SHA256** | `f322a145484f183042206cf42d54bca1311784024ec384a2046df83d8f637880` |
| 解压根布局 | `NY/` `CHI/` `DC/`，内含 `Taxi{CITY}_{pickup\|dropoff}.npy`、`Bike…`、`{CITY}_poi.npy`、`{CITY}_roads.npy` 等 |

### STBaT（等价副本，二选一）

| 项 | 值 |
|---|---|
| 仓 | https://github.com/ShaunHao/stbat |
| **直链** | https://github.com/ShaunHao/stbat/raw/main/data/data.zip |
| **SHA256** | `18f8e2fab2b390498e849aae2ea835b37a314000c99d9b37896f4ef659f8cf21` |
| 解压布局 | `data/NY/`、`data/CHI/`、`data/DC/`（**文件名与 CrossTReS 相同**） |

### 网格形状（float64，解压核验）

| 城市 | `T×H×W` | H×W |
|---|---|---|
| NY | `(8784, 20, 23)` | 20×23 |
| CHI | `(8784, 17, 28)` | 17×28 |
| DC | `(8784, 21, 20)` | 21×20 |

`T = 8784`（366×24）；时间粒度 **1 小时**。

### 映射到本仓库

期望：

```text
data/processed/ny/   → 指向上游 NY（推荐软链 …/upstream）
data/processed/chi/  → CHI
data/processed/dc/   → DC
```

示例（完整步骤见 DATA.md）：

```bash
mkdir -p data/raw data/processed/{ny,chi,dc}
curl -L -o data/raw/crosstres_data.zip \
  https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip
# 校验：sha256sum data/raw/crosstres_data.zip
unzip -q data/raw/crosstres_data.zip -d data/raw/crosstres_extract
ln -sfn "$(pwd)/data/raw/crosstres_extract/NY"  data/processed/ny/upstream
ln -sfn "$(pwd)/data/raw/crosstres_extract/CHI" data/processed/chi/upstream
ln -sfn "$(pwd)/data/raw/crosstres_extract/DC"  data/processed/dc/upstream
python scripts/01_verify_data_layout.py
```

**政策：** zip / npy **不提交**（`.gitignore` 已覆盖 `*.zip` `*.npy` 与 `data/raw/**`、`data/processed/**/*`）。

### 时间划分（与上游一致；实现时勿用错）

CrossTReS / STBaT `split_x_y`：`val_num = test_num = 60 * 24`（各 **1440** 小时）→ 再对目标 train 截 `[-few_shot_days*24:]`（`few_shot_days=3` → **72** 小时）。  
训练切分用 `split_x_y`（`60*24`）。`few_shot_split_indices` 仍是 `30*24` 的旧示意图函数，不要拿它做正式划分。

---

## 5. 代码诚实状态（stub / dry-run / 拒绝）

### 行为契约

| 组件 | 状态 | 行为 |
|---|---|---|
| `STNet` | **已移植 `forward`** | Conv → ResUnit×N → LSTM → Linear → sigmoid；输出 `(B, 1, N_valid)`。无 meta 用的 `functional_forward` |
| `KDWrapper.forward` | **Group 2 已接** | 学生前向 + 冻结教师 `no_grad` 前向；`train()` 保持教师为 `eval` |
| `KDWrapper.kd_loss` / `total_loss` | **可用公式** | 与任务损失相同归约：batch 上 `mean`，格子上 `sum`。`pearson_mask` 仍是 Group 3 hook |
| `freeze_module` | **可用** | `eval()` + `requires_grad=False` |
| `train_teacher` | **Group 1 已跑** | 源城只用训练窗口。`--dry-run` 退出 0。`--max-steps` 只做接线检查 |
| `train_finetune` | **G1-C 已跑** | 源城权重复制到 DC 后全参数微调，无 KD |
| `train_student_kd` | **Group 2 已跑** | 冻结教师 + `L_task + λ L_kd`。`--dry-run` 退出 0。`pearson_multiplier=true` 退出 2 |
| `GridFlowDataset` | **可用** | 窗口对 `(x, y)`；`prepare_city_windows` 做 mask、min-max、`split_x_y` |
| `split_x_y` / `trim_few_shot` | **已对齐上游** | `val=test=60*24`；few-shot 取训练窗口末尾 `days*24` |
| `few_shot_split_indices` | **旧示意图** | 仍用 `30*24`；正式实验不要调用 |
| `upstream_style_cut_lengths` | **可用长度表** | 与上游 `60*24` 一致的切分长度 |
| `metrics.py` | **可用公式** | 仅对真实预测调用；禁止填虚构实验表 |
| `pearson_weekday_profile.py` | **Group 3 hook** | 轮廓 / 行 Pearson 助手；**跨城 matching TBD** — **勿实现 Group 3 最终算法** |

### Scripts `00`–`03` 与 tests

| 脚本 / 测试 | 做什么 | 不做什么 |
|---|---|---|
| `00_check_env.py` | 打印 Python / numpy / torch 等 | 不下载、不训练 |
| `01_verify_data_layout.py` | 检查 `data/{raw,processed/{ny,chi,dc}}` 目录 | **不下载**；空目录仅警告 EMPTY |
| `02_smoke_import.py` | 导入、YAML、合成 Pearson、metrics、split 长度 | 不用真实城市数据出指标 |
| `03_synthetic_shapes.py` | ST-Net 构造与 `forward` 形状、KD loss 形状、`8784` cut lengths | 不用真实城市数据出指标 |
| `tests/test_synthetic_shapes.py` | 同上契约的 pytest | 无真实数据 / 无训练 |

### Pearson hook 位置（Group 3+ only）

- 轮廓与 ρ̃：**`src/utils/pearson_weekday_profile.py`**
- KD 侧乘子入口：`KDWrapper.pearson_mask`；config 键 `kd.pearson_multiplier`（Group 2 YAML 均为 **`false`**）
- **Standing order：** 在顾问定稿 matching / 多通道公式前，**不要实现 Group 3**；保持 hook + TODO。

### 上游对照（移植时打开）

| 用途 | 上游 | 本仓 |
|---|---|---|
| ST-Net | `KL4805/CrossTReS` → `src/model.py` `class STNet` | `src/models/st_net.py` |
| 3-day few-shot | `ShaunHao/stbat` → `utils.split_x_y` + `data_amount` | `src/data/splits.py` |
| KD MSE 写法 | `KL4805/TransGTR` → `engine.distil_trainer` | `src/models/kd.py` |
| Izadi | **无官方公开仓** | 勿发明 URL（PAPERS.md） |

Phase-1 **不做**：CrossTReS region re-weighting 全框架、STBaT RIAM、TransGTR/ST-MLP 传感器图数据。

---

## 6. 下一步任务（有序、可执行）

面向：**本机学生 / 本地 coding agent**。Cloud / docs agent **不要**编造训练结果或把多 GB 数据拉进 git tree。

步骤 1–4 已在 2026-10-28 完成，数字见 [RESULTS.md](RESULTS.md) 与 `notes/progress.md`。下面保留当时的操作顺序。

1. **本机数据落地**  
   按 [DATA.md](DATA.md) 下载 CrossTReS（或 STBaT）zip → 校验 SHA256 → 解压 → 软链到 `data/processed/{ny,chi,dc}/` → `python scripts/01_verify_data_layout.py` → 在 `notes/progress.md` 记 URL / 日期 / SHA。

2. **移植 ST-Net + loaders + splits**  
   对照 [UPSTREAM_MAP.md](UPSTREAM_MAP.md)：`CrossTReS/src/model.py` → `st_net.py`；实现真实 `split_x_y`（`60*24`）与 `GridFlowDataset` 窗口化；`min_max_normalize` 统计量写入 config。

3. **Group 1 — 目标侧 few-shot（及源城教师）**  
   - `G1-src-ny` / `G1-src-chi`：`configs/teacher_{ny,chi}.yaml`  
   - `G1-tgt-3d`：`configs/baseline_target_fewshot.yaml`（仅 DC 3-day）  
   在 `train_teacher.py` 实现真训环；记录真实 val/test 指标到 `notes/progress.md`。

4. **Group 2 — 冻结教师 + 标准 KD**  
   `NY→DC`、`CHI→DC`；`L_task + λ_kd * L_kd`；**`pearson_multiplier=false`**。  
   配置：`configs/student_kd_ny2dc.yaml`、`student_kd_chi2dc.yaml`。

5. **仅此后：Pearson ρ̃ 乘子（Group 3）**  
   等公式 / matching 定稿；再接 `pearson_weekday_profile` → `KDWrapper.pearson_mask`。

**GPU：** 真实训练在学生本机（或自备 GPU）完成。文档 / cloud agent 只维护脚手架与诚实 stub，**禁止**提交假表。

---

## 7. 任何接手 agent 的硬约束

1. **不提交** `*.zip` / `*.npy` / `*.pt` / `*.pth` / `*.ckpt` / 大数据 / 论文 PDF。  
2. **不写虚构指标**；未实跑的表填空或 `—`。  
3. **不把多 GB 数据集下载进将被 commit 的路径**；用 gitignore 下的 `data/raw/`。  
4. **不发明 Izadi 官方代码 URL**（目前仍无；见 PAPERS.md）。  
5. 文档保持 **中文为主、专名英文** 的现有风格。  
6. 有实质代码 / 文档改动时 **开 PR 进 `main`**（不要 silent push 到 main，除非维护者明确要求）。  
7. Groups 1–2 完成前 **不要实现 Group 3 最终 matching**；只留 hook。  
8. 移植 ST-Net 时 **对照上游 `model.py`**，勿凭记忆编层形状；差异必须记入 progress / UPSTREAM_MAP。

---

## 8. Clone 后快速开始（下一位 agent）

```bash
git clone https://github.com/JeromeWang6066/SW.git
cd SW
git log -1 --oneline          # 确认接近 1409c16 或更新的 main

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 训练前按本机 CUDA 安装匹配的 torch（见 requirements.txt 注释）

# 冒烟（无需真实数据）
make env && make smoke && make shapes
# 或：
python scripts/00_check_env.py
python scripts/02_smoke_import.py
python scripts/03_synthetic_shapes.py
python -m pytest tests/test_synthetic_shapes.py -q

python scripts/01_verify_data_layout.py   # 只查目录

# CLI 诚实契约
python -m src.train.train_teacher --config configs/teacher_ny.yaml --dry-run          # exit 0
python -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml --dry-run  # exit 0
# 教师入口无 --dry-run 会开始 Group 1 训练（本机 8GB 请用 config 里的 batch 8）
# 学生 KD 入口无 --dry-run 会开始 Group 2 训练

# 然后按 §6：本机按 docs/DATA.md 下载数据 → 移植 → Group 1 → Group 2
```

可选：只读 clone 上游对照（**不要** vendoring 整树进本仓）：

- https://github.com/KL4805/CrossTReS  
- https://github.com/ShaunHao/stbat  
- https://github.com/KL4805/TransGTR（仅 KD 损失写法参考）

---

## 9. 实验分组速查

| Group | 内容 | 现在该不该做 |
|---|---|---|
| **1** | Source-only / target 3-day ST-Net 基线 | **下一步（移植后）** |
| **2** | 冻结教师 + 标准 KD，NY/CHI→DC | Group 1 可跑后 |
| **3** | 工作日 Pearson ρ̃ 乘子调制 KD | **仅 hook；勿提前实现** |
| **4** | Ablation / matching 变体 | Groups 1–2 稳定后 |

---

## 10. 交接检查清单（接手者自测）

- [ ] 已读本文 + DATA.md + UPSTREAM_MAP.md  
- [ ] `make smoke` / `make shapes` 通过  
- [ ] 理解：`train_teacher` / `train_student_kd` / `train_finetune` 无 `--dry-run` 会真训练；已跑结果见 RESULTS.md  
- [ ] 本机数据按 DATA.md 落地且 **未** `git add` zip/npy  
- [ ] 计划中的下一 PR 只做 Groups 1–2 路径上的一项（移植或基线），不夹带 Group 3 最终公式  
- [ ] 任何实跑指标只写进 `notes/progress.md`，并带来源 seed / config / commit  

---

*本文随仓库演进更新。若与 DATA.md / UPSTREAM_MAP.md 冲突，以那两份的核验表为准，并回修本文。*
