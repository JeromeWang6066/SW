# 复现手册（Phase-1）

> **状态：尚未执行 / 待跑。** 本文档是后续真实实验的操作清单，不是已完成实验的报告。  
> 约束：本阶段脚手架 **不下** 大数据进 git、**不训** 模型、**不写** 虚构指标。  
> 上游对照总表：[UPSTREAM_MAP.md](UPSTREAM_MAP.md) · 数据直链：[DATA.md](DATA.md)  
> Agent / 本地连续接手上下文：[HANDOFF.md](HANDOFF.md)

面向：**Groups 1–2 优先**（ST-Net 基线 + 标准 KD，NY/CHI → DC，STBaT 式 3-day few-shot）。

---

## Day 0 — 前置检查清单

- [ ] Python ≥ 3.9，可用 GPU（CrossTReS 原文约 batch 32 / ~10GB；可按本机缩小 batch）
- [ ] `pip install -r requirements.txt`（PyTorch / CUDA 版本按机器调整，见 `requirements.txt` 注释）
- [ ] `python scripts/00_check_env.py` 通过
- [ ] `python scripts/02_smoke_import.py` 通过
- [ ] `python scripts/03_synthetic_shapes.py` 通过（无真实数据）
- [ ] 已阅读 [HANDOFF.md](HANDOFF.md)、[PAPERS.md](PAPERS.md)、[DATA.md](DATA.md)、[UPSTREAM_MAP.md](UPSTREAM_MAP.md)、[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

```bash
make env && make smoke
python scripts/03_synthetic_shapes.py
python -m src.train.train_teacher --config configs/teacher_ny.yaml --dry-run
python -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml --dry-run
```

---

## Day 1 — 环境与上游对照 clone

```bash
git clone https://github.com/JeromeWang6066/SW.git
cd SW
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 若需 CUDA：按 https://pytorch.org 安装匹配的 torch 轮子，再装其余依赖
python scripts/00_check_env.py
```

可选：单独 clone 上游对照（**不要**把上游整树 vendoring 进本仓库）：

| 用途 | 仓库 | 本仓库映射 |
|---|---|---|
| ST-Net / 网格数据 | https://github.com/KL4805/CrossTReS | → `src/models/st_net.py`，见 UPSTREAM_MAP §1 |
| 3-day few-shot 协议 | https://github.com/ShaunHao/stbat | → `src/data/splits.py`，见 UPSTREAM_MAP §2 |
| KD 损失写法（传感器图） | https://github.com/KL4805/TransGTR | → `src/models/kd.py`，见 UPSTREAM_MAP §3 |
| 同城 KD 对照 | https://github.com/zhangjunfeng1234/ST-MLP | 仅阅读损失组合 |

---

## Day 2 — 数据获取（链接已核实）

**状态：本机下载即可；勿提交 zip/npy。** 完整表见 [DATA.md](DATA.md)。

1. **推荐包（CrossTReS）**  
   - 直链：https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip  
   - SHA256：`f322a145484f183042206cf42d54bca1311784024ec384a2046df83d8f637880`  
   - 解压得 `NY/` `CHI/` `DC/`，内含 `Taxi*_pickup.npy` 等  

2. **等价副本（STBaT）**  
   - 直链：https://github.com/ShaunHao/stbat/raw/main/data/data.zip  
   - 解压布局为 `data/{NY,CHI,DC}/`（文件名相同）  

3. 映射到本仓库：

```bash
mkdir -p data/raw data/processed/{ny,chi,dc}
curl -L -o data/raw/crosstres_data.zip \
  https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip
unzip -q data/raw/crosstres_data.zip -d data/raw/crosstres_extract
ln -sfn "$(pwd)/data/raw/crosstres_extract/NY"  data/processed/ny/upstream
ln -sfn "$(pwd)/data/raw/crosstres_extract/CHI" data/processed/chi/upstream
ln -sfn "$(pwd)/data/raw/crosstres_extract/DC"  data/processed/dc/upstream
python scripts/01_verify_data_layout.py
```

4. 记录到 `notes/progress.md`：日期、URL、SHA256。

5. 形状速查（已核验）：NY `(8784,20,23)` · CHI `(8784,17,28)` · DC `(8784,21,20)` · 1h。

---

## Day 3 — 预处理清单（待实现）

**状态：尚未执行 / 待跑。** 对照 UPSTREAM_MAP：`utils.split_x_y` + `min_max_normalize`。

- [ ] 读入 `Taxi{CITY}_pickup.npy` / `dropoff`（或 Bike）；统一 **1 小时**  
- [ ] 网格：`H×W` 写入 config（见 DATA.md 表）  
- [ ] 时区：包内已是小时网格 —— 勿另猜夏令时偏移  
- [ ] Min-max（percentile 0.999）；显式记录 fit 自源还是目标  
- [ ] 划分（与上游一致）：  
  - val / test 各 **`60*24`** 小时  
  - 目标 train：`[-few_shot_days*24:]`，默认 **3**  
  - 源城：按 CrossTReS/STBaT 源侧惯例（见 UPSTREAM_MAP）  
- [ ] 写出 `configs/*.yaml` 真实路径；`few_shot_days: 3`  
- [ ] 冒烟：`scripts/03_synthetic_shapes.py` + 真实 batch 形状断言（**不**打印假指标）

---

## Day 4+ — Group 1 基线（待跑）

**状态：尚未执行 / 待跑。** 移植 ST-Net：上游 `KL4805/CrossTReS/src/model.py` → `src/models/st_net.py`。

| 实验 ID | 描述 | 配置占位 |
|---|---|---|
| G1-src-ny | 源城 NY 上训练 ST-Net（教师候选） | `configs/teacher_ny.yaml` |
| G1-src-chi | 源城 CHI 上训练 ST-Net | `configs/teacher_chi.yaml` |
| G1-tgt-3d | 仅用 DC 3-day 训练 ST-Net（无迁移） | `configs/baseline_target_fewshot.yaml` |

```bash
# 实现完成后：
python -m src.train.train_teacher --config configs/teacher_ny.yaml
python -m src.train.train_teacher --config configs/baseline_target_fewshot.yaml
```

当前 CLI：**加载 config → 设 seed → `--dry-run` 退出 0；无 dry-run 则明确拒绝并非零退出。**

记录：seed、epoch、best val、checkpoint 路径 → `notes/progress.md`。

---

## Day N — Group 2 标准 KD（待跑）

**状态：尚未执行 / 待跑。** KD 写法对照：TransGTR `engine.distil_trainer`（MSE）、ST-MLP `run_student.py`。

协议要点：

1. 冻结 Group 1 源城教师权重（`requires_grad=False`）。  
2. 目标 DC：STBaT 式 **3-day** 学生训练。  
3. 损失：`L = L_task + λ_kd * L_kd`（λ 见 config；**无** Pearson 乘子 —— Group 3）。  
4. 任务对：`NY→DC`、`CHI→DC`。

```bash
python -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml
python -m src.train.train_student_kd --config configs/student_kd_chi2dc.yaml
```

---

## Groups 3–4（本手册仅占位）

- **Group 3：** 在 KD 上乘 **不可学习** 的工作日 24h Pearson ρ̃（见 `src/utils/pearson_weekday_profile.py`）。  
  **匹配 / 多通道细节 TBD —— 勿在未定稿前写死公式实现。**
- **Group 4：** Ablation / matching 变体 —— 待 Groups 1–2 稳定后设计。

---

## 评估与日志

指标：`src/eval/metrics.py` → MAE / RMSE / MAPE（对照 STBaT `evaluate`；注意反归一化）。

建议日志字段（实现后）：

- `run_id`, `group`, `source`, `target`, `seed`, `few_shot_days`
- `mae`, `rmse`, `mape`（仅真实跑完后填写）
- `checkpoint`, `config_path`, `git_commit`

可选：本地 CSV / TensorBoard；若用 W&B，密钥放 `.env`（已 gitignore）。

---

## 复现“完成”定义（Phase-1）

见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。简要：

1. 能独立从空环境装好依赖；  
2. 数据路径与划分可核对（DATA.md + UPSTREAM_MAP）；  
3. Group 1–2 各至少一组 seed 跑通并记录真实指标；  
4. 无虚构表格；checkpoint 与 config 可回溯。

---

## 已知 TODO

| 项 | 说明 |
|---|---|
| ST-Net 层细节 | 对照 `KL4805/CrossTReS` 的 `src/model.py` 移植，勿凭记忆编造 |
| `splits.py` 与 `60*24` | schema helper 目前用 `30*24` 近似月；实现时改为与上游一致 |
| Pearson / matching | Group 3+；仅 hook |
| Izadi 官方代码 | 仍未找到公开仓（PAPERS.md） |
| 原始开放数据许可链接 | 训练前补 TLC 等（DATA.md） |
