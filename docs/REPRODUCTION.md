# 复现手册（Phase-1）

> **状态：尚未执行 / 待跑。** 本文档是后续真实实验的操作清单，不是已完成实验的报告。  
> 约束：本阶段脚手架 **不下** 大数据、**不训** 模型、**不写** 虚构指标。

面向：**Groups 1–2 优先**（ST-Net 基线 + 标准 KD，NY/CHI → DC，STBaT 式 3-day few-shot）。

---

## 0. 前置检查清单

- [ ] Python ≥ 3.9，可用 GPU（CrossTReS 原文约 batch 32 / ~10GB；可按本机缩小 batch）
- [ ] `pip install -r requirements.txt`（PyTorch / CUDA 版本按机器调整，见 `requirements.txt` 注释）
- [ ] `python scripts/00_check_env.py` 通过
- [ ] `python scripts/02_smoke_import.py` 通过
- [ ] 已阅读 [PAPERS.md](PAPERS.md)、[DATA.md](DATA.md)、[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

---

## 1. 环境

```bash
git clone https://github.com/JeromeWang6066/SW.git
cd SW
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 若需 CUDA：按 https://pytorch.org 安装匹配的 torch 轮子，再装其余依赖
python scripts/00_check_env.py
```

可选：单独 clone 上游对照（**不要**把上游整树 vendoring 进本仓库）：

| 用途 | 仓库 | 备注 |
|---|---|---|
| ST-Net / 网格数据惯例 | https://github.com/KL4805/CrossTReS | Phase-1 骨干参考 |
| 3-day few-shot 协议 | https://github.com/ShaunHao/stbat | 协议对齐，非必须整套复现 STBaT |

---

## 2. 数据获取（待填具体链接）

**状态：尚未执行 / 待跑。**

1. 阅读 [DATA.md](DATA.md) 中的期望布局与许可 TODO。
2. 从 CrossTReS / STBaT 作者数据包获取 NY / CHI / DC 网格流（taxi / bike）。  
   - CrossTReS README：另有 `data` 仓库中的 `crosstres_data.zip` —— **完整 URL 待确认（TODO）**。  
   - STBaT README：另有 `data` 仓库中的 `data.zip` —— **完整 URL 待确认（TODO）**。  
   - STBaT 论文注明数据由 Jin et al. 2022（CrossTReS）整理公开。
3. 解压到本地（示例）：

```text
data/raw/                 # 原始 zip / csv，勿提交
data/processed/ny/
data/processed/chi/
data/processed/dc/
```

4. 跑布局检查（**不下载**）：

```bash
python scripts/01_verify_data_layout.py
```

5. 记录：下载日期、来源 URL、SHA256（写入 `notes/progress.md`）。

---

## 3. 预处理清单（待实现）

**状态：尚未执行 / 待跑。** 下列为规划检查项，对应 `src/data/` stubs。

- [ ] 统一时间粒度（CrossTReS / STBaT：通常 **1 小时**）
- [ ] 网格对齐：各城市 `H×W`（或 region 数）与原论文一致；记录于 config
- [ ] 时区：美东（NY/CHI/DC）—— 确认夏令时处理方式（TODO：对齐上游脚本）
- [ ] Min-max 或 z-score：与所选基线一致（STBaT 文中用 min-max）
- [ ] 划分：
  - 源城（NY 或 CHI）：train / val（STBaT：约 2:1）
  - 目标 DC：test = 最后约 2 个月；val = 其前约 2 个月；train = val 前 **3 天**（few-shot）或 0 天（zero-shot，本 Phase 默认 3 天）
- [ ] 写出 `configs/*.yaml` 中真实路径；`few_shot_days: 3`
- [ ] 冒烟：随机 batch 形状断言（`T×H×W` 或 `N×T`），**不**打印假指标

---

## 4. Group 1 — 基线（待跑）

**状态：尚未执行 / 待跑。**

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

当前 CLI：**加载 config → 设 seed → 打印 “training not implemented” 并非零退出。**

记录：seed、epoch、best val、checkpoint 路径 → `notes/progress.md`。

---

## 5. Group 2 — 标准 KD（待跑）

**状态：尚未执行 / 待跑。**

协议要点：

1. 冻结 Group 1 源城教师权重（`requires_grad=False`）。
2. 目标 DC：STBaT 式 **3-day** 学生训练。
3. 损失：`L = L_task + λ_kd * L_kd`（λ 见 config；**无** Pearson 乘子 —— 那是 Group 3）。
4. 任务对：`NY→DC`、`CHI→DC`（可扩展双向，但 Phase-1 以 →DC 为主）。

```bash
python -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml
python -m src.train.train_student_kd --config configs/student_kd_chi2dc.yaml
```

---

## 6. Groups 3–4（本手册仅占位）

- **Group 3：** 在 KD 上乘 **不可学习** 的工作日 24h Pearson ρ̃（见 `src/utils/pearson_weekday_profile.py`）。  
  **匹配 / 多通道细节 TBD —— 勿在未定稿前写死公式实现。**
- **Group 4：** Ablation / matching 变体 —— 待 Groups 1–2 稳定后设计。

---

## 7. 评估与日志

指标 stub：`src/eval/metrics.py` → MAE / RMSE / MAPE。

建议日志字段（实现后）：

- `run_id`, `group`, `source`, `target`, `seed`, `few_shot_days`
- `mae`, `rmse`, `mape`（仅真实跑完后填写）
- `checkpoint`, `config_path`, `git_commit`

可选：本地 CSV / TensorBoard；若用 W&B，密钥放 `.env`（已 gitignore）。

---

## 8. 复现“完成”定义（Phase-1）

见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。简要：

1. 能独立从空环境装好依赖；
2. 数据路径与划分可核对；
3. Group 1–2 各至少一组 seed 跑通并记录真实指标；
4. 无虚构表格；checkpoint 与 config 可回溯。

---

## 9. 已知 TODO

| 项 | 说明 |
|---|---|
| 数据 zip 直链 | CrossTReS / STBaT `data` 仓库完整 URL 待填 |
| ST-Net 层细节 | 对照 `KL4805/CrossTReS` 的 `model.py` 实现，勿凭记忆编造 |
| 网格尺寸 | 以解压后实际数组为准写入 config |
| Pearson / matching | Group 3+；仅 hook |
