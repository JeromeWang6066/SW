# 实验进度笔记（模板）

> 每次实验 / 数据操作追加一节。不要填写虚构指标。

## 元信息

- 学生：Tianyao Wang (Jerome)
- 仓库：https://github.com/JeromeWang6066/SW
- 当前阶段：脚手架 / Prep（Groups 1–2 尚未实跑）

---

## 2026-10-28 — 交接文档 HANDOFF（Agent / 本地连续接手）

**目标：** 写清冷启动上下文，让另一位（本机）coding agent 或同学不丢上下文地继续；不开真实训练、不下载大数据进 git。  
**配置：** 无训练配置变更  
**Git commit：** （本 PR）  
**机器 / GPU：** Cloud Agent（文档 only）

### 做了什么

- 新增 `docs/HANDOFF.md`：项目身份、PR #1/#2 已完成项、文档地图、DATA.md 核实事实（URL/SHA/形状）、stub/dry-run 契约、有序下一步、硬约束、clone 后命令。
- README 顶部「快速入口」与文档入口突出链接 HANDOFF；`docs/REPRODUCTION.md` 增加指向 HANDOFF 的一行。
- 事实均摘自当时 `main`（`1409c16`）上 DATA / UPSTREAM_MAP / REPRODUCTION，未编造指标或 Izadi URL。

### 数据

- 无新下载；沿用 2026-10-27 已核实直链与 SHA（见上一节 / DATA.md）。

### 问题 / 下一步

- 本机按 HANDOFF §6 / DATA.md：下载 → 软链 → 移植 ST-Net → Group 1 → Group 2  
- Group 3 Pearson：仍仅 hook，勿超前实现  

---

## 2026-10-27 — 核实数据直链 + 上游映射文档

**目标：** 在不下载进 git、不训练、不编造指标的前提下，钉死 NY/CHI/DC 数据包 URL，并写清上游↔本仓映射。  
**配置：** 无训练配置变更（仍用 stub YAML）  
**Git commit：** （本 PR）  
**机器 / GPU：** Cloud Agent（无 GPU 训练）

### 做了什么

- 核实 CrossTReS / STBaT / SFMGTL 仓内 zip 直链；在 `/tmp` 解压核验布局与 `npy` 形状（**未**提交数据）。
- 更新 `docs/DATA.md`（URL、SHA256、`T×H×W`、有效格数、本仓映射步骤）。
- 新增 `docs/UPSTREAM_MAP.md`（ST-Net / few-shot / KD / metrics；stub vs 下一步）。
- 扩展 `docs/REPRODUCTION.md` Day 0–N；同步 README / PAPERS / `data/README.md` TODO 指针。
- 新增合成形状冒烟：`scripts/03_synthetic_shapes.py`、`tests/test_synthetic_shapes.py`。
- Izadi 官方代码：**仍未找到** → 保持 PAPERS.md TODO。

### 数据

- CrossTReS 来源 URL：https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip  
- SHA256：`f322a145484f183042206cf42d54bca1311784024ec384a2046df83d8f637880`  
- STBaT 等价 URL：https://github.com/ShaunHao/stbat/raw/main/data/data.zip  
- SHA256：`18f8e2fab2b390498e849aae2ea835b37a314000c99d9b37896f4ef659f8cf21`  
- 布局检查：本机仅验目录骨架；真实 zip **不入库**。

### 运行命令

```bash
python scripts/02_smoke_import.py
python scripts/03_synthetic_shapes.py
python -m src.train.train_teacher --config configs/teacher_ny.yaml --dry-run
python -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml --dry-run
```

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 本阶段无训练 |
| test | — | — | — | 本阶段无训练 |

### 问题 / 下一步

- 本机按 DATA.md 下载 zip → 软链到 `data/processed/{ny,chi,dc}`  
- 移植 CrossTReS `src/model.py` → `STNet.forward`  
- 将 `splits.py` 与上游 `60*24` 对齐后实现 Group 1  

---

## YYYY-MM-DD — 标题

**目标：**  
**配置：** `configs/...yaml`  
**Git commit：**  
**机器 / GPU：**  

### 做了什么

- 

### 数据

- 来源 URL：
- SHA256：
- 布局检查：`scripts/01_verify_data_layout.py` 结果：

### 运行命令

```bash
```

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | | | | |
| test | | | | |

### 问题 / 下一步

- 

---

## 待办速记

- [x] 确认 CrossTReS / STBaT 数据 zip 直链并写入 `docs/DATA.md`
- [x] 编写 `docs/UPSTREAM_MAP.md`
- [x] 编写 `docs/HANDOFF.md`（Agent / 本地交接）
- [ ] 本机下载数据（不提交）并跑 `01_verify_data_layout.py`
- [ ] 实现 ST-Net 最小训练环（对照上游 `model.py`）
- [ ] Group 1 基线实跑
- [ ] Group 2 标准 KD 实跑
- [ ] Group 3 Pearson 乘子：等公式定稿
