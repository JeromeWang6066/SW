# 实验进度笔记（模板）

> 每次实验 / 数据操作追加一节。不要填写虚构指标。

## 元信息

- 学生：Tianyao Wang (Jerome)
- 仓库：https://github.com/JeromeWang6066/SW
- 当前阶段：Group 1、G1-C、Group 2 和微调加蒸馏已实跑。新 MAPE 只统计真实流量大于 0 的格子。旧实验的 MAPE 仍是放大后的数字，待重算。

---

## 2026-10-28 — 移植 ST-Net 与 Group 1 训练入口

**目标：** 按 CrossTReS `STNet.forward` 接上模型、`60*24` 窗口和教师训练循环，并用很少的步数确认本机 GPU 能反向传播。  
**配置：** `configs/teacher_ny.yaml`（`batch_size: 8`，`weight_decay: 5e-5`）  
**Git commit：** 未提交  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop，`torch 2.11.0+cu128`

### 做了什么

- `src/models/st_net.py`：移植 `ResUnit` 与 `STNet.forward`（无 `functional_forward`，不做 region re-weighting）。
- `src/data/splits.py`：`split_x_y` 与 `trim_few_shot` 对齐上游 `60*24` 和最后 `few_shot_days*24` 个窗口。
- `src/data/dataset.py`：原始流量上计算 `sum>0` mask，再做 percentile 0.999 min-max。源城训练拼接 train/val/test 窗口；验证集仍单独保留。
- `src/train/train_teacher.py`：MSE 与 CrossTReS 一样先对 batch 取均值再对有效格求和；按验证 RMSE 存 checkpoint。`--max-steps` 只做接线检查。
- 形状测试 7 项通过。NY 上 `--max-steps 2` 退出码 0：有效格 458，拼接后训练窗口 8778，验证/测试各 1440。该次损失不是 Group 1 结果，下表不填。

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 仅 2 step 接线，不记为实验 |
| test | — | — | — | 未做全量评估 |

### 问题 / 下一步

- 全量跑 Group 1：`teacher_ny.yaml`、`teacher_chi.yaml`、`baseline_target_fewshot.yaml`（各 50 epoch，早停 patience 5）
- 再实现 Group 2 学生 KD

---

## 2026-10-28 — 本机环境 + CrossTReS 数据落地

**目标：** 在学生本机装好 CUDA PyTorch，下载并核对 NY/CHI/DC 数据包，跑通冒烟。不训练、不移植模型。  
**配置：** 无训练配置变更  
**Git commit：** 未提交（`.venv` 与 `data/raw` 已 gitignore）  
**机器 / GPU：** Windows 本机，NVIDIA GeForce RTX 4060 Laptop GPU，显存 8188 MB，驱动 610.88

### 做了什么

- 创建 `.venv`（Python 3.11.9）。`requirements.txt` 里的 PyPI CPU 轮子下载中断后，改为安装其余依赖，再从 `https://download.pytorch.org/whl/cu128` 安装 `torch==2.11.0+cu128`。
- `torch.cuda.is_available()` 为真，设备名为 RTX 4060 Laptop GPU。
- GitHub 直连超时。数据包字节来自同一仓库路径的 jsDelivr CDN：`https://cdn.jsdelivr.net/gh/KL4805/CrossTReS@main/data/crosstres_data.zip`。SHA256 与 DATA.md 记录的官方包一致。
- 解压到 `data/raw/crosstres_extract/{NY,CHI,DC}/`。目录联接到 `data/processed/{ny,chi,dc}/upstream`。

### 数据

- 官方来源 URL：https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip
- 实际下载 URL（内容经 SHA256 核对为同一文件）：https://cdn.jsdelivr.net/gh/KL4805/CrossTReS@main/data/crosstres_data.zip
- SHA256：`f322a145484f183042206cf42d54bca1311784024ec384a2046df83d8f637880`
- `Taxi{CITY}_pickup.npy`：NY `(8784, 20, 23)`、CHI `(8784, 17, 28)`、DC `(8784, 21, 20)`，均为 float64
- 布局检查：`scripts/01_verify_data_layout.py` 三城均为 `HAS FILES`，退出码 0

### 运行命令

```bash
.\.venv\Scripts\python.exe scripts/00_check_env.py
.\.venv\Scripts\python.exe scripts/02_smoke_import.py
.\.venv\Scripts\python.exe scripts/03_synthetic_shapes.py
.\.venv\Scripts\python.exe scripts/01_verify_data_layout.py
.\.venv\Scripts\python.exe -m src.train.train_teacher --config configs/teacher_ny.yaml --dry-run
.\.venv\Scripts\python.exe -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml --dry-run
```

上述命令均退出码 0。`STNet.forward` 仍为 `NotImplementedError`。

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 本阶段无训练 |
| test | — | — | — | 本阶段无训练 |

### 问题 / 下一步

- 移植 CrossTReS `src/model.py` → `STNet.forward`
- 将 `splits.py` 与上游 `60*24` 对齐后实现 Group 1
- 真训练时按 8 GB 显存调小 `batch_size`（当前占位仍为 32）

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

## 2026-10-28 — Group 1 正式实跑

**目标：** 源城教师只在训练窗口上训练，验证集早停，测试集只评估一次；再跑 DC 3 天 few-shot。  
**配置：** `configs/teacher_ny.yaml`、`configs/teacher_chi.yaml`、`configs/baseline_target_fewshot.yaml`（seed 42，batch 8，最多 50 epoch，patience 5，weight decay 5e-5）  
**Git commit：** 未提交  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop，`torch 2.11.0+cu128`。02:28–02:51，约 22.5 分钟。

### 做了什么

- 源城 `concatenate_all_for_train=False`。NY / CHI 训练窗口 5898，验证和测试各 1440。这和 CrossTReS 把源城 train+val+test 拼进预训练不同。
- DC：`few_shot_days=3`，训练窗口 72，验证和测试各 1440。
- 指标是反归一化后的有效格 RMSE / MAE / MAPE。早停看验证 RMSE。
- 命令：`.\.venv\Scripts\python.exe -u scripts\run_group1.py`，退出码 0。

### 真实结果（仅实跑后填写）

反归一化。MAPE 分母为 `max(|y|, 1e-6)`，大量真实流量接近 0，所以百分比被放大到千万量级，不要拿来和论文对比。

| run | best epoch | split | MAE | RMSE | MAPE | checkpoint |
|---|---|---|---|---|---|---|
| G1-src-ny | 15 | val | 4.1881 | 15.4229 | 48988483.78 | `checkpoints/teacher_ny_stnet.pt` |
| G1-src-ny | 15 | test | 4.3245 | 15.5533 | 47367021.10 | 同上 |
| G1-src-chi | 21 | val | 1.8711 | 6.0326 | 33278616.81 | `checkpoints/teacher_chi_stnet.pt` |
| G1-src-chi | 21 | test | 1.8539 | 6.3722 | 33311818.57 | 同上 |
| G1-tgt-3d | 16 | val | 2.1333 | 6.2564 | 40887661.58 | `checkpoints/baseline_dc_3day_stnet.pt` |
| G1-tgt-3d | 16 | test | 1.8701 | 5.5528 | 40406908.02 | 同上 |

### 问题 / 下一步

- Group 2 学生 KD 仍未实现
- 若要可比较的 MAPE，需要另定零流量格的处理方式后再算，不改这一轮 checkpoint

---

## 2026-10-28 — Group 2 标准 KD

**目标：** 冻结源城教师，在华盛顿 3 天数据上训练学生。`L = L_task + 1.0 * L_kd`，两边都是先对 batch 取平均再对有效格求和。不做区域匹配，Pearson 乘子关闭。  
**配置：** `configs/student_kd_ny2dc.yaml`、`configs/student_kd_chi2dc.yaml`（seed 42，batch 8，weight decay 5e-5，最多 50 epoch，patience 5）  
**Git commit：** 未提交  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop。11:56–11:59，约 2.5 分钟。

### 做了什么

- 教师直接吃华盛顿归一化后的 6 小时窗口，输出落在华盛顿 207 个有效格上。加载权重时去掉源城 mask。
- 学生随机初始化。构造学生之后再加载教师，并恢复随机数状态，使种子 42 下的学生起点与 Group 1 的华盛顿模型对齐。
- 训练前各做了一次教师零样本验证，不更新参数，不作为蒸馏结果。
- 命令：`.\.venv\Scripts\python.exe -u scripts\run_group2.py`，退出码 0。

### 真实结果（仅实跑后填写）

反归一化。对照是 Group 1 华盛顿 3 天：测试 MAE 1.8701，RMSE 5.5528。MAPE 仍不可用。

| run | best epoch | split | MAE | RMSE | MAPE | checkpoint |
|---|---|---|---|---|---|---|
| 教师 NY 零样本 | — | val | 2.5829 | 9.1696 | 28926066.11 | 不更新参数 |
| G2-NY2DC | 17 | val | 2.0248 | 6.3896 | 39899453.12 | `checkpoints/student_kd_ny2dc.pt` |
| G2-NY2DC | 17 | test | 1.7519 | 5.5064 | 40611829.72 | 同上 |
| 教师 CHI 零样本 | — | val | 2.3813 | 6.9052 | 43752821.55 | 不更新参数 |
| G2-CHI2DC | 12 | val | 2.0314 | 6.1986 | 38983657.93 | `checkpoints/student_kd_chi2dc.pt` |
| G2-CHI2DC | 12 | test | 1.7485 | 5.4130 | 38588288.12 | 同上 |

相对华盛顿 3 天测试集：纽约蒸馏 MAE −0.118、RMSE −0.046；芝加哥蒸馏 MAE −0.122、RMSE −0.140。

### 问题 / 下一步

- Group 3 Pearson 乘子仍未实现
- λ 扫描不在这一轮

---

## 2026-10-28 — G1-C 源城权重微调

**目标：** 把纽约或芝加哥的 ST-Net 权重复制到华盛顿，只用任务损失在 3 天数据上微调。没有冻结教师，也没有蒸馏。  
**配置：** `configs/finetune_ny2dc.yaml`、`configs/finetune_chi2dc.yaml`（seed 42，batch 8，学习率 0.001，weight decay 5e-5，最多 50 epoch，patience 5）  
**Git commit：** 未提交  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop。18:46–18:48，约 1.5 分钟，退出码 0。

### 做了什么

- 加载源城 checkpoint，去掉源城 mask，接到华盛顿 207 个有效格上。全部参数可训练。
- 损失与 Group 1 相同。早停看华盛顿验证 RMSE。
- 命令：`.\.venv\Scripts\python.exe -u scripts\run_g1c.py`

### 真实结果（仅实跑后填写）

反归一化。对照测试集：只训华盛顿 MAE 1.8701 / RMSE 5.5528；纽约蒸馏 1.7519 / 5.5064；芝加哥蒸馏 1.7485 / 5.4130。MAPE 仍不可用。

| run | best epoch | split | MAE | RMSE | MAPE | checkpoint |
|---|---|---|---|---|---|---|
| G1C-NY2DC | 3 | val | 2.0548 | 6.1937 | 38473834.11 | `checkpoints/finetune_ny2dc.pt` |
| G1C-NY2DC | 3 | test | 1.7153 | 5.2786 | 36567647.09 | 同上 |
| G1C-CHI2DC | 1 | val | 1.7130 | 5.5809 | 12147899.51 | `checkpoints/finetune_chi2dc.pt` |
| G1C-CHI2DC | 1 | test | 1.4773 | 5.0578 | 11440216.23 | 同上 |

相对只训华盛顿的测试集：纽约微调 MAE −0.155、RMSE −0.274；芝加哥微调 MAE −0.393、RMSE −0.495。两条都好于对应的 Group 2 蒸馏。

### 问题 / 下一步

- 这一轮里，复制源城权重再微调比随机初始化加蒸馏下降更多
- Group 3 仍未做

---

## 2026-10-29 — 微调加蒸馏（种子 42，λ = 1）

**目标：** 学生复制源城教师权重后再做标准蒸馏，看华盛顿 3 天协议上是否同时好于随机初始化蒸馏和纯微调。  
**配置：** `configs/student_kd_ny2dc_init.yaml`、`configs/student_kd_chi2dc_init.yaml`（`init_from_teacher: true`，λ = 1，seed 42，batch 8，weight decay 5e-5，最多 50 epoch，patience 5）  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop。12:35–12:37，约 2.5 分钟，退出码 0。

### 做了什么

- MAPE 改为只在反归一化后真实流量大于 0 的格子上计算，分母用真实流量。RMSE / MAE 的累加方式不变。
- `kd.init_from_teacher` 把教师权重复制到学生（去掉 `spatial_mask`），教师另存一份并冻结。损失仍是 `L_task + λ L_kd`。
- 三个训练入口增加 `--seed`。传入时结果 JSON 和 checkpoint 文件名带种子后缀，不覆盖种子 42 的已有文件。本项没有传 `--seed`，用的是配置里的 42 和新的 checkpoint 名。
- 命令：`.\.venv\Scripts\python.exe -u -m src.train.train_student_kd --config configs/student_kd_ny2dc_init.yaml --resume`，随后同样跑芝加哥配置。`tests/test_synthetic_shapes.py` 8 项通过。

### 真实结果（仅实跑后填写）

数字来自 `logs/G2-NY2DC-init.json` 与 `logs/G2-CHI2DC-init.json`，四位小数。MAPE 是修正后的定义。

| run | best epoch | split | MAE | RMSE | MAPE | checkpoint |
|---|---|---|---|---|---|---|
| G2-NY2DC-init | 5 | val | 2.0879 | 6.6986 | 66.3591 | `checkpoints/student_kd_ny2dc_init.pt` |
| G2-NY2DC-init | 5 | test | 1.7360 | 5.6687 | 64.4227 | 同上 |
| G2-CHI2DC-init | 17 | val | 2.0078 | 5.7935 | 83.1331 | `checkpoints/student_kd_chi2dc_init.pt` |
| G2-CHI2DC-init | 17 | test | 1.7242 | 5.1286 | 82.2489 | 同上 |

### 问题 / 下一步

- 种子 43、44 的华盛顿七类设定尚未重跑
- 种子 42 旧 checkpoint 的 MAPE 尚未按新定义重算

---

## 2026-10-29 — 种子 43 与 44（华盛顿七类设定）

**目标：** 在不重跑种子 42、不重训源城教师的前提下，看七类华盛顿设定换种子后的测试 MAE / RMSE / 修正 MAPE，并给出三种子均值。  
**配置：** `baseline_target_fewshot.yaml`、`finetune_ny2dc.yaml`、`finetune_chi2dc.yaml`、`student_kd_ny2dc.yaml`、`student_kd_chi2dc.yaml`、`student_kd_ny2dc_init.yaml`、`student_kd_chi2dc_init.yaml`，各加 `--seed 43` 和 `--seed 44`。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop。12:41–13:02，14 次顺序训练，退出码 0。教师仍是 `checkpoints/teacher_ny_stnet.pt` 与 `checkpoints/teacher_chi_stnet.pt`。

### 做了什么

- 每次命令都带 `--resume`。结果 JSON 已存在且 `status=complete` 的运行会跳过；这一轮 14 个文件原先都不存在。
- `--seed` 让 checkpoint 与 `logs/<id>_seed<N>.json` 带种子后缀，种子 42 的 `baseline_dc_3day_stnet.pt` 等文件没有被覆盖。
- 种子 42 的 MAE / RMSE 沿用已有 JSON。五个旧实验的种子 42 MAPE 用现有 checkpoint 按新定义重算（只评估，不训练）。初始化加蒸馏的种子 42 数字来自上一节。

### 真实结果（仅实跑后填写）

测试集，四位小数。顺序是 MAE / RMSE / MAPE。均值是种子 42、43、44 的算术平均。

| 设定 | 42 | 43 | 44 | 均值 |
|---|---|---|---|---|
| 华盛顿从零，3 天 | 1.8701 / 5.5528 / 75.9919 | 1.8694 / 5.4060 / 75.2559 | 1.9156 / 5.6114 / 79.2501 | 1.8850 / 5.5234 / 76.8326 |
| 纽约微调 | 1.7153 / 5.2786 / 72.3947 | 1.6344 / 5.1229 / 72.7873 | 1.6242 / 5.1752 / 73.4110 | 1.6580 / 5.1922 / 72.8643 |
| 芝加哥微调 | 1.4773 / 5.0578 / 73.3157 | 1.5319 / 5.0045 / 68.4041 | 1.5422 / 5.1079 / 74.8422 | 1.5171 / 5.0568 / 72.1873 |
| 纽约蒸馏，随机初始化 | 1.7519 / 5.5064 / 59.2878 | 2.0160 / 5.6691 / 76.2539 | 2.0585 / 5.8437 / 84.4423 | 1.9422 / 5.6730 / 73.3280 |
| 芝加哥蒸馏，随机初始化 | 1.7485 / 5.4130 / 65.3868 | 1.8960 / 5.7250 / 66.4677 | 1.8897 / 5.5718 / 70.9988 | 1.8447 / 5.5700 / 67.6177 |
| 纽约初始化加蒸馏 | 1.7360 / 5.6687 / 64.4227 | 1.7025 / 5.8389 / 62.0886 | 1.8518 / 5.7513 / 70.7795 | 1.7634 / 5.7530 / 65.7636 |
| 芝加哥初始化加蒸馏 | 1.7242 / 5.1286 / 82.2489 | 1.6582 / 5.3156 / 76.0493 | 1.6701 / 5.2404 / 77.3214 | 1.6842 / 5.2282 / 78.5399 |

### 问题 / 下一步

- 源城教师和华盛顿旧 checkpoint 的验证集 MAPE 还没全部写入主表
- 芝加哥 λ 扫描尚未开始

---

## 2026-10-29 — 用修正 MAPE 重算种子 42 的旧 checkpoint

**目标：** 不重新训练，把 Group 1 三城、两组蒸馏、两组微调的已有 checkpoint 按新 MAPE 重算验证集和测试集。  
**配置：** 无新训练。checkpoint 为 `teacher_ny_stnet.pt`、`teacher_chi_stnet.pt`、`baseline_dc_3day_stnet.pt`、`student_kd_ny2dc.pt`、`student_kd_chi2dc.pt`、`finetune_ny2dc.pt`、`finetune_chi2dc.pt`。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop。只做评估。

### 做了什么

- 调用更新后的 `evaluate`。RMSE / MAE 与原先 `logs/*.json` 在打印到小数点后六位时一致。
- 早先 progress 里千万量级的 MAPE 是旧分母 `max(|y|, 1e-6)` 的记录，这一节的数字取代它们用于比较。

### 真实结果（仅实跑后填写）

| run | split | MAE | RMSE | MAPE |
|---|---|---|---|---|
| G1-src-ny | val | 4.1881 | 15.4229 | 59.9577 |
| G1-src-ny | test | 4.3245 | 15.5533 | 57.2310 |
| G1-src-chi | val | 1.8711 | 6.0326 | 49.3779 |
| G1-src-chi | test | 1.8539 | 6.3722 | 51.6097 |
| G1-tgt-3d | val | 2.1333 | 6.2564 | 75.1337 |
| G1-tgt-3d | test | 1.8701 | 5.5528 | 75.9919 |
| G2-NY2DC | val | 2.0248 | 6.3896 | 57.9803 |
| G2-NY2DC | test | 1.7519 | 5.5064 | 59.2878 |
| G2-CHI2DC | val | 2.0314 | 6.1986 | 64.7428 |
| G2-CHI2DC | test | 1.7485 | 5.4130 | 65.3868 |
| G1C-NY2DC | val | 2.0548 | 6.1937 | 75.6398 |
| G1C-NY2DC | test | 1.7153 | 5.2786 | 72.3947 |
| G1C-CHI2DC | val | 1.7130 | 5.5809 | 72.9796 |
| G1C-CHI2DC | test | 1.4773 | 5.0578 | 73.3157 |

### 问题 / 下一步

- 芝加哥 λ 扫描尚未开始

---

## 2026-10-29 — 芝加哥到华盛顿的 λ 扫描

**目标：** 种子 42 下比较 λ = 0.1、0.5、5，随机初始化和教师初始化各一条。λ = 1 已有结果，不重跑。  
**配置：** `configs/student_kd_chi2dc_l0.1.yaml`、`l0.5`、`l5`，以及 `student_kd_chi2dc_init_l0.1.yaml`、`init_l0.5`、`init_l5`。batch 8，weight decay 5e-5，华盛顿 3 天。教师是 `checkpoints/teacher_chi_stnet.pt`。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机，RTX 4060 Laptop。13:10–13:25，六次顺序训练，退出码 0。

### 做了什么

- 每条命令带 `--resume`。新的 checkpoint 名不覆盖 `student_kd_chi2dc.pt` 和 `student_kd_chi2dc_init.pt`。
- λ = 1 的随机初始化 MAPE 用已重算的 64.7428 / 65.3868，不用旧 JSON 里的千万量级数字。

### 真实结果（仅实跑后填写）

| 初始化 | λ | epoch | split | MAE | RMSE | MAPE |
|---|---|---|---|---|---|---|
| 随机 | 0.1 | 13 | val | 2.2243 | 6.2333 | 79.7766 |
| 随机 | 0.1 | 13 | test | 2.0019 | 5.6585 | 82.6521 |
| 随机 | 0.5 | 13 | val | 2.3874 | 6.2772 | 94.1952 |
| 随机 | 0.5 | 13 | test | 2.1140 | 5.6829 | 94.2071 |
| 随机 | 1 | 12 | val | 2.0314 | 6.1986 | 64.7428 |
| 随机 | 1 | 12 | test | 1.7485 | 5.4130 | 65.3868 |
| 随机 | 5 | 19 | val | 2.2775 | 6.3289 | 82.0384 |
| 随机 | 5 | 19 | test | 1.9895 | 5.7151 | 80.8705 |
| 教师初始化 | 0.1 | 5 | val | 1.7349 | 5.6594 | 64.9936 |
| 教师初始化 | 0.1 | 5 | test | 1.4783 | 4.9331 | 65.9970 |
| 教师初始化 | 0.5 | 5 | val | 1.8418 | 5.8142 | 71.5902 |
| 教师初始化 | 0.5 | 5 | test | 1.5640 | 5.0545 | 71.9325 |
| 教师初始化 | 1 | 17 | val | 2.0078 | 5.7935 | 83.1331 |
| 教师初始化 | 1 | 17 | test | 1.7242 | 5.1286 | 82.2489 |
| 教师初始化 | 5 | 5 | val | 2.2758 | 6.4084 | 97.7349 |
| 教师初始化 | 5 | 5 | test | 1.9444 | 5.6785 | 95.6167 |

### 问题 / 下一步

- Group 3 Pearson 乘子仍未实现

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
- [x] 本机下载数据（不提交）并跑 `01_verify_data_layout.py`
- [x] 实现 ST-Net 最小训练环（对照上游 `model.py`）
- [x] Group 1 基线实跑
- [x] Group 2 标准 KD 实跑
- [ ] Group 3 Pearson 乘子：等公式定稿
