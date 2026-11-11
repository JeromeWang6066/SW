# 实验进度笔记（模板）

> 每次实验 / 数据操作追加一节。不要填写虚构指标。

## 元信息

- 学生：Tianyao Wang (Jerome)
- 仓库：https://github.com/JeromeWang6066/SW
- 当前阶段：Group 1、G1-C、Group 2、教师初始化蒸馏、种子 43/44 和芝加哥 λ 扫描已实跑。比较用的 MAPE 只统计真实流量大于 0 的格子。10 月 28 日各节里的千万级 MAPE 是旧分母，比较时用 10 月 29 日重算的那一节。

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

## 2026-11-05 — 导师会议：分流规则与格子功能

**目标：** 记录会上的三条建议，并写进实验计划。不训练、不改模型。  
**配置：** 无  
**Git commit：** 未提交  
**机器 / GPU：** 文档

### 做了什么

- 新增 `docs/advisor/2026-11-05-meeting.md`。三条建议分开写：微调已更好时为何还蒸馏；用 Pearson 相关选择微调或蒸馏；用兴趣点、路网和上下车给格子做功能预划分。
- `docs/EXPERIMENT_PLAN.md` 增加 Group 5（分流）和 Group 6（功能标签）。原 Group 3 的 ρ̃ 乘子改到分流实验之后。研究问题改成“按相关选择方法”，不再默认蒸馏是主方法。
- 0.8 和 0.5 记为导师举例。ρ 低于下阈值时退回只训目标城，会上没定，标成待确认。

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 本阶段无训练 |
| test | — | — | — | 本阶段无训练 |

### 问题 / 下一步

- 离线统计三城兴趣点占比、路网度、上下车比和工作日轮廓 Pearson。不训练
- 六个有向城市对上补从零、微调、标准蒸馏。阈值只在验证集上定
- 下次确认：低相关是否退回只训目标城；ρ 是城市级还是格子级；华盛顿可否作源城

---

## 2026-11-06 — 三城每个格子的 24 小时轮廓

**目标：** 计算纽约、芝加哥、华盛顿每个格子的出租车上车量轮廓。不训练，不做跨城匹配。  
**配置：** 无训练配置  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读 `Taxi{CITY}_pickup.npy`，不使用 GPU。

### 做了什么

- `hourly_calendar` 与 `grid_hour_profile`：第 0 小时为 2016-01-01 00:00，工作日为周一到周五，不做夏令时插删。纽约日期来自 CrossTReS 附录 Table 2；芝加哥和华盛顿因同为 8784 小时而共用该起点。
- `scripts/04_grid_profiles.py` 写出 `docs/profiles/{ny,chi,dc}_taxi_pickup.json`。每城含全年和训练段前缀（前 5904 小时）的工作日与周末轮廓，形状 `(H, W, 24)`，无效格保留。
- `tests/test_grid_profiles.py` 3 项通过。

### 数据

- 来源仍是已核验的 CrossTReS 包，未重新下载。
- 有效格与 DATA.md 一致：纽约 458 / 460，芝加哥 270 / 476，华盛顿 207 / 420。
- 全年每个钟点：工作日 261 个样本，周末 105 个。训练段每个钟点：工作日 176，周末 70。
- 有效格平均后的 24 个数见 `docs/profiles/README.md`。JSON 保留完整精度。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_grid_profiles.py -q
.\.venv\Scripts\python.exe scripts\04_grid_profiles.py
```

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 本阶段无训练 |
| test | — | — | — | 本阶段无训练 |

### 问题 / 下一步

- 下车、单车、兴趣点和路网还没算
- 城市之间的 Pearson 和 ρ̃ 乘子还没做

---

## 2026-11-06 — 协议内的轮廓相似度

**目标：** 用出租车上车量的工作日轮廓计算城市级和格子级 Pearson，并检查 72 小时轮廓与 5904 小时轮廓是否一致。不训练。  
**配置：** 无训练配置  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读 `Taxi{CITY}_pickup.npy` 和已有 `docs/profiles/*_taxi_pickup.json`。

### 做了什么

- 目标城轮廓取最后 72 个训练标签，原始下标 `[5832, 5904)`，日期 2016-08-31 至 2016-09-02。每个钟点有 3 个工作日样本。源城轮廓用前 5904 小时。
- 重算的 5904 小时轮廓与已提交的 `train_prefix.weekday_profile` 一致，否则脚本失败。
- 常数轮廓记为无定义。0.5 和 0.8 只描述格子级最大相关的分布。
- 结果在 `docs/profiles/similarity.md` 与 `similarity.json`。`tests/test_profile_similarity.py` 3 项通过。

### 数据

- 同一座城，72 小时对 5904 小时的城市平均轮廓相关：纽约 0.9174，芝加哥 0.9883，华盛顿 0.9944。格子中位数分别是 0.5646、0.6810、0.6054。纽约剔除 111 个常数格，芝加哥 110 个，华盛顿 18 个。
- 协议内城市级相关随方向变化。纽约→华盛顿 0.8598，华盛顿→纽约 0.8519；芝加哥→华盛顿 0.9444，华盛顿→芝加哥 0.9643。
- 格子级最大相关的中位数在 0.6559 到 0.7951 之间。六个方向里，落在 0.5 以下的格子比例最高是华盛顿→纽约的 0.2536，其余方向大多低于 0.03。完整比例见 `similarity.md`。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_profile_similarity.py -q
.\.venv\Scripts\python.exe scripts\05_profile_similarity.py
```

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 本阶段无训练 |
| test | — | — | — | 本阶段无训练 |

### 问题 / 下一步

- 下车、单车、兴趣点和路网仍未算
- 这些相关还没有用来选择微调或蒸馏

---

## 2026-11-09 — 格子分档是否稳定

**目标：** 看华盛顿每个格子用 3 天轮廓和用训练段轮廓是否分到同一档。达到 0.7 才对「有一点像」的格子做蒸馏。不改功能区。  
**配置：** 无训练配置。门槛 0.7 在看结果之前定下。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读出租车上车量，不使用 GPU。

### 做了什么

- `scripts/06_bin_agreement.py` 对纽约→华盛顿、芝加哥→华盛顿各算两套最大相关。源城用前 5904 小时。协议内的华盛顿用下标 `[5832, 5904)`。对照的华盛顿用前 5904 小时。
- 分档是不超过 0.5、大于 0.5 且不超过 0.8、大于 0.8。常数轮廓无定义。一致比例的分母是两次都有定义的格子。
- `tests/test_bin_agreement.py` 2 项通过。

### 数据

- 纽约→华盛顿：189 个格子两次都有定义，88 个档位相同，一致比例 0.4656。
- 芝加哥→华盛顿：189 个格子两次都有定义，95 个档位相同，一致比例 0.5026。
- 两个方向都低于 0.7，所以没有训练。三座城的城市级相关都在 0.85 以上，已有实验里微调误差更低。这次格子分档不稳定，所以不估 0.5 这条线。
- 完整人数见 `docs/profiles/bin_agreement.md`。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_bin_agreement.py -q
.\.venv\Scripts\python.exe scripts\06_bin_agreement.py
```

### 真实结果（仅实跑后填写）

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 未训练。分档未过 0.7 |
| test | — | — | — | 未训练。分档未过 0.7 |

### 问题 / 下一步

- 部分蒸馏没有跑
- 功能区、六个新方向、下车和单车仍未做

---

## 2026-11-09 — 工作日和周末分开的初步训练

**目标：** 源城分别训练工作日和周末加节假日。华盛顿用 3 个工作日加 2 个周末日，和连续 3 个工作日的旧结果对照，并看分开微调是否降低周末误差。  
**配置：** `configs/calendar/`。种子 42。权重写入 `checkpoints/calendar/`，日志写入 `logs/calendar/`。旧 checkpoint 只读。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机，CUDA。

### 做了什么

- 标签小时过滤默认关闭。日历从 2016-01-01 起，不做夏令时插删。周末一类包含周六、周日，以及 1 月 1 日、1 月 18 日、2 月 15 日、5 月 30 日、7 月 4 日、9 月 5 日。
- 源城训练段：工作日 4104 小时，周末加节假日 1794 小时。华盛顿周末标签 `[5736, 5784)`，工作日标签 `[5832, 5904)`。劳动节下标 5952，在验证集。
- 训练了纽约和芝加哥各两个教师，以及华盛顿从零、旧教师微调、随机初始化蒸馏（λ = 1）。再按类别用新教师微调，评价时拼接。
- `tests/test_calendar_mask.py` 3 项通过。全部单元测试 19 项通过。

### 数据

- 沿用已核对的出租车上车量，没有改 `npy`。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_calendar_mask.py -q
.\.venv\Scripts\python.exe scripts\07_calendar_counts.py
.\.venv\Scripts\python.exe -u scripts\run_calendar_seed42.py
.\.venv\Scripts\python.exe scripts\09_calendar_results.py
```

### 真实结果（仅实跑后填写）

测试集全部小时。旧行是连续 3 个工作日，新行是 3 个工作日加 2 个周末日。完整拆分在 `docs/CALENDAR_RESULTS.md`。

| 设定 | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 旧，华盛顿从零 | 1.8701 | 5.5528 | 75.9919 |
| 新，华盛顿从零 | 1.7470 | 5.3581 | 66.1857 |
| 旧，纽约微调 | 1.7153 | 5.2786 | 72.3947 |
| 新，纽约一个模型 | 1.7648 | 5.1637 | 74.1446 |
| 新，纽约分开拼接 | 1.6509 | 5.0619 | 71.1090 |
| 旧，芝加哥微调 | 1.4773 | 5.0578 | 73.3157 |
| 新，芝加哥一个模型 | 1.5781 | 5.0816 | 72.4985 |
| 新，芝加哥分开拼接 | 1.4510 | 4.8467 | 63.0408 |
| 旧，纽约随机初始化蒸馏 | 1.7519 | 5.5064 | 59.2878 |
| 新，纽约随机初始化蒸馏 | 1.8410 | 5.8103 | 62.2484 |
| 旧，芝加哥随机初始化蒸馏 | 1.7485 | 5.4130 | 65.3868 |
| 新，芝加哥随机初始化蒸馏 | 1.8539 | 5.2222 | 82.7596 |

周末加节假日的测试 MAE：纽约分开拼接 1.5682，高于同一个 120 小时、一个模型的 1.5179。芝加哥分开拼接 1.2789，低于一个模型的 1.3914。按事先定下的标准，分开训练在芝加哥的周末行上有用，在纽约上没有。

### 问题 / 下一步

- 种子 43 和 44 的华盛顿重跑见下一节
- 功能区、六个新方向、λ 扫描没有做

---

## 2026-11-09 — 种子 43 和 44 复核日历实验

**目标：** 看种子 42 的华盛顿结果换种子后是否还一样。  
**配置：** 同一套 `configs/calendar/`，命令行 `--seed 43` 和 `--seed 44`。源城教师不重训。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机，CUDA。

### 做了什么

- 华盛顿从零、旧教师微调、随机初始化蒸馏，以及四个分开微调，各种子跑了一遍。权重文件名带 `_seed43`、`_seed44`。
- 种子 42 的 `dc_scratch_wd3we2.pt` 仍是 18:35，`teacher_ny_weekday.pt` 仍是 18:10，旧的 `teacher_ny_stnet.pt` 仍是 10 月 28 日。

### 运行命令

```bash
.\.venv\Scripts\python.exe -u scripts\run_calendar_extra_seeds.py
.\.venv\Scripts\python.exe scripts\09_calendar_results.py
```

### 真实结果（仅实跑后填写）

测试集全部小时的 MAE / RMSE。均值是三个种子的算术平均。

| 设定 | 种子 42 | 种子 43 | 种子 44 | 均值 |
|---|---:|---:|---:|---:|
| 华盛顿从零 | 1.7470 / 5.3581 | 1.7464 / 5.1944 | 1.7639 / 5.2566 | 1.7524 / 5.2697 |
| 纽约一个模型 | 1.7648 / 5.1637 | 1.5540 / 5.3627 | 1.7908 / 5.8419 | 1.7032 / 5.4561 |
| 芝加哥一个模型 | 1.5781 / 5.0816 | 1.4656 / 4.9358 | 1.4139 / 4.7716 | 1.4859 / 4.9297 |
| 纽约随机初始化蒸馏 | 1.8410 / 5.8103 | 1.8366 / 5.7937 | 1.7720 / 5.5343 | 1.8165 / 5.7128 |
| 芝加哥随机初始化蒸馏 | 1.8539 / 5.2222 | 1.8187 / 5.5476 | 1.7827 / 5.1613 | 1.8184 / 5.3103 |
| 纽约分开拼接 | 1.6509 / 5.0619 | 1.5759 / 4.9136 | 1.6363 / 4.9692 | 1.6211 / 4.9816 |
| 芝加哥分开拼接 | 1.4510 / 4.8467 | 1.5216 / 4.7737 | 1.4355 / 4.7138 | 1.4693 / 4.7781 |

周末加节假日测试 MAE，分开拼接相对一个模型：纽约是高于、高于、低于；芝加哥是低于、高于、低于。事先定下的周末标准没有在三个种子上保持同一个方向。

全部小时的测试 RMSE，分开拼接相对一个模型：纽约三个种子都更低；芝加哥三个种子都更低。

从零训练的测试 MAE 停在 1.75 附近。三种子均值里，两座城的随机初始化蒸馏都高于对应的微调。芝加哥一个模型的均值 1.4859 低于纽约一个模型的 1.7032，也低于从零训练的 1.7524。

### 问题 / 下一步

- 源城教师没有换种子
- 功能区、六个新方向、λ 扫描没有做

---

## 2026-11-09 — 模态之间的城市级相关

**目标：** 看出租上车、出租下车、单车上车之间，有没有城市级工作日相关落到 0.8 以下。不训练。  
**配置：** 无训练配置。工作日是周一到周五。源侧前 5904 小时，目标侧 `[5832, 5904)`。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读 `npy`，不使用 GPU。

### 做了什么

- `scripts/10_modality_similarity.py` 读 9 条序列。形状与 `docs/DATA.md` 一致。出租上车有效格 458 / 270 / 207，单车上车 108 / 229 / 125。出租下车是 460 / 284 / 208。
- 已有的出租上车跨城相关与 `docs/profiles/similarity.md` 的差不超过 `1e-4`。
- `tests/test_modality_similarity.py` 4 项通过。常数轮廓无定义，相同非常数轮廓为 1，72 小时每个钟点 3 个样本。

### 数据

- 沿用 `data/processed/{city}/upstream/` 里的 `Taxi*_pickup.npy`、`Taxi*_dropoff.npy`、`Bike*_pickup.npy`。单车下车没算。没有改文件。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_modality_similarity.py -q
.\.venv\Scripts\python.exe scripts\10_modality_similarity.py
```

### 真实结果（仅实跑后填写）

同窗口相关里，出租上车和出租下车，包括跨城，都高于 0.8。出租和单车的同窗口相关多数在 0.5 到 0.8 之间。纽约出租下车对芝加哥单车上车是 0.4854，是唯一不超过 0.5 的一对。

72 个有向对里，34 个同窗口相关不超过 0.8，记为以后的训练候选。纽约单车上车到芝加哥出租上车的协议内相关是 0.7897，同窗口是 0.8342，记成短窗口压低，不是候选。这一步没有训练。

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 未训练 |
| test | — | — | — | 未训练 |

### 问题 / 下一步

- 训练候选都涉及单车，还没有跑从零、微调、蒸馏
- 单车下车和功能区没有做

---

## 2026-11-09 — 五日窗口的格子分档一致比例

**目标：** 看 3 个工作日加 2 个周末日的短轮廓，和前 5904 小时的同一种分类，是否分到同一档。分类和源城训练一致：工作日去掉联邦假日，周末加上这些假日。门槛仍是 0.7。不训练。  
**配置：** 无训练配置。工作日 `[5832, 5904)`，周末 `[5736, 5784)`。前 5904 小时里有 120 个假日小时改到周末。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读出租车上车量，不使用 GPU。

### 做了什么

- `scripts/11_calendar_bin_agreement.py` 对纽约→华盛顿、芝加哥→华盛顿各算工作日和周末两套最大相关。
- 华盛顿短窗口里没有假日。源城和长窗口把 1 月 1 日、1 月 18 日、2 月 15 日、5 月 30 日、7 月 4 日从工作日改到周末。
- `tests/test_calendar_bin_agreement.py` 1 项通过。周末每个钟点 2 个样本，工作日每个钟点 3 个样本，训练段假日小时为 120。

### 数据

- 沿用已核对的出租车上车量，没有改 `npy`。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_calendar_bin_agreement.py -q
.\.venv\Scripts\python.exe scripts\11_calendar_bin_agreement.py
```

### 真实结果（仅实跑后填写）

| 方向 | 分类 | 一致比例 | 档位相同 / 分母 | 是否过 0.7 |
|---|---|---:|---|---|
| 纽约→华盛顿 | 工作日 | 0.4815 | 91 / 189 | 否 |
| 芝加哥→华盛顿 | 工作日 | 0.4921 | 93 / 189 | 否 |
| 纽约→华盛顿 | 周末加节假日 | 0.6127 | 106 / 173 | 否 |
| 芝加哥→华盛顿 | 周末加节假日 | 0.4855 | 84 / 173 | 否 |

四行都低于 0.7。没有训练。

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 未训练。四行都未过 0.7 |
| test | — | — | — | 未训练。四行都未过 0.7 |

### 问题 / 下一步

- 部分蒸馏没有跑
- 单车和功能区没有做

---

## 2026-11-09 — 档位不同是源格换了，还是只差一点点

**目标：** 对档位不同的华盛顿格子，区分最大源格有没有换，并看两个窗口的最大相关差了多少、格子自己的轮廓变了多少。三天工作日和五日窗口都算。不训练。  
**配置：** 无训练配置。三天工作日是周一到周五，含假日，短窗口 `[5832, 5904)`。五日窗口用训练时的假日分类，工作日短窗口同上，周末短窗口 `[5736, 5784)`。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读出租车上车量，不使用 GPU。

### 做了什么

- `max_source_match` 在最大相关之外记下达到最大的源格。差在 1e-8 以内算并列。
- `scripts/12_bin_disagreement.py` 对纽约、芝加哥各算三天工作日，以及五日窗口的工作日和周末。
- 事先核对一致比例：三天工作日 88/189、95/189；五日窗口 91/189、93/189、106/173、84/173。
- `tests/test_bin_disagreement.py` 3 项通过。小差距用 0.02、0.05、0.10，只作描述。

### 数据

- 沿用已核对的出租车上车量，没有改 `npy`。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_bin_disagreement.py -q
.\.venv\Scripts\python.exe scripts\12_bin_disagreement.py
```

### 真实结果（仅实跑后填写）

档位不同的格子里，最大源格相同的每组只有 1 到 3 个。其余都换了源格。换源格的格子，相似度差中位数 0.1875 到 0.2307，不超过 0.05 的每组最多 2 个。自身轮廓相关中位数 0.4045 到 0.4907。

| 设定 | 方向 | 档位不同 | 源格相同 | 源格不同 | 源格不同的相似度差中位数 | 其中不超过 0.05 |
|---|---|---:|---:|---:|---:|---:|
| 三天工作日 | 纽约 | 101 | 3 | 98 | 0.2042 | 2 |
| 三天工作日 | 芝加哥 | 94 | 1 | 93 | 0.2084 | 1 |
| 五日，工作日 | 纽约 | 98 | 2 | 96 | 0.2077 | 2 |
| 五日，工作日 | 芝加哥 | 96 | 1 | 95 | 0.1998 | 1 |
| 五日，周末加节假日 | 纽约 | 67 | 3 | 64 | 0.1875 | 1 |
| 五日，周末加节假日 | 芝加哥 | 89 | 2 | 87 | 0.2307 | 1 |

没有训练。

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 未训练 |
| test | — | — | — | 未训练 |

### 问题 / 下一步

- 部分蒸馏没有跑
- 单车和功能区没有做

---

## 2026-11-10 — 三天格子分流

**目标：** 用三天窗口的出租上车分档，在已经训好的整城模型之间给每个华盛顿格子选预测，并和从零、纯微调、纯蒸馏比较。不训练。  
**配置：** 种子 42。从零 `checkpoints/baseline_dc_3day_stnet.pt`，微调 `finetune_ny2dc.pt`、`finetune_chi2dc.pt`，随机初始化蒸馏 `student_kd_ny2dc.pt`、`student_kd_chi2dc.pt`。大于 0.8 用微调，大于 0.5 且不超过 0.8 用蒸馏，其余用从零。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读已有权重，没有新训练。

### 做了什么

- `scripts/13_bin_route.py` 先复核三天分档：纽约 88/189，短窗口 3、111、75、无定义 18；芝加哥 95/189，短窗口 3、100、86、无定义 18。
- 五个模型按训练时的求和重放，验证集和测试集 MAE、RMSE 与已发表四位小数相差不超过 0.0001。
- 按格累加后再拼接。纽约 75 格微调、111 格蒸馏、21 格从零。芝加哥 86、100、21。
- `tests/test_bin_route.py` 检查切档和拼接。

### 数据

- 沿用已核对的出租车上车量，没有改 `npy`，也没有改权重。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_bin_route.py -q
.\.venv\Scripts\python.exe scripts\13_bin_route.py
```

### 真实结果（仅实跑后填写）

验证集 MAE：纽约拼接 2.0483，低于该源城纯微调 2.0548，高于纯蒸馏 2.0247。芝加哥拼接 1.8637，高于纯微调 1.7130。

测试集 MAE：纽约拼接 1.7353，高于纯微调 1.7153。芝加哥拼接 1.6309，高于纯微调 1.4773。

芝加哥验证集中档格子，微调 MAE 0.4367，蒸馏 0.6783。不超过 0.5 或无定义的格子，微调 0.0723，从零 0.4075。这两档被送去蒸馏和从零，所以芝加哥拼接比一律微调差。

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | 2.0483 / 1.8637 | 6.1693 / 5.5843 | 69.4263 / 65.0473 | 纽约拼接 / 芝加哥拼接 |
| test | 1.7353 / 1.6309 | 5.2661 / 5.0623 | 68.2411 / 65.6085 | 纽约拼接 / 芝加哥拼接。没有改切点 |

### 问题 / 下一步

- 种子 43 和 44 没有跑
- 没有新训混合损失模型

---

## 2026-11-10 — 1km 格子每天有多少上车

**目标：** 看三天轮廓和长轮廓分不到同一档，是不是因为 1km 格子每天的上车太少，短窗口里的少数几次上车撑起了整条轮廓。不训练。  
**配置：** 无训练配置。出租上车，工作日含假日。短窗口 [5832, 5904)，长窗口是前 5904 小时里的 176 个工作日。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。只读 `npy`，不使用 GPU。

### 做了什么

- `scripts/14_cell_volume.py` 按工作日日均上车量把有效格分成低于 1、1–9、10–49、50–199、不低于 200。
- 复算已发表的自身相关中位数（纽约 0.5646、芝加哥 0.6810、华盛顿 0.6054）和三天分档（88/189、95/189），对不上就停。
- 另外看了连续三个工作日的 174 个窗口、邻域求和、2km 和 3km 整块、早晚高峰小时、对数轮廓，以及只给高流量格子分档。
- `tests/test_cell_volume.py` 6 项通过。

### 数据

- 沿用已核对的出租车上车量。计数是整数。没有改 `npy`。

### 运行命令

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_cell_volume.py -q
.\.venv\Scripts\python.exe scripts\14_cell_volume.py
```

### 真实结果（仅实跑后填写）

华盛顿有效格日均中位数 8.02，平均数 152.4。日均低于 10 的有 114/207，占上车量 1.3%。日均不低于 200 的 27 格占上车量 87.8%。

纽约→华盛顿：1–9 档 21/82，10–49 档 12/42，50–199 档 21/24，不低于 200 档 27/27。101 个档位不同的格子里，98 个日均低于 50。按上车量加权的一致比例是 0.9627。芝加哥方向加权是 0.9610；94 个档位不同的格子里，90 个日均低于 50。

华盛顿 1–9 档这三天共中位数 12 次上车，24 小时里中位数 16 小时为 0。去掉最大的那个小时后，和长轮廓的相关中位数下降 0.0785。

174 个连续三天窗口里，格子相关中位数的中位数是 0.6652。协议这三天是 0.6054，只有 11 个窗口不高于它。

日均不低于 50 的格子：纽约 48/51，芝加哥 47/51。3×3 邻域求和：150/207 和 169/207。完整 2km 块：34/56 和 33/56。

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | — | — | — | 未训练 |
| test | — | — | — | 未训练 |

### 问题 / 下一步

- 稀疏格子还没有改成「不单独分档，沿用整城决定」
- 没有用 3×3 轮廓重新训练或重新拼接预测

---

## 2026-11-10 — 上车量门和 3×3 轮廓重放

**目标：** 预测仍用种子 42 的 1km 输出。看两种分流是否改变华盛顿的误差：三天上车不少于 150 才按 1km 轮廓分档，以及每个格子按华盛顿 3×3 短轮廓分档。  
**配置：** 不训练。切点 0.5、0.8、150 不改。没过门的格子用该源城微调。3×3 只平滑华盛顿轮廓，不和上车量门叠加。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。不使用 GPU。缓存来自 logs/bin_route_cache。

### 做了什么

- scripts/16_grid_route.py 读五个已有模型的验证集和测试集逐格误差，按三条规则拼接。
- 写报告前核对纯模型 MAE/RMSE、不做门的 1km 拼接、50 个过门格子、3×3 同档 150/207 和 169/207，以及芝加哥测试集上车量门 MAE 1.4737。
- 	ests/test_grid_route.py 3 项通过。

### 数据

- 沿用已核对的出租车上车量。没有改 
py，没有改权重。

### 运行命令

`ash
.\.venv\Scripts\python.exe -m pytest tests/test_grid_route.py -q
.\.venv\Scripts\python.exe scripts/16_grid_route.py
`

### 真实结果（仅实跑后填写）

过门的格子是 50 个。两条源城的上车量门都只把 4 个格子送到蒸馏，没有格子送到从零。3×3 把纽约 67 格、芝加哥 43 格送到蒸馏，从零仍是 0。

验证集 MAE：纽约一律微调 2.0548，上车量门 2.0509，3×3 2.0740。芝加哥一律微调 1.7130，上车量门 1.7079，3×3 1.7635。

测试集 MAE：纽约一律微调 1.7153，上车量门 1.7132，3×3 1.7389。芝加哥一律微调 1.4773，上车量门 1.4737，3×3 1.5283。

没过门的 157 格在上车量门下和一律微调相同。3×3 把误差加在这 157 格上。芝加哥测试集这 157 格的 MAE 是 0.3615，一律微调是 0.2942。过门的 50 格在 3×3 下仍全部用微调。

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | 1.7130 | 5.5809 | 72.9802 | 芝加哥一律微调。上车量门 1.7079。3×3 1.7635 |
| test | 1.4773 | 5.0578 | 73.3165 | 芝加哥一律微调。上车量门 1.4737。3×3 1.5283 |

### 问题 / 下一步

- 没有按这些表改 150、0.5 或 0.8
- 没有重训，也没有做 2km 或 3km 预测
- 种子 43 和 44 没有重放

---

## 2026-11-10 — 用最相似的源城格子做单格预报

**目标：** 相关高于 0.8 且三天上车不少于 150 的华盛顿格子，只用匹配到的那一个源城格子学 6 小时滞后形状，再用这 72 小时校正水平。验证集和整城微调比。  
**配置：** 不看邻域，不改 ST-Net。岭回归惩罚 1。源城小时是 [0, 5832)，按华盛顿三天均值缩放。验证集不严格更低就不算测试集。  
**Git commit：** 本 PR  
**机器 / GPU：** Windows 本机。不使用 GPU。

### 做了什么

- scripts/17_matched_cell.py 对纽约、芝加哥各 46 个格子分别拟合。
- 对照是同一批格子的整城微调缓存，以及只用 72 小时的岭回归。
- 	ests/test_matched_cell.py 5 项通过。

### 运行命令

`ash
.\.venv\Scripts\python.exe -m pytest tests/test_matched_cell.py -q
.\.venv\Scripts\python.exe scripts/17_matched_cell.py
`

### 真实结果（仅实跑后填写）

验证集，46 格平均 MAE。纽约：匹配格子 6.4362，整城微调 6.8981，只用三天 6.4334。29 格低于微调。芝加哥：匹配格子 6.5973，整城微调 6.2626，只用三天 6.4442。20 格低于微调。

纽约验证集打开了测试集。测试集这 46 格：匹配格子 5.9586，整城微调 5.7476，只用三天 6.0064。芝加哥测试集未计算。

缩放倍数中位数：纽约 0.2621，芝加哥 0.6298。

| split | MAE | RMSE | MAPE | notes |
|---|---|---|---|---|
| val | 6.4362 | — | — | 纽约匹配格子，46 格。整城微调 6.8981。只用三天 6.4334 |
| test | 5.9586 | — | — | 纽约匹配格子。整城微调 5.7476。芝加哥未计算 |

### 问题 / 下一步

- 没有因为纽约验证集更低去改 0.8 或 150
- 芝加哥测试集按规则没有计算


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
- [ ] Group 5：六个方向的粗 Pearson，以及从零 / 微调 / 蒸馏（阈值未定，0.8 与 0.5 只是举例）
- [ ] Group 6：兴趣点、路网度、上下车比的功能标签，按类报告误差
- [ ] Group 3 Pearson 乘子：等 Group 5 说明中等相关区间里蒸馏更好，再定公式
