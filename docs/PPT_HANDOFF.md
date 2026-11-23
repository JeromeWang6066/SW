# 给下一位模型的阅读交接（做 PPT）

> 写于 2026-11-04。读者是接手本项目、负责做汇报幻灯片的另一个大模型。  
> 中文为主。论文名、仓库名、实验 ID 保持英文。  
> 本文是阅读地图加已核对事实，不是新的实验结果。数字以 GitHub `main` 上的 `docs/RESULTS.md` 为准（见 §0）。不要用本地工作区里那份旧的 `RESULTS.md` 做表。

学生：王天遥 / Tianyao Wang (Jerome)。项目：杜克昆山大学 CS Signature Work。  
公开仓库：https://github.com/JeromeWang6066/SW

---

## 0. 先看这一段：本地和线上哪一份是新的

本机目录：

`C:\Users\Jerom\Desktop\Signature Work\实验复现\sw`

2026-11-04 已 `git pull --ff-only`。本地 `main` 与 `origin/main` 同为 `056f5c9`（*Record the Chicago-to-DC lambda sweep.*）。本地 `docs/RESULTS.md` 和 `notes/progress.md` 就是这版，可以直接读。

这 4 个提交把结果补全（从旧到新）：

| 提交 | 说明 |
|---|---|
| `08e2d08` | 记录教师初始化后再蒸馏（λ = 1） |
| `4f747b7` | 记录种子 43、44 的华盛顿测试指标 |
| `aaff13c` | 把种子 42 的 MAPE 改成“只在流量大于 0 的格子上计算” |
| `056f5c9` | 记录芝加哥 → 华盛顿的 λ 扫描 |

同一天又改了几份会误导进度的说明，这些改动还只在本机，没有推到 GitHub：`README.md`、`docs/HANDOFF.md`、`docs/EXPERIMENT_PLAN.md`、`docs/UPSTREAM_MAP.md`、`docs/REPRODUCTION.md`、`notes/progress.md` 开头的阶段一句，以及本文件。

权重、原始 `npy`、以及 `logs/*.json` **不在 GitHub 上**（`.gitignore`）。它们只在本机 `sw\checkpoints\` 和 `sw\logs\`。旧 JSON 里种子 42 主实验的 `mape` 字段仍是旧分母（千万量级），**不要把那些 JSON 里的 MAPE 画进 PPT**。MAE / RMSE 与后来重算一致，可以用。`notes/progress.md` 里 10 月 28 日各节保留了当时的千万级 MAPE，比较时用 10 月 29 日重算的那一节，或 `docs/RESULTS.md`。

---

## 1. 这项工作实际在做什么

一句话：纽约或芝加哥的网格出租车上车量上训练一个 ST-Net 教师，再在华盛顿只用 3 天数据，比较“从零训练 / 复制权重微调 / 知识蒸馏”。

已经跑完的是基线和标准蒸馏。选题里的核心方法——用工作日 24 小时轮廓的 Pearson 相关当蒸馏乘子，并做跨城一对一匹配——**还没有实现**。

| 组 | 选题里的意思 | 代码里实际做了什么 |
|---|---|---|
| 1 | 华盛顿 3 天、随机初始化 | 已做。另有纽约、芝加哥各自的源城教师 |
| 2 | 复制教师权重，λ = 0，只微调 | 已做，实验 ID 是 **G1-C**，不是 “Group 2” |
| 3 | 同一配对上均匀蒸馏，乘子恒为 1 | **未按选题做**。现有 Group 2 是输出上的 MSE 蒸馏，教师直接吃华盛顿窗口，**没有区域配对** |
| 4 | 同一配对，乘子为截断后的 Pearson ρ̃ | **未做**。`pearson_weekday_profile.py` 只有 hook |

做幻灯片时要把“开题时计划的四组”和“已经跑完的对照”分成两页。不要把现有 Group 2 写成“区域乘子实验”。

---

## 2. 建议阅读顺序

做结果汇报，读到第 6 步就可以写。做文献页再往下读。

1. 本文。
2. GitHub `main` 的 `docs/RESULTS.md`（全部可引用数字）。
3. GitHub `main` 的 `notes/progress.md` 里 2026-10-28 与 2026-10-29 的条目（机器、命令、和论文协议的差别）。
4. 本机 `docs/advisor/选题汇报.html`（顾问已经看过的 14 页选题稿，浏览器打开，方向键翻页）。
5. 本机 `docs/EXPERIMENT_PLAN.md`（分组定义；文件头写明不含结果数字）。
6. 本机 `docs/PAPERS.md`（正式引用链接）。
7. 本机 `README.md`（项目身份和目录）。需要改代码时再读 `docs/HANDOFF.md`、`docs/DATA.md`、`docs/UPSTREAM_MAP.md`、`docs/REPRODUCTION.md`。

`docs/HANDOFF.md` 是给 coding agent 的冷启动说明。项目身份、数据 URL、城市形状和硬约束仍然有用。2026-11-04 已把文首进度和分组表改成与 `RESULTS.md` 一致。历史小节里仍保留 PR #1 / #2 当时的写法。

---

## 3. 本地文件：路径和内容

根目录：`C:\Users\Jerom\Desktop\Signature Work\实验复现\sw`

### 写 PPT 时要打开的

| 路径 | 内容 | 使用方式 |
|---|---|---|
| `docs/advisor/选题汇报.html` | 14 页选题幻灯片，标题《基于日周期轮廓相关的跨城市交通流知识蒸馏》。问题、五篇相关工作、六步流程、数据划分、Pearson 匹配、损失、四组对照、风险。写于实验之前 | 文献页和方法页的措辞可以沿用。不要把它当成已有结果 |
| `docs/RESULTS.md` | 结果表。**本地这份是旧的** | 以 GitHub `main` 为准，见本文 §5 |
| `notes/progress.md` | 按日期的实验笔记。本地这份停在 G1-C；GitHub 上还有 10-29 的四节 | 写“和原文协议差在哪里”时用它 |
| `docs/EXPERIMENT_PLAN.md` | 研究问题和 Groups 1–4 的完成标准 | 讲计划，不讲数字 |
| `docs/PAPERS.md` | CrossTReS、STBaT、TransGTR、SFMGTL、ST-MLP、Izadi 的题名、DOI、代码 | 引用页 |
| `README.md` | 学生、仓库、分组、目录、诚实状态 | 封面信息 |
| `docs/HANDOFF.md` | Agent 冷启动：身份、PR、数据事实、代码契约、禁止事项 | 背景；分组进度已过时 |
| `logs/*.json` | 每次实跑的 MAE / RMSE / 种子 / λ / checkpoint 路径。本机有，不入库 | MAE、RMSE 可核对。种子 42 主实验 JSON 的 MAPE 作废 |
| `checkpoints/*.pt` | 本机权重，不入库 | PPT 不需要打开 |

### 只有改代码或核对协议时才需要

| 路径 | 内容 |
|---|---|
| `docs/DATA.md` | CrossTReS / STBaT zip 直链、SHA256、解压布局、网格形状。2026-10-27 核验 |
| `docs/UPSTREAM_MAP.md` | 上游文件 ↔ 本仓文件。ST-Net 已移植；Pearson 仍是 hook |
| `docs/REPRODUCTION.md` | 从环境到 Group 1 / 2 的操作清单 |
| `data/README.md` | 数据目录约定。大文件不进 git |
| `configs/*.yaml` | 教师、3 天基线、微调、标准 KD，以及教师初始化和芝加哥 λ 扫描 |
| `src/models/st_net.py` | CrossTReS ST-Net 的 `forward` |
| `src/models/kd.py` | 冻结教师 + 任务损失与蒸馏损失。Pearson 乘子未接上 |
| `src/train/train_teacher.py` | Group 1 |
| `src/train/train_finetune.py` | G1-C |
| `src/train/train_student_kd.py` | Group 2，以及后来的 `init_from_teacher` |
| `src/data/dataset.py`、`src/data/splits.py` | 窗口、mask、min-max、`60*24` 划分、3 天截断 |
| `src/eval/metrics.py` | MAE / RMSE / MAPE。MAPE 只统计真实流量大于 0 的格子 |
| `src/utils/pearson_weekday_profile.py` | Group 3 的轮廓与相关助手。最终匹配没有实现 |
| `scripts/run_group1.py`、`run_group2.py`、`run_g1c.py` | 正式训练入口 |

### 选题 HTML 的 14 页（不要重新发明结构）

文件：`docs/advisor/选题汇报.html`。浏览器打开即可，不必改代码。

| 页 | 标题 | 这一页在讲什么 |
|---|---|---|
| 01 | 封面 | 教师冻结；目标城只有数天；蒸馏按非学习的日周期轮廓相关加权 |
| 02 | 网格不对齐下的少样本预测 | 区域编号不能互拷；过去 6 小时预测下一小时；目标城训练约 72 小时 |
| 03 | CrossTReS | 源域区域重加权是学出来的。本课题只用数据和 ST-Net |
| 04 | STBaT | 3 天协议和层宽。它的相似度不跨到目标网格 |
| 05 | TransGTR | 长序列向短序列蒸馏。场景是传感器图，不是网格 |
| 06 | MDTLGCN | 全体节点同一个蒸馏权重。页脚写的是另一篇题名，见 §7 的引用警告 |
| 07 | ST-MLP（Zhang 等） | 同城蒸馏的训练形式：教师冻结，标签仍是观测 |
| 08 | 本课题的位置 | 已有工作要么在源域损失里选区域，要么全体节点同等对齐。计划改动是把选择放到蒸馏项上 |
| 09 | 六步 | 数据 → 教师 → 轮廓 → 匹配 → 学生 → 评价。轮廓和匹配只用训练段 |
| 10 | 数据与划分 | 沿用 CrossTReS 网格。主实验 NY→DC、CHI→DC，3 天；7 天只是补充（**7 天还没跑**） |
| 11 | 轮廓、相关与匹配 | 工作日 24 维平均轮廓；ρ̃ = max(ρ, 0)；一对一匹配 |
| 12 | 损失 | 计划中的损失是预测误差加“乘子 × 隐藏表示距离”。**已跑实验不是这个损失** |
| 13 | 四组对照 | 见本文 §1 的对照表。指标计划是上车、下车各自 MAE/RMSE 再平均 |
| 14 | 范围 | 第一阶段只回答乘子是否有效。POI、多教师、图生成都不进第一阶段 |

---

## 4. 协议事实（写方法页时用这些，不要用选题稿里的近似数字）

来自 `DATA.md`、`RESULTS.md`、`progress.md`，均已在本机核对过形状。

- 数据：CrossTReS 的 `crosstres_data.zip`。直链 https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip  
  SHA256 `f322a145484f183042206cf42d54bca1311784024ec384a2046df83d8f637880`  
  本机因 GitHub 直连超时，实际字节来自 jsDelivr，SHA256 与上面相同：  
  https://cdn.jsdelivr.net/gh/KL4805/CrossTReS@main/data/crosstres_data.zip
- 解压后在 `data/raw/crosstres_extract/{NY,CHI,DC}/`，并链接到 `data/processed/{ny,chi,dc}/upstream`。
- 出租车 pickup 形状：纽约 `(8784, 20, 23)`，芝加哥 `(8784, 17, 28)`，华盛顿 `(8784, 21, 20)`。`T = 8784` 小时。
- 训练只用 **Taxi pickup** 这一通道。选题稿写的“上车和下车都预测”还没有做。
- 有效格（流量和 > 0）：进度笔记里纽约 458。华盛顿学生侧是 207。选题稿里的 460 / 476 / 420 是 Hao 等统计的网格数，**不要和有效格混成一个数**。
- 输入 6 小时，预测下一小时。验证、测试各 1440 小时（`60*24`）。
- 源城教师只在 5898 个训练窗口上训练，**没有**把 train+val+test 拼在一起。这和 CrossTReS 原文不同，必须在 PPT 里写出来。
- 华盛顿训练窗口是切分之后的最后 72 小时。
- 归一化：percentile 0.999 的 min-max。指标在反归一化之后、有效格上计算。
- 优化：batch 8，学习率 0.001，weight decay `5e-5`，最多 50 epoch，验证 RMSE 连续 5 个 epoch 不下降则停。机器：Windows，RTX 4060 Laptop 8GB，`torch 2.11.0+cu128`。
- 流量尺度上限：纽约 1212，芝加哥 536，华盛顿 234。源城绝对误差不要和华盛顿横比。
- 现有蒸馏：冻结教师，教师读取的是**华盛顿**归一化窗口（不是选题里“按小时缓存源城隐藏状态”）。损失与任务损失同一归约：batch 上平均，格子上求和。`L = L_task + λ L_kd`。没有区域匹配。

MAPE 定义（GitHub `main`，提交 `aaff13c`）：只在反归一化后真实流量大于 0 的格子上计算，分母是该流量。种子 42 主表的 MAPE 是用旧 checkpoint **重算**的，没有重训。RMSE / MAE 覆盖全部有效格。

---

## 5. 可以放进 PPT 的结果

下面抄自 GitHub `main` 的 `docs/RESULTS.md`（截至 `056f5c9`）。单位是每小时上车量。四舍五入到两位小数的是主表；需要更细就打开那份文件里的四位小数表。

### 5.1 种子 42，华盛顿测试集（主对照）

| 实验 | 设定 | 最佳 epoch | MAE | RMSE | MAPE |
|---|---|---|---|---|---|
| G1-tgt-3d | 华盛顿从零训练，3 天 | 16 | 1.87 | 5.55 | 75.99 |
| G2-NY2DC | 冻结纽约教师，学生随机初始化，λ = 1 | 17 | 1.75 | 5.51 | 59.29 |
| G2-CHI2DC | 冻结芝加哥教师，学生随机初始化，λ = 1 | 12 | 1.75 | 5.41 | 65.39 |
| G1C-NY2DC | 复制纽约权重后全参数微调，无蒸馏 | 3 | 1.72 | 5.28 | 72.39 |
| G1C-CHI2DC | 复制芝加哥权重后全参数微调，无蒸馏 | 1 | 1.48 | 5.06 | 73.32 |

这一轮里，复制权重再微调低于随机初始化蒸馏。芝加哥微调是华盛顿上 MAE、RMSE 最低的一组。

源城教师测试集（不要和上表横比）：纽约 MAE 4.32 / RMSE 15.55 / MAPE 57.23；芝加哥 1.85 / 6.37 / 51.61。

### 5.2 教师初始化后再蒸馏，种子 42，λ = 1，华盛顿测试集

学生先复制源城权重，教师另存一份并冻结，损失仍是 `L_task + 1.0 * L_kd`。

| 实验 | MAE | RMSE | MAPE |
|---|---|---|---|
| G2-NY2DC-init | 1.74 | 5.67 | 64.42 |
| G2-CHI2DC-init | 1.72 | 5.13 | 82.25 |

纽约这一组的 MAE 落在“随机初始化蒸馏”和“纯微调”之间，RMSE 比两者都高。芝加哥这一组优于随机初始化蒸馏，仍差于纯微调。

### 5.3 三种子均值（42 / 43 / 44），只重跑了华盛顿

没有重训源城教师。下表是测试集均值（MAE / RMSE / MAPE）：

| 设定 | 均值 |
|---|---|
| 华盛顿从零，3 天 | 1.885 / 5.523 / 76.83 |
| 纽约微调 | 1.658 / 5.192 / 72.86 |
| 芝加哥微调 | **1.517 / 5.057 / 72.19** |
| 纽约蒸馏，随机初始化 | 1.942 / 5.673 / 73.33 |
| 芝加哥蒸馏，随机初始化 | 1.845 / 5.570 / 67.62 |
| 纽约初始化加蒸馏 | 1.763 / 5.753 / 65.76 |
| 芝加哥初始化加蒸馏 | 1.684 / 5.228 / 78.54 |

三种子看，芝加哥微调仍然最低。纽约随机初始化蒸馏的均值 MAE 1.942，**高于**从零训练的 1.885。种子 42 上“蒸馏略好于从零训练”不能写成稳定结论。

### 5.4 芝加哥 → 华盛顿，λ 扫描，只有种子 42

λ = 1 没有重跑，沿用已有结果。测试集：

| 初始化 | λ = 0.1 | λ = 0.5 | λ = 1 | λ = 5 |
|---|---|---|---|---|
| 随机，MAE / RMSE | 2.00 / 5.66 | 2.11 / 5.68 | **1.75 / 5.41** | 1.99 / 5.72 |
| 教师初始化，MAE / RMSE | **1.48 / 4.93** | 1.56 / 5.05 | 1.72 / 5.13 | 1.94 / 5.68 |

教师初始化且 λ = 0.1 是这一扫描里最低的：MAE 1.478、RMSE 4.933。它和芝加哥纯微调（1.477 / 5.058）的 MAE 几乎一样，RMSE 更低。这是**一个种子**的结果，PPT 里要标明。

### 5.5 还没有的东西（幻灯片上写“未做”）

- Group 3 / 4：Pearson 乘子和一对一匹配。
- 下车量通道、单车通道。
- 7 天补充实验。
- 纽约方向的 λ 扫描。
- 与 CrossTReS、STBaT 完整系统的对照。
- 隐藏状态蒸馏（选题第 12 页的公式）。现有蒸馏是预测输出上的误差。

---

## 6. 线上材料

### 本仓库

- 仓库：https://github.com/JeromeWang6066/SW
- 结果（请用 main）：https://github.com/JeromeWang6066/SW/blob/main/docs/RESULTS.md
- 进度笔记：https://github.com/JeromeWang6066/SW/blob/main/notes/progress.md
- 冷启动说明：https://github.com/JeromeWang6066/SW/blob/main/docs/HANDOFF.md

### 方法论文与代码（`docs/PAPERS.md` 已核验）

| 工作 | 本课题里的角色 | 论文 | 代码 |
|---|---|---|---|
| CrossTReS，KDD 2022 | 数据、ST-Net 骨干。不复现 region re-weighting | https://doi.org/10.1145/3534678.3539250 ；作者 PDF https://kl4805.github.io/files/final_kdd22.pdf | https://github.com/KL4805/CrossTReS |
| STBaT，IJCAI 2025 | 3 天 few-shot 协议和层宽。不复现 RIAM | https://doi.org/10.24963/ijcai.2025/319 ；PDF https://www.ijcai.org/proceedings/2025/0319.pdf | https://github.com/ShaunHao/stbat |
| TransGTR，KDD 2023 | 蒸馏写法参考。传感器图，不是本阶段骨干 | https://doi.org/10.1145/3580305.3599529 | https://github.com/KL4805/TransGTR |
| SFMGTL | 背景。不是实现目标 | https://arxiv.org/abs/2302.11774 | https://github.com/zeonchen/SFMGTL |
| ST-MLP，IEEE TITS 2024 | 同城蒸馏形式。学生在本课题里仍是 ST-Net，不是 MLP | https://doi.org/10.1109/tits.2024.3424808 | https://github.com/zhangjunfeng1234/ST-MLP |
| Izadi 等，arXiv:2401.11798 | 同城压缩背景 | https://arxiv.org/abs/2401.11798 | **没有官方代码，不要编 URL** |

数据包（与代码仓里的 zip 相同，二选一）：

- https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip
- https://github.com/ShaunHao/stbat/raw/main/data/data.zip （SHA256 `18f8e2fab2b390498e849aae2ea835b37a314000c99d9b37896f4ef659f8cf21`）

ST-Net 对照源码：https://github.com/KL4805/CrossTReS/blob/main/src/model.py 里的 `class STNet`。

---

## 7. 桌面上的其他 Signature Work 材料

目录：`C:\Users\Jerom\Desktop\Signature Work\`  
索引页：`C:\Users\Jerom\Desktop\Signature Work\索引.html`（2026-09-21 生成的 PDF 归档说明，25 个 PDF。它不是实验结果）。

做**这一次实验汇报**时建议打开：

| 文件 | 为什么相关 |
|---|---|
| `2026-03\SW_Proposal_TianyaoWang_260311.pdf` | 3 月 11 日开题定稿。和 HTML 选题稿是同一条题目线，HTML 更新 |
| `2026-03\Cross-City traffic prediction based on deep domain adaptive.pdf` | 选题第 6 页脚注指向的域适应跨城预测。做文献页时打开核对作者和题名 |
| `2026-03\Cross-City_Multi-Granular_Adaptive_Transfer_Learning_for_Traffic_Flow_Prediction.pdf` | 跨城多粒度迁移，背景 |
| `2026-06\城市场景下基于域适应迁移学习的交通流预测方法.pdf` | DATE，通信学报 2025，域适应交通流预测 |

`2026-02\` 里的视频异常检测、车联网协同感知、无人机轨迹，以及 `2026-06\` 的 CrossLight / X-Light（交通信号灯），**不是这条网格流量蒸馏线**。除非用户明确要求“文献综述全景”，否则不要放进这次 PPT。

索引里提到、但本机 Signature Work 文件夹中没有副本的 docx / pptx（X-Light 演讲稿、DATE 演讲稿）同样不要当成这次实验的材料。

### 引用警告

选题 HTML 第 6 页的大标题是 MDTLGCN，页脚却是 *Cross-City Traffic Prediction Based on Deep Domain Adaptive Transfer Learning, TRC 176, 105152*。这两处可能不是同一篇。写引用页时打开 §6 的 PDF 和 `docs/PAPERS.md`，不要直接复制这一页的页脚。

---

## 8. 建议的幻灯片边界

听众是顾问，材料是“选题之后的实验进展”，不是把 14 页选题稿再讲一遍。

1. 封面：题目、学生、日期、阶段（基线与标准蒸馏已跑完，Pearson 乘子未做）。
2. 问题：网格不对齐，华盛顿只有 3 天。沿用选题第 2 页，改成“目前只预测出租车上车量”。
3. 沿用什么：CrossTReS 的 ST-Net 和数据，STBaT 的 3 天切分。写明两处实现差别：源城没有拼接 val/test；蒸馏发生在预测输出上，没有区域匹配。
4. 已经比较的设定：从零训练、纯微调、随机初始化蒸馏、教师初始化加蒸馏。用一张表对应选题里的组 1 和组 2，并标明组 3、组 4 未做。
5. 种子 42 的华盛顿测试表（§5.1）。
6. 三种子均值（§5.3）。结论写：微调稳定优于从零训练；随机初始化蒸馏不稳定，纽约方向均值并不更好。
7. 芝加哥 λ 扫描（§5.4），标题上写“单种子”。
8. 还没做的事：工作日轮廓、一对一匹配、ρ̃ 乘子、隐藏状态蒸馏、下车通道、7 天、与原文全模型对比。
9. 下一步就一件：在现有微调基线之上实现选题第 11–13 页，而不是再换骨干。

不要出现的内容：虚构的 Pearson 实验结果、种子 42 旧 JSON 里的千万级 MAPE、把源城 RMSE 和华盛顿 RMSE 画在同一根柱上、把 CrossTReS 原文“约降低 8%”写成我们的结果。

---

## 9. 接手时不要做的事

- 不要把 `logs/`、`checkpoints/`、`data/raw/` 提交到 git。
- 不要编 Izadi 的代码链接。
- 不要为了幻灯片好看改指标定义或重训。缺的实验就写未做。
- 本地 `RESULTS.md` 与 GitHub `main` 冲突时，用 GitHub `main`。
- `docs/HANDOFF.md` 与 `docs/RESULTS.md` 冲突时，进度以 `RESULTS.md` 为准，数据 URL 和形状以 `DATA.md` 为准。
