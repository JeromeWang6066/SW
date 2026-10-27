# 数据说明

> **政策：** 不将大型原始 trip CSV / 完整城市网格提交进 git。  
> 本文件描述**期望布局与张量约定**；下载链接未最终确认处标 **TODO**。

## 城市与任务

| 代码 | 城市 | Phase-1 角色 |
|---|---|---|
| `ny` | New York | Source（教师） |
| `chi` | Chicago | Source（教师） |
| `dc` | Washington, D.C. | Target（3-day few-shot 学生） |

模态：taxi / bike 网格流量（pickup / dropoff 等，以实现与上游一致为准）。

时间粒度（文献惯例）：**1 小时**。

## 期望目录布局

```text
data/
  README.md                 # 本说明的短版也可放在此
  raw/                      # 原始 zip / csv（gitignore）
    crosstres_data.zip      # 示例名；实际文件名以作者包为准
  processed/
    ny/                     # 处理后的 npz/pt/npy 等
    chi/
    dc/
```

检查脚本（只验目录，不下载）：

```bash
python scripts/01_verify_data_layout.py
```

## 下载链接（待填）

| 来源 | 说明 | URL |
|---|---|---|
| CrossTReS | README：`data` 仓库中的 `crosstres_data.zip` | **TODO** — 确认完整 GitHub/网盘链接后填入 |
| STBaT | README：`data` 仓库中的 `data.zip`；文称数据来自 Jin et al. 2022 | **TODO** |
| 官方页 | CrossTReS 代码仓 | https://github.com/KL4805/CrossTReS |
| 官方页 | STBaT 代码仓 | https://github.com/ShaunHao/stbat |

获取后请在 `notes/progress.md` 记录：URL、日期、校验和、许可条款摘要。

## 张量 / 形状约定（占位）

实现前以解压后的真实数组为准；下列为**工作约定**，写入 config 时可覆盖。

| 表示 | 形状 | 说明 |
|---|---|---|
| 网格流 | `T × H × W` 或 `T × H × W × C` | `T` 时间步；`C` 可为 in/out 通道 |
| 展平区域 | `T × N`（`N = H*W`） | 便于与 region-wise Pearson 对齐 |
| 窗口样本 | `B × L × H × W` | `L` = 历史长度（CrossTReS/STBaT 常用 horizon 相关超参） |

**网格尺寸 `H,W`：** **TODO**（解压后填写；各城市可能不同）。

**Region 索引：** 行优先或作者脚本顺序 —— 与 Pearson 轮廓、matching hook 共用同一映射。

## 时间划分（STBaT 风格，待脚本固化）

目标城 DC（few-shot）：

1. **Test：** 时间轴末尾约 **2 个月**  
2. **Val：** test 之前约 **2 个月**  
3. **Train：** val 之前 **`few_shot_days: 3`**（零样本为 0，本 Phase 默认 3）

源城 NY/CHI：train:val ≈ **2:1**（无 3-day 限制）。

配置占位见 `configs/*.yaml`。

## 时区与日历

- NY / CHI / DC：美国相关时区（美东等）—— **夏令时处理 TODO**：对齐上游预处理脚本，避免手工猜偏移。  
- **工作日轮廓：** Group 3 的 Pearson ρ̃ 仅用 weekday；周末策略另定（TODO）。

## 归一化

STBaT：min-max。CrossTReS：以实现脚本为准。  
**要求：** 源/目标归一化统计量来源写进 config（fit 在源 train？或各自？）—— 实现时显式记录，禁止默默混用。

## 许可与署名（TODO）

| 项 | 状态 |
|---|---|
| 原始 trip 开放数据许可（TLC 等） | TODO：按城市数据源补链接与引用 |
| CrossTReS 整理包使用条款 | TODO：遵循作者 README |
| 本仓库 | 仅存布局与文档；处理后的大数组本地保留 |

引用论文时至少包含 Jin et al. KDD 2022（数据整理）与 Hao et al. IJCAI 2025（协议参考）。
