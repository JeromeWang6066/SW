# 数据说明

> **政策：** 不将大型原始 trip CSV / 完整城市网格提交进 git。  
> 本文件描述**期望布局与张量约定**；链接均经本地解压核验（2026-10-27），勿把 zip 提交进本仓库。

## 城市与任务

| 代码 | 城市 | Phase-1 角色 |
|---|---|---|
| `ny` | New York | Source（教师） |
| `chi` | Chicago | Source（教师） |
| `dc` | Washington, D.C. | Target（3-day few-shot 学生） |

模态：taxi / bike 网格流量（`pickup` / `dropoff`）。  
时间粒度：**1 小时**；上游包内 `T = 8784`（366×24，约一年含闰日）。

## 已核实下载链接（优先用 CrossTReS）

上游 README 写 “Go to `data` repo…”，实际就是**各自代码仓内的 `data/` 目录**（非独立 GitHub org）。下列 URL 已下载并解压核验。

### 推荐：CrossTReS（Jin et al. KDD 2022）

| 项 | 值 |
|---|---|
| 来源论文 / 仓 | CrossTReS · https://github.com/KL4805/CrossTReS |
| README 步骤 | Step 1: 解压仓内 `data/crosstres_data.zip`（见 [`data/data_readme.md`](https://github.com/KL4805/CrossTReS/blob/main/data/data_readme.md)） |
| **直链（已核验）** | https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip |
| 备用（raw） | https://raw.githubusercontent.com/KL4805/CrossTReS/main/data/crosstres_data.zip |
| 浏览页 | https://github.com/KL4805/CrossTReS/blob/main/data/crosstres_data.zip |
| 归档名 | `crosstres_data.zip`（≈ 19.6 MB 压缩；解压后≈ 370 MB） |
| SHA256（2026-10-27） | `f322a145484f183042206cf42d54bca1311784024ec384a2046df83d8f637880` |

**解压后上游布局（zip 根下）：**

```text
NY/  CHI/  DC/
  Taxi{CITY}_pickup.npy
  Taxi{CITY}_dropoff.npy
  Bike{CITY}_pickup.npy
  Bike{CITY}_dropoff.npy
  {CITY}_poi.npy
  {CITY}_roads.npy
  Taxi{CITY}_ODPairs / Bike{CITY}_ODPairs   # 非 npy，RegionTrans / 图构造用
```

（zip 内另有 `__MACOSX/` 噪音，可忽略。）

### 等价副本：STBaT（Hao et al. IJCAI 2025）

| 项 | 值 |
|---|---|
| 来源 | STBaT · https://github.com/ShaunHao/stbat |
| README 步骤 | Step 1: 解压仓内 `data/data.zip` |
| **直链（已核验）** | https://github.com/ShaunHao/stbat/raw/main/data/data.zip |
| 备用（raw） | https://raw.githubusercontent.com/ShaunHao/stbat/main/data/data.zip |
| 归档名 | `data.zip`（≈ 18.7 MB） |
| SHA256（2026-10-27） | `18f8e2fab2b390498e849aae2ea835b37a314000c99d9b37896f4ef659f8cf21` |
| 解压布局 | `data/NY/`、`data/CHI/`、`data/DC/`（文件名与 CrossTReS **相同**） |
| 备注 | 论文写明数据由 Jin et al. 2022 整理公开；与 CrossTReS 包内容一致，二选一即可 |

### 可选参考：SFMGTL（同城网格 + 额外邻接）

| 项 | 值 |
|---|---|
| 来源 | https://github.com/zeonchen/SFMGTL |
| 直链（已核验存在） | https://github.com/zeonchen/SFMGTL/raw/main/data/dataset.zip |
| 备注 | 同为 NY/CHI/DC `Taxi*/Bike*` npy，另含 `*_poi_adj.npy` 等；**非 Phase-1 必需** |

### 非本任务数据（勿混用）

| 来源 | 说明 |
|---|---|
| TransGTR | METR-LA / PEMS-BAY / PEMSD7M（传感器速度图），见 [DL-Traff](https://github.com/deepkashiwa20/DL-Traff-Graph) — **不是** NY/CHI/DC 网格流 |
| ST-MLP | METR-LA / PEMS-BAY（Google Drive / 百度云）— 同城传感器 KD 对照，非跨城网格 |

## 建议下载与映射到本仓库

**不要**把 zip / npy 提交进 git（已在 `.gitignore`）。本地操作示例：

```bash
# 1) 下载到 raw（任选 CrossTReS 或 STBaT）
mkdir -p data/raw
curl -L -o data/raw/crosstres_data.zip \
  https://github.com/KL4805/CrossTReS/raw/main/data/crosstres_data.zip

# 2) 解压到 raw 临时目录
mkdir -p data/raw/crosstres_extract
unzip -q data/raw/crosstres_data.zip -d data/raw/crosstres_extract
# CrossTReS: data/raw/crosstres_extract/{NY,CHI,DC}/
# STBaT:     解压后为 data/.../{NY,CHI,DC}/ —— 按实际路径调整

# 3) 映射到本仓库 processed（小写城市码；可软链或复制）
mkdir -p data/processed/{ny,chi,dc}
# 示例：软链整城目录（推荐，省磁盘）
ln -sfn "$(pwd)/data/raw/crosstres_extract/NY"  data/processed/ny/upstream
ln -sfn "$(pwd)/data/raw/crosstres_extract/CHI" data/processed/chi/upstream
ln -sfn "$(pwd)/data/raw/crosstres_extract/DC"  data/processed/dc/upstream

# 4) 布局检查（不下载）
python scripts/01_verify_data_layout.py
```

**本仓库期望布局：**

```text
data/
  raw/
    crosstres_data.zip          # 或 stbat data.zip
    crosstres_extract/{NY,CHI,DC}/...
  processed/
    ny/   # 预处理产物或指向上游的软链（gitignore）
    chi/
    dc/
```

上游脚本期望路径（对照用，见 [UPSTREAM_MAP.md](UPSTREAM_MAP.md)）：

- CrossTReS：`../data/{CITY}/{Taxi\|Bike}{CITY}_{pickup\|dropoff}.npy`
- STBaT：`{data_dir}{CITY}/{Taxi\|Bike}{CITY}_{pickup\|dropoff}.npy`

检查脚本：

```bash
python scripts/01_verify_data_layout.py
```

## 已核实网格形状（解压核验，float64）

| 城市 | 数组形状 `T×H×W` | H×W | Taxi 有效格（sum>0） | Bike 有效格 |
|---|---|---|---|---|
| NY | `(8784, 20, 23)` | 20×23 | 458 | 108 |
| CHI | `(8784, 17, 28)` | 17×28 | 270 | 229 |
| DC | `(8784, 21, 20)` | 21×20 | 207 | 125 |

辅助特征：

| 文件 | 形状 |
|---|---|
| `{CITY}_poi.npy` | `(H, W, 14)` |
| `{CITY}_roads.npy` | `(H, W, H, W)` |

文献：1 km×1 km 网格、1 小时；STBaT Table 1 区域数（NY 460 / CHI 476 / DC 420）与 Taxi 有效格量级一致（计数口径可能含多模态或不同 mask）。

**Region 索引：** 与上游一致时，空间维展平为行优先 `H×W`；Pearson / matching hook 须共用同一映射。

## 时间划分（与上游脚本一致）

CrossTReS / STBaT 共用 `split_x_y`（`val_num = test_num = 60 * 24`）：

1. **Test：** 时间轴末尾 **1440** 小时（≈ 60 天 / “2 个月”）  
2. **Val：** test 之前 **1440** 小时  
3. **Train：** 其余前缀；目标 few-shot 时再截 `train[-data_amount*24:]`（`data_amount=3` → **72** 小时）

源城：CrossTReS 常把 source train/val/test 拼回再训；STBaT 文中写 train:val ≈ 2:1（实现上会把 train+val 拼给源预训练，见上游 `main.py`）。本仓库 Phase-1 目标侧以 **`few_shot_days: 3`** 为准（`configs/*.yaml`）。

## 时区与日历

- 三城均为美东相关开放数据源；**夏令时处理：** 对齐上游预处理（包内已是小时网格），勿在本仓库另行猜偏移。  
- **工作日轮廓：** Group 3 Pearson ρ̃ 仅用 weekday；周末策略另定（TODO）。

## 归一化

STBaT / CrossTReS：`min_max_normalize`（默认 percentile 0.999）。  
**要求：** 源/目标归一化统计量来源写进 config（各自 fit？）—— 实现时显式记录。

## 许可与署名

| 项 | 状态 |
|---|---|
| CrossTReS / STBaT 整理包 | 遵循作者 README / 论文；本仓库仅文档化链接，不 redistrib zip |
| 原始 trip 开放数据（TLC / Divvy / Capital Bikeshare 等） | TODO：按城市开放平台补许可链接（训练前完成即可） |
| 本仓库 | MIT；处理后大数组本地保留 |

至少引用：Jin et al. KDD 2022（数据整理 + ST-Net）与 Hao et al. IJCAI 2025（few-shot 协议）。

获取后请在 `notes/progress.md` 记录：URL、日期、校验和、许可摘要。
