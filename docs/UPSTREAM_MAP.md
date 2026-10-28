# 上游仓库 ↔ 本仓库映射

> 中文主文档。路径均来自 2026-10-27 拉取的 **live GitHub 树**。ST-Net、Group 1、G1-C 与 Group 2 已在本机跑完，指标见 [RESULTS.md](RESULTS.md)。  
> 相关： [DATA.md](DATA.md) · [REPRODUCTION.md](REPRODUCTION.md) · [PAPERS.md](PAPERS.md)

## 总览

| 本仓库 | 上游对照 | 状态 |
|---|---|---|
| `src/models/st_net.py` | CrossTReS / STBaT `model.py` 中 `STNet` | **已移植** `forward`（无 `functional_forward`） |
| `src/train/train_teacher.py` | CrossTReS `run_crosstres.py` 源训环；STBaT `main.py` 源预训练段 | **Group 1 训环已接**；`--dry-run` 仍只加载配置。不含 region re-weighting |
| `src/data/splits.py` | CrossTReS/STBaT `utils.py::split_x_y` + `data_amount` 截断 | **已对齐** `60*24`；`few_shot_split_indices` 仍是旧的 `30*24` 示意图 |
| `configs/baseline_target_fewshot.yaml` | STBaT `--data_amount 3` / CrossTReS 同名参数 | **G1-tgt-3d 已跑** |
| `src/models/kd.py` | TransGTR `engine.py::distil_trainer`；ST-MLP student 训环 | **Group 2 已接**；Pearson 乘子仍是 hook |
| `src/train/train_student_kd.py` | 上列 KD 训环 + STBaT 3-day 目标侧 | **Group 2 已跑** |
| `src/train/train_finetune.py` | 源城权重在目标城上继续训练 | **G1-C 已跑**，无 KD |
| `src/eval/metrics.py` | CrossTReS/STBaT `evaluate`；TransGTR `util.masked_*` | **可用公式**（仅对真实预测调用） |
| `src/utils/pearson_weekday_profile.py` | （本工作新增 hook） | Group 3+；**勿超前实现最终匹配** |
| `src/data/dataset.py` | 上游 `np.load("../data/...")` | **可加载** Taxi pickup，并做 min-max 与窗口 |

---

## 1. CrossTReS ST-Net → `src/models/st_net.py` / 教师训练

**上游仓：** https://github.com/KL4805/CrossTReS  

| 上游路径 | 内容 | 本仓库 |
|---|---|---|
| `src/model.py` → `class STNet`（约 L369+） | Conv2d → `ResUnit`×`num_convs` → LSTM(64→128) → Linear → sigmoid；可选 `return_feat` | `src/models/st_net.py`（**已移植 `forward`**） |
| `src/model.py` → `STNet_nobn` / `STResNet*` | 无 BN / ResNet 变体；`run_crosstres.py` 默认常为 `STNet_nobn` | 移植时按 config 选型；Phase-1 优先 **STNet** 与 STBaT 对齐 |
| `src/model.py` → `ResUnit` / `functional_lstm` | 残差块与 meta 用 functional LSTM | 随 ST-Net 一并移植 |
| `src/run_crosstres.py` | 数据加载、`split_x_y`、源训、区域重加权等全流程 | `src/train/train_teacher.py`（源城拼接训练 + 目标 3-day）；**不**复现 region re-weighting |
| `src/utils.py` | `split_x_y`、`min_max_normalize`、图构造 | `src/data/splits.py` + 日后 preprocess |
| `src/run_crosstres_rt.py` / `gen_rt_dict.py` | RegionTrans 微调 | Phase-1 非必须 |
| `configs/teacher_{ny,chi}.yaml` | （本仓）源城教师超参占位 | 对照 `run_crosstres.py` argparse |

**ST-Net 输入约定（上游）：**

- `lag = [-6,-5,-4,-3,-2,-1]` → 历史 6 小时  
- batch `X`：`(B, L, H, W)`（`L=6`，单通道时 `num_channels=1`）  
- 输出：有效格上的 `(B, 1, N_valid)`（mask 后）或特征  

**下一步必须移植：** `STNet.forward` + `ResUnit` + 与 mask 一致的输出形状；在 `train_teacher.py` 实现真实 train/val 环。

---

## 2. STBaT 3-day few-shot → `src/data/splits.py` / 目标配置

**上游仓：** https://github.com/ShaunHao/stbat  

| 上游路径 | 内容 | 本仓库 |
|---|---|---|
| `utils.py` → `split_x_y` | `val_num=test_num=60*24`；按时间切 train/val/test 窗口 | `few_shot_split_indices`（用 `30*24` 近似「月」—— **移植时应改为与上游相同的 `60*24` 小时**） |
| `main.py` L161–163 | `if data_amount != 0: train = train[-data_amount*24:]` | `few_shot_days: 3` → 72 小时 |
| `main.py` argparse `--data_amount` | `0` 全量；`30/7/3` 天 | `configs/baseline_target_fewshot.yaml`、`student_kd_*.yaml` |
| `main.py` L156–157 | `np.load("%s%s/%s%s_%s.npy" % (data_dir, city, dataname, city, datatype))` | `src/data/dataset.py` 期望路径见 DATA.md |
| `model.py` → `STNet` | 与 CrossTReS 同思路的网格 ST-Net（`num_logits` 变体） | 移植时可交叉对照；**以 CrossTReS `model.py` 为 Phase-1 主源** |
| `main.py` → `evaluate` | RMSE / MAE（反归一化打印） | `src/eval/metrics.py` |

**协议要点（目标城）：**

1. 先 `split_x_y` → test / val 各 1440 小时  
2. 再对 **train** 截最后 `few_shot_days * 24`  
3. Phase-1：`few_shot_days=3`；零样本 `0` 非本阶段默认  

`split_x_y` 与 `trim_few_shot` 已按上游 `60*24` 和 `data_amount*24` 实现。归一化统计量保存在 checkpoint 里（`max_val` / `min_val`），不写回 YAML。

---

## 3. KD 模式 → `src/models/kd.py` / `train_student_kd.py`

Phase-1 **不**跑 TransGTR/ST-MLP 数据；只借 **冻结教师 + 蒸馏损失** 写法。

### TransGTR（传感器图 KD 参考）

**仓：** https://github.com/KL4805/TransGTR  

| 上游路径 | 可借鉴点 | 本仓库 |
|---|---|---|
| `train_distil.py` | CLI：`--teacher_model_path`、`--lambda_d`、源/目标数据路径 | `train_student_kd.py` argparse / config |
| `engine.py` → `distil_trainer` | `teacher.eval()`；student 训；`distil_loss = (s-t).pow(2).mean()` | `KDWrapper.kd_loss`（已实现 MSE；`forward` 仍 TODO） |
| `util.py` → `masked_mae/rmse/mape` | 带 null mask 的指标 | 网格任务可先用无 mask 版；有效格 mask 后对齐 STBaT |

### ST-MLP（同城压缩 KD 参考）

**仓：** https://github.com/zhangjunfeng1234/ST-MLP  

| 上游路径 | 可借鉴点 | 本仓库 |
|---|---|---|
| `run_student.py` | 加载冻结教师 → 建 student → `trainer` 联合损失 | Group 2 训环结构 |
| `models/losses.py` | 任务损失 + KD 项组合 | `KDWrapper.total_loss` |
| `student_model/` | MLP student（同城） | **勿**直接当跨城骨干；本 Phase 学生仍为 ST-Net 级网格模型 |

### 本工作 Group 2 / 3

| 组件 | 说明 |
|---|---|
| Group 2 | `freeze_module(teacher)` + `L_task + λ_kd * L_kd`；`pearson_multiplier=false` |
| Group 3 hook | `KDWrapper.pearson_mask` × residual；轮廓自 `pearson_weekday_profile.py`；**匹配公式 TBD** |

---

## 4. 指标 → `src/eval/metrics.py`

| 上游 | 行为 | 本仓库 |
|---|---|---|
| CrossTReS `run_crosstres.py` → `evaluate` | 在 spatial mask / 权重下算误差 | 移植训环时调用 |
| STBaT `main.py` → `evaluate` | 返回 rmse, mae；打印时乘 `(max-min)` 反归一化 | 记录指标时注明是否反归一化 |
| TransGTR `util.py` | `masked_mae` / `masked_rmse` / `masked_mape` | 可作 mask 版扩展参考 |
| `src/eval/metrics.py` | NumPy `mae` / `rmse` / `mape` / `summarize` | **已实现公式**；禁止填虚构实验表 |

---

## 5. Stub vs 下一步必做（行动清单）

| 优先级 | 项 | 说明 |
|---|---|---|
| P0 | 按 [DATA.md](DATA.md) 本地下载 zip（勿提交） | 链接已核实 |
| P0 | 移植 `STNet`（CrossTReS `src/model.py`） | 已做 `forward`；未做 `functional_forward` |
| P0 | `split_x_y` + `data_amount` 对齐 | 已做 |
| P0 | `train_teacher` 真训环 | Group 1 已实跑，见 RESULTS.md |
| P1 | `KDWrapper.forward` + `train_student_kd` | Group 2 已实跑 |
| P1 | `train_finetune` | G1-C 已实跑 |
| P1 | Dataset 读 `Taxi{CITY}_pickup.npy` 等 | 见 DATA.md 文件名 |
| P2 | Pearson 乘子 | 仅 hook；等顾问定稿 |
| — | Izadi 代码 | **仍无官方仓**（见 PAPERS.md） |
| — | CrossTReS region re-weighting / STBaT RIAM | Phase-1 **不**做 |

---

## 6. 合成接口测试（无真实数据）

```bash
python scripts/03_synthetic_shapes.py
# 或
python -m pytest tests/test_synthetic_shapes.py -q
```

仅断言 ST-Net 构造、KD `kd_loss` / `total_loss` 张量形状与 split 长度；**不**调用未实现的 `forward` 真预测。
