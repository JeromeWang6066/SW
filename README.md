# Duke Kunshan Signature Work · 跨城市网格交通流知识蒸馏

**学生：** Tianyao Wang (Jerome) · DKU CS  
**仓库：** [JeromeWang6066/SW](https://github.com/JeromeWang6066/SW)  
**阶段：** 训练结果、分档重放、匹配源格、按格仿射和形状混合都已写入文档。2026-11-16 的方法全文和数据流见 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)。Pearson 乘子仍未实现。数字见 [docs/RESULTS.md](docs/RESULTS.md)。大数据与权重不入库。

### 快速入口

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — 当前方法与每一步的数据流
- **[docs/PAPER_COMPARISON.md](docs/PAPER_COMPARISON.md)** — 相对目标城直接训练的下降比例，含论文摘录
- **[docs/HANDOFF.md](docs/HANDOFF.md)** — 接手说明（文首指向 2026-11-16 的文档；后文有早期 PR 记录）
- **[docs/advisor/2026-11-formula-meeting.md](docs/advisor/2026-11-formula-meeting.md)** — 录音 7 整理：公式要能当场解释，并在现有流程前加误差反馈
- [docs/DATA.md](docs/DATA.md) — 已核实 zip 直链 / SHA256 / 网格形状
- [docs/UPSTREAM_MAP.md](docs/UPSTREAM_MAP.md) — 上游 ↔ 本仓移植清单
- [docs/REPRODUCTION.md](docs/REPRODUCTION.md) — Day 0–N 真实复现手册

---

## English abstract

Cross-city few-shot **grid traffic flow** prediction on taxi pickups. The written pipeline as of 2026-11-16 is [docs/METHODOLOGY.md](docs/METHODOLOGY.md). Phase-1 backbone is **CrossTReS ST-Net** with an **STBaT-style 3-day** target window. Fine-tuning from Chicago is the strongest completed Washington setting. The non-learnable weekday Pearson multiplier is still not implemented.

---

## 研究主题（已获顾问认可）

跨城市少样本网格交通流 / 需求预测。2026-11-05 导师会后的主线是按相似度选择方法，而不是默认蒸馏：

1. **工作日轮廓 Pearson 相关高**：把源城权重复制到目标城，直接微调
2. **相关中等、仍有可借鉴结构**：再用冻结教师做蒸馏
3. 格子事先做功能标定（兴趣点、路网、上下车），看功能区是否影响迁移
4. 目标侧仍对齐 **STBaT 式 3 天 few-shot**。Phase-1 骨干：**CrossTReS ST-Net**
5. **ρ̃ 乘子**留到分流实验说明蒸馏确实在中等相关区间更好之后

> 0.5 和 0.8 在后来的重放里当作事先定下的切点，没有在验证集上再搜。ρ̃ 乘子仍未实现。已经写完的数据流见 [docs/METHODOLOGY.md](docs/METHODOLOGY.md)。

## 实验分组

| Group | 内容 | 阶段 |
|---|---|---|
| **1** | Source-only / target few-shot 基线（ST-Net，CrossTReS 风格） | 已实跑，见 RESULTS |
| **2** | 标准 KD（冻结教师 → 学生），STBaT 3-day，NY/CHI→DC | 已实跑，见 RESULTS |
| **1-C** | 源城权重在 DC 上微调，无 KD | 已实跑，见 RESULTS |
| 初始化 + KD | 复制源城权重后再蒸馏。芝加哥还扫了 λ = 0.1 / 0.5 / 1 / 5 | 已实跑，见 RESULTS |
| 5 | 按 Pearson 相关选择微调或蒸馏 | 已有重放，见 profiles/bin_route.md、grid_route.md。相对微调的测试 MAE 变化小于种子波动 |
| 6 | 兴趣点 / 路网的格子功能标签 | 已有分层，见 profiles/function_cells.md。没有变成新的训练目标 |
| 3 | 冻结教师 + 工作日 Pearson ρ̃ 乘子调制 KD | 未实现 |
| 后处理 | 匹配源格、按格仿射、形状混合 | 已有汇总，见 profiles/。公式说明见 METHODOLOGY |

## 仓库状态（诚实声明）

| 项目 | 状态 |
|---|---|
| 文档与配置骨架 | ✅ 本仓库已提供 |
| 代码 stub / smoke 检查 | ✅ 可导入、可加载 YAML、可检查目录 |
| 真实训练循环 | Group 1、G1-C、Group 2、教师初始化蒸馏、芝加哥 λ 扫描已在本机跑完。华盛顿另有种子 43 与 44 |
| 数据集下载 / 预处理产物 | ❌ **不在本仓库；勿提交大文件** |
| 实验结果表 | ✅ 真实指标见 [docs/RESULTS.md](docs/RESULTS.md)；权重不入库 |

## 快速开始（仅环境与冒烟）

```bash
# 建议 Python ≥ 3.9；CUDA / PyTorch 版本视本机 GPU 而定
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/00_check_env.py
python scripts/02_smoke_import.py
python scripts/03_synthetic_shapes.py     # 合成张量形状；无需真实数据
python scripts/01_verify_data_layout.py   # 只检查目录，不下载
```

`--dry-run` 只加载配置并退出 0。下面的命令会在本机 GPU 上真训练：

```bash
python -m src.train.train_teacher --config configs/teacher_ny.yaml --dry-run
python -u scripts/run_group1.py
python -u scripts/run_group2.py
python -u scripts/run_g1c.py
```

**数据：** 勿提交大文件。已核实直链与解压布局见 [docs/DATA.md](docs/DATA.md)（CrossTReS `crosstres_data.zip` / STBaT `data.zip`）。

## 目录结构

```
SW/
├── README.md
├── LICENSE                 # MIT
├── requirements.txt
├── configs/                # YAML 占位（路径、城市、3-day、KD λ、seed）
├── data/                   # 布局说明 + .gitkeep；大文件 gitignore
├── docs/
│   ├── HANDOFF.md          # 接手说明；文首指向 2026-11-16 文档
│   ├── METHODOLOGY.md      # 方法与数据流
│   ├── PAPER_COMPARISON.md # 相对直接训练的下降比例
│   ├── RESULTS.md          # 已实跑指标
│   ├── REPRODUCTION.md     # 复现操作清单
│   ├── UPSTREAM_MAP.md     # 上游 ↔ 本仓映射
│   ├── EXPERIMENT_PLAN.md
│   ├── PAPERS.md
│   ├── DATA.md             # 已核实下载直链
│   ├── advisor/2026-11-05-meeting.md
│   ├── advisor/2026-11-formula-meeting.md
│   └── advisor/选题汇报.html
├── notes/progress.md       # 实验笔记（逐步记录）
├── scripts/                # 环境 / 数据布局 / 导入 / 合成形状冒烟
├── tests/                  # 无真实数据的形状断言
├── src/
│   ├── data/               # 窗口、mask、min-max
│   ├── models/             # ST-Net 与 KD
│   ├── train/              # Group 1 / G1-C / Group 2
│   ├── eval/               # MAE / RMSE / MAPE
│   └── utils/              # seed, logging, pearson_weekday_profile
└── checkpoints/            # 本地权重目录（不入库）
```

## 文档入口

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — 方法与数据流（2026-11-16）
- **[docs/PAPER_COMPARISON.md](docs/PAPER_COMPARISON.md)** — 相对直接训练的下降比例
- **[docs/HANDOFF.md](docs/HANDOFF.md)** — 接手说明
- **[docs/RESULTS.md](docs/RESULTS.md)** — 已跑完的指标（含三种子与芝加哥 λ 扫描）
- **[docs/profiles/README.md](docs/profiles/README.md)** — 三城每个格子的出租车上车量 24 小时轮廓
- **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)** — 复现操作清单
- **[docs/DATA.md](docs/DATA.md)** — **已核实** CrossTReS / STBaT zip 直链、网格形状、本仓布局
- **[docs/UPSTREAM_MAP.md](docs/UPSTREAM_MAP.md)** — 上游文件路径 ↔ 本仓 stub / 下一步移植清单
- **[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)** — Groups 1–4 与完成标准
- **[docs/PAPERS.md](docs/PAPERS.md)** — 论文 / 代码链接
- **[docs/advisor/选题汇报.html](docs/advisor/选题汇报.html)** — 选题汇报幻灯片
- **[docs/advisor/2026-11-05-meeting.md](docs/advisor/2026-11-05-meeting.md)** — 11 月 5 日会议：按相关选择微调或蒸馏
- **[docs/advisor/2026-11-formula-meeting.md](docs/advisor/2026-11-formula-meeting.md)** — 录音 7：按格公式与误差反馈。不新增实验数字

## 许可

MIT License。第三方论文代码与数据集遵循各自原许可；本仓库不 vendoring 上游整树，也不提交论文 PDF。

## 致谢与引用

Phase-1 复现将依赖 CrossTReS（ST-Net）与 STBaT（3-day few-shot 协议）。详见 [docs/PAPERS.md](docs/PAPERS.md)。
