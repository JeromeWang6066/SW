# Duke Kunshan Signature Work · 跨城市网格交通流知识蒸馏

**学生：** Tianyao Wang (Jerome) · DKU CS  
**仓库：** [JeromeWang6066/SW](https://github.com/JeromeWang6066/SW)  
**阶段：** 实验前脚手架（Prep only）—— **未训练、未下载大数据、无虚构结果**

### 快速入口

- **[docs/HANDOFF.md](docs/HANDOFF.md)** — ★ **Agent / 本地接手冷启动**（项目身份、已完成 PR、数据事实、stub 契约、下一步任务）
- [docs/DATA.md](docs/DATA.md) — 已核实 zip 直链 / SHA256 / 网格形状
- [docs/UPSTREAM_MAP.md](docs/UPSTREAM_MAP.md) — 上游 ↔ 本仓移植清单
- [docs/REPRODUCTION.md](docs/REPRODUCTION.md) — Day 0–N 真实复现手册

---

## English abstract

Cross-city few-shot **grid traffic flow / demand** prediction with knowledge distillation: a **frozen teacher** on a source city (NY or CHI), a **non-learnable weekday 24h Pearson ρ̃ multiplier** modulating KD by region/hour profile similarity, and a **STBaT-style 3-day few-shot** target protocol on DC. Phase-1 backbone: **CrossTReS ST-Net**. Groups 1–2 first (baselines + standard KD); Groups 3–4 (ρ̃ multiplier / matching) later.

---

## 研究主题（已获顾问认可）

跨城市少样本网格交通流 / 需求预测，采用知识蒸馏：

1. **源城市冻结教师**（NY 或 CHI）
2. **不可学习的工作日 24 小时 Pearson ρ̃ 乘子**，按区域 / 小时轮廓相似度调制 KD 强度
3. 目标城市 **DC**，对齐 **STBaT 式 3 天 few-shot** 协议
4. Phase-1 骨干：**CrossTReS ST-Net**（先复现，不先造新网络）

> Matching 算法与多通道轮廓细节仍为 TBD —— 代码中仅留 hooks / TODO，不发明最终公式。

## 实验分组

| Group | 内容 | 阶段 |
|---|---|---|
| **1** | Source-only / target few-shot 基线（ST-Net，CrossTReS 风格） | 已实跑，见 RESULTS |
| **2** | 标准 KD（冻结教师 → 学生），STBaT 3-day，NY/CHI→DC | 已实跑，见 RESULTS |
| **1-C** | 源城权重在 DC 上微调，无 KD | 已实跑，见 RESULTS |
| 3 | 冻结教师 + 工作日 Pearson ρ̃ 乘子调制 KD | Groups 1–2 之后 |
| 4 | Ablation / matching 变体（TBD） | 更后 |

## 仓库状态（诚实声明）

| 项目 | 状态 |
|---|---|
| 文档与配置骨架 | ✅ 本仓库已提供 |
| 代码 stub / smoke 检查 | ✅ 可导入、可加载 YAML、可检查目录 |
| 真实训练循环 | Group 1、G1-C、Group 2 已在本机跑完 |
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
│   ├── HANDOFF.md          # ★ Agent / 本地接手冷启动
│   ├── REPRODUCTION.md     # 后续真实复现主手册
│   ├── UPSTREAM_MAP.md     # 上游 ↔ 本仓映射
│   ├── EXPERIMENT_PLAN.md
│   ├── PAPERS.md
│   ├── DATA.md             # 已核实下载直链
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

- **[docs/HANDOFF.md](docs/HANDOFF.md)** — ★ Agent / 本地接手交接（冷启动必读）
- **[docs/RESULTS.md](docs/RESULTS.md)** — 已跑完的 Group 1 / G1-C / Group 2 指标
- **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)** — 复现操作清单
- **[docs/DATA.md](docs/DATA.md)** — **已核实** CrossTReS / STBaT zip 直链、网格形状、本仓布局
- **[docs/UPSTREAM_MAP.md](docs/UPSTREAM_MAP.md)** — 上游文件路径 ↔ 本仓 stub / 下一步移植清单
- **[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)** — Groups 1–4 与完成标准
- **[docs/PAPERS.md](docs/PAPERS.md)** — 论文 / 代码链接
- **[docs/advisor/选题汇报.html](docs/advisor/选题汇报.html)** — 选题汇报幻灯片

## 许可

MIT License。第三方论文代码与数据集遵循各自原许可；本仓库不 vendoring 上游整树，也不提交论文 PDF。

## 致谢与引用

Phase-1 复现将依赖 CrossTReS（ST-Net）与 STBaT（3-day few-shot 协议）。详见 [docs/PAPERS.md](docs/PAPERS.md)。
