# 论文与代码阅读清单

> 链接已于脚手架阶段做**快速网页核验**（2026-10）。不确定处标 **TODO**，不编造。  
> 本仓库不提交论文 PDF（版权）；请用 DOI / arXiv / 作者页。

---

## Phase-1 核心（必读）

### 1. CrossTReS（KDD 2022）— 骨干 ST-Net / 网格跨城设定

- **题名：** Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting  
- **作者：** Yilun Jin, Kai Chen, Qiang Yang  
- **DOI：** https://doi.org/10.1145/3534678.3539250  
- **作者 PDF：** https://kl4805.github.io/files/final_kdd22.pdf  
- **代码（已核验）：** https://github.com/KL4805/CrossTReS  

**对本 SW 的作用：** Phase-1 **预测骨干采用其 ST-Net**；NY/CHI/DC 网格 taxi/bike 数据惯例与跨城实验设定的主要来源。本工作不优先复现其 region re-weighting 全框架，而是借用骨干与数据形态。

```bibtex
@inproceedings{jin2022crosstres,
  title={Selective Cross-City Transfer Learning for Traffic Prediction via Source City Region Re-Weighting},
  author={Jin, Yilun and Chen, Kai and Yang, Qiang},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2022}
}
```

### 2. STBaT（IJCAI 2025）— 3-day few-shot 协议

- **题名：** Balancing Imbalance: Data-Scarce Urban Flow Prediction via Spatio-Temporal Balanced Transfer Learning  
- **作者：** Xinyan Hao, Huaiyu Wan, Shengnan Guo, Youfang Lin  
- **DOI：** https://doi.org/10.24963/ijcai.2025/319  
- **PDF：** https://www.ijcai.org/proceedings/2025/0319.pdf  
- **代码（已核验）：** https://github.com/ShaunHao/stbat  

**对本 SW 的作用：** **目标侧少样本协议**（target 最后约 2 个月 test、其前约 2 个月 val、再前 **3 天** train）。Phase-1 对齐该协议；不必先完整复现 STBaT 的 imbalance 模块。

```bibtex
@inproceedings{hao2025balancing,
  title={Balancing imbalance: data-scarce urban flow prediction via spatio-temporal balanced transfer learning},
  author={Hao, Xinyan and Wan, Huaiyu and Guo, Shengnan and Lin, Youfang},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  pages={2865--2873},
  year={2025}
}
```

---

## 相关阅读（非 Phase-1 骨干）

### 3. TransGTR（KDD 2023）— 图结构 + KD（传感器图）

- **题名：** Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities  
- **DOI：** https://doi.org/10.1145/3580305.3599529  
- **代码（已核验）：** https://github.com/KL4805/TransGTR  
- **作用：** 跨城 + 蒸馏相关阅读；场景偏 **传感器速度图**，不是本 Phase 的网格 ST-Net 骨干。

### 4. SFMGTL（arXiv 2302.11774 / TRC 2024）— 多粒度 / memory

- **题名：** Semantic-Fused Multi-Granularity Cross-City Traffic Prediction  
- **arXiv：** https://arxiv.org/abs/2302.11774  
- **代码（已核验）：** https://github.com/zeonchen/SFMGTL  
- **作用：** 跨城语义融合与 memory 思路参考；非 Phase-1 实现目标。

### 5. ST-MLP（Zhang et al., IEEE TITS 2024）— 同城 STGNN→MLP KD 压缩

- **题名：** Knowledge Distillation-Based Spatio-Temporal MLP Model for Real-Time Traffic Flow Prediction  
- **DOI：** https://doi.org/10.1109/tits.2024.3424808  
- **代码（已核验）：** https://github.com/zhangjunfeng1234/ST-MLP  
- **作用：** 同城蒸馏 / 压缩；**非跨城**。用于理解 KD 损失设计，不作为跨城骨干。

### 6. Izadi et al.（arXiv:2401.11798）— 同城 KD + pruning

- **题名：** Knowledge Distillation on Spatial-Temporal Graph Convolutional Network for Traffic Prediction  
- **arXiv：** https://arxiv.org/abs/2401.11798  
- **代码：** **TODO**（公开页未在快速检索中确认独立官方实现仓库；有代码时补链）  
- **作用：** 同城压缩相关；仅作背景。

---

## 数据相关说明

CrossTReS / STBaT README 均指向独立 **`data` 仓库 / zip**（`crosstres_data.zip` / `data.zip`）。  
**完整下载 URL：TODO**（勿猜测组织名或私有链接）。确认后写入 [DATA.md](DATA.md)。

---

## 阅读优先级（建议）

1. CrossTReS 方法节 + ST-Net 结构 + 数据设定  
2. STBaT 实验设定（few-shot / zero-shot 划分）  
3. 选题汇报 `docs/advisor/选题汇报.html`  
4. TransGTR / SFMGTL 按需扫 KD 与迁移段落  
5. ST-MLP / Izadi 仅作同城 KD 对照  
