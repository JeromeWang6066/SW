# 实验计划

> Prep 阶段文档。不含任何实验结果数字。

## 研究问题（一句话）

在跨城市网格交通流少样本设定下，**冻结源城教师 +（后续）工作日 Pearson ρ̃ 调制的 KD**，能否相对标准 KD / 无迁移基线稳定提升 DC 上的预测？

## 城市与协议

| 角色 | 城市 | 备注 |
|---|---|---|
| Source | NY 或 CHI | 教师在充足数据上训练后冻结 |
| Target | DC | STBaT 风格 **3-day few-shot** |
| Backbone | CrossTReS **ST-Net** | Phase-1 不换骨干 |

划分原则（对齐 STBaT / CrossTReS 惯例，实现时以数据脚本为准）：

- 源：train/val ≈ 2:1  
- 目标：test ≈ 最后 2 个月；val ≈ 其前 2 个月；train = val 前 **3 天**

## 分组详情

### Group 1 — 复现优先：基线

| ID | 设定 | 成功标准（定性） |
|---|---|---|
| G1-A | Source-only ST-Net（NY / CHI） | 训练收敛；val 曲线合理；可导出冻结权重 |
| G1-B | Target-only 3-day ST-Net（DC） | 在极少样本下可跑通评估管线 |
| G1-C（可选） | Source→Target fine-tune 无 KD | 作为迁移下限对照 |

**Done（复现）：** 代码与 config 能复跑；指标来自真实 eval；与文献数量级可对照（允许实现差异，须记录差异原因）。

### Group 2 — 复现优先：标准 KD

| ID | 设定 |
|---|---|
| G2-NY2DC | 冻结 NY 教师 → DC 学生，`L_task + λ L_kd` |
| G2-CHI2DC | 冻结 CHI 教师 → DC 学生 |

**Done（复现）：** 教师确实冻结；λ 可扫；相对 G1-B 有可解释的变化（变好/变差都如实记录）。

### Group 3 — 贡献方向：Pearson ρ̃ 乘子

- 不可学习；基于源–目标 **工作日 24h** 区域轮廓 Pearson 相关。  
- 用于调制 KD（区域 / 小时），**不是**另训一个 weighting 网络。  
- Matching / 多通道轮廓：**TBD**，仅保留 hook。

**Done（贡献）：** 乘子定义写入论文级公式 + 消融证明其非随机噪声；与 G2 公平对比。

### Group 4 — 后续 Ablation

- ρ̃ 开关、λ、few-shot 天数（1/3/7）、匹配策略变体等。  
- 在 Groups 1–2 稳定前不展开。

## “复现完成” vs “贡献完成”

| | 复现（Phase-1） | 贡献（Phase-2+） |
|---|---|---|
| 范围 | Groups 1–2 | Groups 3–4 |
| 骨干 | 对齐 CrossTReS ST-Net | 仍以 ST-Net 为主，除非有强理由 |
| 协议 | STBaT 3-day | 同协议上的方法增益 |
| 交付 | 可复跑脚本 + 真实表 + 差异说明 | 方法描述 + 消融 + 局限 |

## 风险与依赖

- 数据包 URL / 网格尺寸未最终钉死 → 见 DATA.md TODO  
- ST-Net 实现细节依赖上游 → 对照官方 `model.py`  
- Group 3 公式未定稿 → 禁止在 stub 里假装已实现完整算法  

## 时间盒（技术范围，非日历）

1. 数据落地 + 形状核对  
2. ST-Net 最小可训实现 + G1  
3. 冻结教师 KD + G2  
4. 再开 Group 3 设计评审  
