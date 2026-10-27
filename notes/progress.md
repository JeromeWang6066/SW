# 实验进度笔记（模板）

> 每次实验 / 数据操作追加一节。不要填写虚构指标。

## 元信息

- 学生：Tianyao Wang (Jerome)
- 仓库：https://github.com/JeromeWang6066/SW
- 当前阶段：脚手架 / Prep（Groups 1–2 尚未实跑）

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

- [ ] 确认 CrossTReS / STBaT 数据 zip 直链并写入 `docs/DATA.md`
- [ ] 实现 ST-Net 最小训练环（对照上游 `model.py`）
- [ ] Group 1 基线实跑
- [ ] Group 2 标准 KD 实跑
- [ ] Group 3 Pearson 乘子：等公式定稿
