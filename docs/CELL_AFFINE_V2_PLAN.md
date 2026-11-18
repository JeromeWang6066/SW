# 按格仿射第二版：中心化收缩与旧公式的对照实验

日期：2026-11-17。本页是实验说明。旧公式的定义和已发表数字见 [profiles/cell_affine.md](profiles/cell_affine.md) 和 `scripts/17_cell_affine.py`。跑完后的结果写在 [profiles/cell_affine_v2.md](profiles/cell_affine_v2.md)，不回写本页。

## 0. 先读这一节

要回答的问题只有两个：

1. 把惩罚改成「中心化、按水平换算」之后，按格校正在验证集和测试集上是否比旧公式 λ=72 更好，或者至少不更差但更可解释。
2. 新公式的收缩强度 κ 取多少合适。这个 κ 只能用 72 个训练小时来选。

**要找的量分清楚：**

- κ 是唯一的超参数。每个源城、每个种子选一个，50 个过门格共用。
- \(a\)、\(c\)（以及 \(b\)）不是调出来的。给定 κ 之后，每个过门格各有一对闭式解。50 格就有 50 对，不存在「最合适的一个 a」。报告的是它们的逐格表和分布。

**硬约束：**

- 不训练 ST-Net，只读已有的微调权重做前向。
- 不修改 `scripts/17_cell_affine.py`、`docs/profiles/cell_affine.md`、`docs/profiles/cell_affine.json`、`docs/RESULTS.md`、`docs/METHODOLOGY.md`、`docs/PAPER_COMPARISON.md`。新代码写在新文件里。
- κ、旧公式的 λ 倍数、网格、门槛、判定规则都按本页事先固定。不在验证集或测试集上选任何东西。
- 不实现 Group 3 的工作日 Pearson 乘子。不把 \(a\) 当成形状混合的 \(w\)。
- 不把预测裁成非负，不把 \(a\) 裁到任何区间。旧公式也没有裁，保持可比。
- 当前工作区没有 `checkpoints/` 和 `data/processed/`。必须在有这些文件的机器上执行。

## 1. 三个公式

对一个过门格，记 72 个训练小时的微调反归一化预测为 \(p_t\)，真实上车为 \(y_t\)，单位是每小时上车量。下面 \(n\) 是参与拟合的小时数：最终拟合时是 72，留一天交叉验证时是 48。

### 1.1 旧公式（O）

\[
J_O(a,b)=\sum_t (y_t-ap_t-b)^2+\lambda(a-1)^2+\lambda b^2,\qquad \lambda=m\cdot n.
\]

\(m\) 是倍数。已发表的 λ=72 就是 \(m=1\)、\(n=72\)。交叉验证里 \(n=48\) 时 λ 取 \(48m\)，这样「λ 等于小时数的 \(m\) 倍」的含义不变。正规方程与 `17_cell_affine.py` 的 `solve_ab` 相同。\(m=\infty\) 时直接返回 \((1,0)\)。

### 1.2 只改尺度（S，消融）

\[
J_S(a,b)=\sum_t (y_t-ap_t-b)^2+(p\cdot p)(a-1)^2+n\,b^2,
\]

\[
\begin{bmatrix}2\,p\cdot p&\sum p\\ \sum p&2n\end{bmatrix}\begin{bmatrix}a\\b\end{bmatrix}=\begin{bmatrix}p\cdot y+p\cdot p\\ \sum y\end{bmatrix}.
\]

没有可调参数。它只用来把「改尺度」和「中心化」两个因素分开。

### 1.3 新公式（N，主方法）

\[
\bar p=\tfrac1n\sum_t p_t,\quad p_{c,t}=p_t-\bar p,\quad r_t=y_t-p_t,\quad S=\sum_t p_{c,t}^2,
\]

\[
J_N(\tilde a,c)=\sum_t\big(r_t-\tilde a\,p_{c,t}-c\big)^2+\kappa n\bar p^{2}\tilde a^{2}+\kappa n c^{2}.
\]

闭式解：

\[
\tilde a=\frac{\sum_t p_{c,t}r_t}{S+\kappa n\bar p^2},\qquad c=\frac{\sum_t r_t}{n(1+\kappa)},\qquad a=1+\tilde a,\qquad b=c-\tilde a\,\bar p.
\]

应用时 \(p'_t=ap_t+b=p_t+\tilde a(p_t-\bar p)+c\)。\(\bar p\) 只用拟合小时算，之后固定，验证和测试小时不重算。

- \(\kappa=0\) 时等于不加惩罚的最小二乘。
- \(\kappa=\infty\) 时返回 \((1,0)\)。
- \(S+\kappa n\bar p^2\le 0\) 时（只可能在 \(\kappa=0\) 且 \(p\) 为常数时出现）令 \(\tilde a=0\)。

## 2. 实验臂

每个源城（纽约、芝加哥）、每个种子（42、43、44）都跑下面七行。没过 150 门的 157 格在所有行里都是 \(a=1\)、\(b=0\)。

| 代号 | 公式 | 强度怎么定 | 角色 |
|---|---|---|---|
| B0 | 不校正，一律微调 | 无 | 参照 |
| O72 | 旧公式 | \(m=1\)，即 λ=72 | 已发表的旧方法，主对照 |
| O\* | 旧公式 | \(m^\*\) 由第 3 节留一天交叉验证选 | 旧公式在同一选择规则下的最好情况 |
| S | 只改尺度 | 无参数 | 消融：只修量纲 |
| N1 | 新公式 | \(\kappa=1\)，事先固定 | **主方法** |
| N\* | 新公式 | \(\kappa^\*\) 由第 3 节留一天交叉验证选 | 「最合适的 κ」 |
| N0 | 新公式 | \(\kappa=0\) | 不加惩罚的最小二乘，看收缩是否有用 |

N0 和 O 在 \(m=0\) 时都是最小二乘，逐格 \((a,b)\) 必须一致，这是第 5 节的检查之一。

## 3. κ 和 m 的选择规则（只用 72 个训练小时）

**事先登记的网格**，新旧公式共用同一组倍数：

```text
GRID = (0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, inf)
```

新公式里它是 κ，旧公式里它是 \(m\)（λ = \(m\times n\)）。

**留一天交叉验证：**

1. 训练窗 72 行按标签小时排好：第 0–23 行是 2016-08-31，第 24–47 行是 09-01，第 48–71 行是 09-02。三天就是三折。
2. 对网格里的每个强度、每一折、每个过门格：用另外 48 行拟合 \((a,b)\)，其中 \(\bar p\) 也只用这 48 行重算；然后预测留出的 24 行。
3. 把 50 格 × 72 行（每行恰好被留出一次）的绝对误差加总，除以 3600，得到该强度的交叉验证 MAE。同时记录交叉验证 RMSE。
4. 交叉验证 MAE 最小的强度就是 κ\* 或 \(m^\*\)。若差值小于 1e-9，取较大的强度，也就是更靠近不校正的那个。
5. 用选出的强度在全部 72 行上重新拟合，得到最终逐格 \((a,b)\)。

每个源城 × 种子独立选，一共 6 个 κ\* 和 6 个 \(m^\*\)。

**这条规则的局限，要写进报告：** 只有三折，留出的都是同一周的工作日。它衡量的是周内稳定性，看不到到 11–12 月测试段的漂移，所以选出的强度可能偏松。训练窗上的 \(p\) 是样本内预测，因为微调就是在这 72 个标签上训的。

**「下一轮的默认 κ」怎么定，事先写死：** 6 个 κ\* 里至少 5 个相同且不等于 1，下一轮就把它作为新的固定默认值。否则保持 κ=1。本轮的判定仍按第 7 节。

## 4. 输入数据与执行顺序

**权重**（只在本机 `checkpoints/`）：

| 源城 | 种子 42 | 种子 43 | 种子 44 |
|---|---|---|---|
| 纽约 | `finetune_ny2dc.pt` | `finetune_ny2dc_seed43.pt` | `finetune_ny2dc_seed44.pt` |
| 芝加哥 | `finetune_chi2dc.pt` | `finetune_chi2dc_seed43.pt` | `finetune_chi2dc_seed44.pt` |

文件名规则来自 `src/train/train_finetune.py`：种子不是 42 时，在文件名主干后加 `_seed{seed}`。如果本机文件名不同，停下来报告，不要猜。

**数据**：与 `17_cell_affine.py` 相同。`prepare_city_windows("dc", processed_dir, few_shot_days=3)`；门槛用 `14_cell_volume.py` 的 `_prepare(raw, raw_valid)["three_sum"].sum(axis=1) >= 150`，必须恰好 50 格。网络加载用 `13_bin_route.py` 的 `_load_net`，模型配置读 `configs/default.yaml`。

**分三个阶段跑，顺序不能换：**

1. `--stage predict`：对 6 个权重各做一次前向，拿到 train、val、test 三段的反归一化预测和标签，存到 `logs/cell_affine_v2_cache/{source}_seed{seed}_{split}.npz`（`logs/` 已被 gitignore）。同时做第 5 节的基线核对。这一步只给未校正的微调打分，不拟合任何 \((a,b)\)。
2. `--stage fit`：**只读 train 缓存**。算第 3 节的交叉验证曲线，选 κ\* 和 \(m^\*\)，给七个实验臂拟合最终逐格 \((a,b)\)。写出 `docs/profiles/cell_affine_v2_fit.json`。这一步的代码路径里不允许出现 val 或 test 缓存的读取。
3. 建议在这里先 `git commit` 这个 fit 文件，把选出的强度固定在提交历史里，然后再打分。
4. `--stage score`：读 fit 文件和 val、test 缓存，把每个臂的 \((a,b)\) 用到验证小时和测试小时，算指标和第 6 节的诊断。fit 文件只读，不允许改写。

## 5. 必须先通过的核对

任何一条不过，就停下来报告，不要继续往后跑。

1. **门槛**：过门格数等于 50；原始门槛和模型标签门槛不一致的格子数为 0（与 `cell_affine.md` 一致）。
2. **训练窗**：train 段恰好 72 行。第 \(t\) 行的标签等于华盛顿截断后计数在小时 \(5832+t\) 的值，误差小于 1e-3。这一条是确认三折确实按日期切开。
3. **未校正微调复现已发表数字**，MAE 和 RMSE 误差都不超过 1e-3：
   - 种子 42 用 `13_bin_route.py` 的 `PUBLISHED`：纽约 val 2.0548 / 6.1937，test 1.7153 / 5.2786；芝加哥 val 1.7130 / 5.5809，test 1.4773 / 5.0578。
   - 种子 43、44 的 test 用 `RESULTS.md`：纽约 1.6344 / 5.1229、1.6242 / 5.1752；芝加哥 1.5319 / 5.0045、1.5422 / 5.1079。
   - 种子 43、44 的 val 若在 `logs/` 的结果 JSON 里有，就一并核对；没有就如实写「未核对」。
4. **O72 复现 `cell_affine.json`**，只对种子 42，全城 MAE 误差不超过 1e-5：

   | 源城 | train | val | test |
   |---|---|---|---|
   | 纽约 | 1.4845197 | 2.0360146 | 1.7054193 |
   | 芝加哥 | 1.4103619 | 1.6863351 | 1.4415295 |

   \(a\) 的中位数也要对上，误差不超过 1e-5：纽约 0.9889595，芝加哥 0.9390470。
5. **两个极限**：N0 与 \(m=0\) 的 O 逐格 \((a,b)\) 差小于 1e-8；κ=∞ 和 \(m=\infty\) 的输出与 B0 完全相同。

## 6. 指标与诊断

### 6.1 主指标

对每个臂、每段（train / val / test）计算：

- 全城 207 格的 MAE、RMSE、MAPE，用 `17_cell_affine.py` 的 `score_arrays`，口径和已发表的一样。
- 过门 50 格子集和未过门 157 格子集的 MAE。157 格子集在各臂之间必须相同，这本身就是一个检查。

MAPE 单独一列，不参与第 7 节的判定。

### 6.2 配对块自助法（测试集和验证集都做，只报告）

以 B0 或 O72 为参照，对每个臂计算全城 MAE 差。测试段 1440 小时等于 60 天，以「天」为块重抽 2000 次，`numpy.random.default_rng(0)`。报告点估计和 2.5%、97.5% 分位。

```python
def day_block_bootstrap(pred_arm, pred_ref, target, n_rep=2000, seed=0):
    hours, cells = target.shape
    if hours % 24:
        raise ValueError("hours must be whole days")
    days = hours // 24
    arm = np.abs(pred_arm - target).reshape(days, 24, cells).sum(axis=(1, 2))
    ref = np.abs(pred_ref - target).reshape(days, 24, cells).sum(axis=(1, 2))
    idx = np.random.default_rng(seed).integers(0, days, size=(n_rep, days))
    diff = (arm[idx].sum(axis=1) - ref[idx].sum(axis=1)) / (days * 24 * cells)
    point = (arm.sum() - ref.sum()) / (days * 24 * cells)
    return float(point), float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))
```

验证段从 2016-09-03 开始，同样是整天对齐的 1440 小时。

### 6.3 回答导师质疑的诊断

这些量只用训练小时就能算出，按格写进 JSON。以前缺的正是这些。

| 名字 | 定义 | 报告方式 |
|---|---|---|
| `p_bar` | 72 小时预测均值 | 50 格的中位数、10 分位、90 分位 |
| `cv2` | \(S/(n\bar p^2)\) | 同上 |
| `p_dot_p` | \(\sum p_t^2\) | 同上，并报告 \(72/(p\cdot p)\) 的同样三个分位 |
| `a_ols`、`b_ols` | κ=0 的解 | 同上 |
| 反向格数 | \(\operatorname{sign}(a-1)\ne\operatorname{sign}(a_{\rm OLS}-1)\) 的格数 | 每个臂一个数 |
| 水平由 \(a\) 承担的比例 | \(|(a-1)\bar p|\,/\,(|(a-1)\bar p|+|b|)\) | 每个臂的中位数 |
| `a` 的最小值、\(a<0.5\) 的格数 | | 每个臂 |
| 测试段负预测的点数 | \(p'<0\) 的小时 × 格数 | 每个臂 |

再报两个测试段切片，只描述不判定：

- 用 `src/data/calendar_mask.py` 的 `is_weekend_or_holiday(label_hours("test"))` 分成「周一到周五」和「周六日」两段，各算一次 MAE。注意这个函数的节日表只到劳动节，11–12 月的节日不在里面，表头写清楚。
- 验证集上 N 和 O 的完整强度曲线，也就是网格每一点的验证集 MAE，标题写「敏感性，未用于选择」。测试集**只**算七个实验臂，不画强度曲线。

## 7. 判定规则（运行前已固定）

**主比较**：种子 42，N1 对 O72，测试集，纽约和芝加哥分别看。

- **N1 更低**：两个源城的测试 MAE 差的块自助 97.5% 分位都小于 0，并且测试 RMSE 都不高于 O72。
- **N1 更高**：任一源城的 2.5% 分位大于 0。
- 其余情况写**无法区分**。

**稳定性**：对种子 43、44 重复同样的比较。6 个「源城 × 种子」上 MAE 差的点估计符号全部相同，才写「方向稳定」。

**次要比较**，只报告，不改写主结论：

- N\* 对 O\*：两套公式各自用训练小时选到的最好强度。
- S 对 O72：只修量纲的效果。
- N1 对 S：在已经修了量纲的基础上，中心化的效果。
- N0 对 N1：收缩本身有没有用。
- 各臂对 B0：校正相对一律微调的幅度。

**不允许的说法：**

- 只看种子 42 一个点估计，就写「新方法更好」。
- 用验证集或测试集的强度曲线去挑 κ，再把那个 κ 的测试 MAE 当成主结果。
- 把 MAPE 的下降当作 MAE 或 RMSE 的改进。
- 把 1.7153→1.7054 这类差值写成迁移方法的成败。它们是微调之上的小幅校正。

## 8. 新文件

| 文件 | 内容 | 是否入库 |
|---|---|---|
| `scripts/19_cell_affine_v2.py` | 三个阶段、三个求解器、交叉验证、打分、诊断 | 是 |
| `tests/test_cell_affine_v2.py` | 第 9 节的单元测试 | 是 |
| `docs/profiles/cell_affine_v2_fit.json` | 交叉验证曲线、κ\*、\(m^\*\)、逐格 \((a,b)\) 和第 6.3 节的训练窗统计 | 是 |
| `docs/profiles/cell_affine_v2.json` | 全部指标、自助区间、诊断 | 是 |
| `docs/profiles/cell_affine_v2.md` | 第 10 节格式的报告 | 是 |
| `logs/cell_affine_v2_cache/*.npz` | 逐小时预测缓存 | 否，已被 gitignore |

逐格 JSON 只写和、均值、分位、系数，不写逐小时数组。格子编号用华盛顿有效格在行优先展平后的序号，再附上 `(row, col)`。

`docs/profiles/README.md` 末尾只加一行链接，指向 `cell_affine_v2.md`。其他已有文档不改。

### 求解器参考实现

新脚本里把小时数和惩罚分开写。旧脚本第 191 行用 `int(LAM)` 校验小时数，把两件事绑在了一起，新脚本不要沿用这个写法。

```python
N_HOURS = 72
DAY_HOURS = 24
GATE = 150
GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)


def _vec(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def solve_old(prediction, target, mult):
    p, y = _vec(prediction), _vec(target)
    if math.isinf(mult):
        return 1.0, 0.0
    lam = mult * p.size
    gram = np.array([[p @ p + lam, p.sum()], [p.sum(), p.size + lam]])
    rhs = np.array([p @ y + lam, y.sum()])
    a, b = np.linalg.solve(gram, rhs)
    return float(a), float(b)


def solve_scale_only(prediction, target):
    p, y = _vec(prediction), _vec(target)
    pp = float(p @ p)
    gram = np.array([[2.0 * pp, p.sum()], [p.sum(), 2.0 * p.size]])
    rhs = np.array([p @ y + pp, y.sum()])
    a, b = np.linalg.solve(gram, rhs)
    return float(a), float(b)


def solve_centered(prediction, target, kappa):
    p, y = _vec(prediction), _vec(target)
    if math.isinf(kappa):
        return 1.0, 0.0
    n = p.size
    p_bar = float(p.mean())
    pc = p - p_bar
    r = y - p
    denom = float(pc @ pc) + kappa * n * p_bar ** 2
    a_dev = 0.0 if denom <= 0.0 else float(pc @ r) / denom
    level = float(r.sum()) / (n * (1.0 + kappa))
    return 1.0 + a_dev, level - a_dev * p_bar


def lodo_curve(pred_train, target_train, eligible, solver, grid=GRID):
    hours = pred_train.shape[0]
    if hours != N_HOURS:
        raise ValueError(f"train hours {hours} != {N_HOURS}")
    folds = [np.arange(k * DAY_HOURS, (k + 1) * DAY_HOURS) for k in range(hours // DAY_HOURS)]
    cells = np.flatnonzero(eligible)
    curve = []
    for strength in grid:
        abs_sum = sq_sum = 0.0
        count = 0
        for hold in folds:
            keep = np.setdiff1d(np.arange(hours), hold)
            for j in cells:
                a, b = solver(pred_train[keep, j], target_train[keep, j], strength)
                gap = a * pred_train[hold, j] + b - target_train[hold, j]
                abs_sum += float(np.abs(gap).sum())
                sq_sum += float((gap ** 2).sum())
                count += gap.size
        curve.append({"strength": strength, "mae": abs_sum / count, "rmse": (sq_sum / count) ** 0.5})
    return curve


def pick_strength(curve):
    best = min(item["mae"] for item in curve)
    return max(item["strength"] for item in curve if item["mae"] - best < 1e-9)
```

`lodo_curve` 的参数里只有 train 数组。`fit` 阶段不得把 val 或 test 数组传进任何函数。JSON 里的 `inf` 写成字符串 `"inf"`。

## 9. 单元测试

`tests/test_cell_affine_v2.py` 不读城市数组，也不读权重。下面的期望值已经用 Python 核对过，容差 5e-4。

| 用例 | 输入 | 期望 |
|---|---|---|
| 旧公式手算例 1 | \(p=(10,10.2,9.8,10.1)\)，\(y=(12,12.4,11.6,12.2)\)，\(m=1\) | \((1.1984, 0.0180)\) |
| 旧公式手算例 2 | \(p=(4,6,5)\)，\(y=(8,9,7)\)，\(m=1\) | \((1.5059, 0.2353)\) |
| 只改尺度，例 1 / 例 2 | 同上 | \((1.0675, 0.6744)\) / \((1.1845, 1.0386)\) |
| 新公式 κ=1，例 1 / 例 2 | 同上 | \((1.0002, 1.0103)\) / \((0.9870, 1.5649)\) |
| κ=0 等于最小二乘 | 例 2 | \((0.5, 5.5)\)；且等于 `solve_old(..., 0)` |
| 极限 | 任意 | κ=∞ 与 \(m=\infty\) 都返回 \((1,0)\) |
| 常数预测 | \(p\) 全为 5，κ 取 0 和 1 | \(a=1\) |
| 单位不变性 | \(p,y\) 同乘 7 | 新公式与只改尺度：\(a\) 不变，\(b\) 乘 7 |
| 与旧脚本一致 | 随机 \(p,y\)，\(m=1\)，\(n=72\) | `solve_old` 等于 `17_cell_affine.solve_ab(p, y, 72)` |
| 交叉验证极限 | 强度 ∞ | 交叉验证 MAE 等于这 50 格训练窗上 B0 的 MAE |
| 自助法 | 两臂完全相同 | 点估计与两个分位都是 0 |

旧测试 `tests/test_cell_affine.py` 也要一起跑，确认没被动到。

## 10. 报告格式（`cell_affine_v2.md`）

风格跟 [profiles/cell_affine.md](profiles/cell_affine.md) 一致：短句，四位小数，先写协议再写表。按顺序包含：

1. 协议：通道、种子、门槛、三个公式、网格、交叉验证规则、判定规则，并写明这些都在打分前固定。
2. 核对结果：第 5 节五条，逐条写「通过」，或写实际差值。
3. 每个源城 × 种子一张强度选择表：网格每一点的交叉验证 MAE / RMSE，以及选出的 κ\* 和 \(m^\*\)。
4. 主表，每个源城 × 种子一张：

   | 臂 | 强度 | train MAE | val MAE | test MAE | test RMSE | test MAPE | 过门 50 格 test MAE | 相对 O72 的 test MAE 差 [2.5%, 97.5%] |
   |---|---|---:|---:|---:|---:|---:|---:|---|

5. 三种子汇总：每个臂测试 MAE / RMSE 的三种子均值，以及第 7 节主比较和稳定性的结论，只写「更低」「更高」或「无法区分」之一。
6. 系数表：每个臂的 \(a\)、\(b\)、\(\tilde a\)、\(c\) 的中位数和 10–90 分位；第 6.3 节的反向格数和「水平由 \(a\) 承担的比例」。
7. 训练窗统计：\(\bar p\)、CV²、\(p\cdot p\)、\(72/(p\cdot p)\) 的分位。
8. 测试段的周一到周五 / 周六日切片，以及验证集强度曲线，标题写「敏感性，未用于选择」。
9. 局限：样本内预测；三折都是同一周的工作日；测试段在 11–12 月；种子 43、44 的 O72 是这次第一次算。
10. 重算命令。

## 11. 命令

```bash
python -m pytest tests/test_cell_affine_v2.py tests/test_cell_affine.py
python scripts/19_cell_affine_v2.py --stage predict --seeds 42 43 44
python scripts/19_cell_affine_v2.py --stage fit --seeds 42 43 44
git add docs/profiles/cell_affine_v2_fit.json && git commit -m "Freeze cell affine v2 strengths before scoring"
python scripts/19_cell_affine_v2.py --stage score --seeds 42 43 44
```

只需前向，不训练。显卡不可用时用 `--device cpu`。

## 12. 验收清单

- [ ] 第 5 节五条核对全部通过，或已停下并报告差值。
- [ ] `cell_affine_v2_fit.json` 先于打分写出，打分阶段没有改动它。
- [ ] 6 个 κ\* 和 6 个 \(m^\*\) 全部来自训练窗交叉验证，报告里给出完整曲线。
- [ ] 主表七个臂齐全，157 格子集在各臂之间相同。
- [ ] 第 7 节主比较、稳定性、次要比较都已按规则写出结论，没有超出规则的措辞。
- [ ] 逐格系数和训练窗统计写进 JSON，没有逐小时数组。
- [ ] 旧脚本和已有文档未改，只在 `profiles/README.md` 末尾加了一行链接。
