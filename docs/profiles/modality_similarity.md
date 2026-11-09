# 模态之间的城市级相关

通道是出租上车、出租下车、单车上车。单车下车没有算。工作日是周一到周五，不把假日从工作日里拿出去。
源侧用前 5904 小时的工作日。目标侧用最后 72 个训练标签，下标 [5832, 5904)，日期 2016-08-31 至 2016-09-02。
这 72 小时每个钟点有 3 个工作日样本。常数轮廓记为无定义，不记成 0。
城市轮廓是该通道自己的有效格上的 24 小时均值。格子不在城市之间对齐。
0.5 和 0.8 只用来分档。同窗口相关不超过 0.8，并且协议内相关也有定义，才是以后的训练候选。
协议内不超过 0.8、同窗口仍高于 0.8 的，记成短窗口把相关压低了，不是候选。这一步没有训练。

有向协议内相关 72 个，无序同窗口相关 36 个。
训练候选 34 个。短窗口压低 1 个。

## 序列

- ny:taxi_pickup：形状 8784×20×23，有效格 458
- ny:taxi_dropoff：形状 8784×20×23，有效格 460
- ny:bike_pickup：形状 8784×20×23，有效格 108
- chi:taxi_pickup：形状 8784×17×28，有效格 270
- chi:taxi_dropoff：形状 8784×17×28，有效格 284
- chi:bike_pickup：形状 8784×17×28，有效格 229
- dc:taxi_pickup：形状 8784×21×20，有效格 207
- dc:taxi_dropoff：形状 8784×21×20，有效格 208
- dc:bike_pickup：形状 8784×21×20，有效格 125

## 协议内相关

源侧是 5904 小时，目标侧是 72 小时。

- ny:taxi_pickup → ny:taxi_dropoff：0.9407（gt_0.8），同窗口 0.9955，both_high
- ny:taxi_pickup → ny:bike_pickup：0.7063（gt_0.5_le_0.8），同窗口 0.6550，candidate
- ny:taxi_pickup → chi:taxi_pickup：0.9359（gt_0.8），同窗口 0.9353，both_high
- ny:taxi_pickup → chi:taxi_dropoff：0.9678（gt_0.8），同窗口 0.9667，both_high
- ny:taxi_pickup → chi:bike_pickup：0.5532（gt_0.5_le_0.8），同窗口 0.5429，candidate
- ny:taxi_pickup → dc:taxi_pickup：0.8598（gt_0.8），同窗口 0.8746，both_high
- ny:taxi_pickup → dc:taxi_dropoff：0.8783（gt_0.8），同窗口 0.8947，both_high
- ny:taxi_pickup → dc:bike_pickup：0.5881（gt_0.5_le_0.8），同窗口 0.6031，candidate
- ny:taxi_dropoff → ny:taxi_pickup：0.8870（gt_0.8），同窗口 0.9955，both_high
- ny:taxi_dropoff → ny:bike_pickup：0.6580（gt_0.5_le_0.8），同窗口 0.6024，candidate
- ny:taxi_dropoff → chi:taxi_pickup：0.9178（gt_0.8），同窗口 0.9130，both_high
- ny:taxi_dropoff → chi:taxi_dropoff：0.9615（gt_0.8），同窗口 0.9572，both_high
- ny:taxi_dropoff → chi:bike_pickup：0.4960（le_0.5），同窗口 0.4854，candidate
- ny:taxi_dropoff → dc:taxi_pickup：0.8330（gt_0.8），同窗口 0.8445，both_high
- ny:taxi_dropoff → dc:taxi_dropoff：0.8573（gt_0.8），同窗口 0.8707，both_high
- ny:taxi_dropoff → dc:bike_pickup：0.5319（gt_0.5_le_0.8），同窗口 0.5475，candidate
- ny:bike_pickup → ny:taxi_pickup：0.7944（gt_0.5_le_0.8），同窗口 0.6550，candidate
- ny:bike_pickup → ny:taxi_dropoff：0.7635（gt_0.5_le_0.8），同窗口 0.6024，candidate
- ny:bike_pickup → chi:taxi_pickup：0.7897（gt_0.5_le_0.8），同窗口 0.8342，attenuated
- ny:bike_pickup → chi:taxi_dropoff：0.7308（gt_0.5_le_0.8），同窗口 0.7561，candidate
- ny:bike_pickup → chi:bike_pickup：0.9326（gt_0.8），同窗口 0.9386，both_high
- ny:bike_pickup → dc:taxi_pickup：0.7571（gt_0.5_le_0.8），同窗口 0.7799，candidate
- ny:bike_pickup → dc:taxi_dropoff：0.7357（gt_0.5_le_0.8），同窗口 0.7577，candidate
- ny:bike_pickup → dc:bike_pickup：0.9638（gt_0.8），同窗口 0.9784，both_high
- chi:taxi_pickup → ny:taxi_pickup：0.9322（gt_0.8），同窗口 0.9353，both_high
- chi:taxi_pickup → ny:taxi_dropoff：0.9412（gt_0.8），同窗口 0.9130，both_high
- chi:taxi_pickup → ny:bike_pickup：0.8768（gt_0.8），同窗口 0.8342，both_high
- chi:taxi_pickup → chi:taxi_dropoff：0.9783（gt_0.8），同窗口 0.9852，both_high
- chi:taxi_pickup → chi:bike_pickup：0.7471（gt_0.5_le_0.8），同窗口 0.7379，candidate
- chi:taxi_pickup → dc:taxi_pickup：0.9444（gt_0.8），同窗口 0.9601，both_high
- chi:taxi_pickup → dc:taxi_dropoff：0.9495（gt_0.8），同窗口 0.9660，both_high
- chi:taxi_pickup → dc:bike_pickup：0.7669（gt_0.5_le_0.8），同窗口 0.7741，candidate
- chi:taxi_dropoff → ny:taxi_pickup：0.9226（gt_0.8），同窗口 0.9667，both_high
- chi:taxi_dropoff → ny:taxi_dropoff：0.9437（gt_0.8），同窗口 0.9572，both_high
- chi:taxi_dropoff → ny:bike_pickup：0.8022（gt_0.8），同窗口 0.7561，candidate
- chi:taxi_dropoff → chi:taxi_pickup：0.9722（gt_0.8），同窗口 0.9852，both_high
- chi:taxi_dropoff → chi:bike_pickup：0.6407（gt_0.5_le_0.8），同窗口 0.6358，candidate
- chi:taxi_dropoff → dc:taxi_pickup：0.9081（gt_0.8），同窗口 0.9242，both_high
- chi:taxi_dropoff → dc:taxi_dropoff：0.9247（gt_0.8），同窗口 0.9419，both_high
- chi:taxi_dropoff → dc:bike_pickup：0.6735（gt_0.5_le_0.8），同窗口 0.6908，candidate
- chi:bike_pickup → ny:taxi_pickup：0.6843（gt_0.5_le_0.8），同窗口 0.5429，candidate
- chi:bike_pickup → ny:taxi_dropoff：0.6405（gt_0.5_le_0.8），同窗口 0.4854，candidate
- chi:bike_pickup → ny:bike_pickup：0.9471（gt_0.8），同窗口 0.9386，both_high
- chi:bike_pickup → chi:taxi_pickup：0.7235（gt_0.5_le_0.8），同窗口 0.7379，candidate
- chi:bike_pickup → chi:taxi_dropoff：0.6383（gt_0.5_le_0.8），同窗口 0.6358，candidate
- chi:bike_pickup → dc:taxi_pickup：0.6884（gt_0.5_le_0.8），同窗口 0.7111，candidate
- chi:bike_pickup → dc:taxi_dropoff：0.6550（gt_0.5_le_0.8），同窗口 0.6737，candidate
- chi:bike_pickup → dc:bike_pickup：0.9892（gt_0.8），同窗口 0.9698，both_high
- dc:taxi_pickup → ny:taxi_pickup：0.8519（gt_0.8），同窗口 0.8746，both_high
- dc:taxi_pickup → ny:taxi_dropoff：0.8557（gt_0.8），同窗口 0.8445，both_high
- dc:taxi_pickup → ny:bike_pickup：0.8407（gt_0.8），同窗口 0.7799，candidate
- dc:taxi_pickup → chi:taxi_pickup：0.9643（gt_0.8），同窗口 0.9601，both_high
- dc:taxi_pickup → chi:taxi_dropoff：0.9270（gt_0.8），同窗口 0.9242，both_high
- dc:taxi_pickup → chi:bike_pickup：0.7369（gt_0.5_le_0.8），同窗口 0.7111，candidate
- dc:taxi_pickup → dc:taxi_dropoff：0.9907（gt_0.8），同窗口 0.9959，both_high
- dc:taxi_pickup → dc:bike_pickup：0.7281（gt_0.5_le_0.8），同窗口 0.7079，candidate
- dc:taxi_dropoff → ny:taxi_pickup：0.8523（gt_0.8），同窗口 0.8947，both_high
- dc:taxi_dropoff → ny:taxi_dropoff：0.8625（gt_0.8），同窗口 0.8707，both_high
- dc:taxi_dropoff → ny:bike_pickup：0.8218（gt_0.8），同窗口 0.7577，candidate
- dc:taxi_dropoff → chi:taxi_pickup：0.9686（gt_0.8），同窗口 0.9660，both_high
- dc:taxi_dropoff → chi:taxi_dropoff：0.9431（gt_0.8），同窗口 0.9419，both_high
- dc:taxi_dropoff → chi:bike_pickup：0.7004（gt_0.5_le_0.8），同窗口 0.6737，candidate
- dc:taxi_dropoff → dc:taxi_pickup：0.9913（gt_0.8），同窗口 0.9959，both_high
- dc:taxi_dropoff → dc:bike_pickup：0.6946（gt_0.5_le_0.8），同窗口 0.6787，candidate
- dc:bike_pickup → ny:taxi_pickup：0.7555（gt_0.5_le_0.8），同窗口 0.6031，candidate
- dc:bike_pickup → ny:taxi_dropoff：0.7155（gt_0.5_le_0.8），同窗口 0.5475，candidate
- dc:bike_pickup → ny:bike_pickup：0.9668（gt_0.8），同窗口 0.9784，both_high
- dc:bike_pickup → chi:taxi_pickup：0.7360（gt_0.5_le_0.8），同窗口 0.7741，candidate
- dc:bike_pickup → chi:taxi_dropoff：0.6734（gt_0.5_le_0.8），同窗口 0.6908，candidate
- dc:bike_pickup → chi:bike_pickup：0.9553（gt_0.8），同窗口 0.9698，both_high
- dc:bike_pickup → dc:taxi_pickup：0.6780（gt_0.5_le_0.8），同窗口 0.7079，candidate
- dc:bike_pickup → dc:taxi_dropoff：0.6503（gt_0.5_le_0.8），同窗口 0.6787，candidate

## 同窗口相关

两边都是前 5904 小时，一对只有一个数。

- ny:taxi_pickup 与 ny:taxi_dropoff：0.9955（gt_0.8）
- ny:taxi_pickup 与 ny:bike_pickup：0.6550（gt_0.5_le_0.8）
- ny:taxi_pickup 与 chi:taxi_pickup：0.9353（gt_0.8）
- ny:taxi_pickup 与 chi:taxi_dropoff：0.9667（gt_0.8）
- ny:taxi_pickup 与 chi:bike_pickup：0.5429（gt_0.5_le_0.8）
- ny:taxi_pickup 与 dc:taxi_pickup：0.8746（gt_0.8）
- ny:taxi_pickup 与 dc:taxi_dropoff：0.8947（gt_0.8）
- ny:taxi_pickup 与 dc:bike_pickup：0.6031（gt_0.5_le_0.8）
- ny:taxi_dropoff 与 ny:bike_pickup：0.6024（gt_0.5_le_0.8）
- ny:taxi_dropoff 与 chi:taxi_pickup：0.9130（gt_0.8）
- ny:taxi_dropoff 与 chi:taxi_dropoff：0.9572（gt_0.8）
- ny:taxi_dropoff 与 chi:bike_pickup：0.4854（le_0.5）
- ny:taxi_dropoff 与 dc:taxi_pickup：0.8445（gt_0.8）
- ny:taxi_dropoff 与 dc:taxi_dropoff：0.8707（gt_0.8）
- ny:taxi_dropoff 与 dc:bike_pickup：0.5475（gt_0.5_le_0.8）
- ny:bike_pickup 与 chi:taxi_pickup：0.8342（gt_0.8）
- ny:bike_pickup 与 chi:taxi_dropoff：0.7561（gt_0.5_le_0.8）
- ny:bike_pickup 与 chi:bike_pickup：0.9386（gt_0.8）
- ny:bike_pickup 与 dc:taxi_pickup：0.7799（gt_0.5_le_0.8）
- ny:bike_pickup 与 dc:taxi_dropoff：0.7577（gt_0.5_le_0.8）
- ny:bike_pickup 与 dc:bike_pickup：0.9784（gt_0.8）
- chi:taxi_pickup 与 chi:taxi_dropoff：0.9852（gt_0.8）
- chi:taxi_pickup 与 chi:bike_pickup：0.7379（gt_0.5_le_0.8）
- chi:taxi_pickup 与 dc:taxi_pickup：0.9601（gt_0.8）
- chi:taxi_pickup 与 dc:taxi_dropoff：0.9660（gt_0.8）
- chi:taxi_pickup 与 dc:bike_pickup：0.7741（gt_0.5_le_0.8）
- chi:taxi_dropoff 与 chi:bike_pickup：0.6358（gt_0.5_le_0.8）
- chi:taxi_dropoff 与 dc:taxi_pickup：0.9242（gt_0.8）
- chi:taxi_dropoff 与 dc:taxi_dropoff：0.9419（gt_0.8）
- chi:taxi_dropoff 与 dc:bike_pickup：0.6908（gt_0.5_le_0.8）
- chi:bike_pickup 与 dc:taxi_pickup：0.7111（gt_0.5_le_0.8）
- chi:bike_pickup 与 dc:taxi_dropoff：0.6737（gt_0.5_le_0.8）
- chi:bike_pickup 与 dc:bike_pickup：0.9698（gt_0.8）
- dc:taxi_pickup 与 dc:taxi_dropoff：0.9959（gt_0.8）
- dc:taxi_pickup 与 dc:bike_pickup：0.7079（gt_0.5_le_0.8）
- dc:taxi_dropoff 与 dc:bike_pickup：0.6787（gt_0.5_le_0.8）

## 读法

训练候选是同窗口相关不超过 0.8 的有向对。这一步仍然不训练。
- 候选 ny:taxi_pickup → ny:bike_pickup：协议内 0.7063，同窗口 0.6550
- 候选 ny:taxi_pickup → chi:bike_pickup：协议内 0.5532，同窗口 0.5429
- 候选 ny:taxi_pickup → dc:bike_pickup：协议内 0.5881，同窗口 0.6031
- 候选 ny:taxi_dropoff → ny:bike_pickup：协议内 0.6580，同窗口 0.6024
- 候选 ny:taxi_dropoff → chi:bike_pickup：协议内 0.4960，同窗口 0.4854
- 候选 ny:taxi_dropoff → dc:bike_pickup：协议内 0.5319，同窗口 0.5475
- 候选 ny:bike_pickup → ny:taxi_pickup：协议内 0.7944，同窗口 0.6550
- 候选 ny:bike_pickup → ny:taxi_dropoff：协议内 0.7635，同窗口 0.6024
- 候选 ny:bike_pickup → chi:taxi_dropoff：协议内 0.7308，同窗口 0.7561
- 候选 ny:bike_pickup → dc:taxi_pickup：协议内 0.7571，同窗口 0.7799
- 候选 ny:bike_pickup → dc:taxi_dropoff：协议内 0.7357，同窗口 0.7577
- 候选 chi:taxi_pickup → chi:bike_pickup：协议内 0.7471，同窗口 0.7379
- 候选 chi:taxi_pickup → dc:bike_pickup：协议内 0.7669，同窗口 0.7741
- 候选 chi:taxi_dropoff → ny:bike_pickup：协议内 0.8022，同窗口 0.7561
- 候选 chi:taxi_dropoff → chi:bike_pickup：协议内 0.6407，同窗口 0.6358
- 候选 chi:taxi_dropoff → dc:bike_pickup：协议内 0.6735，同窗口 0.6908
- 候选 chi:bike_pickup → ny:taxi_pickup：协议内 0.6843，同窗口 0.5429
- 候选 chi:bike_pickup → ny:taxi_dropoff：协议内 0.6405，同窗口 0.4854
- 候选 chi:bike_pickup → chi:taxi_pickup：协议内 0.7235，同窗口 0.7379
- 候选 chi:bike_pickup → chi:taxi_dropoff：协议内 0.6383，同窗口 0.6358
- 候选 chi:bike_pickup → dc:taxi_pickup：协议内 0.6884，同窗口 0.7111
- 候选 chi:bike_pickup → dc:taxi_dropoff：协议内 0.6550，同窗口 0.6737
- 候选 dc:taxi_pickup → ny:bike_pickup：协议内 0.8407，同窗口 0.7799
- 候选 dc:taxi_pickup → chi:bike_pickup：协议内 0.7369，同窗口 0.7111
- 候选 dc:taxi_pickup → dc:bike_pickup：协议内 0.7281，同窗口 0.7079
- 候选 dc:taxi_dropoff → ny:bike_pickup：协议内 0.8218，同窗口 0.7577
- 候选 dc:taxi_dropoff → chi:bike_pickup：协议内 0.7004，同窗口 0.6737
- 候选 dc:taxi_dropoff → dc:bike_pickup：协议内 0.6946，同窗口 0.6787
- 候选 dc:bike_pickup → ny:taxi_pickup：协议内 0.7555，同窗口 0.6031
- 候选 dc:bike_pickup → ny:taxi_dropoff：协议内 0.7155，同窗口 0.5475
- 候选 dc:bike_pickup → chi:taxi_pickup：协议内 0.7360，同窗口 0.7741
- 候选 dc:bike_pickup → chi:taxi_dropoff：协议内 0.6734，同窗口 0.6908
- 候选 dc:bike_pickup → dc:taxi_pickup：协议内 0.6780，同窗口 0.7079
- 候选 dc:bike_pickup → dc:taxi_dropoff：协议内 0.6503，同窗口 0.6787

下面这些协议内相关落到 0.8 及以下，同窗口仍高于 0.8。

- ny:bike_pickup → chi:taxi_pickup：协议内 0.7897，同窗口 0.8342

重算：

```bash
python scripts/10_modality_similarity.py
```
