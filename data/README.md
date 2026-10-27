# data/

大文件不入库。期望布局：

```text
raw/                 # 原始 zip（gitignore）
processed/ny|chi|dc/ # 预处理产物或指向上游解压目录的软链
```

**下载直链已核实**（CrossTReS / STBaT）：见 [docs/DATA.md](../docs/DATA.md)。  
上游文件名示例：`TaxiNY_pickup.npy`、`BikeDC_dropoff.npy`。

```bash
python scripts/01_verify_data_layout.py
```
