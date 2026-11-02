# data/

三份表格，用途完全不同：

| 文件 | 行数 | 说明 |
|---|---|---|
| `dataset.xlsx` | 6 | 文献可核实的起始集，逐行标注 Source/DOI（2PACz、MeO-2PACz、Me-4PACz、I-2PACz、4PADCB、PhPAPy）。**不足以训练模型**（n<8 时 step2 会拒绝），用于验证 schema 与链路 |
| `dataset_template.xlsx` | 0 | 空白模板，`README` 工作表标了哪些字段必填 |
| `dataset_demo.xlsx` | 27 | **合成数据**（`tests/make_demo_dataset.py` 生成），只用于跑通管线，禁止引用为实验值 |
| `candidates.csv` | 30 | 设计候选库（咔唑膦酸系列 + TPA-PA / BPh-PA / Naph-PA），供 step3 筛选 |

## 字段

| 字段 | 必填 | 含义 |
|---|---|---|
| `Molecule` | 是 | 唯一化学名 |
| `Abbrev` | 否 | 图中简称 |
| `SMILES` | 是 | RDKit 可解析的 SMILES |
| `Anchor` | 否 | 锚定基（PO3H2 / COOH / SO3H） |
| `Device_Structure` | 否 | 器件堆叠 |
| `Processing` | 否 | 成膜方式 |
| `Control_PCE` | 否 | 对照器件 PCE；填够 ≥3 行后自动启用提升比目标 |
| `Modified_PCE` | 是 | SAM 器件 PCE |
| `Improved_Ratio` | 否 | 留空自动算 `(Modified-Control)/Control` |
| `Voc` / `Jsc` / `FF` | 否 | 其他光伏参数 |
| `Source` / `DOI` | 否 | 每行出处，建议必填以便复核 |

## 换成自己的实验数据

1. 复制 `dataset_template.xlsx` 为 `dataset.xlsx`，按上表填行；
2. 建议与 Su 的字段对齐（分子名 / SMILES / 器件结构 / 对照 PCE / 改性 PCE / 提升比），方便后人复用；
3. 至少 30 行走完整的 15% hold-out + 5 折 CV；8–29 行走 LOOCV；
4. 数据获取优先级：官方仓库 > 论文附表 OCR > 邮件索取；拿到数据**先查 LICENSE**（MIT 可再分发，需保留版权声明）；
5. 投稿时在 SI 的 Data Availability 写本仓库地址。
