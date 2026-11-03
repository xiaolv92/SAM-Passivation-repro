# SAM-Passivation-repro

面向钙钛矿太阳能电池 **自组装单分子层（SAM）空穴传输/钝化材料** 的"数据 → 特征 → 回归 → 虚拟筛选 → 量化排序"可复现管线。

骨架与 [Perovskite-Vapor-Passivation-repro](https://github.com/xiaolv92/Perovskite-Vapor-Passivation-repro) 同构，把"气相钝化分子"换成"SAM 锚定分子"，并针对 SAM 的化学约束重写了筛选层与量化层（Gaussian 簇模型 + NiO / PbI2 表面）。

> 本仓库同时是一份可套用的**方法模板**：换成你自己的 `data/dataset.xlsx`，跑同五步，就得到投稿配套的"官方仓库"。

---

## 数据声明（Data statement）

**本仓库的示例数据与测试数据均为合成数据，不得作为实验值引用。**

| 文件 | 性质 | 可否引用 |
|---|---|---|
| `data/dataset_demo.xlsx`（27 行） | **合成**：脚本随机生成，仅用于跑通链路 | ❌ 禁止引用为实验值 |
| `qc_test/` 下的 `.log` / `OUTCAR`（`tests/make_fixtures.py` 生成） | **合成**：虚构的 SCF 能量与几何，仅用于解析器冒烟测试 | ❌ 禁止引用 |
| `data/candidates.csv`（30 个设计候选） | **合成**：按规则枚举的设计分子，无实测器件数据 | ❌ 仅作筛选演示例 |
| `data/dataset.xlsx`（6 行） | 文献已发表数值，逐行标注 DOI，可核实 | ✅ 可引用，版权归原作者 |

本仓库**不包含**任何未发表的 P 系列候选排序、DFT/VASP 结果、论文图表或器件数据；这些内容保留在本地，论文接收后再按期刊要求发布。仓库中的模型指标（第 5 节）来自上述合成 demo 数据，只证明管线可跑通，不代表真实材料性能。

---

## 1. 管线总览

| 步骤 | 脚本 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| ① 数据 | `src/step1_dataset.py` | `data/dataset.xlsx` | `outputs/fingerprints_train.csv`、`features.json`、`step1_log.json` | SMILES 校验 → MACCS + 10 物理描述符 + 3 锚定基标记 → 稀疏/相关性剔除 |
| ② 建模 | `src/step2_features_train.py` | 上一步产物 | `model_metrics*.csv`、`shap_ranking.csv`、`model_rf.joblib`、`fig2_*.png` | 六个回归模型 + CV + SHAP 重要性 |
| ③ 筛选 | `src/step3_screen.py` | 上一步产物 + `data/candidates.csv` | `screening_results.csv`、`qc_candidates.csv`、`feature_resolution.txt` | 锚定基/尺寸/柔性/极性/适用域闸门 + A/B/C 分档 |
| ④ 量化输入 | `src/step4_qc_inputs.py` | `outputs/qc_candidates.csv` | `qc/gaussian/*.gjf`、`qc/vasp/<case>/*` | Gaussian 单体/锚定簇 + VASP NiO(100)、PbI2 终止钙钛矿板 |
| ⑤ 量化后处理 | `src/step5_analyze_qc.py` | `qc/**/*.log`、`OUTCAR` | `outputs/qc_descriptors.csv` | 解析 dE_anchor / Eads / gap / dipole → 加权评分 |

所有路径都由 [`src/common.py`](src/common.py) 里的 `ROOT = src 的上一级` 推导，仓库 clone 到任何机器都能直接跑，不含绝对路径。

## 2. 为什么 SAM 要单独一套闸门

SAM 必须同时干两件事：

1. **化学吸附在电极上**（ITO / NiOx）——分子必须带膦酸 `PO3H2`、羧酸 `COOH` 或磺酸 `SO3H` 锚定基，否则成不了单分子层；
2. **钝化钙钛矿表面的 Pb 位点**——锚定基与分子骨架的偶极还要能改善电极功函数。

因此 `step3` 的漏斗是：

```
锚定基必需 → MW 150–800 → 可旋转键 ≤12 → TPSA ≤120
          → MACCS Tanimoto ≥0.40（适用域）→ ML 预测分 → A/B/C 三档
```

这里还检查**指纹塌缩**：不同分子得到完全相同的 MACCS 向量时，ML 无法区分，必须交给 `step4/5` 的量化层排序。`step3` 会把塌缩组写进 `outputs/feature_resolution.txt`。

## 3. 快速开始

```powershell
pip install -r requirements.txt

# 起步：文献可核实的 6 行数据（本仓库自带）
python src\step1_dataset.py
python src\step2_features_train.py        # n=6 < 8，会提示"数据不够"并停止
python src\step3_screen.py                # 需先成功训练出 model_rf.joblib

# 想先看完整链路：用合成 demo 数据（27 行，禁止引用）
python tests\make_demo_dataset.py
python src\step1_dataset.py --dataset dataset_demo
python src\step2_features_train.py
python src\step3_screen.py
python src\step4_qc_inputs.py --top 6      # 生成 13 个 Gaussian + 14 个 VASP 算例
python tests\make_fixtures.py              # 合成 QC 输出，冒烟测试解析器
python src\step5_analyze_qc.py --qc qc_test
```

`data/dataset.xlsx` 换成你的实验数据（字段见 [data/README.md](data/README.md)）即可复用全部脚本。

## 4. 数据 schema

`data/dataset.xlsx` 的 `data` 工作表（必填 `Molecule`、`SMILES`、`Modified_PCE`）：

| 字段 | 含义 |
|---|---|
| `Molecule` / `Abbrev` | 化学名 / 图中简称 |
| `SMILES` | RDKit 可解析的 SMILES（必填） |
| `Anchor` | 锚定基类型（PO3H2 / COOH / SO3H） |
| `Device_Structure` / `Processing` | 器件堆叠 / 成膜方式 |
| `Control_PCE` / `Modified_PCE` | 对照器件 PCE / SAM 器件 PCE |
| `Improved_Ratio` | 留空即由 `(Modified-Control)/Control` 自动计算 |
| `Voc` / `Jsc` / `FF` | 其他光伏参数（可选） |
| `Source` / `DOI` | 每行出处（**必须可核实**） |

目标变量选择（[`common.py`](src/common.py) 的 `target_name`）：`Improved_Ratio` 至少 3 行有效时回归提升比，否则退回回归 `Modified_PCE`。

## 5. 实测基准（本机跑通，2026-11-02；以下数值全部来自合成 demo 数据）

环境：Python 3.13.2；rdkit 2026.09.1 / ase 3.29.0 / xgboost 3.4.1 / shap 0.54.1 / scikit-learn 1.9.1 / pandas 3.1.0 / numpy 2.2.2。

**step1**

| 数据集 | 行数 | 可用 | MACCS 保留 | 剔除 | 总特征 |
|---|---|---|---|---|---|
| `dataset.xlsx`（文献 6 行） | 6 | 6 | 26 | 稀疏<2:119，相关:22 | 39 |
| `dataset_demo.xlsx`（合成 27 行） | 27 | 27 | 26 | 稀疏<9:117，相关:24 | 39 |

39 = 26 MACCS + 10 物理描述符 + 3 锚定基标记。

**step2（demo，n=27，LOOCV，target=Modified_PCE）**

| 模型 | RMSE train/LOOCV | MAE train/LOOCV |
|---|---|---|
| RF | 0.222 / 0.460 | 0.152 / 0.355 |
| GB | 0.031 / 0.507 | 0.025 / 0.406 |
| SVR | 0.434 / 0.488 | 0.312 / 0.380 |
| MLP | 0.268 / 1.224 | 0.205 / 0.653 |
| GP | 0.046 / 0.515 | 0.034 / 0.371 |
| XGB | 0.023 / 0.504 | 0.018 / 0.378 |

SHAP 前五（5 个种子平均）：`NumRotatableBonds` 0.0835、`FractionCSP3` 0.0681、`TPSA` 0.0504、`MolWt` 0.0352、`MolLogP` 0.0299。

**step3（30 个设计候选）**

| 指标 | 结果 |
|---|---|
| 通过闸门 | 30 / 30 |
| 分档 | A 10 / B 10 / C 10 |
| 预测 PCE 区间 | 24.78 – 25.03 |
| Tanimoto 适用域 | 0.86 – 0.98 |
| 指纹塌缩组 | 0 |

**step4（--top 6）**

| 产物 | 数量/校验 |
|---|---|
| Gaussian 输入 | 13（1 个 Pb2I4 参考 + 6 单体 + 6 锚定簇） |
| VASP 算例 | 14（NiO/PbI2 各 1 参考 + 各 6 吸附） |
| POSCAR | 标准"唯一元素行 + 计数行"；NiO 108 Ni + 108 O（3×3×3，Ni:O=1:1）；PbI2 为中性 PbI2 计量 |
| 分子几何 | 锚定 P=O 氧朝下，距顶部 Ni 2.10 Å / 顶部 Pb 2.40 Å，xy 取模留在周期盒内 |

**step5（合成 fixture 冒烟测试，数值为虚构，仅证明解析链路）**

| molecule | gap_eV | dipole_D | dE_anchor | Eads_oxide | score |
|---|---|---|---|---|---|
| I-C6-PACz | 4.082 | 9.0 | −0.348 | −0.607 | 0.70 |
| OMe-C4-PACz | 4.082 | 8.0 | −0.348 | −0.557 | 0.35 |
| H-C2-PACz | 4.082 | 7.0 | −0.348 | −0.507 | 0.00 |

## 6. 与上游文献/模板的差异（显式声明）

1. **目标变量**：上游回归 99 个 SAM 的提升比；本仓库起始集只有改性 PCE，默认回归 `Modified_PCE`，填够 `Control_PCE` 后自动切回提升比。
2. **小样本协议**：n≥30 用 15% hold-out（`random_state=31`）+ 5 折 CV；8–29 用 LOOCV；n<8 拒绝拟合、清理旧模型并提示补数据；n<25 自动剔除 GP/MLP。上游固定 99→78 训练集。
3. **特征块**：MACCS（稀疏阈值 `min(30, max(2, n//3))`，相关 |r|>0.90 仅作用于 MACCS）+ 10 物理描述符 + 3 锚定基标记；上游用 11 个片段 + FA/MA/Cs/Rb。物理描述符与锚定基标记是本仓库为 SAM 新增。
4. **新增虚拟筛选层**：锚定基硬闸门 + MW/可旋转键/TPSA + Tanimoto 适用域 + A/B/C 分档；上游无此层。
5. **量化层重写**：Gaussian `Pb2I4` 簇（膦酸锚定）+ VASP `NiO(100) 3×3×3` 与 `PbI2` 终止的 `FAPbI3(001) 2×2×2` 模型；A 位阳离子为控制成本省略，评分权重 0.5（氧化物吸附）/0.3（Pb 侧锚定）/0.2（偶极）。
6. **重要性**：n≥20 用 5 个种子的 SHAP 平均并报告种子标准差；n<20 退回 permutation importance。

## 7. 量化协议

**Gaussian 16**

```
#p B3LYP/GENECP em=GD3BJ opt freq int=ultrafine pop=MK iop(6/33=2) scf=xqc
```
- 基组：C/H/N/O/P/S/F/Cl/Br → 6-31G(d,p)；Pb/I → Def2TZVP（GENECP 卡片顺序：基组块 → 空行 → 空 ECP 段）
- 单体：`opt freq`；锚定簇：`opt=(calcfc,maxstep=8,maxcycles=300) freq`
- 参考簇 `Pb2I4`：2 Pb + 4 I，中性单重态

**VASP**

| 参数 | 值 |
|---|---|
| 表面 | NiO(100) 3×3×3 岩盐板（a=4.177 Å）；PbI2 终止 FAPbI3(001) 2×2×2（a=6.362 Å） |
| 真空 | ≥15 Å |
| ENCUT / EDIFF / EDIFFG | 520 eV / 1e-5 / −0.03 |
| ISIF / IBRION / NSW | 2 / 2 / 300 |
| K 点 | Γ 中心 2×2×1 |
| 色散 | IVDW = 12 |

**评分**

```
score = 0.5·norm(−Eads_oxide) + 0.3·norm(−dE_anchor) + 0.2·norm(|dipole_D|)
```

## 8. 已知局限

- 文献起始集仅 6 行，**不足以训练**；step2 会明确拒绝（n<8）。请补数据或先用 demo 验证链路。
- `dataset_demo.xlsx`（27 行）、`data/candidates.csv`（30 个候选）与 `qc_test/` 下的 QC 输出均为**合成数据**，禁止引用为实验值或材料性能证据；它们只用于验证脚本能跑通。
- NiO 板未做自旋极化/U 校正；PbI2 板省略 A 位阳离子；这些都会影响绝对值，只适合同体系内相对排序。
- 量化输入是初猜几何，需在集群上弛豫后再用 step5 解析。

## 9. 引用与许可

相关文献（数据来源逐行标注在 `dataset.xlsx`）：

- ACS Energy Lett. **2025**, 10, 4882−4910. DOI: [10.1021/acsenergylett.5c02024](https://doi.org/10.1021/acsenergylett.5c02024)
- Angew. Chem. Int. Ed. **2025**, e202505876. DOI: [10.1002/anie.202505876](https://doi.org/10.1002/anie.202505876)
- 上游模板：Su et al., *Adv. Energy Mater.* **2026**, aenm.71732；代码骨架 [Perovskite-Vapor-Passivation-repro](https://github.com/xiaolv92/Perovskite-Vapor-Passivation-repro)

代码 MIT（见 [LICENSE](LICENSE)）；`data/dataset.xlsx` 中的文献数值版权归原作者/出版商，本仓库仅作可核实的起始集引用。
