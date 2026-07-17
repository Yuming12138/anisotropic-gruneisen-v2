# 各向异性 Grüneisen 热膨胀计算 v2 设计文档

状态：v2 独立实现已开始；公共核心、runner、batch、全量 preflight 和 smoke 验证已完成，生产参数代表材料尚未启动
日期：2026-07-17
目标脚本：

- `run_gruneisen_thermal_expansion.py`
- `batch_gruneisen_thermal_expansion.py`

建议实施方式：保留当前脚本作为 legacy 版本，新增独立的 v2 runner 和 batch 脚本，所有新结果写入版本化子目录，避免覆盖已有数据。

当前实现进度和验证证据见：

```text
ANISOTROPIC_GRUNEISEN_V2_IMPLEMENTATION_STATUS.md
```

## 1. 背景与目标

审稿人指出，等方性与各向异性材料的负热膨胀可能由不同物理机制控制：

- 等方性体积热膨胀主要由总体 Grüneisen 参数的符号决定；
- 各向异性材料中，六个应变 Grüneisen 分量与弹性柔度张量的非对角项共同决定热膨胀；
- 因此，各向异性材料可能在部分应变 Grüneisen 参数为正时仍产生某些方向的 NTE，反之亦然。

本项目需要建立一个能够真正实现下式的计算流程：

\[
\alpha_i(T)=\frac{1}{V_0}\sum_{j=1}^{6}S_{ij}I_j(T),
\qquad i=1,\ldots,6,
\]

其中

\[
I_j(T)=\frac{1}{N_q}\sum_{\mathbf q\nu}
w_{\mathbf q}C_{\mathbf q\nu}(T)\gamma^j_{\mathbf q\nu},
\]

\[
\gamma^j_{\mathbf q\nu}
=-\left.\frac{\partial\ln\omega_{\mathbf q\nu}}
{\partial\eta_j}\right|_{\eta=0}.
\]

最终目标包括：

1. 在统一坐标系中计算完整六分量 \(\gamma_j\)、\(S_{ij}\) 和 \(\alpha_i\)；
2. 得到坐标无关的体积热膨胀 \(\alpha_V\)；
3. 将热膨胀张量投影到实际晶格 \(a,b,c\) 方向；
4. 输出机械稳定性、虚频、应变导数收敛性和数据来源等质量信息；
5. 为后续等方性/各向异性数据集拆分和审稿回复提供可审计结果。

## 2. 当前脚本不能直接用于最终论文结论的原因

当前 `run_gruneisen_thermal_expansion.py` 存在以下关键限制：

1. 对所有晶体只拟合 \(C_{11},C_{12},C_{44}\)，并人为构造立方形式的弹性张量；
2. 没有读取每个材料真实的 `elastic/ELASTIC_TENSOR`；
3. 声子使用根目录 `POSCAR`，弹性部分又独立进行一次完整晶胞松弛，两部分可能对应不同结构和体积；
4. 只计算三个所谓的 `gamma_a/gamma_b/gamma_c`，没有计算六个工程应变分量；
5. `cell[axis] *= 1 + strain` 缩放的是晶格基矢，不是一般非正交晶胞中的笛卡尔正应变；
6. `primitive_matrix="auto"` 对每个应变结构独立选择原胞，可能改变原子数、原子顺序和模的定义；
7. 当前直接按频率数组下标相减，没有处理简并模和模交换；
8. 异常或过大的 \(\gamma\) 被直接设为零，会隐藏软模、应变不稳定或数值错误；
9. 应变结构没有在固定晶格下松弛内部原子坐标；
10. 当前输出可能覆盖材料目录中的 `ELASTIC_TENSOR`；
11. 缓存只检查文件是否存在，没有验证结构、模型、参数和脚本是否一致；
12. 默认模型为 MatterSim-5M，有限位移为 0.001 Å，与论文 Methods 中的 1M 和 0.01 Å 不一致。

因此，现有结果可以作为历史测试数据，但不能作为严格六分量各向异性热膨胀的最终证据。

## 3. 输入数据契约

每个材料至少需要以下文件：

```text
material/
├── POSCAR                         # 数据集/热膨胀方向定义所用结构
├── thermal_expansion.dat          # 原有等方性 QHA 结果，可选但建议保留
└── elastic/
    ├── ELASTIC_TENSOR             # 完整 6×6 VASPKIT 刚度张量，单位 GPa
    ├── POSCAR                     # 生成该张量时使用的已松弛平衡结构，必需
    ├── CONTCAR                    # 建议保留，通常与 POSCAR 相同或为备份
    ├── BM_SS.log                  # 建议保留，用于来源和拟合审计
    └── calculation_metadata.json  # 建议新增，记录模型和计算参数
```

其中 `elastic/POSCAR` 必须是生成 `ELASTIC_TENSOR` 时的参考平衡结构。只有张量而没有对应结构时，无法严格确定张量所属的笛卡尔坐标系，该材料应标记为：

```text
missing_elastic_reference_structure
```

不得自动使用根目录 `POSCAR` 代替，也不得仅根据相同化学式猜测坐标关系。

### 3.1 全量弹性结果导入时必须检查

当前 `D:\9.Project\10.recalcu_elastic_nte\auto_elastic_sh\elastic_calculator.py` 的默认模型仍是 MatterSim-5M。论文 Methods 要求弹性和声子使用 MatterSim-1M。因此，导入全量重算结果时必须同时核对：

- 实际模型路径或模型 SHA256；
- 是否显式使用 1M checkpoint；
- MatterSim、ASE、PyTorch、VASPKIT 版本；
- 完整松弛阈值和固定晶格离子松弛阈值；
- VASPKIT 应变点数和应变范围；
- 弹性参考 POSCAR 的 SHA256。

若缺少日志，应标记模型来源为 `unverified`，不能仅凭目录名推断使用了 1M。

## 4. 统一坐标策略

### 4.1 基本决定

以 `elastic/POSCAR` 作为所有六分量 Grüneisen 声子计算的参考结构。

这样：

- `ELASTIC_TENSOR` 在弹性参考结构的笛卡尔坐标系中；
- 六个应变都在同一笛卡尔坐标系中施加；
- \(\gamma_j\)、\(S_{ij}\) 和 \(\alpha_i\) 不需要相互猜测轴标签或先行旋转；
- 根目录 `POSCAR` 只负责定义需要报告的实际晶格 \(a,b,c\) 方向。

### 4.2 根目录晶格方向映射

使用 `collect_mechanical_properties_correct.py` 中已验证的思路：

1. 读取根目录结构和弹性参考结构；
2. 使用 `target_lattice.find_mapping(source_lattice)` 寻找晶格映射；
3. 将根目录结构的 \(a,b,c\) 方向旋转到弹性参考结构的笛卡尔坐标系；
4. 用这些单位方向向量投影最终热膨胀张量。

如果映射失败：

- 笛卡尔六分量和 \(\alpha_V\) 仍可输出；
- `alpha_a/alpha_b/alpha_c` 输出为 NaN；
- 质量报告写入 `cte_axis_to_elastic_lattice_mapping_failed`；
- 不得根据轴名直接把 `x/y/z` 当成 `a/b/c`。

## 5. Voigt 与工程剪切约定

采用 VASPKIT 常用顺序：

```text
1 = xx
2 = yy
3 = zz
4 = yz
5 = xz
6 = xy
```

工程应变向量定义为：

\[
\boldsymbol\eta=
(\varepsilon_{xx},\varepsilon_{yy},\varepsilon_{zz},
2\varepsilon_{yz},2\varepsilon_{xz},2\varepsilon_{xy}).
\]

给定工程应变 \(\eta_j=h\)，对称小应变张量为：

\[
E(\boldsymbol\eta)=
\begin{pmatrix}
\eta_1 & \eta_6/2 & \eta_5/2\\
\eta_6/2 & \eta_2 & \eta_4/2\\
\eta_5/2 & \eta_4/2 & \eta_3
\end{pmatrix}.
\]

变形梯度采用：

\[
F=I+E.
\]

ASE 的 `cell.array` 以行为晶格矢量，因此正确实现为：

```python
new_cell = old_cell @ F.T
strained.set_cell(new_cell, scale_atoms=True)
```

中心差分使用 \(+h\) 和 \(-h\)。对剪切分量，传给 Grüneisen 导数的总工程应变差仍然是 `2*h`，不再额外乘或除剪切因子。

## 6. 参考结构与应变结构松弛

### 6.1 参考结构

默认直接使用生成弹性张量的 `elastic/POSCAR`，避免再次独立进行完整晶胞松弛后使结构与 \(C_{ij}\) 不一致。

开始声子计算前检查：

- 最大残余力；
- 应力张量；
- 体积和最短原子间距；
- 是否存在 NaN/inf；
- 原子元素和顺序。

如果残余力或应力明显超标，应标记参考结构不一致，优先重新计算弹性和声子共同参考结构，而不是静默再松弛一次。

### 6.2 六个正负应变结构

对每个 \(j=1,\ldots,6\)：

1. 从同一个参考结构复制；
2. 施加 \(+h\) 或 \(-h\) 的笛卡尔工程应变；
3. 固定应变后的晶格；
4. 只松弛内部原子坐标；
5. 保留元素和原子顺序；
6. 计算力常数。

内部松弛默认建议：

```text
fmax = 1e-3 eV/Å
max_steps = 1000
```

每个目录应保存松弛前 `POSCAR`、松弛后 `CONTCAR`、最终最大力和能量。

## 7. 声子计算参数

为与论文 Methods 一致，默认参数为：

```text
MatterSim model      = 1M
dtype                = float64（先在代表材料上验证成本与稳定性）
finite displacement  = 0.01 Å
base supercell       = 2 × 2 × 2
minimum cell length  = 12 Å
q-point mesh         = 30 × 30 × 30
temperature range    = 10–1000 K, step 10 K
reference strain h   = 0.005
fallback strain h    = 0.0025
primitive_matrix     = identity
```

### 7.1 超胞选择

至少使用 \(2\times2\times2\)。如果某一超胞晶格矢量长度仍小于 12 Å，则增加对应方向重复数：

```python
n_i = max(2, ceil(12.0 / lattice_length_i))
```

最终超胞矩阵必须对零应变和全部十二个正负应变结构保持完全一致。

### 7.2 固定原胞与原子顺序

使用：

```python
primitive_matrix = np.eye(3)
```

不得对零应变和不同应变结构分别使用 `primitive_matrix="auto"`。所有结构必须保持：

- 相同原子数；
- 相同元素顺序；
- 相同超胞矩阵；
- 相同 primitive matrix；
- 相同质量顺序。

## 8. 简并模与 Grüneisen 参数算法

### 8.1 禁止的实现

不得继续采用以下方法：

- 按频率数组下标直接相减；
- 只按频率最近邻配对；
- 只按单模特征向量重叠做 Hungarian 匹配；
- 把异常 \(\gamma\) 直接替换成零。

这些方法在模交叉和简并子空间中不稳定。

### 8.2 采用 Phonopy 原生简并微扰方法

使用 Phonopy 4.3.1 中：

```python
phonopy.gruneisen.core.GruneisenBase
phonopy.gruneisen.mesh.GruneisenMesh
phonopy.phonon.degeneracy.lift_degeneracy
```

对每个 q 点，零应变动力学矩阵满足：

\[
D_0 e_\nu=\omega_\nu^2 e_\nu.
\]

每个应变分量构造：

\[
\Delta D_j=D_j^{+}-D_j^{-}.
\]

对于简并子空间，将 \(\Delta D_j\) 投影到该子空间并对角化，而不是人为选择某一个简并本征矢：

\[
\Delta D_j^{(\mathrm{deg})}
=E_{\mathrm{deg}}^\dagger\Delta D_jE_{\mathrm{deg}}.
\]

然后计算：

\[
\gamma^j_{\mathbf q\nu}
=-\frac{\langle e_\nu|\Delta D_j|e_\nu\rangle}
{2\omega_{\mathbf q\nu}^2(2h)}.
\]

Phonopy 原生算法已在现有 `261.Si3NiP4` 力常数上完成只读验证，能够避免简单模配对产生的非物理跳变。

### 8.3 简并分量的解释限制

对于完全简并的模，不同 \(j\) 分量可能在简并子空间中选择不同的“良好基”。因此：

- \(I_j(T)\) 和总热膨胀是良定义的；
- 简并子空间的迹或加和是基不变量；
- 不应把同一数组下标下的六个 \(\gamma_j\) 强行解释为一个唯一的六维单模向量；
- 输出逐模数据时应同时保存 `degenerate_group_id` 或标记其解释限制。

## 9. 软模、虚频和应变收敛

原生简并微扰算法解决的是模匹配问题，但不能消除真实或数值性的应变不稳定。

现有 `261.Si3NiP4` 测试中，在 \(30^3\) 网格的 \(q=(1/30,0,0)\) 处发现：

```text
reference frequency ≈ 0.0357 THz
+0.5% strained frequency ≈ -0.1197 THz
perturbative gamma ≈ 1307
```

因此不得设置统一的 `GAMMA_ABS_MAX` 后直接清零。建议质量控制如下：

1. 排除 Γ 点的三个平移声学零模；
2. 记录参考结构所有非 Γ 虚频；
3. 记录每个正负应变结构中新出现的虚频；
4. 记录 \(|\gamma|\) 的中位数、P90、P95、P99 和最大值；
5. 记录被排除模式所占的热容权重；
6. 对出现应变诱导虚频、极端 \(\gamma\) 或异常 \(I_j\) 的分量，自动补算 \(h=0.0025\)；
7. 比较 \(h=0.005\) 和 \(h=0.0025\) 的 \(I_j(100\,\mathrm K)\) 及最终 \(\alpha_V(100\,\mathrm K)\)；
8. 若结果不收敛，标记 `strain_derivative_unresolved`，不得用于最终分类。

建议区分：

```text
valid
warning_soft_mode
reference_imaginary
strained_imaginary
strain_derivative_unresolved
```

阈值最终应通过代表性晶体的收敛测试确定，而不是在代码中任意硬编码。

## 10. 弹性张量处理

读取完整 \(6\times6\) 刚度矩阵后执行：

1. 检查单位是否为 GPa；
2. 检查所有元素有限；
3. 检查矩阵非对称程度；
4. 只有在非对称误差足够小时才使用
   \(C\leftarrow(C+C^T)/2\)，并记录修正量；
5. 计算刚度本征值；
6. 检查正定性和条件数；
7. 通过稳定线性代数方法得到 \(S=C^{-1}\)。

不稳定或病态张量应分别标记：

```text
elastic_not_positive_definite
elastic_ill_conditioned
elastic_parse_error
elastic_coordinate_source_missing
```

不得使用当前脚本中的立方化 \(C_{11}/C_{12}/C_{44}\) 替代一般张量。

## 11. 热膨胀计算

模式热容为：

\[
C_{\mathbf q\nu}(T)=k_Bx^2
\frac{e^x}{(e^x-1)^2},
\qquad
x=\frac{h\nu_{\mathbf q\nu}}{k_BT}.
\]

数值实现应使用 `expm1` 或小 \(x\) 展开避免消减误差。

每个应变分量：

\[
I_j(T)=\frac{
\sum_{\mathbf q\nu}w_{\mathbf q}
C_{\mathbf q\nu}(T)\gamma^j_{\mathbf q\nu}}
{\sum_{\mathbf q}w_{\mathbf q}}.
\]

单位换算后：

\[
\boldsymbol\alpha_{\mathrm{Voigt}}(T)
=\frac{1}{V_0}S_{\mathrm{Pa}^{-1}}\mathbf I(T).
\]

将工程热应变恢复成三维对称热膨胀张量：

\[
\boldsymbol\alpha=
\begin{pmatrix}
\alpha_1 & \alpha_6/2 & \alpha_5/2\\
\alpha_6/2 & \alpha_2 & \alpha_4/2\\
\alpha_5/2 & \alpha_4/2 & \alpha_3
\end{pmatrix}.
\]

### 11.1 体积热膨胀

体积热膨胀必须按迹计算：

\[
\alpha_V(T)=\operatorname{Tr}(\boldsymbol\alpha)
=\alpha_1+\alpha_2+\alpha_3.
\]

### 11.2 实际晶格方向热膨胀

根目录晶格方向映射到弹性坐标系后，对单位方向 \(n_a,n_b,n_c\)：

\[
\alpha_a=n_a^T\boldsymbol\alpha n_a,
\quad
\alpha_b=n_b^T\boldsymbol\alpha n_b,
\quad
\alpha_c=n_c^T\boldsymbol\alpha n_c.
\]

对于单斜和三斜晶体，\(a,b,c\) 通常不是正交方向，因此：

\[
\alpha_a+\alpha_b+\alpha_c
\neq \alpha_V
\]

一般成立。禁止再通过三个晶格方向投影值相加得到体积热膨胀。

## 12. 建议输出目录和文件

默认结果目录：

```text
material/gruneisen_aniso_1M_v2/
```

建议文件：

```text
reference/
├── POSCAR
├── residual_force_stress.json
└── structure_mapping.json

work/
├── strain_0/
├── eta1_minus/
├── eta1_plus/
├── ...
├── eta6_minus/
└── eta6_plus/

thermal_expansion_cartesian.dat
thermal_expansion_directional.dat
gruneisen_temperature_voigt.dat
gruneisen_integrals.dat
gruneisen_mesh.h5
elastic_tensor_used.dat
quality_report.json
run_metadata.json
calculation_complete.json
```

### 12.1 输出表建议

`thermal_expansion_cartesian.dat`：

```text
T_K alpha_xx alpha_yy alpha_zz alpha_yz_eng alpha_xz_eng alpha_xy_eng alpha_volume
```

`thermal_expansion_directional.dat`：

```text
T_K alpha_a alpha_b alpha_c alpha_volume axis_mapping_status
```

`gruneisen_temperature_voigt.dat`：

```text
T_K G_xx G_yy G_zz G_yz G_xz G_xy
```

所有热膨胀默认报告单位为 \(10^{-6}\,\mathrm K^{-1}\)，文件头必须明确单位和剪切分量是否为工程分量。

不得在根目录或 `elastic/` 中写入新的 `ELASTIC_TENSOR`。若需要保存实际使用的张量，应写为结果目录中的 `elastic_tensor_used.dat`，并附源文件路径和 SHA256。

## 13. 元数据与缓存规则

`run_metadata.json` 至少记录：

- schema/version；
- runner 和 batch 脚本 SHA256；
- 根 POSCAR SHA256；
- 弹性参考 POSCAR SHA256；
- ELASTIC_TENSOR SHA256；
- MatterSim checkpoint 路径和 SHA256；
- Python、MatterSim、PyTorch、ASE、Phonopy、spglib、pymatgen、NumPy、SciPy 版本；
- device 和 dtype；
- 应变幅度；
- 位移幅度；
- 超胞矩阵；
- primitive matrix；
- q 网格；
- 松弛阈值；
- 温度范围；
- 创建时间；
- 输入结构原子数、元素顺序和体积。

缓存复用不能只检查 `FORCE_CONSTANTS` 是否存在。只有当前计算 fingerprint 与缓存元数据完全匹配时才能复用。

不匹配时应：

- 默认拒绝复用并报告差异；或
- 写入新的版本化结果目录；
- 不得静默使用旧缓存。

## 14. Batch 脚本设计

建议新建：

```text
run_gruneisen_thermal_expansion_v2.py
batch_gruneisen_thermal_expansion_v2.py
```

Batch 脚本负责调度和汇总，不承担科学核心计算。

### 14.1 建议功能

- `--preflight-only`：只检查输入、弹性正定性、模型和结构映射；
- `--roots`：选择 NTE/PTE 根目录；
- `--materials` 和 `--materials-file`；
- `--result-subdir gruneisen_aniso_1M_v2`；
- `--model`，默认明确指向 1M；
- `--dtype float64`；
- `--strain 0.005`；
- `--fallback-strain 0.0025`；
- `--displacement 0.01`；
- `--mesh 30 30 30`；
- `--min-supercell-length 12`；
- `--resume`；
- `--force`；
- `--chunk-count/--chunk-index`；
- `--stop-on-error`；
- 每个材料独立日志和最终总表。

### 14.2 当前 batch 脚本需要避免的问题

当前材料发现顺序先 NTE 后 PTE，`--limit` 在分块前生效，因此小规模测试可能只选到 NTE。v2 应采用以下任一方式：

- 每个 root 独立 limit；
- 交错 NTE/PTE 材料；
- 先稳定排序并分块，再对块应用 limit。

### 14.3 Batch 总表建议字段

```text
root
material
preflight_status
calc_status
quality_status
elastic_positive_definite
elastic_min_eigenvalue_GPa
elastic_condition_number
axis_mapping_status
reference_imaginary_count
strained_imaginary_count
strain_converged
alphaV_iso_100K
alphaV_aniso_100K
sign_iso
sign_aniso
sign_match
result_dir
metadata_path
quality_report_path
log_path
```

## 15. 质量判据与分类边界

### 15.1 可用于物理解释的最低条件

一个材料的 v2 结果至少应满足：

1. 完整弹性张量存在且可解析；
2. 弹性参考结构存在；
3. 刚度矩阵机械稳定且非病态；
4. 六个应变分量的结构、原子顺序和超胞一致；
5. 参考结构不存在显著非 Γ 虚频；
6. 应变导数达到预设收敛标准；
7. 被排除模式的热容权重足够小；
8. 坐标和模型来源可追溯。

否则结果可以保留用于诊断，但不得作为最终物理分类依据。

### 15.2 NTE 标签

项目现有标签规则保持不变：

- 以体积热膨胀 \(\alpha_V(100\,\mathrm K)\) 的符号作为参考；
- NTE 还要求连续负热膨胀温区 \(T_w>100\,\mathrm K\)。

各向异性计算需要额外区分：

- volumetric NTE：\(\alpha_V<0\)；
- directional NTE：至少一个实际晶格方向 \(\alpha_a,\alpha_b,\alpha_c<0\)；
- mixed-axis response：不同方向符号不同但 \(\alpha_V\) 可能为正或负。

不能把“某一方向 NTE”直接等同于“体积 NTE”。

## 16. 建议实施阶段

### 阶段 A：导入与预检

1. 导入全量重算弹性结果；
2. 确认每个材料具有 `elastic/POSCAR` 和完整张量；
3. 核对实际使用的 MatterSim-1M；
4. 生成输入审计 CSV；
5. 统计结构映射、正定性和模型来源问题。

### 阶段 B：v2 核心 runner

1. 实现输入和元数据校验；
2. 实现六分量工程应变；
3. 实现固定晶格内部松弛；
4. 实现固定 primitive/atom-order 声子计算；
5. 调用 Phonopy 简并微扰算法；
6. 计算完整 \(I_j\)、\(\alpha_i\)、\(\alpha_V\)；
7. 实现实际晶格方向投影；
8. 输出质量报告和版本化结果。

### 阶段 C：代表材料验证

覆盖以下晶体系统：

- cubic；
- tetragonal；
- orthorhombic；
- hexagonal/trigonal；
- monoclinic；
- triclinic。

验证项目：

- 应变 ±0.25%、±0.5%、必要时 ±0.75%；
- 位移 0.005、0.01 Å，必要时比较 0.001 Å；
- 固定 \(2^3\) 与满足 12 Å 的超胞；
- float32 与 float64；
- \(\alpha_V(100\,\mathrm K)\) 与等方性 QHA 的符号；
- 软模和应变诱导虚频的处理。

### 阶段 D：Batch 与全量运行

代表材料通过后，再运行全量数据集。全量运行不得先于阶段 C 的收敛验证。

## 17. 实施时应优先复用的现有代码

坐标与张量工具优先参考：

```text
D:\9.Project\25.Anisotropy_thermal_expansion\collect_mechanical_properties_correct.py
```

重点函数：

- `voigt_to_full_3x3x3x3`
- `lattice_axis_unit_vectors_in_target_lattice`
- `cte_axis_unit_vectors_in_elastic_frame`
- `directional_linear_compressibility`
- `directional_shear_modulus`

声子和有限位移流程可参考：

```text
0.scripts/prepare_three_phonons.py
0.scripts/qha_calcu/qha_calcu.py
```

其中 `prepare_three_phonons.py` 已包含固定体积下内部坐标松弛的思路，但 v2 仍需修正坐标、六分量应变、primitive 和缓存规则。

## 18. 当前决定汇总

以下决定视为后续实现默认约束：

1. 不直接修补并覆盖旧脚本结果，优先新增 v2；
2. 以弹性参考 POSCAR 作为 Grüneisen 计算坐标基准；
3. 完整计算六个工程应变分量；
4. 使用 Phonopy 原生简并微扰，不使用简单模匹配；
5. 应变结构必须固定晶格松弛内部坐标；
6. \(\alpha_V\) 按热膨胀张量迹计算；
7. 实际 \(a,b,c\) 膨胀通过方向投影获得；
8. 不截断并清零大 \(\gamma\)，而是进行应变收敛和虚频诊断；
9. 默认参数与论文保持一致：1M、0.01 Å、\(2^3\) 或 ≥12 Å、\(30^3\)；
10. 所有缓存必须有完整 fingerprint；
11. 永不覆盖权威弹性张量；
12. 全量运行前必须完成代表材料收敛验证。

## 19. 尚待导入全量弹性结果后确认的事项

1. 全量重算是否全部实际使用 MatterSim-1M；
2. 每个结果是否都保存了生成张量的 `elastic/POSCAR`；
3. `ELASTIC_TENSOR` 的具体格式和 Voigt 顺序是否完全统一；
4. 是否存在多个弹性参考结构版本；
5. 是否需要将旧恢复的 173 个张量整体替换或仅作为备份；
6. 最终质量阈值应由哪些代表材料的收敛结果确定；
7. v2 输出是否需要继续兼容旧的五列表 `thermal_expansion_gruneisen.dat`。

在上述事项确认前，可以实现通用框架和预检工具，但不应启动全量正式计算。
