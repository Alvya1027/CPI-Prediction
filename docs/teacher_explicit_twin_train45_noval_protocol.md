# 显式孪生光储备池：Train45 / Test47 无验证集协议

## 1. 当前数据划分（2026-08-07）

当前正式环比实验取消独立验证集：

| split | 目标月份 | 目标数 | 用途 |
| --- | --- | ---: | --- |
| train | 2018-09 至 2022-05 | 45 | 拟合输入变换、状态标准化和共享输出权重；作为全部参考窗口池 |
| test | 2022-06 至 2026-04 | 47 | 配置冻结后只生成状态并评价一次 |

每个目标仍使用连续前12个月的 `actual` 预测下一个月：

```text
x(t) = [actual(t-12), ..., actual(t-1)]
y(t) = actual(t)
```

没有独立验证集不代表可以用测试集调参。当前正式入口把
`alpha=100`、`lambda_pair=0.1`、`K=5`、`aggregation=mean`、
`gap=1` 和单光读出 `alpha=2` 声明为测试前固定常量。查看测试指标后如需
改变这些值，必须建立新的实验版本，不能覆盖本次结果。

训练关系对只由45个训练窗口内部构造。`gap=1` 时得到561条派生关系，
它们不是561个新增独立样本。验证和测试参考都不再存在跨集合选择：测试目标
只能从固定的train45支持库中按输入窗口距离选择参考，且不得读取测试标签排序。

## 2. 独立目录

新协议不覆盖历史50/45/47数据，使用：

```text
matlab/optical_reservoir_cpi_mom_train45_noval_20260807/
├── data/
├── inputs_twin/
├── states_twin/
└── audits_twin/
```

数据可在仓库根目录重新生成：

```powershell
python scripts/prepare_optical_reservoir_mom_train45_noval.py
```

## 3. 强制运行顺序

### 阶段A：MATLAB只生成train45

在 `matlab/optical_reservoir_cpi/` 运行：

```matlab
outputs = run_teacher_twin_train45_noval();
```

### 阶段B：Python固定配置并授权测试状态

在仓库根目录运行：

```powershell
python scripts/run_teacher_explicit_twin_mom_train45_noval.py
```

该阶段只能读取train45，随后写出：

```text
results/siamese_optical_mom_teacher_twin_train45_noval_20260807/
└── tables/fixed_configuration.json

matlab/optical_reservoir_cpi_mom_train45_noval_20260807/
└── inputs_twin/test_generation_authorization.json
```

### 阶段C：MATLAB生成不变的test47

```matlab
outputs = run_teacher_twin_test45_frozen();
```

### 阶段D：Python只评价一次

```powershell
python scripts/run_teacher_explicit_twin_mom_train45_noval.py `
  --frozen-test results/siamese_optical_mom_teacher_twin_train45_noval_20260807/tables/fixed_configuration.json
```

单光与孪生必须读取同一批新显式Twin状态。测试完成后不得根据测试结果改参
并在同一实验名下重复测试。
