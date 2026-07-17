# MATLAB / Simulink 目录

- `optical_reservoir_cpi/`：CPI 光储备池仿真的独立工作目录。这里使用老师模型的副本，不修改 `ESN/` 中的原始文件。
- `README_matlab.md`：仓库原有的 MATLAB 使用说明。

当前采用孪生光储备池回归：共享光储备池提取两个窗口的状态，回归读出层预测连续 CPI 差值，再用参考 CPI 还原目标值。
