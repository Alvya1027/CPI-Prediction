# MATLAB 环境说明与光储备池仿真框架

> 本文档为第4阶段"光储备池仿真"提前准备。内容涵盖 MATLAB 安装配置、仿真框架说明、以及关键参考公式。

---

## 1. MATLAB 安装

### 1.1 获取 MATLAB

- 学校通常提供 MATLAB 校园授权，可通过学校信息中心或软件正版化平台获取
- 也可使用 [MATLAB Online](https://matlab.mathworks.com/)（在线版，无需安装，功能基本够用）

### 1.2 必需工具箱

本项目第4阶段光储备池仿真需要的工具箱：

| 工具箱 | 用途 |
|--------|------|
| MATLAB 基础 | 数组运算、绘图 |
| Optimization Toolbox | 可选，用于岭回归求解 |
| Parallel Computing Toolbox | 可选，用于参数扫描加速 |

如果只用基础版 MATLAB，岭回归可以手动实现（`W_out = (X'X + βI)\(X'y)`），不需要 Optimization Toolbox。

---

## 2. 光储备池仿真框架

### 2.1 整体流程

```
输入 CPI 数据
    │
    ▼
掩码预处理（Sample & Hold + Masking）
    │
    ▼
非线性节点动力学仿真（L-K 速率方程 + RK4 求解）
    │
    ▼
虚节点状态读取（在延迟环中采样）
    │
    ▼
输出层训练（岭回归）
    │
    ▼
预测与评估
```

### 2.2 核心方程：L-K 速率方程

来自光学学报综述（黄灿等，2025），式(5)-(6)：

$$\frac{dE}{dt} = \frac{1}{2}(1+i\alpha)\left[\frac{g(N-N_0)}{1+\varepsilon|E|^2} - \frac{1}{\tau_p}\right]E + kE(t-\tau)e^{-i\omega\tau} + k_{inj}E_{inj}(t)e^{i\Delta\omega t}$$

$$\frac{dN}{dt} = J - \frac{N}{\tau_s} - \frac{g(N-N_0)}{1+\varepsilon|E|^2}|E|^2$$

**参数说明**：

| 参数 | 符号 | 物理含义 | 典型值 |
|------|------|----------|--------|
| 线宽增强因子 | α | 载流子引起的折射率变化 | 3~5 |
| 增益系数 | g | 激光器增益 | 1.5×10⁴ s⁻¹ |
| 透明载流子数 | N₀ | 激光阈值处的载流子数 | 1.5×10⁸ |
| 增益饱和系数 | ε | 抑制增益饱和 | 2.5×10⁻⁸ |
| 光子寿命 | τp | 光子在腔内的平均寿命 | 2 ps |
| 载流子寿命 | τs | 载流子复合时间 | 2 ns |
| 反馈强度 | k | 反馈光的强度 | 0~0.1 ps⁻¹ |
| 注入强度 | kinj | 注入光的强度 | 0~0.1 ps⁻¹ |
| 延迟时间 | τ | 反馈环的延迟 | 与虚节点数相关 |
| 注入电流 | J | 激光器偏置电流 | 1.2×Jth |
| 频率失谐 | Δω | DL 与 RL 的频率差 | -10~10 GHz |

### 2.3 RK4 数值求解

L-K 方程是延迟微分方程（DDE），需要用数值方法求解。RK4（四阶龙格-库塔）是最常用的方法。

MATLAB 伪代码框架：

```matlab
% 光储备池仿真主函数
function [states, y_pred] = optical_reservoir(X_train, y_train, X_test, params)

    % 参数初始化
    N = params.n_virtual;      % 虚节点数量
    tau = params.tau;           % 延迟时间
    theta = tau / N;            % 虚节点间隔
    dt = theta / 10;            % 仿真步长（比 theta 更细）
    k_fb = params.feedback_gain; % 反馈强度
    k_inj = params.input_gain;  % 注入强度
    
    % 随机掩码
    mask = 2 * (rand(N, 1) > 0.5) - 1;  % {-1, +1}
    
    % 初始化状态变量
    E = 0.01;  % 光场（复振幅）
    N_carrier = params.N0;  % 载流子密度
    
    % 存储历史（用于延迟反馈）
    E_history = zeros(ceil(tau/dt), 1);
    
    % 收集虚节点状态
    states = [];
    
    for sample_idx = 1:length(X_train)
        % 输入保持（Sample & Hold）
        u = X_train(sample_idx);
        I = u * ones(N, 1);  % 保持一个延迟周期
        J = mask .* I;        % 掩码
        
        % 在每个延迟周期内仿真空节点
        virtual_states = zeros(N, 1);
        
        for v = 1:N
            % 当前虚节点的输入
            J_current = J(v);
            
            % RK4 求解一个 theta 窗
            E, N_carrier, E_history = rk4_step_optical(...
                E, N_carrier, E_history, ...
                J_current, k_fb, k_inj, tau, dt, theta, params);
            
            % 读取虚节点状态（光强 = |E|²）
            virtual_states(v) = abs(E)^2;
        end
        
        states = [states; virtual_states'];
    end
    
    % 训练输出权重（岭回归）
    W_out = (states' * states + params.beta * eye(N)) \ (states' * y_train);
    
    % 预测
    y_pred = states * W_out;
end
```

### 2.4 虚节点读取

虚节点是延迟反馈环中按时间间隔 θ 采样的点。关键参数：

- **θ 的选择**：θ ≈ 0.2T（T 为非线性节点的特征时间尺度）。θ 太小 → 过度平均化，θ 太大 → 虚节点耦合不足
- **N 的选择**：N = τ/θ。N 越大，储备池维度越高，但计算量也越大。通常 N 在 50~400 之间
- **状态读取**：读取每个虚节点处的光强 |E|² 作为储备池状态

### 2.5 工作点扫描

光储备池的性能敏感依赖于反馈强度 k 和注入强度 kinj。需要扫描这些参数找到最优工作点：

```matlab
% 参数扫描框架
k_fb_values = 0.01:0.01:0.1;
k_inj_values = 0.01:0.01:0.1;

results = zeros(length(k_fb_values), length(k_inj_values));

for i = 1:length(k_fb_values)
    for j = 1:length(k_inj_values)
        params.feedback_gain = k_fb_values(i);
        params.input_gain = k_inj_values(j);
        
        [~, y_pred] = optical_reservoir(X_train, y_train, X_val, params);
        results(i, j) = rmse(y_val, y_pred);
    end
end

% 可视化参数空间
imagesc(k_inj_values, k_fb_values, results);
xlabel('注入强度 k_{inj}');
ylabel('反馈强度 k');
colorbar;
title('RMSE 在参数空间的分布');
```

---

## 3. 简化方案：Mackey-Glass 非线性节点

如果不直接用 L-K 方程（复杂度较高），可以先用 Mackey-Glass 方程作为简化版非线性节点，验证整体框架：

$$\dot{x}(t) = -x(t) + \frac{\eta \cdot (x(t-\tau) + \gamma \cdot J(t))}{1 + (x(t-\tau) + \gamma \cdot J(t))^p}$$

- η：反馈增益（类似谱半径，通常 0.8~1.0）
- γ：输入增益（类似输入缩放，通常 0.5~1.0）
- p：非线性指数（通常 7，决定非线性程度）
- τ：延迟时间

Mackey-Glass 方程比 L-K 方程简单，不需要处理复数，可以先用它验证延时 RC 框架的正确性，再升级到 L-K 方程。

---

## 4. 文件结构建议

```
matlab/
├── README_matlab.md          # 本文档
├── src/
│   ├── optical_reservoir.m   # 光储备池主函数（L-K 方程）
│   ├── rk4_step.m            # RK4 单步求解
│   ├── mg_reservoir.m        # 简化版：Mackey-Glass 非线性节点
│   ├── train_output.m        # 岭回归训练
│   └── params.m              # 参数配置
├── data/
│   ├── load_cpi_data.m       # 加载 CPI 数据
│   └── cpi_monthly.csv       # CPI 数据副本
└── results/
    ├── run_optical_rc.m      # 主运行脚本
    └── plot_results.m        # 结果可视化
```

---

## 5. 参考文献

1. Appeltant, L. et al. "Information processing using a single dynamical node as complex system." *Nature Communications* 2, 468 (2011). — 延时 RC 原始论文，包含 Mackey-Glass 节点的详细参数。

2. 黄灿等. "半导体激光非线性动力学与模拟光计算（特邀）." *光学学报* 45(14), 1420007 (2025). — L-K 方程和光储备池仿真框架的详细说明。

3. Lang, R. & Kobayashi, K. "External optical feedback effects on semiconductor injection laser properties." *IEEE J. Quantum Electron.* 16(3), 347-355 (1980). — L-K 方程原始论文。