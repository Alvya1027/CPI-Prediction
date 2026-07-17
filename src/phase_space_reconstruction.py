from config import DATA_PROCESSED_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 1. 加载数据
# ------------------------------
data = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastmonth=100.csv')
ts = np.array(data['actual'])
'''
x_train = np.load(DATA_PROCESSED_DIR / 'cpi_seq_train.npy')
x_val = np.load(DATA_PROCESSED_DIR / 'cpi_seq_val.npy')
x_test = np.load(DATA_PROCESSED_DIR / 'cpi_seq_test.npy')
x = np.append(x_train, x_val, axis=0)
ts = np.append(x, x_test, axis=0)
'''
print(ts)
# 若为一维数组则直接使用，若多维可根据需要展平或取第一列
if ts.ndim > 1:
    ts = ts.ravel()


# ------------------------------
# 2. 互信息法求延迟时间 tau
# ------------------------------
def mutual_information(ts, max_lag, bins=20):
    """
    计算时间序列在不同延迟下的互信息
    ts: 一维时间序列
    max_lag: 最大延迟
    bins: 直方图的箱数
    返回: lags, mi (互信息值)
    """
    n = len(ts)
    mi = np.zeros(max_lag)
    # 数据归一化到 [0,1] 以便分箱
    ts_norm = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-10)
    for lag in range(1, max_lag + 1):
        x = ts_norm[:n - lag]
        y = ts_norm[lag:]
        # 二维直方图联合概率
        joint, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
        joint = joint / np.sum(joint)  # 联合概率
        # 边缘概率
        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)
        # 只保留非零项
        non_zero = joint > 0
        # 互信息 I(X;Y) = sum P(x,y) log(P(x,y)/(P(x)P(y)))
        mi[lag - 1] = np.sum(joint[non_zero] *
                             np.log(joint[non_zero] / (px[:, None] * py)[non_zero]))
    lags = np.arange(1, max_lag + 1)
    return lags, mi


def find_first_minimum(mi):
    """寻找互信息曲线的第一个极小值对应的索引"""
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i - 1] and mi[i] < mi[i + 1]:
            return i + 1  # lag = index+1
    # 如果找不到明显的极小值，取下降变缓的点（经验：取连续下降后首次回升的前一点）
    diffs = np.diff(mi)
    for i in range(len(diffs) - 1):
        if diffs[i] < 0 and diffs[i + 1] > 0:
            return i + 2
    # 最终回退到延迟 1
    return 1


# 计算延迟时间
max_lag = 50  # 根据序列长度可调
lags, mi = mutual_information(ts, max_lag)
tau = find_first_minimum(mi)
print(f"互信息法确定的延迟时间 tau = {tau}")


# ------------------------------
# 3. Cao 方法求嵌入维数 m
# ------------------------------
def cao_method(ts, tau, max_dim=10):
    """
    Cao方法确定嵌入维数
    ts: 一维时间序列
    tau: 延迟时间
    max_dim: 最大尝试嵌入维数
    返回: dims, E1(d) (E1接近饱和时对应的 d 即为嵌入维数)
    """
    n = len(ts)
    # 计算最大的相空间点数
    N = n - (max_dim - 1) * tau
    if N < 10:
        raise ValueError("时间序列太短，无法计算到 max_dim 维")

    E1 = np.zeros(max_dim - 1)  # 存储维度从2开始到max_dim
    for d in range(2, max_dim + 1):
        # 构建 d 维相空间
        m = d
        N_d = n - (m - 1) * tau
        # 向量 (N_d x m)
        vectors = np.array([ts[i: i + N_d] for i in range(m)]).T  # shape (N_d, m)

        # 计算最近邻距离（欧几里得）以及下一维度的距离
        # 为了提高效率，可以直接计算
        a = np.zeros(N_d)
        for i in range(N_d):
            # 排除自身
            diff = vectors - vectors[i]
            # 欧氏距离
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            dist[i] = np.inf
            nn_idx = np.argmin(dist)
            # d 维最近邻距离
            dist_d = dist[nn_idx]
            # d+1 维中的距离（需要时间序列的额外点）
            if i + m * tau < n and nn_idx + m * tau < n:
                # d+1 维向量的第 d 个坐标是 ts[i + m*tau] (这里 m = d, 所以实际是 ts[i+d*tau])
                # 注意: 延迟坐标: [x(i), x(i+tau), ..., x(i+(d-1)*tau)]
                dist_d1 = np.sqrt(dist_d ** 2 + (ts[i + d * tau] - ts[nn_idx + d * tau]) ** 2)
            else:
                # 如果超出长度，忽略该点
                dist_d1 = np.inf
            if dist_d > 0:
                a[i] = dist_d1 / dist_d
            else:
                a[i] = 1.0  # 避免除零
        # E(d) = 1/(N_d) * sum a(i)
        valid = np.isfinite(a)
        if np.sum(valid) == 0:
            E1[d - 2] = np.nan
        else:
            E1[d - 2] = np.mean(a[valid])

    # 计算 E1(d) = E(d+1)/E(d) 的变化，通常画图找饱和点
    # Cao 方法通常看 E1(d) 是否停止变化（接近1），或者计算 E2(d)
    # 简化：找到 E1(d) 变化小于阈值且后续稳定的维数
    # 这里计算 E1 的差分比值
    dims = np.arange(2, max_dim + 1)

    # 寻找第一个满足 |E1(d) - E1(d-1)| 很小的点
    threshold = 0.05  # 可根据实际情况调整
    for i in range(1, len(E1)):
        if np.abs(E1[i] - E1[i - 1]) < threshold and E1[i] < 1.2:
            m_opt = dims[i]
            break
    else:
        # 没找到就取 E1 最小值对应的维度
        m_opt = dims[np.argmin(np.abs(E1 - 1))]
    return dims, E1, m_opt

#tau = 2

max_dim = 30
dims, E1, m = cao_method(ts, tau, max_dim)
print(f"Cao 方法确定的嵌入维数 m = {m}")
print(f"不同嵌入维数下的 E1 值: {dict(zip(dims, E1))}")

#m=10

# 最终结果
print(f"\n相空间重构参数: 延迟时间 τ = {tau}, 嵌入维数 m = {m}")

# 相空间重构
N = len(ts)
num_vectors = N - (m - 1) * tau
X_reconstructed = np.array([ts[i: i + num_vectors] for i in range(0, m * tau, tau)]).T

# 对齐特征与目标：舍弃最后一个无法预测的相点
features = X_reconstructed[:-1]          # 前 3612 个相点
start_idx = (m - 1) * tau
targets = ts[start_idx + 1 : start_idx + 1 + num_vectors - 1]

print(f"特征形状: {features.shape}, 目标形状: {targets.shape}")

# 将特征和目标按列合并
data_total = np.column_stack((features, targets))  # 最后一列为目标值
# 使用 pandas 保存（带列名更清晰）
df = pd.DataFrame(data_total, columns=[f'feature_{i}' for i in range(features.shape[1])] + ['target'])
df.to_csv('reconstructed_data.csv', index=False)

np.set_printoptions(precision=2, suppress=True, linewidth=120)  # 控制输出格式
print(features)
print(targets)

features_list = np.arange(m)

#3 数据划分
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.1, shuffle=False)

transfer = StandardScaler()
x_train_standard = transfer.fit_transform(x_train)
x_test_standard = transfer.transform(x_test)
y_train_standard = transfer.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_standard = transfer.transform(y_test.reshape(-1, 1)).ravel()

#4 机器学习模型
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

def model_metrics(model_type, y_test, y_predict):
    MAE = mean_absolute_error(y_test, y_predict)
    RMSE = root_mean_squared_error(y_test, y_predict)
    MAPE = mean_absolute_percentage_error(y_test, y_predict)
    MASE = MAE / naive_MAE
    print(f"{model_type}_MAE:{MAE:.4f}, {model_type}_RMSE:{RMSE:.4f}, "
          f"{model_type}_MAPE:{(MAPE * 100):.3f}%, {model_type}_MASE:{(MASE * 100):.2f}%")
    return MAE, RMSE, MAPE, MASE

#4.1 naive模型
naive_predict = np.roll(y_test, 1) # 简单用上期值预测当期
naive_MAE = mean_absolute_error(y_test, naive_predict)

naive_MAE, naive_RMSE, naive_MAPE, naive_MASE = model_metrics('naive', y_test, naive_predict)

#4.2 线性回归模型
from sklearn.linear_model import LinearRegression, Lasso, Ridge

#lr = LinearRegression()
#lr = Lasso(alpha = 0.0022)
lr = Ridge(alpha = 2.858)
lr.fit(x_train, y_train)

lr_train_predict = lr.predict(x_train)
lr_predict = lr.predict(x_test)


'''
alpha=np.arange(2,3,0.001)
grid = {'alpha':alpha}
#lr_searchCV = RandomizedSearchCV(estimator=svr, param_distributions=grid, cv=3,
#                                n_iter=100, scoring='neg_mean_squared_error', verbose=2)
lr_searchCV = GridSearchCV(estimator=lr, param_grid=grid, cv=3,
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
lr_searchCV.fit(x_train, y_train)
print(lr_searchCV.best_params_)
lr_predict = lr_searchCV.predict(x_test)
'''

'''
#lr = LinearRegression()
#lr = Lasso(alpha = 0.0277)
lr = Ridge(alpha = 45.387)
lr.fit(x_train_standard, y_train_standard)

lr_predict_standard = lr.predict(x_test_standard)
lr_predict = transfer.inverse_transform(lr_predict_standard.reshape(-1, 1)).ravel()
'''

'''
alpha=np.arange(0.01,0.03,0.0001)
grid = {'alpha':alpha}
#lr_searchCV = RandomizedSearchCV(estimator=svr, param_distributions=grid, cv=3,
#                                n_iter=100, scoring='neg_mean_squared_error', verbose=2)
lr_searchCV = GridSearchCV(estimator=lr, param_grid=grid, cv=3,
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
lr_searchCV.fit(x_train_standard, y_train_standard)
print(lr_searchCV.best_params_)
lr_predict_standard = lr_searchCV.predict(x_test_standard)
lr_predict = transfer.inverse_transform(lr_predict_standard.reshape(-1, 1)).ravel()
'''

lr_MAE, lr_RMSE, lr_MAPE, lr_MASE = model_metrics('lr', y_test, lr_predict)

#4.3 随机森林模型
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=83, max_depth=12, min_samples_split=11,
                           min_samples_leaf=2, max_features=0.7, random_state=42)
rf.fit(x_train, y_train)

rf_predict = rf.predict(x_test)

'''
n_estimators=np.arange(80,90,1)
max_depth=[11,12,13]
min_samples_split=[10,11,12]
min_samples_leaf=[2,3]
max_features=[0.7]  # 覆盖各种随机程度
grid = {'n_estimators':n_estimators, 'max_depth':max_depth, 'min_samples_split':min_samples_split,
        'min_samples_leaf':min_samples_leaf, 'max_features':max_features}
rf_searchCV = GridSearchCV(estimator=rf, param_grid=grid, cv=3,
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
rf_searchCV.fit(x_train, y_train)
print(rf_searchCV.best_params_)
rf_predict = rf_searchCV.predict(x_test)
'''

rf_MAE, rf_RMSE, rf_MAPE, rf_MASE = model_metrics('rf', y_test, rf_predict)

#4.4 支持向量机模型
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

'''
svr = SVR(kernel='rbf', gamma='scale', C=330000, epsilon=0.164)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train, y_train)

svr_predict = svr.predict(x_test)
'''

'''
C = np.arange(300000,350000,1000)
#gamma = [0.001, 0.01, 0.1, 1, 10]
epsilon = np.arange(0.1,0.2,0.001)
grid = {'C':C, 'epsilon':epsilon}

svr_searchCV = GridSearchCV(estimator=svr, param_grid=grid, cv=3,
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
svr_searchCV.fit(x_train, y_train)
print(svr_searchCV.best_params_)
svr_predict = svr_searchCV.predict(x_test)
'''


svr = SVR(kernel='rbf', gamma='scale', C=1.291, epsilon=0.637)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train_standard, y_train_standard)

svr_predict_standard = svr.predict(x_test_standard)
svr_predict = transfer.inverse_transform(svr_predict_standard.reshape(-1, 1)).ravel()


'''
C = np.arange(1.28,1.3,0.001)
#gamma = [0.001, 0.01, 0.1, 1, 10]
epsilon = np.arange(0.63,0.65,0.001)
grid = {'C':C, 'epsilon':epsilon}

svr_searchCV = GridSearchCV(estimator=svr, param_grid=grid, cv=3,
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
svr_searchCV.fit(x_train_standard, y_train_standard)
print(svr_searchCV.best_params_)
svr_predict_standard = svr_searchCV.predict(x_test_standard)
svr_predict = transfer.inverse_transform(svr_predict_standard.reshape(-1, 1)).ravel()
'''

svr_MAE, svr_RMSE, svr_MAPE, svr_MASE = model_metrics('svr', y_test, svr_predict)

#4.5 XGBoost模型
import xgboost as xgb

xgb_model = xgb.XGBRegressor(max_depth=2, n_estimators=31, learning_rate=0.3249, random_state=42, objective='reg:squarederror')
xgb_model.fit(x_train, y_train)

xgb_predict = xgb_model.predict(x_test)

'''
max_depth=[1,2,3]
n_estimators=np.arange(25,35,1)
learning_rate=[0.3249]
grid = {'max_depth':max_depth, 'n_estimators':n_estimators, 'learning_rate':learning_rate}

xgb_searchCV = GridSearchCV(estimator=xgb_model, param_grid=grid, cv=3,
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
xgb_searchCV.fit(x_train, y_train)
print(xgb_searchCV.best_params_)
xgb_predict = xgb_searchCV.predict(x_test)
'''

xgb_MAE, xgb_RMSE, xgb_MAPE, xgb_MASE = model_metrics('xgb', y_test, xgb_predict)

#4.6 SARIMAX模型
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

y_train_series = pd.Series(
    y_train,
    index=pd.date_range(start='2001-08', periods=len(y_train), freq='MS')  # MS = 月初
)

# ========== 拟合最优均值模型 ==========
final_model = SARIMAX(y_train_series, order=(0, 0, 0), trend='c',
                      enforce_stationarity=False, enforce_invertibility=False)
final_results = final_model.fit(disp=False)

# ========== 残差白噪声检验 ==========
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(final_results.resid, lags=12, return_df=True)
'''
print("\n残差白噪声检验 (p值全部 > 0.05 即代表模型合格)：")
print(lb_test)
print("\n检验结论：所有 p 值均远大于 0.05，残差为纯随机白噪声，模型通过验证！")
'''

# ========== 未来预测 ==========
forecast_steps = len(y_test)
forecast_result = final_results.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()  # 95%置信区间

# 构造预测时间索引
forecast_index = pd.date_range(
    start=y_train_series.index[-1] + pd.DateOffset(months=1),
    periods=forecast_steps,
    freq='MS'
)

# 整理预测结果为DataFrame
forecast_df = pd.DataFrame({
    '预测值': forecast_mean.values,
    '置信区间下界': forecast_ci.iloc[:, 0].values,
    '置信区间上界': forecast_ci.iloc[:, 1].values
}, index=forecast_index)

sarimax_MAE, sarimax_RMSE, sarimax_MAPE, sarimax_MASE = model_metrics('sarimax', y_test, forecast_mean)

'''
print("\n未来12个月预测结果：")
print(forecast_df.round(2))
'''

#5 特征重要性可视化
#5.1 随机森林模型
# 获取特征重要性（默认基于不纯度减少的 MDI 值）
importances = rf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features_list,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()

plt.xlabel('重要性分数 (基于不纯度减少)', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.title('随机森林特征重要性', fontsize=14)

for i, (feature, imp) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
    plt.text(imp + 0.001, i, f'{imp:.3f}', va='center', fontsize=9)

# 显示网格
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#5.2 XGBoost模型
# 获取重要性数据（以gain为基准）
importance_gain = xgb_model.get_booster().get_score(importance_type='gain')

importance_df = pd.DataFrame({
    'Feature': list(importance_gain.keys()),
    'Importance': list(importance_gain.values())
}).sort_values(by='Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()

for i, v in enumerate(importance_df['Importance']):
    plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

plt.xlabel('平均信息增益 (Gain)')
plt.title('XGBoost 特征重要性')
plt.tight_layout()
plt.show()

#xgb.plot_importance(xgb_model, importance_type='gain')  # 可以指定类型
#plt.show()

#6 各模型预测值与实际值可视化
x_seq = np.arange(len(y_test))

fig, axes = plt.subplots(1, 6, figsize=(18, 5))

# 子图1: Naive模型
axes[0].plot(x_seq, y_test, label='Actual', color='blue', lw=1.5)
axes[0].plot(x_seq, naive_predict, label='Predicted', color='red', lw=1.5, alpha=0.7)
axes[0].set_xlabel('Sequence')
axes[0].set_ylabel('CPI')
axes[0].set_title('Naive Model')
axes[0].legend()

# 子图2: 线性回归模型
axes[1].plot(x_seq, y_test, label='Actual', color='blue', lw=1.5)
axes[1].plot(x_seq, lr_predict, label='Predicted', color='red', lw=1.5, alpha=0.7)
axes[1].set_xlabel('Sequence')
axes[1].set_ylabel('CPI')
axes[1].set_title('Linear Regression')
axes[1].legend()

# 子图3: 随机森林模型
axes[2].plot(x_seq, y_test, label='Actual', color='blue', lw=1.5)
axes[2].plot(x_seq, rf_predict, label='Predicted', color='red', lw=1.5, alpha=0.7)
axes[2].set_xlabel('Sequence')
axes[2].set_ylabel('CPI')
axes[2].set_title('Random Forest')
axes[2].legend()

# 子图4: 支持向量机模型
axes[3].plot(x_seq, y_test, label='Actual', color='blue', lw=1.5)
axes[3].plot(x_seq, svr_predict, label='Predicted', color='red', lw=1.5, alpha=0.7)
axes[3].set_xlabel('Sequence')
axes[3].set_ylabel('CPI')
axes[3].set_title('SVR')
axes[3].legend()

# 子图5: XGBoost模型
axes[4].plot(x_seq, y_test, label='Actual', color='blue', lw=1.5)
axes[4].plot(x_seq, xgb_predict, label='Predicted', color='red', lw=1.5, alpha=0.7)
axes[4].set_xlabel('Sequence')
axes[4].set_ylabel('CPI')
axes[4].set_title('XGBoost')
axes[4].legend()

# 子图6: SARIMAX模型
axes[5].plot(x_seq, y_test, label='Actual', color='blue', lw=1.5)
axes[5].plot(x_seq, forecast_mean, label='Predicted', color='red', lw=1.5, alpha=0.7)
axes[5].set_xlabel('Sequence')
axes[5].set_ylabel('CPI')
axes[5].set_title('SARIMAX')
axes[5].legend()

plt.tight_layout()
plt.show()

#7 实际值与预测值差值（残差）曲线
residuals_dict = {
    'Naive': y_test - naive_predict,
    'LR': y_test - lr_predict,
    'RF': y_test - rf_predict,
    'SVR': y_test - svr_predict,
    'XGBoost': y_test - xgb_predict,
    'SARIMAX': y_test - forecast_mean
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (model_name, residuals) in enumerate(residuals_dict.items()):
    ax = axes[idx]
    ax.plot(range(len(residuals)), residuals, marker='o', linestyle='-', color='darkorange', markersize=4, linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)  # 参考零线
    ax.set_title(f'{model_name} 残差', fontsize=12)
    ax.set_xlabel('测试集序列号')
    ax.set_ylabel('残差 (实际 - 预测)')
    ax.grid(True, linestyle=':', alpha=0.6)
    # 显示残差基本统计量
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.text(0.02, 0.95, f'均值: {mean_res:.4f}\n标准差: {std_res:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

#8 各模型MAE/RMSE/MAPE/MASE对比图
metrics_dict = {
    'Naive':   [naive_MAE, naive_RMSE, naive_MAPE, naive_MASE],
    'LR':      [lr_MAE, lr_RMSE, lr_MAPE, lr_MASE],
    'RF':      [rf_MAE, rf_RMSE, rf_MAPE, rf_MASE],
    'SVR':     [svr_MAE, svr_RMSE, svr_MAPE, svr_MASE],
    'XGBoost': [xgb_MAE, xgb_RMSE, xgb_MAPE, xgb_MASE],
    'SARIMAX': [sarimax_MAE, sarimax_RMSE, sarimax_MAPE, sarimax_MASE],
}
metric_names = ['MAE', 'RMSE', 'MAPE', 'MASE']
model_names = list(metrics_dict.keys())
colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (ax, metric_name) in enumerate(zip(axes.flat, metric_names)):
    values = [metrics_dict[model][idx] for model in model_names]
    bars = ax.bar(model_names, values, color=colors, edgecolor='gray', linewidth=0.8)
    ax.set_title(metric_name, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name)
    # 在柱状图上标注数值
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f}' if metric_name != 'MAPE' and metric_name != 'MASE' else f'{v*100:.3f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

#9 趋势预测正确性可视化
# 计算实际CPI的变化方向
actual_diff = np.diff(y_test)
actual_direction = np.sign(actual_diff)   # 1:上升, -1:下降, 0:持平（长度 = len(y_test)-1）

# 各模型预测值
model_predictions = {
    'Naive': naive_predict,
    'LR': lr_predict,
    'RF': rf_predict,
    'SVR': svr_predict,
    'XGBoost': xgb_predict,
    'SARIMAX': forecast_mean
}

trend_accuracy = {}
for name, pred in model_predictions.items():
    pred_diff = np.diff(pred)
    pred_direction = np.sign(pred_diff)
    # 比较实际趋势与预测趋势（完全相同符号视为正确）
    correct = (actual_direction == pred_direction)
    accuracy = np.mean(correct) * 100
    trend_accuracy[name] = accuracy

# 绘制柱状图
plt.figure(figsize=(10, 6))
model_names = list(trend_accuracy.keys())
accuracies = list(trend_accuracy.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=0.8)
plt.ylabel('趋势预测准确率 (%)', fontsize=12)
plt.title('各模型趋势方向（上升/下降）预测准确率对比', fontsize=14)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 在柱顶显示准确率数值
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("\n各模型趋势预测准确率：")
for name, acc in trend_accuracy.items():
    print(f"{name}: {acc:.2f}%")