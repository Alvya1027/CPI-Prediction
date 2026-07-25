from config import DATA_PROCESSED_DIR, TABLES_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#1 数据加载
#cpi_raw = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_yoy.csv')
cpi_raw = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastmonth=100.csv')

#cpi_raw = cpi_raw.iloc[183:]
cpi_raw = cpi_raw.iloc[200:297]
#print(cpi_raw)

#2 数据预处理
cpi_data = pd.get_dummies(cpi_raw, columns=['month'])
cpi = np.array(cpi_data['actual'])
features = cpi_data.drop('actual', axis=1)
features_list = list(features.columns)

#3 数据划分
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(features, cpi, test_size=0.48, shuffle=False)
#print(y_test)

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

def model_searchCV(model, grid, x_train, y_train, x_test):
    tscv = TimeSeriesSplit(n_splits=5, max_train_size=24)  # 可根据数据量调整
    model_searchCV = GridSearchCV(estimator=model, param_grid=grid, cv=tscv,
                               scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
    model_searchCV.fit(x_train, y_train)
    print(model_searchCV.best_params_)
    model_predict = model_searchCV.predict(x_test)
    return model_predict

#4.1 naive模型
naive_predict = np.roll(y_test, 1) # 简单用上期值预测当期
naive_MAE = mean_absolute_error(y_test, naive_predict)

#4.2 线性回归模型
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline

'''
#lr = LinearRegression()
#lr = Lasso(alpha = 0.0022)
lr = Ridge(alpha = 2.403)
lr.fit(x_train, y_train)

lr_train_predict = lr.predict(x_train)
lr_predict = lr.predict(x_test)
'''

'''
alpha=np.arange(0.1,5,0.001)
grid = {'alpha':alpha}
lr_predict = model_searchCV(lr, grid, x_train, y_train, x_test)
'''


#lr = LinearRegression()
#lr = Lasso(alpha = 0.0277)
lr = Ridge(alpha = 19.1)
lr.fit(x_train_standard, y_train_standard)

lr_predict_standard = lr.predict(x_test_standard)
lr_predict = transfer.inverse_transform(lr_predict_standard.reshape(-1, 1)).ravel()


'''
alpha=np.arange(0.01,0.03,0.0001)
grid = {'alpha':alpha}
lr_predict_standard = model_searchCV(lr, grid, x_train_standard, y_train_standard, x_test_standard)
lr_predict = transfer.inverse_transform(lr_predict_standard.reshape(-1, 1)).ravel()
'''

'''
# 定义 Pipeline：先标准化，再训练 SVR
lr = Pipeline([
    ('scaler', StandardScaler()),          # 标准化特征
    ('lr', Ridge(alpha = 19.1))             # RBF 核 SVR
])
lr.fit(x_train, y_train)

lr_predict = lr.predict(x_test)
'''

'''
# 定义超参数搜索空间（注意加前缀 'svr__'）
grid = {
    'lr__alpha': np.arange(10,20,0.1),          # 惩罚参数，建议对数增长
}

lr_predict = model_searchCV(lr, grid, x_train, y_train, x_test)
'''

#4.3 随机森林模型
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=110, max_depth=7, min_samples_split=10,
                           min_samples_leaf=1, max_features=0.65, random_state=42)
rf.fit(x_train, y_train)

rf_predict = rf.predict(x_test)

'''
n_estimators=[105,110,115]
max_depth=[6,7,8]
min_samples_split=[9,10,11]
min_samples_leaf=[1,2]
max_features=[0.6,0.65,0.7]  # 覆盖各种随机程度
grid = {'n_estimators':n_estimators, 'max_depth':max_depth, 'min_samples_split':min_samples_split,
        'min_samples_leaf':min_samples_leaf, 'max_features':max_features}
rf_predict = model_searchCV(rf, grid, x_train, y_train, x_test)
'''

#4.4 支持向量机模型
from sklearn.svm import SVR
#from sklearn.kernel_ridge import KernelRidge

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
svr_predict = model_searchCV(svr, grid, x_train, y_train, x_test)
'''

'''
svr = SVR(kernel='rbf', gamma=0.001, C=10, epsilon=0.1)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train_standard, y_train_standard)

svr_predict_standard = svr.predict(x_test_standard)
svr_predict = transfer.inverse_transform(svr_predict_standard.reshape(-1, 1)).ravel()
'''

'''
C = [1, 10, 100]
gamma = [0.001, 0.01, 0.1, 1, 10]
epsilon = [0.001, 0.01, 0.1, 0.2]
grid = {'C':C, 'gamma':gamma, 'epsilon':epsilon}

#svr_predict = model_searchCV(svr, grid, x_train_standard, y_train, x_test_standard)
svr_predict_standard = model_searchCV(svr, grid, x_train_standard, y_train_standard, x_test_standard)
svr_predict = transfer.inverse_transform(svr_predict_standard.reshape(-1, 1)).ravel()
'''


# 定义 Pipeline：先标准化，再训练 SVR
svr = Pipeline([
    ('scaler', StandardScaler()),          # 标准化特征
    ('svr', SVR(kernel='rbf', gamma=0.001, C=10, epsilon=0.1))             # RBF 核 SVR
])
svr.fit(x_train, y_train)

svr_predict = svr.predict(x_test)


'''
# 定义超参数搜索空间（注意加前缀 'svr__'）
grid = {
    'svr__C': [1, 10, 100],          # 惩罚参数，建议对数增长
    'svr__gamma': [0.001,0.01, 0.1, 1, 'scale'], # RBF 核宽度
    'svr__epsilon': [0.001, 0.01, 0.1, 0.2]      # 不敏感损失带
}

svr_predict = model_searchCV(svr, grid, x_train, y_train, x_test)
'''

#4.5 XGBoost模型
import xgboost as xgb

xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=550, learning_rate=0.0008, random_state=42, objective='reg:squarederror')
xgb_model.fit(x_train, y_train)

xgb_predict = xgb_model.predict(x_test)

'''
max_depth=[2,3,4]
n_estimators=np.arange(400,700,10)
learning_rate=np.arange(0.0007,0.0009,0.00005)
grid = {'max_depth':max_depth, 'n_estimators':n_estimators, 'learning_rate':learning_rate}

xgb_predict = model_searchCV(xgb_model, grid, x_train, y_train, x_test)
'''

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

'''
print("\n未来12个月预测结果：")
print(forecast_df.round(2))
'''


#4.7 将环比预测值换算为同比（上年同月=100）
# 读取完整环比序列
hist_mom = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastmonth=100.csv')['actual'].values[:-1]  #去掉最后1行
hist_yoy = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastyear=100.csv')['actual'].values
test_start_mom = len(hist_mom) - len(y_test)
test_start_yoy = len(hist_yoy) - len(y_test)

def to_yoy(pred_mom):
    """
    利用【上个月真实同比 + 本月预测环比 + 去年同月实际环比】递推预测同比

    参数:
        pred_mom : ndarray, 模型预测的环比序列 (上月=100)
        hist_mom : ndarray, 完整的历史环比序列 (上月=100)
        hist_yoy : ndarray, 完整的历史同比序列 (上年同月=100)
        test_start: int, 预测起点在历史序列中的索引
                    (要求 test_start >= 1, 且 test_start-12 >= 0)
    返回:
        pred_yoy : ndarray, 预测的同比序列 (上年同月=100)
    """
    n = len(pred_mom)
    pred_yoy = np.empty(n, dtype=float)

    for i in range(n):
        idx_mom = test_start_mom + i  # 当前预测月份在历史数组中的绝对位置
        idx_yoy = test_start_yoy + i

        prev_yoy = hist_yoy[idx_yoy - 1]
        mom_last_year = hist_mom[idx_mom - 12]
        pred_yoy[i] = prev_yoy * (pred_mom[i] / mom_last_year)
    return pred_yoy

# 对各模型预测值进行换算
y_test = hist_yoy[test_start_yoy:]
naive_predict = np.roll(y_test, 1)
naive_MAE = mean_absolute_error(y_test, naive_predict)
lr_predict = to_yoy(lr_predict)
rf_predict = to_yoy(rf_predict)
svr_predict = to_yoy(svr_predict)
xgb_predict = to_yoy(xgb_predict)
forecast_mean = to_yoy(np.array(forecast_mean))


#4.8 计算模型各项误差指标
naive_MAE, naive_RMSE, naive_MAPE, naive_MASE = model_metrics('naive', y_test, naive_predict)
lr_MAE, lr_RMSE, lr_MAPE, lr_MASE = model_metrics('lr', y_test, lr_predict)
rf_MAE, rf_RMSE, rf_MAPE, rf_MASE = model_metrics('rf', y_test, rf_predict)
svr_MAE, svr_RMSE, svr_MAPE, svr_MASE = model_metrics('svr', y_test, svr_predict)
xgb_MAE, xgb_RMSE, xgb_MAPE, xgb_MASE = model_metrics('xgb', y_test, xgb_predict)
sarimax_MAE, sarimax_RMSE, sarimax_MAPE, sarimax_MASE = model_metrics('sarimax', y_test, forecast_mean)

#5 特征重要性可视化
#5.1 随机森林模型
# 获取特征重要性（默认基于不纯度减少的 MDI 值）
importances = rf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features_list,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

#print(importance_df)

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

#print(importance_df)

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
import matplotlib.dates as mdates

date = pd.to_datetime(cpi_raw[['year', 'month']].assign(day=1))
y_test_len = len(y_test)
date = date.tail(y_test_len)

interval = 1
start = date.min().replace(day=1)
end = date.max().replace(day=1)
xticks_dates = pd.date_range(start=start, end=end, freq=f'{interval}MS')

# 定义模型列表
models = [
    ('Naive', naive_predict),
    ('Linear Regression', lr_predict),
    ('Random Forest', rf_predict),
    ('SVR', svr_predict),
    ('XGBoost', xgb_predict),
    ('SARIMAX', forecast_mean)
]

fig, axes = plt.subplots(3, 2, figsize=(24, 12))

for idx, (name, pred) in enumerate(models):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    ax.plot(date, y_test, label='Actual', color='blue', lw=1.5)
    ax.plot(date, pred, label='Predicted', color='red', lw=1.5, alpha=0.7)
    ax.set_title(f'{name} Model')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('CPI')

for ax in axes.flat:
    ax.margins(x=0.05)
    ax.set_xticks(xticks_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.3, linestyle='--')

fig.autofmt_xdate(rotation=60)
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

fig, axes = plt.subplots(3, 2, figsize=(24, 12))
axes = axes.flatten()

for idx, (model_name, residuals) in enumerate(residuals_dict.items()):
    ax = axes[idx]
    ax.plot(date, residuals, marker='o', linestyle='-', color='darkorange', markersize=4, linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)  # 参考零线
    ax.set_title(f'{model_name} 残差', fontsize=12)
    ax.set_xlabel('日期')
    ax.set_ylabel('残差 (实际 - 预测)')
    ax.grid(True, linestyle=':', alpha=0.6)
    # 显示残差基本统计量
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.text(0.02, 0.95, f'均值: {mean_res:.4f}\n标准差: {std_res:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

for ax in axes:
    ax.margins(x=0.05)
    ax.set_xticks(xticks_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

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

'''
print("\n各模型趋势预测准确率：")
for name, acc in trend_accuracy.items():
    print(f"{name}: {acc:.2f}%")
'''

# 10 生成预测结果对比表格
results_df = pd.DataFrame({
    'actual': y_test,
    'Naive': naive_predict,
    'Ridge': lr_predict,
    'RandomForest': rf_predict,
    'SVR': svr_predict,
    'XGBoost': xgb_predict,
    'SARIMAX': forecast_mean
})

#print("\n各模型预测值对比（同比，上年同月=100）：")
#print(results_df)

# 可选：保存为CSV
#results_df.to_csv(TABLES_DIR / "baseline_predictions.csv", index=False)

# 11 生成预测误差对比表格
model_error_dict = {
    'Naive':       [naive_MAE, naive_RMSE, naive_MAPE, naive_MASE],
    'Ridge':       [lr_MAE,   lr_RMSE,   lr_MAPE,   lr_MASE],
    'RandomForest':[rf_MAE,   rf_RMSE,   rf_MAPE,   rf_MASE],
    'SVR':         [svr_MAE,  svr_RMSE,  svr_MAPE,  svr_MASE],
    'XGBoost':     [xgb_MAE,  xgb_RMSE,  xgb_MAPE,  xgb_MASE],
    'SARIMAX':     [sarimax_MAE, sarimax_RMSE, sarimax_MAPE, sarimax_MASE],
}

# 转换为 DataFrame，模型名作为行索引
error_df = pd.DataFrame(model_error_dict).T.reset_index()
error_df.columns = ['model', 'MAE', 'RMSE', 'MAPE', 'MASE']

# MAPE 转为百分比形式（原值为小数，例如 0.0123 表示 1.23%）
#error_df['MAPE'] = error_df['MAPE'] * 100

print("\n各模型预测误差对比（同比值，上年同月=100）：")
print(error_df)

# 如需保存为 CSV
#error_df.to_csv(TABLES_DIR / "baseline_results.csv", index=False)