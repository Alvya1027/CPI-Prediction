from config import DATA_PROCESSED_DIR, TABLES_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#1 数据加载
#cpi_raw_yoy = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastyear=100.csv')
cpi_raw_mom = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastmonth=100.csv')

#cpi_data = cpi_raw_yoy.iloc[183:]
cpi_data = cpi_raw_mom.iloc[155:297]
#print(cpi_data)

#2 数据预处理
cpi_data_processed = pd.get_dummies(cpi_data, columns=['month'])
cpi = np.array(cpi_data_processed['actual'])
features = cpi_data_processed.drop('actual', axis=1)
features_list = list(features.columns)

#3 数据划分和标准化
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
x_train, x_val_test, y_train, y_val_test = train_test_split(features, cpi, test_size=0.647, shuffle=False)
#print(y_val_test)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.51, shuffle=False)
#print(y_val)
#print(y_test)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

x_train_standard = scaler_X.fit_transform(x_train)
x_test_standard = scaler_X.transform(x_test)
x_val_standard = scaler_X.transform(x_val)
y_train_standard = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

#4 机器学习模型
#4.1 naive模型
def naive_model(data_type, y_val_or_test):
    if data_type == 'val':
        # 拼接训练集最后值与y_val前n-1个值，作为预测
        naive_predict = np.concatenate([[y_train[-1]], y_val_or_test[:-1]])
    elif data_type == 'test':
        # 拼接验证集最后值与y_test前n-1个值，作为预测
        naive_predict = np.concatenate([[y_val[-1]], y_val_or_test[:-1]])
    return naive_predict

#4.2 线性回归模型
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline

'''
#lr = LinearRegression()
#lr = Lasso(alpha = 0.0022)
lr = Ridge(alpha = 2.403)
lr.fit(x_train, y_train)
'''

'''
#lr = LinearRegression()
#lr = Lasso(alpha = 0.0277)
lr = Ridge(alpha = 15.9777)
lr.fit(x_train_standard, y_train_standard)
'''


# 定义 Pipeline：先标准化，再训练 lr
lr = Pipeline([
    ('scaler', StandardScaler()),          # 标准化特征
    ('lr', Ridge(alpha = 15.9777))
])
lr.fit(x_train, y_train)


#4.3 随机森林模型
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=195, max_depth=6, min_samples_split=2,
                           min_samples_leaf=1, max_features=0.1201, random_state=42)
rf.fit(x_train, y_train)

#4.4 支持向量机模型
from sklearn.svm import SVR
#from sklearn.kernel_ridge import KernelRidge

'''
svr = SVR(kernel='rbf', gamma='scale', C=330000, epsilon=0.164)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train, y_train)
'''

'''
svr = SVR(kernel='rbf', gamma=0.0003569, C=19.0967, epsilon=0.05413)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train_standard, y_train_standard)
'''

# 定义 Pipeline：先标准化，再训练 SVR
svr = Pipeline([
    ('scaler', StandardScaler()),          # 标准化特征
    ('svr', SVR(kernel='rbf', gamma=0.0003569, C=19.0967, epsilon=0.05413))             # RBF 核 SVR
])
svr.fit(x_train, y_train)

#4.5 XGBoost模型
import xgboost as xgb

xgb_model = xgb.XGBRegressor(max_depth=7, n_estimators=227, learning_rate=0.047229, random_state=42, objective='reg:squarederror',
                             colsample_bytree=0.32909, subsample=0.37373)
xgb_model.fit(x_train, y_train)

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
lb_val = acorr_ljungbox(final_results.resid, lags=12, return_df=True)
'''
print("\n残差白噪声检验 (p值全部 > 0.05 即代表模型合格)：")
print(lb_val)
print("\n检验结论：所有 p 值均远大于 0.05，残差为纯随机白噪声，模型通过验证！")
'''

# ========== 未来预测 ==========
forecast_steps = len(y_val) + len(y_test)
forecast_result = final_results.get_forecast(steps=forecast_steps)
sarimax_total_predict = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()  # 95%置信区间

# 构造预测时间索引
forecast_index = pd.date_range(
    start=y_train_series.index[-1] + pd.DateOffset(months=1),
    periods=forecast_steps,
    freq='MS'
)

# 整理预测结果为DataFrame
forecast_df = pd.DataFrame({
    '预测值': sarimax_total_predict.values,
    '置信区间下界': forecast_ci.iloc[:, 0].values,
    '置信区间上界': forecast_ci.iloc[:, 1].values
}, index=forecast_index)

'''
print("\n未来12个月预测结果：")
print(forecast_df.round(2))
'''

# 4.7 模型环比预测
# 验证集环比预测
naive_val_predict = naive_model('val', y_val)
#lr_val_predict = lr.predict(x_val_standard)
#lr_val_predict = scaler_y.inverse_transform(lr_val_predict.reshape(-1, 1)).ravel()
lr_val_predict = lr.predict(x_val)
rf_val_predict= rf.predict(x_val)
svr_val_predict = svr.predict(x_val)
xgb_val_predict = xgb_model.predict(x_val)
sarimax_val_predict = np.array(sarimax_total_predict[:len(y_val)])

# 测试集环比预测
naive_test_predict = naive_model('test', y_test)
#lr_test_predict = lr.predict(x_test_standard)
#lr_test_predict = scaler_y.inverse_transform(lr_test_predict.reshape(-1, 1)).ravel()
lr_test_predict = lr.predict(x_test)
rf_test_predict= rf.predict(x_test)
svr_test_predict = svr.predict(x_test)
xgb_test_predict = xgb_model.predict(x_test)
sarimax_test_predict = np.array(sarimax_total_predict[len(y_val):])

# 4.8 模型超参数优化
from scipy.stats import loguniform, randint, uniform

def model_searchCV(search_type, model, model_param, x_train, y_train, x_val, x_test):
    tscv = TimeSeriesSplit(n_splits=5, max_train_size=24)  # 可根据数据量调整
    if search_type == 'random':
        model_searchCV = RandomizedSearchCV(
            estimator=model,
            param_distributions=model_param,
            n_iter=10, cv=tscv,
            scoring='neg_mean_squared_error',
            #random_state=42,
            verbose=1,
            n_jobs=-1
        )
    elif search_type == 'grid':
        model_searchCV = GridSearchCV(
            estimator=model,
            param_grid=model_param,
            cv=tscv,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
    model_searchCV.fit(x_train, y_train)
    print(model_searchCV.best_params_)
    print(f"Best MSE: {model_searchCV.best_score_:.4f}")
    model_val_predict = model_searchCV.predict(x_val)
    model_test_predict = model_searchCV.predict(x_test)
    return model_val_predict, model_test_predict

# 4.8.1 线性回归模型
'''
ridge_param_dist = {
    'alpha': loguniform(1e-3, 1e2)   # 对数均匀分布，覆盖0.001~100
}
lr_val_predict = model_searchCV('random', lr, ridge_param_dist, x_train, y_train, x_val)
'''

'''
alpha=np.arange(0.01,0.03,0.0001)
grid = {'alpha':alpha}
lr_val_predict_standard = model_searchCV('random', lr, grid, x_train_standard, y_train_standard, x_val_standard)
lr_val_predict = scaler_y.inverse_transform(lr_val_predict_standard.reshape(-1, 1)).ravel()
'''

'''
# 定义 Pipeline：先标准化，再训练 lr
lr = Pipeline([
    ('scaler', StandardScaler()),          # 标准化特征
    ('lr', Ridge(alpha = 14.5282))
])

ridge_param_dist = {
    'lr__alpha': loguniform(1e-3, 1e2)   # 对数均匀分布，覆盖0.001~100
}
lr_val_predict, lr_test_predict = model_searchCV('random', lr, ridge_param_dist, x_train, y_train, x_val, x_test)
'''

'''
ridge_param_dist = {
    'lr__alpha': np.arange(15.8,16,0.0001)
}
lr_val_predict, lr_test_predict = model_searchCV('grid', lr, ridge_param_dist, x_train, y_train, x_val, x_test)
'''

# 4.8.2 随机森林模型
'''
rf_param_dist = {
    'n_estimators': randint(50, 500),   # 整数均匀分布
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.01, 0.9)  # 均匀分布，0.3~1.0
}
rf_val_predict, rf_test_predict = model_searchCV('random', rf, rf_param_dist, x_train, y_train, x_val, x_test)
'''

'''
rf_param_dist = {
    'n_estimators': np.arange(50,70,1),
    'max_depth': np.arange(3,6,1),
    'min_samples_split': np.arange(2,3,1),
    'min_samples_leaf': np.arange(2,4,1),
    'max_features': np.arange(0.36,0.45,0.01)
}
rf_val_predict, rf_test_predict = model_searchCV('grid', rf, rf_param_dist, x_train, y_train, x_val, x_test)
'''

# 4.8.3 支持向量机模型
'''
C = np.arange(300000,350000,1000)
#gamma = [0.001, 0.01, 0.1, 1, 10]
epsilon = np.arange(0.1,0.2,0.001)
grid = {'C':C, 'epsilon':epsilon}
svr_val_predict, svr_test_predict = model_searchCV('random', svr, grid, x_train, y_train, x_val, x_test)
'''

'''
C = [1, 10, 100]
gamma = [0.001, 0.01, 0.1, 1, 10]
epsilon = [0.001, 0.01, 0.1, 0.2]
grid = {'C':C, 'gamma':gamma, 'epsilon':epsilon}

#svr_val_predict = model_searchCV('random', svr, grid, x_train_standard, y_train, x_val_standard)
svr_val_predict_standard, svr_test_predict_standard = model_searchCV('random', svr, grid, x_train_standard, y_train_standard, x_val_standard, x_test_standard)
svr_val_predict = scaler_y.inverse_transform(svr_val_predict_standard.reshape(-1, 1)).ravel()
svr_test_predict = scaler_y.inverse_transform(svr_test_predict_standard.reshape(-1, 1)).ravel()
'''

'''
# 定义 Pipeline：先标准化，再训练 SVR
svr = Pipeline([
    ('scaler', StandardScaler()),  # 标准化特征
    ('svr', SVR(kernel='rbf', gamma=0.0003569, C=19.0967, epsilon=0.05413))  # RBF 核 SVR
])

# 定义超参数搜索空间（注意加前缀 'svr__'）
svr_param_dist = {
    'svr__C': loguniform(1, 1e5),
    'svr__gamma': loguniform(1e-4, 1e0),   # 也可包含 'scale', 'auto'
    'svr__epsilon': uniform(0.01, 0.2)
}
svr_val_predict, svr_test_predict = model_searchCV('random', svr, svr_param_dist, x_train, y_train, x_val, x_test)
'''

# 4.8.4 XGBoost模型
'''
xgb_param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(2, 10),
    'learning_rate': loguniform(1e-4, 0.1),
    'subsample': uniform(0.1, 0.5),   # 0.6~1.0
    'colsample_bytree': uniform(0.1, 0.5)
}
xgb_val_predict, xgb_test_predict = model_searchCV('random', xgb_model, xgb_param_dist, x_train, y_train, x_val, x_test)
'''


#4.9 数据同比换算（上年同月=100）
# 计算测试集和验证集在历史数据中的起始位置
hist_yoy = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastyear=100.csv')['actual'].values
hist_mom = cpi_raw_mom['actual'].values[:-1]  #去掉最后1行
test_start_mom = len(hist_mom) - len(y_test)
test_start_yoy = len(hist_yoy) - len(y_test)
val_start_mom = test_start_mom - len(y_val)
val_start_yoy = test_start_yoy - len(y_val)

# 验证集同比换算
def to_yoy_val(pred_mom):
    """
    利用【上个月真实同比+本月预测环比+去年同月实际环比】递推预测同比

    参数:
        pred_mom : ndarray, 模型预测的环比序列 (上月=100)
        hist_mom : ndarray, 完整的历史环比序列 (上月=100)
        hist_yoy : ndarray, 完整的历史同比序列 (上年同月=100)
        val_start: int, 预测起点在历史序列中的索引
                    (要求 test_start >= 1, 且 test_start-12 >= 0)
    返回:
        pred_yoy : ndarray, 预测的同比序列 (上年同月=100)
    """
    n = len(pred_mom)
    pred_yoy = np.empty(n, dtype=float)

    for i in range(n):
        idx_mom = val_start_mom + i
        idx_yoy = val_start_yoy + i
        prev_yoy = hist_yoy[idx_yoy - 1]
        mom_last_year = hist_mom[idx_mom - 12]
        pred_yoy[i] = prev_yoy * (pred_mom[i] / mom_last_year)
    return pred_yoy

# 对各模型预测值进行换算
y_val = hist_yoy[val_start_yoy:val_start_yoy + len(y_val)]
naive_val_predict = np.concatenate([[hist_yoy[187]], y_val[:-1]]) # 取训练集最后一个元素
lr_val_predict = to_yoy_val(lr_val_predict)
rf_val_predict = to_yoy_val(rf_val_predict)
svr_val_predict = to_yoy_val(svr_val_predict)
xgb_val_predict = to_yoy_val(xgb_val_predict)
sarimax_val_predict = to_yoy_val(sarimax_val_predict)

# 测试集同比换算
def to_yoy(pred_mom):
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
naive_test_predict = np.concatenate([[hist_yoy[232]], y_test[:-1]]) # 取验证集最后一个元素
lr_test_predict = to_yoy(lr_test_predict)
rf_test_predict = to_yoy(rf_test_predict)
svr_test_predict = to_yoy(svr_test_predict)
xgb_test_predict = to_yoy(xgb_test_predict)
sarimax_test_predict = to_yoy(sarimax_test_predict)


#4.10 计算模型各项误差指标
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

def model_metrics(data_type, model_type, y_val_or_test, y_predict):
    MAE = mean_absolute_error(y_val_or_test, y_predict)
    RMSE = root_mean_squared_error(y_val_or_test, y_predict)
    MAPE = mean_absolute_percentage_error(y_val_or_test, y_predict)
    if data_type == 'val':
        MASE = MAE / naive_val_MAE
    elif data_type == 'test':
        MASE = MAE / naive_test_MAE
    print(f"{model_type}_MAE:{MAE:.4f}, {model_type}_RMSE:{RMSE:.4f}, "
          f"{model_type}_MAPE:{(MAPE * 100):.3f}%, {model_type}_MASE:{(MASE * 100):.2f}%")
    return MAE, RMSE, MAPE, MASE

#验证集各项误差指标
print("验证集各项误差指标：")
naive_val_MAE = mean_absolute_error(y_val, naive_val_predict)
naive_val_MAE, naive_val_RMSE, naive_val_MAPE, naive_val_MASE = model_metrics('val', 'naive', y_val, naive_val_predict)
lr_val_MAE, lr_val_RMSE, lr_val_MAPE, lr_val_MASE = model_metrics('val', 'lr', y_val, lr_val_predict)
rf_val_MAE, rf_val_RMSE, rf_val_MAPE, rf_val_MASE = model_metrics('val', 'rf', y_val, rf_val_predict)
svr_val_MAE, svr_val_RMSE, svr_val_MAPE, svr_val_MASE = model_metrics('val', 'svr', y_val, svr_val_predict)
xgb_val_MAE, xgb_val_RMSE, xgb_val_MAPE, xgb_val_MASE = model_metrics('val', 'xgb', y_val, xgb_val_predict)
sarimax_val_MAE, sarimax_val_RMSE, sarimax_val_MAPE, sarimax_val_MASE = model_metrics('val', 'sarimax', y_val, sarimax_val_predict)

#测试集各项误差指标
print("测试集各项误差指标：")
naive_test_MAE = mean_absolute_error(y_test, naive_test_predict)
naive_test_MAE, naive_test_RMSE, naive_test_MAPE, naive_test_MASE = model_metrics('test', 'naive', y_test, naive_test_predict)
lr_test_MAE, lr_test_RMSE, lr_test_MAPE, lr_test_MASE = model_metrics('test', 'lr', y_test, lr_test_predict)
rf_test_MAE, rf_test_RMSE, rf_test_MAPE, rf_test_MASE = model_metrics('test', 'rf', y_test, rf_test_predict)
svr_test_MAE, svr_test_RMSE, svr_test_MAPE, svr_test_MASE = model_metrics('test', 'svr', y_test, svr_test_predict)
xgb_test_MAE, xgb_test_RMSE, xgb_test_MAPE, xgb_test_MASE = model_metrics('test', 'xgb', y_test, xgb_test_predict)
sarimax_test_MAE, sarimax_test_RMSE, sarimax_test_MAPE, sarimax_test_MASE = model_metrics('test', 'sarimax', y_test, sarimax_test_predict)


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

#6 测试集：各模型预测值与实际值可视化
import matplotlib.dates as mdates

date_test = pd.to_datetime(cpi_data[['year', 'month']].assign(day=1)).tail(len(y_test))

interval = 1
start = date_test.min().replace(day=1)
end = date_test.max().replace(day=1)
xticks_dates_test = pd.date_range(start=start, end=end, freq=f'{interval}MS')

models_test = [
    ('Naive', naive_test_predict),
    ('Linear Regression', lr_test_predict),
    ('Random Forest', rf_test_predict),
    ('SVR', svr_test_predict),
    ('XGBoost', xgb_test_predict),
    ('SARIMAX', sarimax_test_predict),
]

fig, axes = plt.subplots(3, 2, figsize=(24, 12))

for idx, (name, pred) in enumerate(models_test):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    ax.plot(date_test, y_test, label='Actual', color='blue', lw=1.5)
    ax.plot(date_test, pred, label='Predicted', color='red', lw=1.5, alpha=0.7)
    ax.set_title(name)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('CPI (MoM)')

for ax in axes.flat:
    ax.margins(x=0.05)
    ax.set_xticks(xticks_dates_test)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.3, linestyle='--')

fig.autofmt_xdate(rotation=60)
plt.tight_layout()
plt.show()

#7 测试集：实际值与预测值差值（残差）曲线
residuals_dict = {
    'Naive': y_test - naive_test_predict,
    'LR': y_test - lr_test_predict,
    'RF': y_test - rf_test_predict,
    'SVR': y_test - svr_test_predict,
    'XGBoost': y_test - xgb_test_predict,
    'SARIMAX': y_test - sarimax_test_predict
}

fig, axes = plt.subplots(3, 2, figsize=(24, 12))
axes = axes.flatten()

for idx, (model_name, residuals) in enumerate(residuals_dict.items()):
    ax = axes[idx]
    ax.plot(date_test, residuals, marker='o', linestyle='-', color='darkorange', markersize=4, linewidth=1)
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
    ax.set_xticks(xticks_dates_test)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')

plt.tight_layout()
plt.show()

#8 测试集：各模型MAE/RMSE/MAPE/MASE对比图
metrics_dict = {
    'Naive':   [naive_test_MAE, naive_test_RMSE, naive_test_MAPE, naive_test_MASE],
    'LR':      [lr_test_MAE, lr_test_RMSE, lr_test_MAPE, lr_test_MASE],
    'RF':      [rf_test_MAE, rf_test_RMSE, rf_test_MAPE, rf_test_MASE],
    'SVR':     [svr_test_MAE, svr_test_RMSE, svr_test_MAPE, svr_test_MASE],
    'XGBoost': [xgb_test_MAE, xgb_test_RMSE, xgb_test_MAPE, xgb_test_MASE],
    'SARIMAX': [sarimax_test_MAE, sarimax_test_RMSE, sarimax_test_MAPE, sarimax_test_MASE],
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

#9 测试集：趋势预测正确性可视化
# 计算实际CPI的变化方向
actual_diff = np.diff(y_test)
actual_direction = np.sign(actual_diff)   # 1:上升, -1:下降, 0:持平（长度 = len(y_test)-1）

# 各模型预测值
model_predictions = {
    'Naive': naive_test_predict,
    'LR': lr_test_predict,
    'RF': rf_test_predict,
    'SVR': svr_test_predict,
    'XGBoost': xgb_test_predict,
    'SARIMAX': sarimax_test_predict
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

# 10 测试集：生成预测结果对比表格
results_df = pd.DataFrame({
    'actual': y_test,
    'Naive': naive_test_predict,
    'Ridge': lr_test_predict,
    'RandomForest': rf_test_predict,
    'SVR': svr_test_predict,
    'XGBoost': xgb_test_predict,
    'SARIMAX': sarimax_test_predict
})

#print("\n各模型预测值对比（同比，上年同月=100）：")
#print(results_df)

# 可选：保存为CSV
#results_df.to_csv(TABLES_DIR / "baseline_predictions.csv", index=False)

# 11 测试集：生成预测误差对比表格
model_error_dict = {
    'Naive':       [naive_test_MAE, naive_test_RMSE, naive_test_MAPE, naive_test_MASE],
    'Ridge':       [lr_test_MAE,   lr_test_RMSE,   lr_test_MAPE,   lr_test_MASE],
    'RandomForest':[rf_test_MAE,   rf_test_RMSE,   rf_test_MAPE,   rf_test_MASE],
    'SVR':         [svr_test_MAE,  svr_test_RMSE,  svr_test_MAPE,  svr_test_MASE],
    'XGBoost':     [xgb_test_MAE,  xgb_test_RMSE,  xgb_test_MAPE,  xgb_test_MASE],
    'SARIMAX':     [sarimax_test_MAE, sarimax_test_RMSE, sarimax_test_MAPE, sarimax_test_MASE],
}

# 转换为 DataFrame，模型名作为行索引
error_df = pd.DataFrame(model_error_dict).T.reset_index()
error_df.columns = ['model', 'MAE', 'RMSE', 'MAPE', 'MASE']

# MAPE 转为百分比形式（原值为小数，例如 0.0123 表示 1.23%）
#error_df['MAPE'] = error_df['MAPE'] * 100

print("\n测试集各模型预测误差对比（同比值，上年同月=100）：")
print(error_df)

# 如需保存为 CSV
#error_df.to_csv(TABLES_DIR / "baseline_results.csv", index=False)

#12 验证集：各模型预测值与实际值可视化
import matplotlib.dates as mdates

date_val = pd.to_datetime(cpi_data[['year', 'month']].assign(day=1)).iloc[len(y_train):len(y_train) + len(y_val)]

start = date_val.min().replace(day=1)
end = date_val.max().replace(day=1)
xticks_dates_val = pd.date_range(start=start, end=end, freq=f'1MS')

models_val = [
    ('Naive', naive_val_predict),
    ('Linear Regression', lr_val_predict),
    ('Random Forest', rf_val_predict),
    ('SVR', svr_val_predict),
    ('XGBoost', xgb_val_predict),
    ('SARIMAX', sarimax_val_predict),
]

fig, axes = plt.subplots(3, 2, figsize=(24, 12))

for idx, (name, pred) in enumerate(models_val):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    ax.plot(date_val, y_val, label='Actual', color='blue', lw=1.5)
    ax.plot(date_val, pred, label='Predicted', color='red', lw=1.5, alpha=0.7)
    ax.set_title(f'{name} - Validation Set')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('CPI (YoY)')

for ax in axes.flat:
    ax.margins(x=0.05)
    ax.set_xticks(xticks_dates_val)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(True, alpha=0.3, linestyle='--')

fig.autofmt_xdate(rotation=60)
plt.tight_layout()
plt.show()

#13 验证集：各模型MAE/RMSE/MAPE/MASE对比图
val_metrics_dict = {
    'Naive':   [naive_val_MAE, naive_val_RMSE, naive_val_MAPE, naive_val_MASE],
    'LR':      [lr_val_MAE,   lr_val_RMSE,   lr_val_MAPE,   lr_val_MASE],
    'RF':      [rf_val_MAE,   rf_val_RMSE,   rf_val_MAPE,   rf_val_MASE],
    'SVR':     [svr_val_MAE,  svr_val_RMSE,  svr_val_MAPE,  svr_val_MASE],
    'XGBoost': [xgb_val_MAE,  xgb_val_RMSE,  xgb_val_MAPE,  xgb_val_MASE],
    'SARIMAX': [sarimax_val_MAE, sarimax_val_RMSE, sarimax_val_MAPE, sarimax_val_MASE],
}
metric_names = ['验证集：MAE', '验证集：RMSE', '验证集：MAPE', '验证集：MASE']
val_model_names = list(val_metrics_dict.keys())
colors = plt.cm.Set2(np.linspace(0, 1, len(val_model_names)))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (ax, metric_name) in enumerate(zip(axes.flat, metric_names)):
    values = [val_metrics_dict[model][idx] for model in val_model_names]
    bars = ax.bar(val_model_names, values, color=colors, edgecolor='gray', linewidth=0.8)
    ax.set_title(metric_name, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f}' if metric_name != 'MAPE' and metric_name != 'MASE' else f'{v*100:.3f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

#14 验证集：趋势预测正确性可视化
actual_diff_val = np.diff(y_val)
actual_direction_val = np.sign(actual_diff_val)

val_model_preds = {
    'Naive': naive_val_predict,
    'LR': lr_val_predict,
    'RF': rf_val_predict,
    'SVR': svr_val_predict,
    'XGBoost': xgb_val_predict,
    'SARIMAX': sarimax_val_predict,
}

trend_accuracy_val = {}
for name, pred in val_model_preds.items():
    pred_diff = np.diff(pred)
    pred_direction = np.sign(pred_diff)
    correct = (actual_direction_val == pred_direction)
    accuracy = np.mean(correct) * 100
    trend_accuracy_val[name] = accuracy

plt.figure(figsize=(10, 6))
val_model_names_list = list(trend_accuracy_val.keys())
accuracies_val = list(trend_accuracy_val.values())
colors_val = plt.cm.Set3(np.linspace(0, 1, len(val_model_names_list)))

bars = plt.bar(val_model_names_list, accuracies_val, color=colors_val, edgecolor='black', linewidth=0.8)
plt.ylabel('趋势预测准确率 (%)', fontsize=12)
plt.title('验证集：各模型趋势方向（上升/下降）预测准确率对比', fontsize=14)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar, acc in zip(bars, accuracies_val):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

'''
print("\n验证集各模型趋势预测准确率：")
for name, acc in trend_accuracy_val.items():
    print(f"{name}: {acc:.2f}%")
'''