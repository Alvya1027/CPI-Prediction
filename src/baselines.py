from config import DATA_PROCESSED_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#1 数据加载
#cpi_data = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastyear=100.csv')
cpi_data = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data_lastmonth=100.csv')

#2 数据预处理
cpi_data = pd.get_dummies(cpi_data, columns=['month'])
cpi = np.array(cpi_data['actual'])
features = cpi_data.drop('actual', axis=1)
features_list = list(features.columns)
features = np.array(features)

#3 数据划分
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(features, cpi, test_size=0.1, shuffle=False)

transfer = StandardScaler()
x_train_standard = transfer.fit_transform(x_train)
x_test_standard = transfer.transform(x_test)
y_train_standard = transfer.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_standard = transfer.transform(y_test.reshape(-1, 1)).ravel()

#4 机器学习模型
#4.1 naive模型
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

naive_predict = np.roll(y_test, 1) # 简单用上期值预测当期
naive_MAE = mean_absolute_error(y_test, naive_predict)
naive_RMSE = root_mean_squared_error(y_test, naive_predict)
naive_MAPE = mean_absolute_percentage_error(y_test, naive_predict)
print(f"naive_MAE:{naive_MAE}, naive_RMSE:{naive_RMSE}, naive_MAPE:{naive_MAPE}")

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

lr_MAE = mean_absolute_error(y_test, lr_predict)
lr_RMSE = root_mean_squared_error(y_test, lr_predict)
lr_MAPE = mean_absolute_percentage_error(y_test, lr_predict)
print(f"lr_MAE:{lr_MAE}, lr_RMSE:{lr_RMSE}, lr_MAPE:{lr_MAPE}")

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

rf_MAE = mean_absolute_error(y_test, rf_predict)
rf_RMSE = root_mean_squared_error(y_test, rf_predict)
rf_MAPE = mean_absolute_percentage_error(y_test, rf_predict)
print(f"rf_MAE:{rf_MAE}, rf_RMSE:{rf_RMSE}, rf_MAPE:{rf_MAPE}")

# 各特征重要性
importances = list(rf.feature_importances_)
feature_importances = [(features, round(importance, 2)) for features, importance in zip(features_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#4.4 支持向量机模型
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

'''
svr = SVR(kernel='rbf', gamma='scale', C=1.291, epsilon=0.637)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train_standard, y_train_standard)

svr_predict_standard = svr.predict(x_test_standard)
svr_predict = transfer.inverse_transform(svr_predict_standard.reshape(-1, 1)).ravel()
'''

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


svr = SVR(kernel='rbf', gamma='scale', C=330000, epsilon=0.164)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train, y_train)

svr_predict = svr.predict(x_test)


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

svr_MAE = mean_absolute_error(y_test, svr_predict)
svr_RMSE = root_mean_squared_error(y_test, svr_predict)
svr_MAPE = mean_absolute_percentage_error(y_test, svr_predict)
print(f"svr_MAE:{svr_MAE}, svr_RMSE:{svr_RMSE}, svr_MAPE:{svr_MAPE}")

#4.5 XGBoost模型
import xgboost as xgb

xgb = xgb.XGBRegressor(max_depth=2, n_estimators=31, learning_rate=0.3249, random_state=42, objective='reg:squarederror')
xgb.fit(x_train, y_train)

xgb_predict = xgb.predict(x_test)

'''
max_depth=[1,2,3]
n_estimators=np.arange(25,35,1)
learning_rate=[0.3249]
grid = {'max_depth':max_depth, 'n_estimators':n_estimators, 'learning_rate':learning_rate}

xgb_searchCV = GridSearchCV(estimator=xgb, param_grid=grid, cv=3,
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=8)
xgb_searchCV.fit(x_train, y_train)
print(xgb_searchCV.best_params_)
xgb_predict = xgb_searchCV.predict(x_test)
'''

xgb_MAE = mean_absolute_error(y_test, xgb_predict)
xgb_RMSE = root_mean_squared_error(y_test, xgb_predict)
xgb_MAPE = mean_absolute_percentage_error(y_test, xgb_predict)
print(f"xgb_MAE:{xgb_MAE}, xgb_RMSE:{xgb_RMSE}, xgb_MAPE:{xgb_MAPE}")

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

sarimax_MAE = mean_absolute_error(y_test, forecast_mean)
sarimax_RMSE = root_mean_squared_error(y_test, forecast_mean)
sarimax_MAPE = mean_absolute_percentage_error(y_test, forecast_mean)
print(f"sarimax_MAE:{sarimax_MAE}, sarimax_RMSE:{sarimax_RMSE}, sarimax_MAPE:{sarimax_MAPE}")

'''
print("\n未来12个月预测结果：")
print(forecast_df.round(2))
'''

# ========== 可视化（最终报告图） ==========
plt.figure(figsize=(14, 5))

# 历史数据
plt.plot(y_train_series.index, y_train_series,
         label='历史数据 (均值 ≈ 100)', color='steelblue', linewidth=1.5)

# 预测值
plt.plot(forecast_df.index, forecast_df['预测值'],
         label='预测值 (均值回归)', color='darkred', linestyle='--', linewidth=2)

# 置信区间
plt.fill_between(forecast_df.index,
                 forecast_df['置信区间下界'],
                 forecast_df['置信区间上界'],
                 color='pink', alpha=0.5, label='95% 预测置信区间')

# 添加一条基准线 y=100 作为视觉参考
plt.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='基准线 (100)')

plt.title(f'SARIMAX模型 (数据标准差={round(y_train_series.std(), 3)})', fontsize=14)
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#5 各模型预测值与实际值可视化
x_seq = np.arange(len(y_test))

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

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

plt.tight_layout()
plt.show()