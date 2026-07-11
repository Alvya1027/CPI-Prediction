from config import DATA_PROCESSED_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1 数据加载
cpi_data = pd.read_csv(DATA_PROCESSED_DIR / 'cpi_data.csv')

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
naive_mape = mean_absolute_percentage_error(y_test, naive_predict)
print("naive_mape:{}".format(naive_mape))

#4.2 线性回归模型
from sklearn.linear_model import LinearRegression, Lasso, Ridge

#lr = LinearRegression()
lr = Lasso(alpha = 0.062)
#lr = Ridge(alpha = 2.442)
lr.fit(x_train, y_train)

lr_train_predict = lr.predict(x_train)
lr_predict = lr.predict(x_test)

'''
alpha=np.arange(0.06,0.08,0.001)

grid = {'alpha':alpha}

#lr_searchCV = RandomizedSearchCV(estimator=svr, param_distributions=grid, cv=3,
#                                n_iter=100, scoring='neg_mean_absolute_percentage_error', verbose=2)
lr_searchCV = GridSearchCV(estimator=lr, param_grid=grid, cv=3,
                          scoring='neg_mean_absolute_percentage_error', verbose=2, n_jobs=4)
lr_searchCV.fit(x_train, y_train)
print(lr_searchCV.best_params_)
lr_predict = lr_searchCV.predict(x_test)
'''

lr_mape = mean_absolute_percentage_error(y_test, lr_predict)
print("lr_mape:{}".format(lr_mape))

#4.3 随机森林模型
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=10,
                           min_samples_leaf=4, max_features=0.7, random_state=42)
rf.fit(x_train, y_train)

rf_predict = rf.predict(x_test)

'''
n_estimators=np.arange(70,73,1)
max_depth=[9,10,11]
min_samples_split=[9,10,11]
min_samples_leaf=[3,4,5]
max_features=[0.7]  # 覆盖各种随机程度

grid = {'n_estimators':n_estimators, 'max_depth':max_depth, 'min_samples_split':min_samples_split,
        'min_samples_leaf':min_samples_leaf, 'max_features':max_features}

#rf_searchCV = RandomizedSearchCV(estimator=svr, param_distributions=grid, cv=3,
#                                n_iter=100, scoring='neg_mean_absolute_percentage_error', verbose=2)
rf_searchCV = GridSearchCV(estimator=rf, param_grid=grid, cv=3,
                          scoring='neg_mean_absolute_percentage_error', verbose=2, n_jobs=4)
rf_searchCV.fit(x_train, y_train)
print(rf_searchCV.best_params_)
rf_predict = rf_searchCV.predict(x_test)
'''

rf_mape = mean_absolute_percentage_error(y_test, rf_predict)
print("rf_mape:{}".format(rf_mape))

# 各特征重要性
importances = list(rf.feature_importances_)
feature_importances = [(features, round(importance, 2)) for features, importance in zip(features_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#4.4 支持向量机模型
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

'''
svr = SVR(kernel='rbf', gamma='scale', C=1.26, epsilon=0.2)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train_standard, y_train_standard)

svr_predict_standard = svr.predict(x_test_standard)
svr_predict = transfer.inverse_transform(svr_predict_standard.reshape(-1, 1)).ravel()
'''

svr = SVR(kernel='rbf', gamma='scale', C=510000, epsilon=0.148)
#model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01)
svr.fit(x_train, y_train)

svr_predict = svr.predict(x_test)

'''
C = np.arange(1.2, 1.4, 0.01)
#gamma = [0.001, 0.01, 0.1, 1, 10]
epsilon = np.arange(0.1,0.3,0.01)
grid = {'C':C, 'epsilon':epsilon}

#svr_searchCV = RandomizedSearchCV(estimator=svr, param_distributions=grid, cv=3,
#                                n_iter=100, scoring='neg_mean_absolute_percentage_error', verbose=2)
svr_searchCV = GridSearchCV(estimator=svr, param_grid=grid, cv=3,
                          scoring='neg_mean_absolute_percentage_error', verbose=2)
svr_searchCV.fit(x_train_standard, y_train_standard)
print(svr_searchCV.best_params_)
svr_predict_standard = svr_searchCV.predict(x_test_standard)
svr_predict = transfer.inverse_transform(svr_predict_standard.reshape(-1, 1)).ravel()
'''

svr_mape = mean_absolute_percentage_error(y_test, svr_predict)
print("svr_mape:{}".format(svr_mape))

#4.5 XGBoost模型
import xgboost as xgb

xgb = xgb.XGBRegressor(max_depth=2, n_estimators=725, learning_rate=0.0107, random_state=42, objective='reg:squarederror')
xgb.fit(x_train, y_train)

xgb_predict = xgb.predict(x_test)

'''
max_depth=[1,2,3]
n_estimators=np.arange(720,730,1)
learning_rate=np.arange(0.01,0.011,0.0001)
grid = {'max_depth':max_depth, 'n_estimators':n_estimators, 'learning_rate':learning_rate}

#xgb_searchCV = RandomizedSearchCV(estimator=svr, param_distributions=grid, cv=3,
#                                n_iter=100, scoring='neg_mean_absolute_error', verbose=2)
xgb_searchCV = GridSearchCV(estimator=xgb, param_grid=grid, cv=3,
                          scoring='neg_mean_absolute_error', verbose=2, n_jobs=4)
xgb_searchCV.fit(x_train, y_train)
print(xgb_searchCV.best_params_)
xgb_predict = xgb_searchCV.predict(x_test)
'''

xgb_mape = mean_absolute_percentage_error(y_test, xgb_predict)
print("xgb_mape:{}".format(xgb_mape))

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