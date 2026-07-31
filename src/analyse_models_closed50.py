import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 自动路径设置 ====================
# 获取脚本所在目录（假设在 src/ 下）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（脚本的上一级）
base_dir = os.path.dirname(script_dir)

def find_file(filename, search_dir):
    """在 search_dir 及其子目录中递归查找第一个匹配的文件"""
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

# 需要查找的文件列表
file_list = [
    "classical_models_recent50_predictions.csv",
    "classical_models_recent50_metrics.csv",
    "closed50_test_prediction_comparison.csv",
    "closed50_model_comparison.csv"
]

file_paths = {}
for fname in file_list:
    # 首先在根目录查找
    path = os.path.join(base_dir, fname)
    if os.path.exists(path):
        file_paths[fname] = path
    else:
        # 在 results 目录下递归查找
        results_dir = os.path.join(base_dir, "results")
        if os.path.exists(results_dir):
            found = find_file(fname, results_dir)
            if found:
                file_paths[fname] = found
            else:
                raise FileNotFoundError(f"无法找到文件: {fname}，请确认文件在项目目录中。")
        else:
            raise FileNotFoundError(f"无法找到 results 目录: {results_dir}")

print("找到的数据文件:")
for fname, path in file_paths.items():
    print(f"  {fname}: {path}")

# 读取数据
baseline_pred = pd.read_csv(file_paths["classical_models_recent50_predictions.csv"])
baseline_metrics = pd.read_csv(file_paths["classical_models_recent50_metrics.csv"])
reservoir_pred = pd.read_csv(file_paths["closed50_test_prediction_comparison.csv"])
reservoir_metrics = pd.read_csv(file_paths["closed50_model_comparison.csv"])

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# ===== 修改：输出目录指向 closed50 子文件夹 =====
OUTPUT_DIR = os.path.join(base_dir, "results", "figures", "closed50")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 2. 提取测试集预测数据 ====================
ridge_test = baseline_pred[(baseline_pred['model'] == 'ridge') & (baseline_pred['split'] == 'test')].copy()
naive_test = baseline_pred[(baseline_pred['model'] == 'naive_last_value') & (baseline_pred['split'] == 'test')].copy()

ordinary_test = reservoir_pred[['sample_i_id', 'cpi_actual', 'cpi_predicted_ordinary_recent50']].copy()
siamese_test = reservoir_pred[['sample_i_id', 'cpi_actual', 'cpi_predicted_siamese_closed50']].copy()

ridge_test = ridge_test[['sample_id', 'actual', 'predicted']]
naive_test = naive_test[['sample_id', 'actual', 'predicted']]
ordinary_test = ordinary_test.rename(columns={'sample_i_id': 'sample_id', 'cpi_actual': 'actual',
                                              'cpi_predicted_ordinary_recent50': 'predicted'})
siamese_test = siamese_test.rename(columns={'sample_i_id': 'sample_id', 'cpi_actual': 'actual',
                                            'cpi_predicted_siamese_closed50': 'predicted'})

ridge_test = ridge_test.sort_values('sample_id').reset_index(drop=True)
naive_test = naive_test.sort_values('sample_id').reset_index(drop=True)
ordinary_test = ordinary_test.sort_values('sample_id').reset_index(drop=True)
siamese_test = siamese_test.sort_values('sample_id').reset_index(drop=True)

all_test = pd.DataFrame({
    'sample_id': ridge_test['sample_id'],
    'actual': ridge_test['actual'],
    'ridge': ridge_test['predicted'],
    'naive': naive_test['predicted'],
    'ordinary': ordinary_test['predicted'],
    'siamese': siamese_test['predicted']
})

# ==================== 3. 获取指标数据（泛化分析） ====================
ridge_val_rmse = baseline_metrics[(baseline_metrics['model'] == 'ridge') & (baseline_metrics['split'] == 'val')]['rmse'].values[0]
ridge_test_rmse = baseline_metrics[(baseline_metrics['model'] == 'ridge') & (baseline_metrics['split'] == 'test')]['rmse'].values[0]
naive_val_rmse = baseline_metrics[(baseline_metrics['model'] == 'naive_last_value') & (baseline_metrics['split'] == 'val')]['rmse'].values[0]
naive_test_rmse = baseline_metrics[(baseline_metrics['model'] == 'naive_last_value') & (baseline_metrics['split'] == 'test')]['rmse'].values[0]

ordinary_val_rmse = reservoir_metrics[reservoir_metrics['model'] == 'ordinary_optical_reservoir_closed50']['val_rmse'].values[0]
ordinary_test_rmse = reservoir_metrics[reservoir_metrics['model'] == 'ordinary_optical_reservoir_closed50']['test_rmse'].values[0]
siamese_val_rmse = reservoir_metrics[reservoir_metrics['model'] == 'siamese_optical_reservoir_closed50']['val_rmse'].values[0]
siamese_test_rmse = reservoir_metrics[reservoir_metrics['model'] == 'siamese_optical_reservoir_closed50']['test_rmse'].values[0]

generalization = pd.DataFrame({
    'Model': ['Ridge', 'Naive', 'Ordinary Reservoir', 'Siamese Reservoir'],
    'Validation RMSE': [ridge_val_rmse, naive_val_rmse, ordinary_val_rmse, siamese_val_rmse],
    'Test RMSE': [ridge_test_rmse, naive_test_rmse, ordinary_test_rmse, siamese_test_rmse]
})

# ==================== 4. 图表生成 ====================
# 4.1 预测曲线对比
plt.figure(figsize=(12, 6))
plt.plot(all_test['sample_id'], all_test['actual'], 'k-', linewidth=2, label='Actual CPI', marker='o')
plt.plot(all_test['sample_id'], all_test['ridge'], 'b--', label='Ridge', marker='s')
plt.plot(all_test['sample_id'], all_test['naive'], 'g-.', label='Naive (Last Value)', marker='^')
plt.plot(all_test['sample_id'], all_test['ordinary'], 'm:', label='Ordinary Reservoir', marker='d')
plt.plot(all_test['sample_id'], all_test['siamese'], 'r-', label='Siamese Reservoir', marker='x')
plt.xlabel('Sample ID (Test Months)')
plt.ylabel('CPI')
plt.title('CPI Prediction Curves on Test Set (Closed 50 Windows)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_curves.png'), dpi=300)
plt.close()

# 4.2 残差箱线图
residuals = pd.DataFrame({
    'Ridge': all_test['ridge'] - all_test['actual'],
    'Naive': all_test['naive'] - all_test['actual'],
    'Ordinary': all_test['ordinary'] - all_test['actual'],
    'Siamese': all_test['siamese'] - all_test['actual']
})
plt.figure(figsize=(10, 6))
sns.boxplot(data=residuals, palette="Set2")
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
plt.ylabel('Residual (Prediction - Actual)')
plt.title('Residual Distribution of Four Models (Closed 50)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residual_boxplot.png'), dpi=300)
plt.close()

# 4.4 泛化能力（验证 vs 测试 RMSE）
generalization_melted = generalization.melt(id_vars='Model', var_name='Set', value_name='RMSE')
plt.figure(figsize=(10, 6))
sns.barplot(data=generalization_melted, x='Model', y='RMSE', hue='Set', palette='Set1')
plt.title('Validation vs Test RMSE (Generalization Performance, Closed 50)')
plt.ylabel('RMSE')
plt.legend(title='Dataset')
for i, p in enumerate(plt.gca().patches):
    plt.gca().annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom', fontsize=9, rotation=0)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'generalization_rmse.png'), dpi=300)
plt.close()

# 4.5 误差最大的前5个测试月份（按平均绝对误差）
all_test['ridge_abs'] = np.abs(all_test['ridge'] - all_test['actual'])
all_test['naive_abs'] = np.abs(all_test['naive'] - all_test['actual'])
all_test['ordinary_abs'] = np.abs(all_test['ordinary'] - all_test['actual'])
all_test['siamese_abs'] = np.abs(all_test['siamese'] - all_test['actual'])
all_test['avg_abs_error'] = all_test[['ridge_abs', 'naive_abs', 'ordinary_abs', 'siamese_abs']].mean(axis=1)

top5_samples = all_test.nlargest(5, 'avg_abs_error')[['sample_id', 'avg_abs_error']].copy()
top5_ids = top5_samples['sample_id'].values
top5_data = all_test[all_test['sample_id'].isin(top5_ids)].copy()

top5_melted = top5_data.melt(id_vars=['sample_id'],
                             value_vars=['ridge_abs', 'naive_abs', 'ordinary_abs', 'siamese_abs'],
                             var_name='Model', value_name='Absolute Error')
model_map = {'ridge_abs': 'Ridge', 'naive_abs': 'Naive',
             'ordinary_abs': 'Ordinary', 'siamese_abs': 'Siamese'}
top5_melted['Model'] = top5_melted['Model'].map(model_map)

plt.figure(figsize=(12, 6))
sns.barplot(data=top5_melted, x='sample_id', y='Absolute Error', hue='Model', palette='Set2')
plt.title('Top 5 Test Months with Largest Average Absolute Error (Closed 50)')
plt.xlabel('Sample ID')
plt.ylabel('Absolute Error')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top5_errors.png'), dpi=300)
plt.close()

print("Top 5 months with largest average absolute errors:")
print(top5_samples)

# 4.6 方向趋势预测准确率
def directional_accuracy(pred_series, actual_series):
    diffs_actual = np.diff(actual_series)
    diffs_pred = np.diff(pred_series)
    correct = np.sum(np.sign(diffs_actual) == np.sign(diffs_pred))
    total = len(diffs_actual)
    return correct / total

models = {
    'Ridge': all_test['ridge'],
    'Naive': all_test['naive'],
    'Ordinary': all_test['ordinary'],
    'Siamese': all_test['siamese']
}
accuracies = {name: directional_accuracy(pred.values, all_test['actual'].values) for name, pred in models.items()}

acc_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
plt.figure(figsize=(8, 6))
sns.barplot(data=acc_df, x='Model', y='Accuracy', palette='pastel')
plt.ylim(0, 1)
plt.ylabel('Directional Accuracy')
plt.title('Trend Direction Prediction Accuracy (Closed 50)')
for i, row in acc_df.iterrows():
    plt.text(i, row['Accuracy'] + 0.02, f"{row['Accuracy']:.2%}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'directional_accuracy.png'), dpi=300)
plt.close()

# 饼图
plt.figure(figsize=(8, 8))
plt.pie(accuracies.values(), labels=accuracies.keys(), autopct='%1.1f%%', startangle=90)
plt.title('Directional Accuracy Comparison (Closed 50)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'directional_accuracy_pie.png'), dpi=300)
plt.close()

print("All figures have been saved to:", OUTPUT_DIR)