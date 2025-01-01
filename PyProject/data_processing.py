import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('TkAgg')  # 设置后端为 TkAgg
# 设置字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
data_train = pd.read_csv('train.csv')  # 确保路径正确，去除多余空格
data_test = pd.read_csv('test.csv')

# 查看数据基本信息
print("训练集头部数据:", data_train.head())
print("测试集头部数据:", data_test.head())
print("训练集列名:", data_train.columns)
print("训练集每列的数据格式:")
print(data_train.info())

# 缺失值处理
missing_values = data_train.isnull().sum()
missing_ratio = (missing_values / len(data_train)) * 100
drop_columns = missing_ratio[missing_ratio > 50].index
data_train = data_train.drop(columns=drop_columns)
data_test = data_test.drop(columns=drop_columns)

class_variable = [col for col in data_train.columns if data_train[col].dtypes == 'O']
numerical_variable = [col for col in data_train.columns if data_train[col].dtypes != 'O' and col != 'SalePrice']

imputer = SimpleImputer(strategy='median')
data_train[numerical_variable] = imputer.fit_transform(data_train[numerical_variable])
data_test[numerical_variable] = imputer.transform(data_test[numerical_variable])

data_train[class_variable] = data_train[class_variable].fillna('None')
data_test[class_variable] = data_test[class_variable].fillna('None')

data_train = pd.get_dummies(data_train, columns=class_variable, drop_first=True)
data_test = pd.get_dummies(data_test, columns=class_variable, drop_first=True)

data_train, data_test = data_train.align(data_test, join='left', axis=1)
data_test = data_test.fillna(0)  # 测试集可能有部分列缺失，用 0 填充

print("训练集列名：", data_train.columns)
print("测试集列名：", data_test.columns)

# 构造随机森林模型
X = data_train.drop(['SalePrice'], axis=1)  # 特征
y = data_train['SalePrice']  # 目标变量

# 数据集划分（训练集 S 和 验证集 V）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 超参数调优
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None]  # 替换 'auto' 为 None
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)

# 使用最佳参数训练模型
rf_model = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_features=grid_search.best_params_['max_features'],
    random_state=42
)
rf_model.fit(X_train, y_train)

# 模型评估：训练集
rf_preds_train = rf_model.predict(X_train)
rf_r2_train = r2_score(y_train, rf_preds_train)
rf_rmse_train = np.sqrt(mean_squared_error(y_train, rf_preds_train))

# 模型评估：验证集
rf_preds_val = rf_model.predict(X_val)
rf_r2_val = r2_score(y_val, rf_preds_val)
rf_rmse_val = np.sqrt(mean_squared_error(y_val, rf_preds_val))

# 打印结果
print("\n随机森林模型评估结果：")
print(f"训练集 R²（拟合率）: {rf_r2_train:.2f}")
print(f"训练集 RMSE（均方根误差）: {rf_rmse_train:.2f}")
print(f"验证集 R²（准确率）: {rf_r2_val:.2f}")
print(f"验证集 RMSE（均方根误差）: {rf_rmse_val:.2f}")

# 测试集预测
rf_preds_test = rf_model.predict(data_test)

# 保存预测结果到 Predictions.csv
test_ids = pd.read_csv('test.csv')['Id']  # 确保路径正确
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': rf_preds_test})
submission.to_csv('Predictions.csv', index=False)
print("\n测试集预测结果已保存为 Predictions.csv 文件中")
