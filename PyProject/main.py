import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# 数据类型
data_train_dtypes = data_train.dtypes
print("训练集数据结构为:", data_train_dtypes)

# 统计房价的基本信息
print("\n房价描述统计信息:")
print(data_train['SalePrice'].describe())  # 输出描述性统计信息

# 检查是否有缺失值
print("\n房价列中缺失值数量:")
print(data_train['SalePrice'].isnull().sum())  # 检查缺失值数量

# 查看因变量价格的分布情况
plt.figure(figsize=(10, 6))
sns.histplot(data_train['SalePrice'], kde=True, bins=30)  # distplot
plt.title("房屋价格分布图")
plt.xlabel("房屋价格")
plt.ylabel("频率")
plt.show()

# 按价格统计频率并绘制分布图
price_counts = data_train['SalePrice'].value_counts().reset_index()
price_counts.columns = ['SalePrice', 'Count']

plt.figure(figsize=(12, 6))
sns.barplot(x='SalePrice', y='Count', data=price_counts.head(20))  # 显示频率前20的价格
plt.xticks(rotation=45)
plt.title("房屋价格计数统计 (前20)")
plt.xlabel("房屋价格")
plt.ylabel("频率")
plt.show()

# 缺失值处理逻辑
# 统计缺失值
missing_values = data_train.isnull().sum()
missing_ratio = (missing_values / len(data_train)) * 100
print("\n缺失值统计:")
print(missing_ratio[missing_ratio > 0].sort_values(ascending=False))

# 删除缺失值比例超过50%的列
drop_columns = missing_ratio[missing_ratio > 50].index
print("\n以下列缺失比例超过50%，将被删除：", drop_columns)
data_train = data_train.drop(columns=drop_columns)
data_test = data_test.drop(columns=drop_columns)

# 分类变量和数值变量分离
class_variable = [col for col in data_train.columns if data_train[col].dtypes == 'O']
numerical_variable = [col for col in data_train.columns if data_train[col].dtypes != 'O' and col != 'SalePrice']
print("\n类别型变量:", class_variable)
print("数值型变量:", numerical_variable)

# 填充缺失值
# 数值型变量：用中位数填充
imputer = SimpleImputer(strategy='median')
data_train[numerical_variable] = imputer.fit_transform(data_train[numerical_variable])
data_test[numerical_variable] = imputer.transform(data_test[numerical_variable])

# 类别型变量：填充为 "None"
data_train[class_variable] = data_train[class_variable].fillna('None')
data_test[class_variable] = data_test[class_variable].fillna('None')

# 检查填充后的缺失值情况
print("\n处理后的数据列中缺失值数量：")
print(data_train.isnull().sum().sort_values(ascending=False).head())

# 可视化：房价与某些特征的关系
# CentralAir
plt.figure(figsize=(8, 6))
sns.boxplot(x='CentralAir', y='SalePrice', data=data_train)
plt.title("房价与 CentralAir 的关系")
plt.show()

# MSSubClass
plt.figure(figsize=(10, 6))
sns.boxplot(x='MSSubClass', y='SalePrice', data=data_train)
plt.title("房价与 MSSubClass 的关系")
plt.show()

# MSZoning
plt.figure(figsize=(10, 6))
sns.boxplot(x='MSZoning', y='SalePrice', data=data_train)
plt.title("房价与 MSZoning 的关系")
plt.show()

# SalePrice vs LotArea (散点图)
plt.figure(figsize=(8, 6))
plt.scatter(data_train['SalePrice'], data_train['LotArea'], alpha=0.6)
plt.xlabel('房屋价格')
plt.ylabel('土地面积')
plt.title('房价与土地面积的关系')
plt.show()

# 特征相关性分析
# 仅选择数值型列用于计算相关性
numerical_data_train = data_train.select_dtypes(include=['float64', 'int64'])

# 计算特征与房价的相关性
correlation_matrix = numerical_data_train.corr()

# 选择与目标变量 SalePrice 相关性最高的特征
correlation_with_target = correlation_matrix['SalePrice'].sort_values(ascending=False)
print("与房价相关性最高的特征：\n", correlation_with_target)

# 可视化相关性矩阵
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title("特征相关性热力图")
plt.show()

# 绘制与房价最相关的前10个特征
top_features = correlation_with_target.head(11).index
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_data_train[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("与房价最相关的特征热力图")
plt.show()

# 数据可视化之后再进一步处理数据
# 对类别型变量进行 One-Hot 编码
data_train = pd.get_dummies(data_train, columns=class_variable, drop_first=True)
data_test = pd.get_dummies(data_test, columns=class_variable, drop_first=True)

# 确保训练集和测试集的列一致
data_train, data_test = data_train.align(data_test, join='left', axis=1)
data_test = data_test.fillna(0)  # 测试集可能有部分列缺失，用 0 填充
print("训练集列名：", data_train.columns)
print("测试集列名：", data_test.columns)
print("列名是否一致：", set(data_train.columns) == set(data_test.columns))
print("测试集中是否包含 'SalePrice'：", 'SalePrice' in data_test.columns)

# 构造随机森林模型
X = data_train.drop(['SalePrice'], axis=1)  # 特征
y = data_train['SalePrice']  # 目标变量

# 数据集划分（训练集 S 和 验证集 V）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 超参数调优
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2', None]
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
# 确保测试集和训练集的列一致（处理预测时的列名问题）
data_test = data_test.drop(columns=['SalePrice'], errors='ignore')  # 确保测试集中没有 'SalePrice'
data_test = data_test.reindex(columns=X.columns, fill_value=0)  # 调整列顺序，与训练集一致

# 对测试集进行预测
rf_preds_test = rf_model.predict(data_test)  # 预测房价

# 创建预测结果 DataFrame
test_ids = data_test['Id']  # 提取测试集的 'Id' 列
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': rf_preds_test})

# 保存预测结果到 Predictions.csv
submission.to_csv('Predictions.csv', index=False)
print("\n测试集预测结果已保存为 Predictions.csv 文件中")

# GBDT 模型
param_grid_gbdt = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

grid_search_gbdt = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid_gbdt,
    scoring='r2',
    cv=5,
    n_jobs=-1
)

grid_search_gbdt.fit(X_train, y_train)
print("GBDT最佳参数:", grid_search_gbdt.best_params_)

# GBDT模型训练
gbdt_model = GradientBoostingRegressor(
    n_estimators=grid_search_gbdt.best_params_['n_estimators'],
    learning_rate=grid_search_gbdt.best_params_['learning_rate'],
    max_depth=grid_search_gbdt.best_params_['max_depth'],
    subsample=grid_search_gbdt.best_params_['subsample'],
    random_state=42
)
gbdt_model.fit(X_train, y_train)

# GBDT模型评估
gbdt_preds_val = gbdt_model.predict(X_val)
gbdt_r2_val = r2_score(y_val, gbdt_preds_val)
gbdt_rmse_val = np.sqrt(mean_squared_error(y_val, gbdt_preds_val))
print("\nGBDT模型验证集性能：")
print(f"R²: {gbdt_r2_val:.2f}, RMSE: {gbdt_rmse_val:.2f}")

# 测试集预测
data_test_gbdt = data_test.drop(columns=['SalePrice'], errors='ignore')
gbdt_preds_test = gbdt_model.predict(data_test_gbdt)
submission_gbdt = pd.DataFrame({'Id': data_test['Id'], 'SalePrice': gbdt_preds_test})
submission_gbdt.to_csv('Predictions_GBDT.csv', index=False)
print("GBDT测试集预测结果已保存为 Predictions_GBDT.csv")

# 随机森林和GBDT性能对比
print("\n模型性能对比：")
print(f"随机森林验证集 R²: {rf_r2_val:.2f}, RMSE: {rf_rmse_val:.2f}")
print(f"GBDT验证集 R²: {gbdt_r2_val:.2f}, RMSE: {gbdt_rmse_val:.2f}")



