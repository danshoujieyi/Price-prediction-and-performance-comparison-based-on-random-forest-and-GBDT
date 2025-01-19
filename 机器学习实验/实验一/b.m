clc; clear;

% 读取数据并保持原始列名
data = readtable('50_Startups.csv', 'VariableNamingRule', 'preserve');

% 检查列名，确保正确读取数据
disp(data.Properties.VariableNames);

% 提取数值型特征
X_numeric = data{:, {'R&D Spend', 'Administration', 'Marketing Spend'}}; % 数值型特征

% 对分类变量（State）进行编码
state_idx = grp2idx(data.State); % 将 'State' 转换为数值型

% 将分类变量添加到数值型特征中
X = [X_numeric, state_idx];

% 提取目标变量
y = data.Profit;

% 标准化数值型特征
X(:, 1:3) = (X(:, 1:3) - mean(X(:, 1:3))) ./ std(X(:, 1:3)); % 标准化数值变量

% 划分数据集为训练集和测试集
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% 训练集，20% 测试集
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% 增加一列1以包括截距项（常数项）
X_train = [ones(size(X_train, 1), 1), X_train];
X_test = [ones(size(X_test, 1), 1), X_test];

% 训练多元线性回归模型，计算回归系数
theta = (X_train' * X_train) \ (X_train' * y_train);

% 预测训练集和测试集的结果
y_train_pred = X_train * theta; % 训练集预测值
y_test_pred = X_test * theta;   % 测试集预测值

% 计算均方误差 (MSE)
MSE_train = mean((y_train - y_train_pred).^2);
MSE_test = mean((y_test - y_test_pred).^2);

% 输出结果
disp('训练集均方误差 (MSE):');
disp(MSE_train);
disp('测试集均方误差 (MSE):');
disp(MSE_test);

% 可视化实际值和预测值的对比
figure;
subplot(1, 2, 1);
scatter(y_train, y_train_pred, 'filled');
title('训练集实际值与预测值对比');
xlabel('实际值');
ylabel('预测值');
grid on;

subplot(1, 2, 2);
scatter(y_test, y_test_pred, 'filled');
title('测试集实际值与预测值对比');
xlabel('实际值');
ylabel('预测值');
grid on;
