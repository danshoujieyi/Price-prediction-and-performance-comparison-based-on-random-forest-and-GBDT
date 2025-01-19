
clc; clear;
data = readtable('Salary_Data.csv');

% 提取特征与标签
X = data.YearsExperience; % 特征
y = data.Salary;          % 标签

% 数据划分：80% 用于训练，20% 用于测试
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);  % 80% 训练，20% 测试
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% 标准化训练集特征 
mX = mean(X_train);  % 计算均值
sX = std(X_train);   % 计算标准差
X_train_std = (X_train - mX) / sX; 
X_test_std = (X_test - mX) / sX;  % 使用训练集的均值和标准差标准化测试集

% 增加一列1以包括截距项
X_train_std = [ones(length(X_train_std), 1), X_train_std]; 
X_test_std = [ones(length(X_test_std), 1), X_test_std]; 

% 计算参数 theta
theta = (X_train_std' * X_train_std) \ (X_train_std' * y_train);

% 计算训练集和测试集的拟合值
y_train_fit = X_train_std * theta;
y_test_fit = X_test_std * theta;

% 计算残差平方和 (SSR) 和总离差平方和 (SST)
SSR_train = sum((y_train - y_train_fit).^2);
SST_train = sum((y_train - mean(y_train)).^2);
SSR_test = sum((y_test - y_test_fit).^2);
SST_test = sum((y_test - mean(y_test)).^2);

% 计算拟合优度 R^2
R2_train = 1 - SSR_train / SST_train;
R2_test = 1 - SSR_test / SST_test;

% 计算均方误差 (MSE) 和均方根误差 (RMSE)
MSE_train = mean((y_train - y_train_fit).^2);
RMSE_train = sqrt(MSE_train);
MSE_test = mean((y_test - y_test_fit).^2);
RMSE_test = sqrt(MSE_test);

% 输出结果
disp('训练集参数 theta:');
disp(theta);
disp('训练集拟合优度 R^2:');
disp(R2_train);
disp('测试集拟合优度 R^2:');
disp(R2_test);
disp('训练集均方误差 MSE:');
disp(MSE_train);
disp('训练集均方根误差 RMSE:');
disp(RMSE_train);
disp('测试集均方误差 MSE:');
disp(MSE_test);
disp('测试集均方根误差 RMSE:');
disp(RMSE_test);

% 绘图
figure;
scatter(X, y, 'filled'); % 原始数据点
hold on;

% 绘制拟合线
x_fit = linspace(min(X), max(X), 100)';
x_fit_std = (x_fit - mX) / sX; % 标准化
x_fit_int = [ones(length(x_fit_std), 1), x_fit_std]; % 增加截距项

y_fit_plot = x_fit_int * theta; % 预测拟合值
plot(x_fit, y_fit_plot, 'r-'); % 拟合线

xlabel('工作年限');
ylabel('薪资');
title('线性回归拟合（标准化）');
legend('数据', '拟合线');
grid on;
