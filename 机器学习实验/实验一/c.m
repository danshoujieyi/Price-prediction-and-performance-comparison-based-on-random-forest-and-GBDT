clc;clear;
data = readtable('Salary_Data.csv');

% 提取特征与标签
X = data.YearsExperience; % 特征
y = data.Salary;          % 标签

% 标准化特征 
mX = mean(X);  % 计算均值
sX = std(X);   % 计算标准差
X_std = (X - mX) / sX; 
% 增加一列1以包括截距项
X_std = [ones(length(X_std), 1), X_std]; 

% 计算参数 theta
theta = (X_std' * X_std) \ (X_std' * y);
% 计算拟合值
y_fit = X_std * theta;
% 计算残差平方和 (SSR)
SSR = sum((y - y_fit).^2);
% 计算总离差平方和 (SST)
SST = sum((y - mean(y)).^2);
% 计算拟合优度 R^2
R2 = 1 - SSR / SST;

% 输出
disp('参数 theta:');
disp(theta);
disp('拟合值 y_fit:');
disp(y_fit);
disp('残差平方和 SSR:');
disp(SSR);
disp('总离差平方和 SST:');
disp(SST);
disp('拟合优度 R^2:');
disp(R2);

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




