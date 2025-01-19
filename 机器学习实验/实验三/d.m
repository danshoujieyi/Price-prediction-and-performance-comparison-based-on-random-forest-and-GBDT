clc;
clear;
% 输入数据
data = {
    'X1', 50, 50, 9;
    'X2', 28, 9, 4;
    'X3', 17, 15, 3;
    'X4', 25, 40, 5;
    'X5', 28, 40, 2;
    'X6', 50, 50, 1;
    'X7', 50, 40, 9;
    'X8', 50, 40, 9;
    'X9', 40, 40, 5;
    'X10', 50, 50, 9;
    'X11', 50, 50, 5;
    'X12', 50, 50, 9;
    'X13', 40, 40, 9;
    'X14', 40, 32, 17;
    'X15', 50, 50, 9;
};

% 转换为数组
names = data(:, 1);
X = cell2mat(data(:, 2:end));

% 数据标准化（最小-最大归一化）
min_X = min(X);
max_X = max(X);
X_norm = (X - min_X) ./ (max_X - min_X);
disp(X_norm);
% K-Means 聚类（选择 3 个簇）
k = 3;
[idx, C] = kmeans(X_norm, k);

% 显示聚类结果
result = [names, num2cell(X), num2cell(idx)];

% 按聚类结果排序
[~, sortIdx] = sort(idx);
sortedResult = result(sortIdx, :);

% 打印结果
disp('球队名称, 赛事一, 赛事二, 赛事三, 聚类结果');
disp(sortedResult);

% 绘制三维图，不同队伍不同颜色
figure;
scatter3(X_norm(:, 1), X_norm(:, 2), X_norm(:, 3), 100, idx, 'filled');
xlabel('赛事一');
ylabel('赛事二');
zlabel('赛事三');
title('K-Means 聚类结果');
grid on;

% 设置队伍名字，防止重叠
for i = 1:length(names)
    text(X_norm(i, 1), X_norm(i, 2), X_norm(i, 3), names{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end