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
X_norm = (X - min_X)./ (max_X - min_X);

% 手动设置初始聚类中心
initCenters = X_norm([2,11,14],:); % 对应 X2、X11、X14 的归一化数据

% 定义 k，表示聚类的簇数
k = 3;

% 迭代次数
numIterations = 5;

% 存储每次迭代的结果
allIdx = cell(numIterations, 1);
allC = cell(numIterations, 1);

for iter = 1:numIterations
    % 进行 k-means 聚类
    [idx, C] = kmeans(X_norm, k, 'Start',initCenters, 'MaxIter',100);
    
    % 更新初始中心为本次迭代结果的中心
    initCenters = C;
    
    % 存储结果
    allIdx{iter} = idx;
    allC{iter} = C;
end

% 选择最佳结果（可以根据某种准则，这里简单选择最后一次迭代结果）
bestIdx = allIdx{numIterations};
bestC = allC{numIterations};

% 显示聚类结果
result = [names, num2cell(X), num2cell(bestIdx)];

% 按聚类结果排序
[~, sortIdx] = sort(bestIdx);
sortedResult = result(sortIdx, :);

% 打印结果
disp('球队名称, 赛事一, 赛事二, 赛事三, 聚类结果');
disp(sortedResult);

% 绘制三维图，不同队伍不同颜色
figure;
scatter3(X_norm(:, 1), X_norm(:, 2), X_norm(:, 3), 100, bestIdx, 'filled');
xlabel('赛事一');
ylabel('赛事二');
zlabel('赛事三');
title('K-Means 聚类结果');
grid on;

% 设置队伍名字，防止重叠
for i = 1:length(names)
    text(X_norm(i, 1), X_norm(i, 2), X_norm(i, 3), names{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end