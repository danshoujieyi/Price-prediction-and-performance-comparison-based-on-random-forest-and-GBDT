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

% 设置初始聚类中心 μ_1, μ_2, μ_3
initial_centroids = [
    0.3, 0, 0.19;   % μ_1
    0.7, 0.76, 0.5; % μ_2
    1, 1, 0.5       % μ_3
];

% K-Means 聚类（选择 3 个簇，指定初始聚类中心）
k = 3;
[idx, C] = kmeans(X_norm, k, 'Start', initial_centroids);

% 显示聚类结果
result = [names, num2cell(X), num2cell(idx)];

% 按聚类结果排序
[~, sortIdx] = sort(idx);
sortedResult = result(sortIdx, :);

% 打印结果
disp('球队名称, 赛事一, 赛事二, 赛事三, 聚类结果');
disp(sortedResult);

% 自定义颜色映射（假设相似水平的球队应具有相同颜色）
% 根据 idx 的值进行颜色手动调整，以确保相同水平的球队颜色相近
colorMap = zeros(size(idx, 1), 3); % 初始化颜色映射

% 根据聚类的分配，手动设置颜色
colorMap(idx == 1, :) = repmat([1, 0, 0], sum(idx == 1), 1);  % 红色 - 聚类1
colorMap(idx == 2, :) = repmat([0, 1, 0], sum(idx == 2), 1);  % 绿色 - 聚类2
colorMap(idx == 3, :) = repmat([0, 0, 1], sum(idx == 3), 1);  % 蓝色 - 聚类3

% 绘制三维图
figure;
scatter3(X_norm(:, 1), X_norm(:, 2), X_norm(:, 3), 50, colorMap, 'filled', 'MarkerFaceAlpha', 0.8); % 减小点大小并设置透明度
xlabel('赛事一');
ylabel('赛事二');
zlabel('赛事三');
title('K-Means 聚类结果');
grid on;

% 设置队伍名字
for i = 1:length(names)
    t = text(X_norm(i, 1), X_norm(i, 2), X_norm(i, 3), names{i}, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left'); % 调整文本位置
    t.Color = [0.5,0.5,0.5]; % 设置文本半透明
end
view(30,30); % 调整视角