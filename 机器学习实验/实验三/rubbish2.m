clc;clear;
% 学生按照PPT示例的步骤来实现对十五支球队的聚类分析
% 归一化-指定聚类中心-聚类-绘图
% 输入数据表，表示球队名称及赛事成绩
teamData = {
    'T1', 50, 50, 9;
    'T2', 28, 9, 4;
    'T3', 17, 15, 3;
    'T4', 25, 40, 5;
    'T5', 28, 40, 2;
    'T6', 50, 50, 1;
    'T7', 50, 40, 9;
    'T8', 50, 40, 9;
    'T9', 40, 40, 5;
    'T10', 50, 50, 9;
    'T11', 50, 50, 5;
    'T12', 50, 50, 9;
    'T13', 40, 40, 9;
    'T14', 40, 32, 17;
    'T15', 50, 50, 9;
};
% 将数据拆分为名称和数值部分
teamNames = teamData(:, 1); 
performanceData = cell2mat(teamData(:, 2:end)); 
% 对数据进行归一化处理
dataMin = min(performanceData); % 每列的最小值
dataMax = max(performanceData); % 每列的最大值
normalizedData = (performanceData - dataMin) ./ (dataMax - dataMin); 

% 指定初始聚类中心
initialCenters = [
    0.3, 0.0, 0.19;   % 队伍2
    0.7, 0.76, 0.5;   % 队伍13
    1.0, 1.0, 0.5;   % 队伍15
];

% 使用 K-Means 算法进行聚类分析
numClusters = 3; 
[clusterLabels, clusterCenters] = kmeans(normalizedData, numClusters, 'Start', initialCenters);

% 组合结果，方便排序和输出
analysisResults = [teamNames, num2cell(performanceData), num2cell(clusterLabels)];
[~, sortedIndices] = sort(clusterLabels);
sortedResults = analysisResults(sortedIndices, :);

% 打印最终结果
disp('队伍, 比赛1, 比赛2, 比赛3, 聚类结果:');
disp(sortedResults);

% 定义颜色映射，根据分类分配颜色
% 每个分类分配特定颜色
clusterColors = zeros(size(clusterLabels, 1), 3); % 初始化颜色矩阵
clusterColors(clusterLabels == 1, :) = repmat([0.9, 0.2, 0.2], sum(clusterLabels == 1), 1); % 分类1为红色
clusterColors(clusterLabels == 2, :) = repmat([0.2, 0.9, 0.2], sum(clusterLabels == 2), 1); % 分类2为绿色
clusterColors(clusterLabels == 3, :) = repmat([0.2, 0.2, 0.9], sum(clusterLabels == 3), 1); % 分类3为蓝色

% 绘制三维聚类图
figure;
scatter3(normalizedData(:, 1), normalizedData(:, 2), normalizedData(:, 3), ...
    70, clusterColors, 'filled', 'MarkerFaceAlpha', 0.7); % 设置透明度和点大小
xlabel('比赛1 (归一化成绩)');
ylabel('比赛2 (归一化成绩)');
zlabel('比赛3 (归一化成绩)');
title('球队聚类分析 (K-Means)');
grid on;

% 避免在图表中无法分辨球队，在这里添加标注，
% 记十五支球队分别为T1，T2，T3，T4.....T15，方便观察分析
for i = 1:length(teamNames)
    teamLabel = text(normalizedData(i, 1), normalizedData(i, 2), normalizedData(i, 3), teamNames{i}, ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
    teamLabel.Color = [0.4, 0.4, 0.4];
end

view(45, 35); 
