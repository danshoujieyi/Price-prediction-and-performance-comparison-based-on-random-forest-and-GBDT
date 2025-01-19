clc;clear;
% 学生经过网络搜索和MATLAB官网帮助查阅得知，对于K-means聚类分析，
% MATLAB有专门的函数kmeans(d, n)，前一个参数为数据集，后一个参数为聚类数量;
% 下面学生先使用MATLAB内置函数kmeans(d, n)
% 完成例题4.1对十五支球队分成三个层次水平的聚类分析

% 手动输入PPT例题4.1的数据，每一列代表一支队伍，每一行代表一种赛事
data = [
    50 28 17 25 28 50 50 50 40 50 50 50 40 40 50;
    50 9  15 40 40 50 40 40 40 50 50 50 40 32 50;
     9  4   3  5  2  1  9  9  5  9  5  9  9 17  9
]';

% 聚类的数量，分为3组
number = 3;
% 使用MATLAB内置函数kmeans实现k-means聚类
[idx, C] = kmeans(data, number);
% 聚类结果展示
disp('每个数据点的分类:');
disp(idx);
disp('聚类中心:');
disp(C);

% 构建三维散点图并标注每列数据，实现数据可视化
figure;
hold on;
colors = lines(number); 
for k = 1:number % 按类别绘制散点图
    scatter3(data(idx == k, 1), data(idx == k, 2), data(idx == k, 3), ...
        100, colors(k, :), 'filled', 'DisplayName', ['类别 ', num2str(k)]);
end

% 避免在图表中无法分辨球队，在这里添加标注，
% 记十五支球队分别为T1，T2，T3，T4.....T15，方便观察分析
for i = 1:size(data, 1)
    text(data(i, 1), data(i, 2), data(i, 3), ['T', num2str(i)], ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

% 图表设置
title('K均值聚类分析（三维）');
xlabel('第一维度（特征1）');
ylabel('第二维度（特征2）');
zlabel('第三维度（特征3）');
legend('show'); 
grid on;
view(3); % 输出三维视图
hold off;

