data = [50 28 17 25 28 50 50 50 40 50 50 50 40 40 50;
        50 9 15 40 40 50 40 40 40 50 50 50 40 32 50;
        9 4 3 5 2 1 9 9 5 9 5 9 9 17 9];

% 归一化处理
normalizedData = zeros(size(data));
for i = 1:size(data,1)
    minVal = min(data(i,:));
    maxVal = max(data(i,:));
    normalizedData(i,:) = (data(i,:) - minVal) / (maxVal - minVal);
end

% 输出归一化后的数据（表 4-2）
disp('归一化后的数据：');
disp(normalizedData);

% 初始聚类中心
mu1 = [0.3,0,0.19];
mu2 = [0.7,0.76,0.5];
mu3 = [1,1,0.5];

% 计算每个数据点到每个聚类中心的欧氏距离
distances = zeros(size(normalizedData,2),3);
for i = 1:size(normalizedData,2)
    distances(i,1) = norm(normalizedData(:,i) - mu1);
    distances(i,2) = norm(normalizedData(:,i) - mu2);
    distances(i,3) = norm(normalizedData(:,i) - mu3);
end

% 以更清晰的形式输出结果
fprintf('球队编号\t到中心1距离\t到中心2距离\t到中心3距离\n');
for i = 1:size(normalizedData,2)
    fprintf('%d\t%.4f\t%.4f\t%.4f\n', i, distances(i,1), distances(i,2), distances(i,3));
end