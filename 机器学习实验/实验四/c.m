clc;clear
% 学生先对数据进行预处理，将西瓜数据特征转换为数值
% 假设输入数据如下：
% 颜色：青绿=1，浅白=2，乌黑=3
% 敲击声：浊响=1，清脆=2，沉闷=3
% 标签：好瓜=1，坏瓜=0
% 手动输入数据特征：颜色和敲击声
X_train = [
    1 1;  % 青绿 浊响
    2 1;  % 浅白 浊响
    2 2;  % 浅白 清脆
    1 3;  % 青绿 沉闷
    3 3;  % 乌黑 沉闷
    1 2   % 青绿 清脆
]';

% 输出标签（好瓜=1，坏瓜=0）
Y_train = [1 0 0 0 0 1];

% 初始化神经网络参数
input_size = 2;   % 输入层节点数（西瓜数据集有2个特征）
hidden_size = 2;  % 隐藏层节点数
output_size = 1;  % 输出层节点数（好瓜或坏瓜）

% 初始化参数
W1 = randn(hidden_size, input_size) * 0.01;   % 输入层到隐藏层的权重
b1 = zeros(hidden_size, 1);                   % 隐藏层的偏置
W2 = randn(output_size, hidden_size) * 0.01;  % 隐藏层到输出层的权重
b2 = zeros(output_size, 1);                   % 输出层的偏置

% 激活函数（Sigmoid 和 tanh）
sigmoid = @(x) 1 ./ (1 + exp(-x));  
tanh_activation = @(x) tanh(x);   
sigmoid_derivative = @(x) x .* (1 - x);  

% 训练超参数
learning_rate = 0.1;  
num_iterations = 10000;  % 迭代次数

% 梯度下降训练过程
for i = 1:num_iterations
    % 前向传播
    Z1 = W1 * X_train + b1;  % 隐藏层加权输入
    A1 = tanh_activation(Z1);  % 隐藏层激活输出
    Z2 = W2 * A1 + b2;  % 输出层加权输入
    A2 = sigmoid(Z2);  % 输出层激活输出
    
    % 计算代价（交叉熵损失函数）
    m = size(Y_train, 2);  
    cost = -sum(Y_train .* log(A2) + (1 - Y_train) .* log(1 - A2)) / m;

    % 反向传播
    dA2 = A2 - Y_train;  % 输出层误差
    dZ2 = dA2 .* sigmoid_derivative(A2);  % 输出层梯度
    dW2 = (1 / m) * dZ2 * A1';  % 输出层权重梯度
    db2 = (1 / m) * sum(dZ2, 2);  % 输出层偏置梯度
    
    dA1 = W2' * dZ2;  % 隐藏层误差
    dZ1 = dA1 .* (1 - A1.^2);  % 隐藏层梯度
    dW1 = (1 / m) * dZ1 * X_train';  % 输入层到隐藏层的权重梯度
    db1 = (1 / m) * sum(dZ1, 2);  % 隐藏层偏置梯度
    
    % 更新参数
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    
    % 每1000次输出一次代价函数
    if mod(i, 1000) == 0
        fprintf('迭代次数 %d, 代价函数为：%f\n', i, cost);
    end
end
% 输出训练后的权重和偏置
fprintf('\n训练后的权重和偏置：\n');
fprintf('W1 (输入层到隐藏层的权重)：\n');
disp(W1);
fprintf('b1 (隐藏层的偏置)：\n');
disp(b1);
fprintf('W2 (隐藏层到输出层的权重)：\n');
disp(W2);
fprintf('b2 (输出层的偏置)：\n');
disp(b2);

% 模型评估：使用训练好的参数进行预测
% 测试数据：我们将用所有训练数据来进行预测,包含训练集样本
X_test = X_train;  % 测试数据为所有训练数据
Y_test = Y_train;  % 测试标签为训练数据的真实标签

% 测试数据前向传播
Z1_test = W1 * X_test + b1;
A1_test = tanh_activation(Z1_test);
Z2_test = W2 * A1_test + b2;
A2_test = sigmoid(Z2_test);

% 输出预测结果
fprintf('预测结果：\n');
predictions = (A2_test > 0.5);  % 预测值大于0.5为好瓜（1），否则为坏瓜（0）

% 输出每个样本的预测结果
for i = 1:size(A2_test, 2)
    if predictions(1, i) == 1
        fprintf('样本 %d: 好瓜\n', i);
    else
        fprintf('样本 %d: 坏瓜\n', i);
    end
end

% 计算准确率
correct_predictions = sum(predictions == Y_test);  % 正确预测的数量
accuracy = (correct_predictions / length(Y_test)) * 100;
fprintf('准确率：%.2f%%\n', accuracy);

% 对第七个数据一个乌黑和清脆的西瓜进行预测
X_test = [3 2]';  % 乌黑(3) 和 清脆(2)
Z1_test = W1 * X_test + b1;
A1_test = tanh_activation(Z1_test);
Z2_test = W2 * A1_test + b2;
A2_test = sigmoid(Z2_test);

if A2_test > 0.5
    fprintf('当西瓜特征为乌黑清脆时预测结果：好瓜\n');
else
    fprintf('当西瓜特征为乌黑清脆时预测结果：坏瓜\n');
end

