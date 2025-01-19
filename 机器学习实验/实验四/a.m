% 数据预处理：将西瓜数据特征转换为数值
% 假设输入数据如下：
% 颜色：青绿=1，浅白=2，乌黑=3
% 敲击声：浊响=1，清脆=2，沉闷=3
% 标签：好瓜=1，坏瓜=0

% 输入特征：颜色和敲击声
X = [
    1 1;  % 青绿 浊响
    2 1;  % 浅白 浊响
    2 2;  % 浅白 清脆
    1 3;  % 青绿 沉闷
    3 3;  % 乌黑 沉闷
    1 2   % 青绿 清脆
]';

% 输出标签（好瓜=1，坏瓜=0）
Y = [1 0 0 0 0 1];

% 初始化神经网络参数
input_size = 2;   % 输入层节点数（西瓜数据集有2个特征）
hidden_size = 2;  % 隐藏层节点数
output_size = 1;  % 输出层节点数（好瓜或坏瓜）

% 初始化参数（权重和偏置）
W1 = randn(hidden_size, input_size) * 0.01;  % 输入层到隐藏层的权重
b1 = zeros(hidden_size, 1);                   % 隐藏层的偏置
W2 = randn(output_size, hidden_size) * 0.01;  % 隐藏层到输出层的权重
b2 = zeros(output_size, 1);                   % 输出层的偏置

% 激活函数（Sigmoid 和 tanh）
sigmoid = @(x) 1 ./ (1 + exp(-x));  % Sigmoid 激活函数
tanh_activation = @(x) tanh(x);    % Tanh 激活函数
sigmoid_derivative = @(x) x .* (1 - x);  % Sigmoid 激活函数的导数

% 训练超参数
learning_rate = 0.1;
num_iterations = 10000;  % 迭代次数

% 梯度下降训练过程
for i = 1:num_iterations
    % 前向传播
    Z1 = W1 * X + b1;  % 隐藏层加权输入
    A1 = tanh_activation(Z1);  % 隐藏层激活输出
    Z2 = W2 * A1 + b2;  % 输出层加权输入
    A2 = sigmoid(Z2);  % 输出层激活输出
    
    % 计算代价（交叉熵损失函数）
    m = size(Y, 2);  % 样本数
    cost = -sum(Y .* log(A2) + (1 - Y) .* log(1 - A2)) / m;

    % 反向传播
    dA2 = A2 - Y;  % 输出层误差
    dZ2 = dA2 .* sigmoid_derivative(A2);  % 输出层梯度
    dW2 = (1 / m) * dZ2 * A1';  % 输出层权重梯度
    db2 = (1 / m) * sum(dZ2, 2);  % 输出层偏置梯度
    
    dA1 = W2' * dZ2;  % 隐藏层误差
    dZ1 = dA1 .* (1 - A1.^2);  % 隐藏层梯度
    dW1 = (1 / m) * dZ1 * X';  % 输入层到隐藏层的权重梯度
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

% 模型评估：使用训练好的参数进行预测
% 测试数据：假设我们测试一个乌黑和清脆的西瓜
X_test = [3 2]';  % 乌黑(3) 和 清脆(2)
Z1_test = W1 * X_test + b1;
A1_test = tanh_activation(Z1_test);
Z2_test = W2 * A1_test + b2;
A2_test = sigmoid(Z2_test);

% 输出预测结果
if A2_test > 0.5
    fprintf('预测结果：好瓜\n');
else
    fprintf('预测结果：坏瓜\n');
end




