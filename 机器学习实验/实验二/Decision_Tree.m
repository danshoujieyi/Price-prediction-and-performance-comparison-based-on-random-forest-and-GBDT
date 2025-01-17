clc;
clear all;

% 阅读代码分析决策树，使用信息增益率将ID3算法改为C4.5算法，36-41行

%西瓜数据集
data = ["青绿","蜷缩","浊响","清晰","凹陷","硬滑","是";
    "乌黑","蜷缩","沉闷","清晰","凹陷","硬滑","是";
    "乌黑","蜷缩","浊响","清晰","凹陷","硬滑","是";
    "青绿","蜷缩","沉闷","清晰","凹陷","硬滑","是";
    "浅白","蜷缩","浊响","清晰","凹陷","硬滑","是";
    "青绿","稍蜷","浊响","清晰","稍凹","软粘","是";
    "乌黑","稍蜷","浊响","稍糊","稍凹","软粘","是";
    "乌黑","稍蜷","浊响","清晰","稍凹","硬滑","是";
    "乌黑","稍蜷","沉闷","稍糊","稍凹","硬滑","否";
    "青绿","硬挺","清脆","清晰","平坦","软粘","否";
    "浅白","硬挺","清脆","模糊","平坦","硬滑","否";
    "浅白","蜷缩","浊响","模糊","平坦","软粘","否";
    "青绿","稍蜷","浊响","稍糊","凹陷","硬滑","否";
    "浅白","稍蜷","沉闷","稍糊","凹陷","硬滑","否";
    "乌黑","稍蜷","浊响","清晰","稍凹","软粘","否";
    "浅白","蜷缩","浊响","模糊","平坦","硬滑","否";
    "青绿","蜷缩","沉闷","稍糊","稍凹","硬滑","否"];

label = ["色泽","根蒂","敲声","纹理","脐部","触感","好瓜"];

% 参数预定义
datasetRate = 1; % 设置训练集占总数据集的比例，1表示使用全部数据作为训练集
dataSize = size(data); % 获取数据集的大小，dataSize(1) 表示样本数量，dataSize(2)表示特征数量

% 数据预处理
% index = randperm(dataSize(1,1),round(datasetRate*(dataSize(1,1)-1)));
index =[1:17]; % 手动设置使用前17个行样本作为训练集索引
trainSet = data(index,:); % 根据索引从数据集中提取训练集
testSet = data;% 将测试集初始化为整个数据集
testSet(index,:) = [];% 从测试集中删除训练集中使用的样本，留下用于测试的样本

impurity = calculateImpurity(trainSet);
disp(impurity);

% 所有标签
deepth = ones(1,dataSize(1,2)-1);  % 初始化每个特征的深度信息，深度初始为1，特征数量为 dataSize(1,2)-1（不包括类别标签）
% 生成树
rootNode = makeTree(label,trainSet,deepth,'null');% 调用自定义函数 makeTree 生成决策树，传入特征标签、训练集、深度信息及初始值 'null'
% 画出决策树
drawTree(rootNode);
