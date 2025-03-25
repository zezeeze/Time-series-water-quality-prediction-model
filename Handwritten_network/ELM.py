import numpy as np
from sklearn.preprocessing import OneHotEncoder

class HiddenLayer:
    def __init__(self, x, num):  # x：输入矩阵   num：隐含层神经元个数
        # 构造函数，初始化隐含层
        row = x.shape[0]
        # 获取输入矩阵x的行数
        columns = x.shape[1]
        # 获取输入矩阵x的列数
        rnd = np.random.RandomState(9999)
        # 创建一个随机数生成器，种子为9999
        self.w = rnd.uniform(-1, 1, (columns, num))
        # 随机初始化权重矩阵w，形状为输入特征数x列数到神经元数num
        self.b = np.zeros([row, num], dtype=float)
        # 初始化偏置b为零矩阵，形状为输入样本数x行数到神经元数num
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            # 随机生成偏置值，范围在-0.4到0.4之间
            for j in range(row):
                self.b[j, i] = rand_b
                # 将生成的偏置值赋给b矩阵对应位置
        self.h = self.sigmoid(np.dot(x, self.w) + self.b)
        # 计算隐含层的输出h，使用sigmoid激活函数
        self.H_ = np.linalg.pinv(self.h)
        # 计算h的伪逆矩阵H_

    def sigmoid(self, x):
        # 定义sigmoid激活函数
        return 1.0 / (1 + np.exp(-x))
        # 返回sigmoid函数值

    def regressor_train(self, T):
        # 定义回归模型的训练函数
        C = 2
        # 设置正则化参数C
        I = len(T)
        # 获取目标值T的长度
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        # 计算中间项sub_former
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        # 计算中间项all_m
        self.beta = np.dot(all_m, T)
        # 计算输出权值beta
        return self.beta
        # 返回beta

    def classifisor_train(self, T):
        # 定义分类模型的训练函数
        en_one = OneHotEncoder()
        # 创建一个OneHotEncoder实例
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()
        # 对目标值T进行独热编码
        C = 3
        # 设置正则化参数C
        I = len(T)
        # 获取目标值T的长度
        sub_former = np.dot(np.transpose(self.h), self.h) + I / C
        # 计算中间项sub_former
        all_m = np.dot(np.linalg.pinv(sub_former), np.transpose(self.h))
        # 计算中间项all_m
        self.beta = np.dot(all_m, T)
        # 计算输出权值beta
        return self.beta
        # 返回beta

    def regressor_test(self, test_x):
        # 定义回归模型的测试函数
        b_row = test_x.shape[0]
        # 获取测试数据test_x的行数
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        # 计算测试数据的隐含层输出h
        result = np.dot(h, self.beta)
        # 计算最终的预测结果
        return result
        # 返回结果

    def classifisor_test(self, test_x):
        # 定义分类模型的测试函数
        b_row = test_x.shape[0]
        # 获取测试数据test_x的行数
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        # 计算测试数据的隐含层输出h
        result = np.dot(h, self.beta)
        # 计算最终的预测结果
        result = [item.tolist().index(max(item.tolist())) for item in result]
        # 将预测结果转换为类别索引
        return result
        # 返回结果