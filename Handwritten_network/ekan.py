import torch.nn as nn
import torch
from Handwritten_network.kan import KAN

# 定义 BiLSTM-KAN 模型
class BiLSTM_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM_ekan, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 双向LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 注意：因为是双向，所以全连接层的输入是 hidden_dim * 2
        self.e_kan = KAN([hidden_dim* 2, 10, output_dim])

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播双向LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 取双向的最后一个时间步的输出
        out = self.e_kan(out[:, -1, :])
        return out


# 定义 GRU-KAN 模型
class GRU_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU_ekan, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，与LSTM相同
        self.e_kan = KAN([hidden_dim,10, output_dim])
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0.detach())
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.e_kan(out[:, -1, :])
        return out


class lstm_kan(nn.Module): # 定义一个名为lstm的类，继承自nn.Module
    # 初始化函数，定义模型的各层和参数
    def __init__(self, input_size, hidden_size, num_layers , output_size , dropout, batch_first=True):
        super(lstm_kan, self).__init__()  # 调用父类的构造函数
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size # 设置LSTM的隐藏层大小
        self.input_size = input_size # 设置LSTM的输入特征维度
        self.num_layers = num_layers # 设置LSTM的层数
        self.output_size = output_size  # 设置输出的维度
        self.dropout = dropout  # 设置Dropout概率
        self.batch_first = batch_first # 设置batch_first参数，决定输入输出张量的维度顺序
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout ) # 定义LSTM层
        # self.linear = nn.Linear(self.hidden_size, self.output_size) # 定义线性层
        self.kan = KAN([self.hidden_size,16,self.output_size])

    def forward(self, x):  # 前向传播函数
         # 通过LSTM层，得到输出out和隐藏状态hidden, cell
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # print(f"hidden.shape: {hidden.shape}")
        # print(f"hidden[-1].shape: {hidden[-1].shape}")
        # out = self.linear(hidden.reshape(a * b, c))
        # out = self.linear(hidden) # 将hidden通过线性层
        out = self.kan(hidden[-1])
        return out #  只保留最后一个时间步的输出


# 定义 Transformer-KAN 模型

class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs,hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer_ekan, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space=hidden_space

        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.e_kan = KAN([hidden_space, 10, num_outputs])
        self.transform_layer=nn.Linear(input_dim, hidden_space)

    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)

        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 取最后一个时间步的输出
        x = x[-1, :, :]

        # 全连接层生成最终输出
        x = self.e_kan(x)
        return x

