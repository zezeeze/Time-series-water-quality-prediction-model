# coding=UTF-8
import warnings

warnings.filterwarnings("ignore")  # 取消警告
import numpy as np
import sys
from tensorflow.keras.layers import Concatenate, Lambda, RepeatVector, Activation, Flatten, Permute, Multiply, LSTM, \
    GRU, SimpleRNN, Dense, Input, LeakyReLU, Conv1D, Bidirectional, Reshape, MaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, Model
from Handwritten_network.tcn import TCN
from Handwritten_network.Transformer import TimeSeriesTransformer
import xgboost as xgb
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from Handwritten_network.ELM import HiddenLayer
from sklearn.neural_network import MLPRegressor  # 从sklearn.neural_network导入MLPRegressor，用于创建多层感知器回归模型
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score  # 导入评估指标
import os
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
import joblib
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import keras.backend as K  # 导入keras的后端接口
import torch
import torch.nn as nn
from Handwritten_network.defothermodels import run_train
from Handwritten_network.ekan import BiLSTM_ekan,GRU_ekan,lstm_kan,TimeSeriesTransformer_ekan
class modelss:
    def __init__(self, fit_or_not,plotloss, epochs, batch, learning_rate, X_train, X_test, Y_train, Y_test, scaled_tool, n_out,
                 n_in, or_dim):
        self.fit_or_not = fit_or_not
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.scaled = scaled_tool
        self.epochs = epochs
        self.batch = batch
        self.units = 40
        self.plotloss = plotloss
        self.n_in = n_in
        self.or_dim = or_dim
        self.n_out = n_out
        self.learning_rate = learning_rate

        # 将预测结果保存到表格
        folder_path = "预测结果保存"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.result_model_dir = "预测结果保存/"

        # 将训练模型进行保存
        folder_path = "Save_models"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.trained_model_dir = "Save_models/"

    def attention_layer(inputs, time_steps):
        a = Permute((2, 1))(inputs)
        a = Dense(time_steps, activation='softmax')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def S_attention_layer(inputs, single_attention_vector=False):
        # 注意力机制层的实现
        time_steps = K.int_shape(inputs)[1]
        input_dim = K.int_shape(inputs)[2]
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, time_steps))(a)
        a = Dense(time_steps, activation='softmax')(a)
        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul


    def run_CNN_BiLSTM(self, filters, kernel_size, dro, lstm_units):

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.n_in, self.or_dim))

        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.n_in, self.or_dim))


        if self.fit_or_not == 1:

            inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
            conv1d = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
            maxpooling = MaxPooling1D(pool_size=2)(conv1d)
            reshaped = Reshape((-1, filters * maxpooling.shape[1]))(maxpooling)
            bilstm = Bidirectional(LSTM(lstm_units, activation='selu', return_sequences=False))(reshaped)  # BiLSTM层
            # dense_units=128
            # dense1=Dense(units=dense_units, activation='softmax')(bilstm)
            dropout = Dropout(rate=dro)(bilstm)
            outputs = Dense(self.Y_train.shape[1])(dropout)
            # 配置和训练
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics='mae')
            model.summary()
            history = model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch,
                                validation_split=0.25, verbose=2, )

            if self.plotloss == True:
                plt.plot(history.history['loss'], label='train')
                plt.plot(history.history['val_loss'], label='test')
                plt.gcf().subplots_adjust(bottom=0.201)
                plt.legend(loc='upper right', frameon=False)
                plt.xlabel("epochs", fontsize=5)
                plt.ylabel("loss", fontsize=5)
                plt.title(f"The prediction result of MODEL--CNN_BiLSTM_Attention")
                plt.savefig("预测结果保存/CNN_BiLSTM损失函数.png")
                plt.legend()
                plt.show()

            # save the model
            #joblib.dump(model, self.trained_model_dir + 'CNN_BiLSTM_model.joblib')

        #else:
            #model = joblib.load(self.trained_model_dir + 'CNN_BiLSTM_model.joblib')
        # 训练集预测
        Y1_pre = model.predict(self.X_train)
        try:
            Y1_pre = Y1_pre.reshape(self.X_train.shape[0], self.n_out)
        except:
            print('请把fit_or_not改为1，请重新训练模型试试！')
            sys.exit()
        xunlian_predicted_data = self.scaled.inverse_transform(Y1_pre)  # 反归一化
        np.savetxt(self.result_model_dir + "CNN_BiLSTM训练集预测结果.csv", xunlian_predicted_data, delimiter=",")

        Y2_pre = model.predict(self.X_test)
        Y2_pre = Y2_pre.reshape(self.X_test.shape[0], self.n_out)
        ceshi_predicted_data = self.scaled.inverse_transform(Y2_pre)  # 反归一化
        np.savetxt(self.result_model_dir + "CNN_BiLSTM测试集预测结果.csv", ceshi_predicted_data, delimiter=",")
        return xunlian_predicted_data, ceshi_predicted_data


    def run_BiLSTM(self, dro, BiLSTM_units):
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.n_in, self.or_dim))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.n_in, self.or_dim))

        if self.fit_or_not == 1:

            inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
            BiLSTM = Bidirectional(LSTM(BiLSTM_units, activation='selu', return_sequences=False))(inputs)
            # dense_units=128
            # dense1=Dense(units=dense_units, activation='softmax')(BiLSTM)
            dropout = Dropout(rate=dro)(BiLSTM)
            outputs = Dense(self.Y_train.shape[1])(dropout)  # 全连接层
            model = Model(inputs=inputs, outputs=outputs)
            # 配置和训练
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics='mae')
            model.summary()
            history = model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch,
                                validation_split=0.25, verbose=2, )
            if self.plotloss == True:
                plt.plot(history.history['loss'], label='train')
                plt.plot(history.history['val_loss'], label='test')
                plt.gcf().subplots_adjust(bottom=0.201)
                plt.legend(loc='upper right', frameon=False)
                plt.xlabel("epochs", fontsize=5)
                plt.ylabel("loss", fontsize=5)
                plt.title(f"The prediction result of MODEL--BiLSTM")
                plt.savefig("预测结果保存/BiLSTM损失函数.png")
                plt.legend()
                plt.show()

            # save the model
            #joblib.dump(model, self.trained_model_dir + 'BiLSTM_model.joblib')

            #else:
            #model = joblib.load(self.trained_model_dir + 'BiLSTM_model.joblib')
        # 训练集预测
        Y1_pre = model.predict(self.X_train)
        try:
            Y1_pre = Y1_pre.reshape(self.X_train.shape[0], self.n_out)
        except:
            print('请把fit_or_not改为1，请重新训练模型试试！')
            sys.exit()
        xunlian_predicted_data = self.scaled.inverse_transform(Y1_pre)  # 反归一化
        np.savetxt(self.result_model_dir + "BiLSTM训练集预测结果.csv", xunlian_predicted_data, delimiter=",")
        Y2_pre = model.predict(self.X_test)
        Y2_pre = Y2_pre.reshape(self.X_test.shape[0], self.n_out)
        ceshi_predicted_data = self.scaled.inverse_transform(Y2_pre)  # 反归一化
        np.savetxt(self.result_model_dir + "BiLSTM测试集预测结果.csv", ceshi_predicted_data, delimiter=",")
        return xunlian_predicted_data, ceshi_predicted_data

