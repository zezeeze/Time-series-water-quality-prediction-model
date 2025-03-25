from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score  # 导入额外的评估指标
from matplotlib import rcParams
from math import sqrt  # 从math模块导入sqrt函数，用于计算平方根
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
from prettytable import PrettyTable #可以优美的打印表格结果
import numpy as np  # 导入numpy模块，用于数值计算
from sklearn.preprocessing import MinMaxScaler  # 导入sklearn中的MinMaxScaler，用于特征缩放
def data_collation(data, n_in, n_out, or_dim, scroll_window, num_samples,train_ratio):

    # data是传递进来的原始数据。
    res = np.zeros((num_samples,n_in*or_dim+n_out))
    for i in range(0, num_samples):
        h1 = data[scroll_window*i: n_in+scroll_window*i,0:or_dim]
        h2 = h1.reshape( 1, n_in*or_dim)
        h3 = data[n_in+scroll_window*(i) : n_in+scroll_window*(i)+n_out,-1].T
        h4 = h3[np.newaxis, :]
        h5 = np.hstack((h2,h4))
        res[i,:] = h5


    values = np.array(res)

    n_train_number = int(num_samples * train_ratio)

    Xtrain = values[:n_train_number, :n_in * or_dim]
    Ytrain = values[:n_train_number, n_in * or_dim:]
    Xtest = values[n_train_number:, :n_in * or_dim]
    Ytest = values[n_train_number:, n_in * or_dim:]
    # 对训练集和测试集进行归一化
    m_in = MinMaxScaler()
    vp_train = m_in.fit_transform(Xtrain)  # 注意fit_transform() 和 transform()的区别
    vp_test = m_in.transform(Xtest)  # 注意fit_transform() 和 transform()的区别

    m_out = MinMaxScaler()
    vt_train = m_out.fit_transform(Ytrain)  # 注意fit_transform() 和 transform()的区别
    vt_test = m_out.transform(Ytest)  # 注意fit_transform() 和 transform()的区别

    return vp_train, vp_test, vt_train, vt_test, m_out,Ytest

def mape(y_true, y_pred):
    # 定义一个计算平均绝对百分比误差（MAPE）的函数。
    record = []
    for index in range(len(y_true)):
        # 遍历实际值和预测值。
        if y_true[index] == 0:
            temp_mape = 0
        else:
            temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        # 计算单个预测的MAPE。
        record.append(temp_mape)
        # 将MAPE添加到记录列表中。
    return np.mean(record) * 100
    # 返回所有记录的平均值，乘以100得到百分比。

def evaluate_forecasts(Ytest, predicted_data, n_out):
    # 定义一个函数来评估预测的性能。
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    mse = []
    rmse = []
    mae = []
    MApe = []
    r2 = []
    # 初始化存储各个评估指标的字典。
    table = PrettyTable(['测试集指标','MSE', 'RMSE', 'MAE', 'MAPE','R2'])
    # 计算均方误差（MSE）。
    mse = mean_squared_error(Ytest, predicted_data)
    mse_dic.append(mse)
    # 计算均方根误差（RMSE）。
    rmse = sqrt(mean_squared_error(Ytest, predicted_data))
    rmse_dic.append(rmse)
    # 计算平均绝对误差（MAE）。
    mae = mean_absolute_error(Ytest, predicted_data)
    mae_dic.append(mae)
    # 计算平均绝对百分比误差（MAPE）。
    MApe = mape(Ytest, predicted_data)
    mape_dic.append(MApe)
    # 计算R平方值（R2）。
    r2 = r2_score(Ytest, predicted_data)
    r2_dic.append(r2)
    strr = '预测结果指标：'
    table.add_row([strr, mse, rmse, mae, str(MApe) + '%', str(r2 * 100) + '%'])
    print(table)  # 显示预测指标数值
    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table

def plot_result(valuess,ceshi_predicted_data, Ytest, n_in,n_out,flag):
    ## 画测试集的预测结果图
    # 配置一下画图的参数
    config = {
        "font.family": 'serif',
        "font.size": 10,  # 相当于小四大小
        "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
        "font.serif": ['Times New Roman'],  # Times New Roman
        'axes.unicode_minus': False  # 处理负号，即-号
    }
    rcParams.update(config)
    plt.ion()
    plt.rcParams['axes.unicode_minus'] = False
    # 设置matplotlib的配置，用来正常显示负号。
    # 创建一个图形对象，并设置大小为10x2英寸，分辨率为300dpi。
    fig, ax = plt.subplots(figsize=(5, 1.5), dpi=250)
    # plt.figure(figsize=(10, 2), dpi=300)
    ax.grid(ls="--", lw=0.1, color="#4E616C")
    x = range(1, len(ceshi_predicted_data) + 1)
    # 创建x轴的值，从1到实际值列表的长度。
    # plt.xticks(x[::int((len(ceshi_predicted_data)+1))])
    # 设置x轴的刻度，每几个点显示一个刻度。
    plt.tick_params(labelsize=5)  # 改变刻度字体大小
    # 设置刻度标签的字体大小。
    if n_out == 1:  # 单步预测
        # 调用evaluate_forecasts函数。
        # 传递实际值(inv_y)、预测值(inv_yhat)以及预测的步数(n_out)作为参数。
        # 此函数将计算预测步长的RMSE、MAE、MAPE和R2值。
        mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest, ceshi_predicted_data, n_out)

        ax.plot(x, ceshi_predicted_data, marker="*", mfc="white", ms=1, linestyle="--", linewidth=0.5,
                label='predict')
        ax.plot(x, Ytest, marker="o", mfc="white", ms=1, linestyle="-", linewidth=0.5, label='Real')

        # plt.plot(x, ceshi_predicted_data,linestyle="--",linewidth=0.5, label='predict')
        # plt.plot(x, Ytest, linestyle="-", linewidth=0.5, label='Real')
    elif n_out > 1:  # 多步预测，当程序为超前多步预测时，因为每个测试集都会预测出一段序列，故不可能对所有序列都绘制一个预测结果图，这里取出测试集中的某一个样本进行预测结果的绘制。
        r2_dicc = []
        for i in (range(ceshi_predicted_data.shape[0])):
            r2 = r2_score(Ytest[i, :], ceshi_predicted_data[i, :])
            r2_dicc.append(r2)
        example = r2_dicc.index(max(r2_dicc))  # 选R2指标最好的那个画图
        # example = np.random.choice(range(ceshi_predicted_data.shape[0]))
        # 要查找的子数组
        # 找到这组测试集所在的位置，把历史数据也画上。
        subarray = Ytest[example, :].tolist()
        # 找出子数组在 A中的位置
        A = valuess[:, -1]
        A = A.tolist()
        positions = []
        for i in range(len(A) - len(subarray) + 1):
            if A[i:i + len(subarray)] == subarray:
                positions.append(i)
        ax.plot(range(1, n_in + 2), valuess[int(np.array(positions)) - n_in:int(np.array(positions)) + 1, -1],
                linestyle="-", linewidth=0.8, label='history')
        ax.plot(range(n_in + 1, ceshi_predicted_data.shape[1] + n_in + 1), ceshi_predicted_data[example, :],
                linestyle="-", linewidth=0.5, label='predict', mfc="white", marker='*', ms=1)
        ax.plot(range(n_in + 1, ceshi_predicted_data.shape[1] + n_in + 1), Ytest[example, :], linestyle="-",
                linewidth=0.5, label='Real', mfc="white", marker='o', ms=2)
        # 调用evaluate_forecasts函数。
        # 传递实际值(inv_y)、预测值(inv_yhat)以及预测的步数(n_out)作为参数。
        # 此函数将计算预测步长的RMSE、MAE、MAPE和R2值。
        mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest[example, :],
                                                                                 ceshi_predicted_data[example, :],
                                                                                 n_out)

    plt.rcParams.update({'font.size': 5})  # 改变图例里面的字体大小
    # 更新图例的字体大小。
    plt.gcf().subplots_adjust(bottom=0.201)
    plt.legend(loc='upper right', frameon=False)
    # 显示图例，位置在图形的右上角，没有边框。
    plt.xlabel("Sample points", fontsize=5)
    # 设置x轴标签为"样本点"，字体大小为5。
    plt.ylabel("value", fontsize=5)
    plt.title(f"The prediction result of MODEL-- {flag}")
    # 使用plt.savefig()保存图像
    plt.savefig("预测结果保存/" + flag + "预测结果.png")
    # plt.xlim(xmin=600, xmax=700)  # 显示600-1000的值   局部放大有利于观察
    # 如果需要，可以取消注释这行代码，以局部放大显示600到700之间的值。
    # plt.savefig('figure/预测结果图.png')
    # 如果需要，可以取消注释这行代码，以将图形保存为PNG文件。
    plt.ioff()  # 关闭交互模式
    plt.show()
    # 显示图形。

